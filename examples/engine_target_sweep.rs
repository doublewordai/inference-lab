//! Engine-target sweep + ground-truth validation for the Llama-3.1-8B +
//! EAGLE3 GH200 config.
//!
//! Part 1 — VALIDATION. Before the sweep's headroom targets mean anything,
//! the simulator's fixed-k predictions are checked against the engine's
//! measured static-k envelope: a ShareGPT closed-loop bench of the real
//! sglang/GH200 deployment (job calib-5175604, 2026-06-12), conc x k in
//! {1,4,16,64,256} x {0,1,2,3,4,6,8}. The validation scenario mirrors the
//! bench, not the sweep: real prefill (no prefilled-arrival shortcut),
//! per-request ISL/OSL drawn lognormally and matched to each bench rung's
//! realized means (from the bench jsonl metadata), and throughput counted
//! the way the bench counts it — completed-request output tokens over wall
//! time from t=0, ramp and drain included.
//!
//! Part 2 — TARGET SWEEP. The decode pool in isolation (requests arrive
//! prefilled, ISL=1024/OSL=256 — the cost table's own calibration shape),
//! concurrency 1 -> 256, comparing at each concurrency:
//!   - no speculation
//!   - the best fixed draft length k (over the table's MEASURED widths only;
//!     widths with no measured rows are skipped, not roofline-priced — a
//!     fantasy column priced from the optimistic roofline otherwise wins)
//!   - GoodputBudget   (homogeneous width, bank-mean acceptance signal)
//!   - GatedAggregate  (per-sequence signal, one batch-uniform width -- the
//!                      engine-realizable aggregation)
//!   - GatedBudget     (per-sequence signal, ragged per-sequence widths --
//!                      the unconstrained upper reference)
//! and reports GatedAggregate's headroom over the best fixed k.
//!
//! The scenario comes entirely from the config file: acceptance is the real
//! EAGLE3-on-Llama trace bank and step cost the measured
//! (batch_size, num_draft_tokens) -> step_seconds grid it points at. The
//! no-spec baseline runs as a gamma=0 speculative config so it is priced
//! from the same table's plain-decode (num_draft_tokens = 1) rows — fully
//! commensurate with the speculative columns. Decode batches beyond the
//! table's measured batch range extrapolate nearest-neighbor and are
//! flagged; draft depths with no measured rows are excluded everywhere
//! (fixed columns and policy candidate sets alike).
//!
//! Run: `cargo run --release --example engine_target_sweep --no-default-features`
//!
//! Config selection: `SWEEP_CONFIG=<path>` env var, else the first CLI arg,
//! else the default Llama config. The Part 1 validation tables (measured
//! ShareGPT envelope, native-k acceptance) are ground truth for the default
//! Llama/EAGLE3 deployment only and are skipped for any other config; the
//! bank acceptance summary and Part 2 target sweep run for every config.

use inference_lab::compute::MeasuredCostTable;
use inference_lab::config::{
    AcceptanceModel, ClusterSpec, Config, GammaPolicy, SpeculativeConfig, TraceBank,
};
use inference_lab::request::Request;
use inference_lab::simulation::{simulate_closed_loop, Engine, Topology};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, LogNormal};
use rayon::prelude::*;
use std::collections::BTreeMap;

const DEFAULT_CONFIG_PATH: &str = "configs/llama31-8b-eagle3-gh200.toml";

/// `SWEEP_CONFIG` env var, else first CLI arg, else the default Llama config.
fn config_path() -> String {
    std::env::var("SWEEP_CONFIG")
        .ok()
        .or_else(|| std::env::args().nth(1))
        .unwrap_or_else(|| DEFAULT_CONFIG_PATH.to_string())
}

// ---------------------------------------------------------------------------
// Ground truth: ShareGPT closed-loop bench of the real engine (sglang on
// GH200, same hardware/engine config the measured cost table was fitted
// from). Source: job calib-5175604, 2026-06-12 — output token throughput
// (tok/s) per (concurrency, static k). k=0 is the no-speculation boot.
// ---------------------------------------------------------------------------
const VAL_CONCS: [u32; 5] = [1, 4, 16, 64, 256];
const VAL_KS: [u32; 7] = [0, 1, 2, 3, 4, 6, 8];
const MEASURED_TPS: [[f64; 7]; 5] = [
    [178.2, 202.9, 228.0, 233.9, 236.1, 229.1, 215.0],
    [573.2, 605.5, 689.6, 717.1, 723.1, 698.5, 663.0],
    [1599.5, 991.8, 1046.3, 1053.0, 1038.0, 1004.8, 961.0],
    [3813.8, 2041.0, 2223.9, 2242.0, 2216.4, 1982.1, 1964.3],
    [6714.8, 2779.2, 3056.1, 3127.0, 3081.5, 2897.9, 2670.7],
];
// Realized per-rung mean (ISL, OSL) of the bench's ShareGPT sample, read
// from the bench jsonl metadata (total_input_tokens / completed etc. of the
// k=0 legs). The bench reuses one prompt sample per rung across k.
const RUNG_LENS: [(f64, f64); 5] =
    [(304.0, 211.0), (304.0, 211.0), (316.0, 211.0), (289.0, 189.0), (313.0, 190.0)];
// ShareGPT length spread: sigma of log-length, assumed 1.0 for both ISL and
// OSL. Cross-checked against the bench's mean/median E2E ratio at conc=1
// (1183ms/695ms -> sigma_log(OSL) ~ 1.04).
const SIGMA_LOG: f64 = 1.0;
// Engine-measured native-k acceptance, EXCLUDING the bonus token: the bench
// jsonl's accept_length - 1, averaged over the five concurrencies of each
// static-k leg (same job). Conc-variation is < +/-1%.
const NATIVE_ACCEPT: [(u32, f64); 6] =
    [(1, 0.508), (2, 0.763), (3, 0.887), (4, 0.963), (6, 1.035), (8, 1.057)];

fn topology(cfg: &Config) -> Topology {
    let cluster = ClusterSpec {
        hardware: cfg.hardware.clone(),
        parallel: cfg.parallel.clone(),
        comms: None,
        num_workers: 1,
        node: 0,
    };
    Topology::aggregated(cluster, cfg.model.clone(), cfg.scheduler.clone()).expect("topo")
}

#[derive(Clone, Copy)]
enum Policy {
    NoSpec,
    Fixed(u32),
    Budget(u32),
    GatedAggregate(u32),
    Gated(u32),
}

fn spec_for(p: Policy, base: &SpeculativeConfig) -> Option<SpeculativeConfig> {
    let mk = |gamma: u32, policy: GammaPolicy| {
        let mut s = base.clone();
        s.gamma = gamma;
        s.policy = policy;
        Some(s)
    };
    match p {
        // gamma = 0: behaviourally identical to no speculation (verify width
        // 1, zero drafts), but keeps the measured cost table attached so the
        // baseline's decode steps are priced from the table's plain-decode
        // (num_draft_tokens = 1) rows — commensurate with the spec columns
        // instead of the analytic roofline.
        Policy::NoSpec => mk(0, GammaPolicy::Fixed),
        Policy::Fixed(g) => mk(g, GammaPolicy::Fixed),
        Policy::Budget(g) => mk(g, GammaPolicy::GoodputBudget),
        Policy::GatedAggregate(g) => mk(g, GammaPolicy::GatedAggregate),
        Policy::Gated(g) => mk(g, GammaPolicy::GatedBudget),
    }
}

/// -> (decode tokens/s, mean decode batch). Pure-decode target: fixed-shape
/// requests arriving prefilled, steady-state window.
fn run(cfg: &Config, conc: u32, isl: u32, osl: u32, p: Policy) -> (f64, f64) {
    let base = cfg.speculative.as_ref().expect("config has [speculative]");
    let total = (conc * 2).max(1000);
    let warmup = conc / 2;
    let res = simulate_closed_loop(
        topology(cfg),
        conc,
        isl,
        osl,
        total,
        warmup,
        spec_for(p, base),
        7,
        true, // skip_prefill: pure-decode target, prefill not competing
    )
    .expect("run");
    let dbatch = res.mean_batch_per_pool.first().copied().flatten().unwrap_or(0.0);
    (res.throughput() * osl as f64, dbatch)
}

/// Closed-loop validation run mirroring the ShareGPT bench: real prefill
/// (requests arrive unprefilled and compete for the engine), per-request
/// ISL/OSL sampled lognormally around the bench rung's realized means
/// (SIGMA_LOG spread), and throughput counted as the bench counts it:
/// total completed-request output tokens / wall time from t=0 to the last
/// completion (ramp + drain included; no steady-state windowing). The sim
/// has no client: a finished slot resubmits instantly, where the bench pays
/// response-transfer + next-request turnaround per completion — so the sim
/// is expected to sit slightly above the bench at high churn.
fn run_validation(
    cfg: &Config,
    conc: u32,
    k: u32,
    mean_isl: f64,
    mean_osl: f64,
    num_requests: u32,
    seed: u64,
) -> f64 {
    let base = cfg.speculative.as_ref().expect("config has [speculative]");
    let p = if k == 0 { Policy::NoSpec } else { Policy::Fixed(k) };
    let mut engine = Engine::new(topology(cfg));
    engine.enable_speculative(spec_for(p, base).expect("spec"), seed);

    let isl_dist =
        LogNormal::new(mean_isl.ln() - SIGMA_LOG * SIGMA_LOG / 2.0, SIGMA_LOG).expect("isl dist");
    let osl_dist =
        LogNormal::new(mean_osl.ln() - SIGMA_LOG * SIGMA_LOG / 2.0, SIGMA_LOG).expect("osl dist");
    let mut rng = StdRng::seed_from_u64(seed);
    // Length clamps from the bench's own latency percentiles: at conc=1/k=0,
    // OSL_p50 ~ (medE2E - medTTFT)/medTPOT = 124, OSL_p90 ~ 447, OSL_p99 ~
    // 1125 — a clean sigma_log = 1 lognormal up to ~p99, truncated above
    // (ShareGPT turns are bounded). An unclamped lognormal tail (p99.9 ~
    // 2.5k) manufactures drain-phase stragglers that depress the
    // full-window throughput well below anything the bench's sample
    // contains, so clamp at the empirical ~p99.
    let sample = |rng: &mut StdRng| {
        let isl = isl_dist.sample(rng).round().clamp(4.0, 1200.0) as u32;
        let osl = osl_dist.sample(rng).round().clamp(4.0, 1200.0) as u32;
        (isl, osl)
    };

    let mut submitted = 0u32;
    for _ in 0..conc.min(num_requests) {
        let (isl, osl) = sample(&mut rng);
        engine.submit(Request::new(format!("v{submitted}"), 0, 0.0, isl, osl));
        submitted += 1;
    }
    let mut out_tokens: u64 = 0;
    let mut completed = 0u32;
    let mut t_last = 0.0f64;
    while completed < num_requests {
        if engine.next_event_time().is_none() {
            break; // queue drained (shouldn't happen before num_requests)
        }
        let outcome = engine.step().expect("step");
        for timing in outcome.completions {
            completed += 1;
            out_tokens += timing.num_output_tokens as u64;
            t_last = timing.completion_time;
            if submitted < num_requests {
                let (isl, osl) = sample(&mut rng);
                engine.submit(Request::new(
                    format!("v{submitted}"),
                    0,
                    timing.completion_time,
                    isl,
                    osl,
                ));
                submitted += 1;
            }
        }
    }
    out_tokens as f64 / t_last.max(1e-9)
}

fn main() {
    let config_path = config_path();
    // The hardcoded ground-truth tables (MEASURED_TPS, NATIVE_ACCEPT) belong
    // to the default Llama/EAGLE3 GH200 deployment; for any other config the
    // validation part is skipped rather than compared against the wrong bench.
    let is_default_config = config_path == DEFAULT_CONFIG_PATH;
    let cfg = Config::from_file(&config_path).expect("config");
    let base = cfg.speculative.as_ref().expect("config has [speculative]");
    let gamma_max = base.gamma;

    let mc = base.measured_cost.as_ref().expect("config has measured_cost");
    let table = MeasuredCostTable::load(&mc.path).expect("measured cost table");
    let bank_path = match &base.acceptance {
        AcceptanceModel::TraceRounds { path } => path.clone(),
        other => panic!("expected trace_rounds acceptance, got {other:?}"),
    };
    let bank = TraceBank::load(&bank_path).expect("trace bank");

    // Fixed-k candidates: the table's MEASURED draft depths only. Depths the
    // grid lacks (here ndt = 6 and 8, i.e. k = 5 and 7) are skipped outright
    // — pricing them via the optimistic analytic roofline produced fantasy
    // columns that won the argmax. The engine applies the same exclusion to
    // the adaptive policies' candidate sets.
    let fixed: Vec<u32> = (1..=gamma_max).filter(|&g| table.has_draft(g)).collect();
    let skipped: Vec<String> = (1..=gamma_max)
        .filter(|&g| !table.has_draft(g))
        .map(|g| format!("k={g} (ndt={})", g + 1))
        .collect();

    println!("Engine-target sweep + validation: {}", cfg.model_name());
    println!("config:        {config_path}");
    println!(
        "{} TP{}, gamma_max={gamma_max} c_draft={}",
        cfg.hardware.name, cfg.parallel.tp, base.draft_cost_frac
    );
    println!("acceptance:    trace_rounds ({bank_path})");
    println!(
        "measured cost: {} (ref_seq_len {:?}; unmeasured widths excluded)",
        mc.path, mc.ref_seq_len
    );
    if !skipped.is_empty() {
        println!(
            "NOTE: no measured rows at {}; those draft depths are SKIPPED \
             (excluded from fixed columns and policy candidate sets).",
            skipped.join(", ")
        );
    }
    println!();

    // -----------------------------------------------------------------
    // Acceptance sanity: the bank's E[accepted | g] (mean over rounds of
    // min(commits, g); commits EXCLUDE the bonus token) against the
    // engine's measured native-k accept (bench accept_length - 1). The
    // bank rows are k=8 rounds, so small g carries a truncation bias:
    // min(commits@8, g) undercounts a run that would have continued past
    // an early k=8 chain break. Bias is largest at g=1 and vanishes by
    // g=8.
    // -----------------------------------------------------------------
    println!("E[accepted | g] (excl. bonus): trace bank vs engine native-k");
    println!("{:>3} {:>10} {:>10} {:>8}", "g", "bank", "native-k", "d(1+E)%");
    // NATIVE_ACCEPT is the default Llama deployment's bench; for other
    // configs only the bank column is meaningful.
    let native: BTreeMap<u32, f64> = if is_default_config {
        NATIVE_ACCEPT.into_iter().collect()
    } else {
        BTreeMap::new()
    };
    for g in 1..=gamma_max {
        let e_bank = bank.expected_accepted(g);
        match native.get(&g) {
            Some(&e_nat) => {
                let d = 100.0 * ((1.0 + e_bank) / (1.0 + e_nat) - 1.0);
                println!("{g:>3} {e_bank:>10.3} {e_nat:>10.3} {d:>+8.1}");
            }
            None => println!("{g:>3} {e_bank:>10.3} {:>10} {:>8}", "-", "-"),
        }
    }
    if is_default_config {
        println!(
            "(native-k = bench accept_length - 1, ShareGPT closed-loop bench \
             calib-5175604,\n mean over conc {{1,4,16,64,256}}; bank = k=8 \
             ShareGPT trace rounds, same engine)\n"
        );
    } else {
        println!(
            "(native-k column only available for the default Llama config's \
             bench; bank =\n trace rounds from this config's acceptance file)\n"
        );
    }

    // -----------------------------------------------------------------
    // Step-cost spot checks: what the sim actually pays per step.
    // -----------------------------------------------------------------
    println!("step-cost spot checks: priced step_time(batch, k) from the table");
    for (b, g) in [(1u32, 0u32), (4, 1), (16, 2), (64, 4), (256, 8)] {
        match table.step_time(b, g) {
            Some(t) => println!("  B={b:<4} k={g}  ->  {:7.2} ms", 1e3 * t),
            None => println!("  B={b:<4} k={g}  ->  (no measured rows)"),
        }
    }
    println!();

    // -----------------------------------------------------------------
    // VALIDATION: sim vs the measured ShareGPT static-k envelope. The
    // ground truth is the default Llama deployment's bench, so this part
    // only runs for that config.
    // -----------------------------------------------------------------
    if is_default_config {
        validate_against_llama_bench(&cfg);
    } else {
        println!(
            "VALIDATION skipped: the measured ShareGPT static-k envelope \
             (calib-5175604) is\nground truth for {DEFAULT_CONFIG_PATH} only.\n"
        );
    }

    // -----------------------------------------------------------------
    // TARGET SWEEP (pure decode, prefilled arrivals, table's native shape).
    // -----------------------------------------------------------------
    target_sweep(&cfg, &table, gamma_max, &fixed);
}

/// Part 1: sim vs the measured ShareGPT static-k envelope of the default
/// Llama/EAGLE3 GH200 deployment (MEASURED_TPS, calib-5175604).
fn validate_against_llama_bench(cfg: &Config) {
    let val_tasks: Vec<(usize, usize)> = (0..VAL_CONCS.len())
        .flat_map(|ci| (0..VAL_KS.len()).map(move |ki| (ci, ki)))
        .collect();
    let val_results: Vec<((usize, usize), f64)> = val_tasks
        .par_iter()
        .map(|&(ci, ki)| {
            let conc = VAL_CONCS[ci];
            let k = VAL_KS[ki];
            let (mean_isl, mean_osl) = RUNG_LENS[ci];
            // The bench ran max(12*conc, 64) prompts per rung; at low conc we
            // run more to cut length-sampling noise (same loop semantics).
            let total = (12 * conc).max(512);
            let seed = 7 + 1000 * conc as u64 + k as u64;
            let tps = run_validation(cfg, conc, k, mean_isl, mean_osl, total, seed);
            ((ci, ki), tps)
        })
        .collect();
    let mut sim_tps = [[0.0f64; 7]; 5];
    for ((ci, ki), tps) in val_results {
        sim_tps[ci][ki] = tps;
    }

    println!("VALIDATION — output tok/s, sim vs measured (ShareGPT closed-loop");
    println!("bench, calib-5175604, 2026-06-12). resid = (sim - meas) / meas.");
    print!("{:>5} {:>6} |", "conc", "");
    for k in VAL_KS {
        print!(" {:>8}", format!("k={k}"));
    }
    println!();
    println!("{}", "-".repeat(14 + 9 * VAL_KS.len()));
    for (ci, &conc) in VAL_CONCS.iter().enumerate() {
        print!("{conc:>5} {:>6} |", "sim");
        for ki in 0..VAL_KS.len() {
            print!(" {:>8.0}", sim_tps[ci][ki]);
        }
        println!();
        print!("{:>5} {:>6} |", "", "meas");
        for ki in 0..VAL_KS.len() {
            print!(" {:>8.0}", MEASURED_TPS[ci][ki]);
        }
        println!();
        print!("{:>5} {:>6} |", "", "resid");
        for ki in 0..VAL_KS.len() {
            let r = 100.0 * (sim_tps[ci][ki] - MEASURED_TPS[ci][ki]) / MEASURED_TPS[ci][ki];
            print!(" {:>+7.1}%", r);
        }
        println!();
        println!("{}", "-".repeat(14 + 9 * VAL_KS.len()));
    }
    println!(
        "validation scenario: prefill ON, lognormal ISL/OSL matched to each \
         rung's\nrealized means {RUNG_LENS:?},\nsigma_log={SIGMA_LOG}, tails \
         clamped at the bench's empirical ~p99 (1200), throughput =\n\
         completed output tokens / wall(t=0..last completion), ramp+drain \
         included.\n\
         Known unmodeled bench-side costs (measured from the bench job's \
         telemetry,\nnot fudged into the sim): (1) closed-loop slack — the \
         bench's achieved\nconcurrency is 86-94% of nominal and its \
         time-weighted decode batch ~76-93%\n(client turnaround, response \
         transfer); (2) serving-loop round overhead —\nrealized inter-round \
         time under ShareGPT serving exceeds the calib table's\nstep cost by \
         ~10% (k=0) to ~40-100% (k>=2 at conc>=16: EAGLE draft-extend\nafter \
         each prefill, host overhead under overlap scheduling); (3) the \
         table's\nref_seq_len KV correction is a bandwidth lower bound — the \
         realized KV-length\neffect at B=256 is ~2x larger (attention-kernel \
         shape). (1)+(2) make the sim\nread HIGH at conc>=16 (worst at deep \
         k); (3) makes k=0 read LOW at conc>=64.\n"
    );
}

/// Part 2: the decode-pool target sweep (pure decode, prefilled arrivals,
/// the cost table's native calibration shape).
fn target_sweep(cfg: &Config, table: &MeasuredCostTable, gamma_max: u32, fixed: &[u32]) {
    let isl = 1024u32;
    let osl = 256u32;
    let concs = [1u32, 2, 4, 8, 16, 32, 64, 128, 256];

    // Columns per conc: 0 = nospec, 1..=fixed.len() = measured fixed k, then
    // budget, gated_aggregate, gated (ragged).
    let ncols = fixed.len() + 4;
    let (col_bud, col_agg, col_gat) = (ncols - 3, ncols - 2, ncols - 1);
    let mut tasks: Vec<(u32, usize, Policy)> = Vec::new();
    for &conc in &concs {
        tasks.push((conc, 0, Policy::NoSpec));
        for (i, &g) in fixed.iter().enumerate() {
            tasks.push((conc, 1 + i, Policy::Fixed(g)));
        }
        tasks.push((conc, col_bud, Policy::Budget(gamma_max)));
        tasks.push((conc, col_agg, Policy::GatedAggregate(gamma_max)));
        tasks.push((conc, col_gat, Policy::Gated(gamma_max)));
    }

    let results: Vec<(u32, usize, f64, f64)> = tasks
        .par_iter()
        .map(|&(conc, col, p)| {
            let (tps, db) = run(cfg, conc, isl, osl, p);
            (conc, col, tps, db)
        })
        .collect();

    let mut tps: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    let mut dbatch: BTreeMap<u32, f64> = BTreeMap::new();
    for (conc, col, t, db) in results {
        tps.entry(conc).or_insert_with(|| vec![0.0; ncols])[col] = t;
        if col == 0 {
            dbatch.insert(conc, db);
        }
    }

    // Extrapolation honesty: the table's batch coverage is partial. In this
    // closed loop with prefilled arrivals the decode batch equals the
    // concurrency, so any (conc, ndt) cell where conc falls outside the
    // table's measured batch range at that ndt was priced by
    // nearest-neighbor clamping — flag those concs as provisional. ndt here
    // is the file's verify-width column (g + 1). Only measured widths are
    // checked (unmeasured ones are excluded from the run entirely).
    let mut provisional: BTreeMap<u32, Vec<String>> = BTreeMap::new();
    for &conc in &concs {
        let cells: Vec<String> = std::iter::once(0u32)
            .chain(fixed.iter().copied())
            .filter_map(|g| {
                let (lo, hi) = table.batch_range(g)?;
                if conc < lo {
                    Some(format!("ndt={} (min batch {lo})", g + 1))
                } else if conc > hi {
                    Some(format!("ndt={} (max batch {hi})", g + 1))
                } else {
                    None
                }
            })
            .collect();
        if !cells.is_empty() {
            provisional.insert(conc, cells);
        }
    }

    println!(
        "TARGET SWEEP (pure decode, prefilled arrivals, ISL={isl} OSL={osl} — \
         the cost\ntable's own calibration shape; see VALIDATION above for \
         how sim relates to the\nreal serving envelope)"
    );
    println!(
        "{:>5} {:>7} | {:>8} | {:>13} | {:>8} {:>9} {:>8} | {:>7}",
        "conc", "dbatch", "nospec", "best-fixed", "budget", "gated_agg", "gated", "agg d%"
    );
    println!("{}", "-".repeat(86));
    for &conc in &concs {
        let row = &tps[&conc];
        let ns = row[0];
        let mut best = ns;
        let mut bestk = 0u32;
        for (i, &g) in fixed.iter().enumerate() {
            if row[1 + i] > best {
                best = row[1 + i];
                bestk = g;
            }
        }
        let (bud, agg, gat) = (row[col_bud], row[col_agg], row[col_gat]);
        let d = 100.0 * (agg - best) / best;
        let label = if bestk == 0 { "nospec".into() } else { format!("k{bestk}") };
        let conc_s = format!(
            "{conc}{}",
            if provisional.contains_key(&conc) { "*" } else { "" }
        );
        println!(
            "{conc_s:>5} {:>7.1} | {ns:>8.0} | {label:>5} {best:>7.0} | {bud:>8.0} {agg:>9.0} {gat:>8.0} | {d:>+7.2}",
            dbatch[&conc]
        );
    }
    println!("\ntokens/s = completed-request output tokens per second, steady state.");
    println!("agg d% = gated_aggregate headroom over the best fixed k at that concurrency.");
    if !provisional.is_empty() {
        println!();
    }
    for (conc, cells) in &provisional {
        println!(
            "WARNING: conc {conc}: step_time extrapolated outside the measured table's \
             batch coverage (nearest-neighbor clamp) at {}; target numbers at this \
             concurrency are provisional.",
            cells.join(", ")
        );
    }
}

trait ModelName {
    fn model_name(&self) -> &str;
}
impl ModelName for Config {
    fn model_name(&self) -> &str {
        use inference_lab::config::ModelCosts;
        self.model.name()
    }
}
