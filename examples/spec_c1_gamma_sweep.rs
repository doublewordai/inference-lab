//! C1: the auto-tuner sanity check.
//!
//! Claim under test: a load-aware policy that prices γ through the real cost
//! model (`GoodputBudget`) lands on the best fixed γ at *every* operating point,
//! with no per-point tuning. If true, the BUDGET goodput curve traces the upper
//! envelope of the fixed-γ family across the whole batch sweep. This is a
//! validation of the policy+cost-model coupling (and, implicitly, of the cost
//! model itself), not a win claim -- the wins are C2 (mistuning).
//!
//! Method (one variable: the decode batch):
//!   * Single aggregated pool, closed-loop at concurrency `conc` (= operating
//!     point), chunked prefill on (the vLLM-V1 target).
//!   * Held fixed: model, hardware, acceptance (constant α), ISL/OSL, c_draft, seed.
//!   * Swept: `conc` (the batch) × policy ∈ { nospec, fixed γ∈1..=8, BUDGET }.
//!   * Oracle-best-fixed = max goodput over the fixed-γ family at each conc.
//!   * Readout: goodput = `output_tokens_per_sec` (lossless spec ⇒ committed
//!     output tokens/s), plus bandwidth util to show the regime.
//!   * Pass iff BUDGET ≈ max-over-fixed-γ at every conc.
//!
//! MODEL: DeepSeek-V4-Flash on B200, params derived from the HF config +
//! safetensors shapes (see `deepseek_v4_flash` below).
//!
//! Run: `cargo run --release --example spec_c1_gamma_sweep --no-default-features`

mod common;

use inference_lab::config::{AcceptanceModel, DrafterCost, GammaPolicy, SpeculativeConfig};
use inference_lab::simulation::Simulator;

#[derive(Clone, Copy)]
enum Policy {
    NoSpec,
    Fixed(u32),
    Budget(u32),
}

impl Policy {
    fn spec(&self, alpha: f64, c_draft: f64) -> Option<SpeculativeConfig> {
        let acceptance = AcceptanceModel::Constant { alpha };
        match *self {
            Policy::NoSpec => None,
            Policy::Fixed(g) => Some(SpeculativeConfig {
                gamma: g,
                acceptance,
                policy: GammaPolicy::Fixed,
                measured_cost: None,
                switch: Default::default(),
                drafter: Some(DrafterCost::Fraction { frac: c_draft }),
            }),
            Policy::Budget(g) => Some(SpeculativeConfig {
                gamma: g,
                acceptance,
                policy: GammaPolicy::GoodputBudget,
                measured_cost: None,
                switch: Default::default(),
                drafter: Some(DrafterCost::Fraction { frac: c_draft }),
            }),
        }
    }
}

/// Returns (goodput_tok_s, tpot_ms, bw_util, flops_util).
fn run_point(
    conc: usize,
    isl: u32,
    osl: u32,
    alpha: f64,
    c_draft: f64,
    p: Policy,
) -> (f64, f64, f64, f64) {
    let mut config = common::closed_loop_config(common::deepseek_v4_flash(), 8192, conc, isl, osl);
    config.speculative = p.spec(alpha, c_draft);
    config.finalize();
    let mut sim = Simulator::new(config, None).expect("build sim");
    sim.run_with_callback(|_| {}).expect("run");
    let s = sim.summary();
    (
        s.throughput_metrics.output_tokens_per_sec,
        s.latency_metrics.per_token_ms.mean * 1000.0,
        s.utilization.avg_bandwidth_util,
        s.utilization.avg_flops_util,
    )
}

fn main() {
    // Blog SpecDecOptimalGamma defaults: avg seq len 4096, α 0.75, drafter 10%.
    // ISL 3840 + OSL/2 256 ⇒ mean decode context ≈ 4096.
    let isl: u32 = 3840;
    let osl: u32 = 512;
    let alpha = 0.75;
    let c_draft = 0.10;
    let concs = [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
    let fixed = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let gamma_max = 8u32;

    println!("C1 auto-tuner sanity check  (V4-Flash, B200 TP1/EP1, closed-loop, ISL={isl} OSL={osl}, α={alpha}, c_draft={c_draft})");
    println!("goodput = committed output tokens/s. BUDGET should trace max-over-fixed-γ at every batch.\n");

    // header
    print!("{:>6}", "conc");
    print!("  {:>10}", "nospec");
    for g in fixed {
        print!("  {:>10}", format!("fix γ{g}"));
    }
    print!("  {:>10}", "best-fix");
    print!("  {:>10}", format!("BUDGET≤{gamma_max}"));
    print!("  {:>7}", "Δ%");
    print!("  {:>7}", "argmax");
    print!("  {:>6}", "bw%");
    println!();
    println!("{}", "-".repeat(6 + (3 + fixed.len() + 2) * 12 + 7 + 7 + 6));

    for &conc in &concs {
        let (ns, _, _, _) = run_point(conc, isl, osl, alpha, c_draft, Policy::NoSpec);
        let mut best_fixed = ns;
        let mut best_g = 0u32;
        let mut fixed_gp = Vec::new();
        for &g in &fixed {
            let (gp, _, _, _) = run_point(conc, isl, osl, alpha, c_draft, Policy::Fixed(g));
            if gp > best_fixed {
                best_fixed = gp;
                best_g = g;
            }
            fixed_gp.push(gp);
        }
        let (budget, _tpot, bw, _fl) =
            run_point(conc, isl, osl, alpha, c_draft, Policy::Budget(gamma_max));
        let d_budget = 100.0 * (budget - best_fixed) / best_fixed;

        print!("{conc:>6}");
        print!("  {ns:>10.0}");
        for gp in &fixed_gp {
            print!("  {gp:>10.0}");
        }
        print!("  {best_fixed:>10.0}");
        print!("  {budget:>10.0}");
        print!("  {d_budget:>+7.2}");
        let argmax = if best_g == 0 {
            "nospec".to_string()
        } else {
            format!("γ{best_g}")
        };
        print!("  {argmax:>7}");
        print!("  {:>5.0}%", bw * 100.0);
        println!();
    }

    println!("\nPass criterion: Δ% ≈ 0 (BUDGET matches best fixed γ) at every conc, with the");
    println!(
        "winning fixed γ (argmax) shifting across the sweep -- that shift is what C2 exploits."
    );
}
