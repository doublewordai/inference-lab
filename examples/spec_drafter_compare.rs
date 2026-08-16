//! MTP vs DFlash: does the drafter's cost *shape* change the value of adaptation?
//!
//! The drafter-roofline section of the post establishes two cost shapes on the
//! same verifier (Qwen3.6-35B-A3B): MTP (autoregressive) costs LINEAR in the
//! draft depth γ — it re-streams its weights on each of γ passes — while DFlash
//! (block-parallel diffusion) costs FLAT in γ until the block crosses the ridge,
//! because it streams its weights once. Both borrow the target's vocab head, so
//! the only difference is how many times that read is paid.
//!
//! This sweep prices each drafter with its real roofline (`DrafterCost`) instead
//! of a free drafter, runs both against their measured acceptance
//! banks (MTP D=8 / DFlash D=16, SPEED-Bench), and asks at each operating point:
//! how far does a priced adaptive policy beat the best-in-hindsight fixed γ?
//!
//! Hypothesis: with MTP, adaptation has a real draft-depth lever (deep drafts
//! genuinely cost more), so trimming γ pays on the cost axis as well as the
//! verify axis. With DFlash the drafter is ~free in γ, so that lever weakens and
//! whatever adaptation gain survives is coming from the verify side (the MoE
//! expert tax at small batch, and load).
//!
//! Run: `cargo run --release --no-default-features --example spec_drafter_compare`

mod common;

use inference_lab::config::{AcceptanceModel, DrafterCost, GammaPolicy, SpeculativeConfig};
use inference_lab::simulation::Simulator;

#[derive(Clone, Copy)]
enum Drafter {
    Mtp,
    Dflash,
}

impl Drafter {
    fn label(&self) -> &'static str {
        match self {
            Drafter::Mtp => "MTP (autoregressive, D=8)",
            Drafter::Dflash => "DFlash (block-parallel, D=16)",
        }
    }
    fn bank_path(&self) -> &'static str {
        match self {
            Drafter::Mtp => "data/banks/mtp_speedbench_rounds.csv",
            Drafter::Dflash => "data/banks/dflash_speedbench_rounds.csv",
        }
    }
    fn cost(&self) -> DrafterCost {
        match self {
            // One MoE decoder layer + tied head. Dense = head (2048·248320) +
            // attention (18.87M) + EAGLE fusion (4096·2048). Experts coupon-
            // collect over the batch, 8 of 256 + 1 shared, 3.146M params each.
            Drafter::Mtp => DrafterCost::Autoregressive {
                dense_params: 535_822_336.0,
                expert_params: 3_145_728.0,
                num_experts: 256,
                experts_per_tok: 8,
                shared_experts: 1,
            },
            // Eight dense diffusion layers + fusion + tied head, streamed once.
            Drafter::Dflash => DrafterCost::BlockParallel {
                params: 982_515_712.0,
                block: 16,
            },
        }
    }
    fn gamma_max(&self) -> u32 {
        match self {
            Drafter::Mtp => 8,
            Drafter::Dflash => 16,
        }
    }
    fn fixed_grid(&self) -> Vec<u32> {
        (1..=self.gamma_max()).collect()
    }
}

#[derive(Clone, Copy)]
enum Policy {
    NoSpec,
    Fixed(u32),
    Budget(u32),
}

impl Policy {
    fn spec(&self, d: Drafter) -> Option<SpeculativeConfig> {
        let acceptance = AcceptanceModel::TraceRounds {
            path: d.bank_path().into(),
        };
        let mk = |gamma: u32, policy: GammaPolicy| {
            Some(SpeculativeConfig {
                gamma,
                acceptance: acceptance.clone(),
                policy,
                measured_cost: None,
                switch: Default::default(),
                drafter: Some(d.cost()),
            })
        };
        match *self {
            Policy::NoSpec => None,
            Policy::Fixed(g) => mk(g, GammaPolicy::Fixed),
            Policy::Budget(g) => mk(g, GammaPolicy::GoodputBudget),
        }
    }
}

/// (goodput tok/s, tpot ms, bw util).
fn run_point(conc: usize, isl: u32, osl: u32, d: Drafter, p: Policy) -> (f64, f64, f64) {
    let mut config = common::closed_loop_config(common::qwen36(), 16384, conc, isl, osl);
    config.speculative = p.spec(d);
    config.finalize();
    let mut sim = Simulator::new(config, None).expect("build sim");
    sim.run_with_callback(|_| {}).expect("run");
    let s = sim.summary();
    (
        s.throughput_metrics.output_tokens_per_sec,
        s.latency_metrics.per_token_ms.mean * 1000.0,
        s.utilization.avg_bandwidth_util,
    )
}

fn sweep(d: Drafter, isl: u32, osl: u32, concs: &[usize]) {
    let gmax = d.gamma_max();
    let grid = d.fixed_grid();
    println!(
        "\n=== {}  (Qwen3.6-35B-A3B verifier, B200 TP1/EP1, ISL={isl} OSL={osl}) ===",
        d.label()
    );
    println!(
        "goodput = committed output tok/s. drafter priced by its roofline (not a fixed fraction)."
    );
    println!(
        "{:>6}  {:>9}  {:>9}  {:>9} {:>6}  {:>9} {:>7}  {:>7}",
        "conc",
        "nospec",
        format!("fixγ{gmax}"),
        "best-fix",
        "γ*",
        "adaptive",
        "Δvbest",
        "Δvγmax"
    );
    println!("{}", "-".repeat(82));
    for &conc in concs {
        let (ns, _, _) = run_point(conc, isl, osl, d, Policy::NoSpec);
        let mut best_fixed = ns;
        let mut best_g = 0u32;
        let mut fix_gmax = ns;
        for &g in &grid {
            let (gp, _, _) = run_point(conc, isl, osl, d, Policy::Fixed(g));
            if g == gmax {
                fix_gmax = gp;
            }
            if gp > best_fixed {
                best_fixed = gp;
                best_g = g;
            }
        }
        let (budget, _, _) = run_point(conc, isl, osl, d, Policy::Budget(gmax));
        let d_best = 100.0 * (budget - best_fixed) / best_fixed;
        let d_gmax = 100.0 * (budget - fix_gmax) / fix_gmax;
        let gstar = if best_g == 0 {
            "—".to_string()
        } else {
            format!("γ{best_g}")
        };
        println!(
            "{conc:>6}  {ns:>9.0}  {fix_gmax:>9.0}  {best_fixed:>9.0} {gstar:>6}  {budget:>9.0} {d_best:>+6.1}%  {d_gmax:>+6.1}%"
        );
    }
}

/// Head-to-head: each drafter's adaptive (priced-budget) goodput at each batch,
/// so the crossover is legible. MTP's draft cost is linear in γ, DFlash's is flat;
/// the question is where the cost-shape difference flips the winner.
fn crossover(isl: u32, osl: u32, concs: &[usize]) {
    let drafters = [Drafter::Mtp, Drafter::Dflash];
    let names = ["MTP", "DFlash"];
    println!("\n=== CROSSOVER: adaptive (priced-budget) goodput by drafter, tok/s ===");
    println!(
        "each drafter priced by its own roofline + acceptance bank; winner = highest goodput."
    );
    println!(
        "{:>6}  {:>9}  {:>10}  {:>10}  {:>8}",
        "conc", "nospec", "MTP", "DFlash", "winner"
    );
    println!("{}", "-".repeat(58));
    for &conc in concs {
        let (ns, _, _) = run_point(conc, isl, osl, Drafter::Mtp, Policy::NoSpec);
        let g: Vec<f64> = drafters
            .iter()
            .map(|&d| run_point(conc, isl, osl, d, Policy::Budget(d.gamma_max())).0)
            .collect();
        let widx = (0..2)
            .max_by(|&a, &b| g[a].partial_cmp(&g[b]).unwrap())
            .unwrap();
        println!(
            "{conc:>6}  {ns:>9.0}  {:>10.0}  {:>10.0}  {:>8}",
            g[0], g[1], names[widx]
        );
    }
}

fn main() {
    // ISL=1 makes prefill ~free, isolating the decode (output) side — equivalent
    // to disaggregated prefill for pricing purposes. This removes the budget
    // policy's only blind spot (it prices decode and ignores prefill contention),
    // so the adaptive-vs-best-fixed envelope should hold even at high batch.
    let isl: u32 = 1;
    let osl: u32 = 1024;
    let concs = [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
    println!("Drafter cost-shape comparison: MTP (linear-in-γ) vs DFlash (flat-in-γ).");
    println!(
        "Same verifier and policies; each drafter priced by its own roofline + acceptance bank."
    );
    sweep(Drafter::Mtp, isl, osl, &concs);
    sweep(Drafter::Dflash, isl, osl, &concs);
    crossover(isl, osl, &concs);
    println!(
        "\nRead: Δvs-fix is how much a priced policy beats the best fixed γ chosen in hindsight."
    );
    println!("Hypothesis — MTP's linear cost gives adaptation a draft-depth lever; DFlash's flat");
    println!("cost removes it, leaving only the verify-side (expert-tax / load) gain.");
}
