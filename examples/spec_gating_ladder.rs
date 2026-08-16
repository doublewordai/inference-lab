//! Acceptance gating: does the drafter's per-round confidence buy anything over
//! the homogeneous priced policy that uses only the average acceptance curve?
//!
//! The envelope experiment showed the homogeneous priced policy (`GoodputBudget`)
//! lands on the best fixed gamma. This one keeps everything fixed and walks a
//! ladder of policies that use progressively more of the per-round signal:
//!
//!   homogeneous   GoodputBudget    one width for the batch, average curve only
//!   realizable    GatedAggregate   per-round confidence -> one batch-uniform width
//!   ragged        GatedBudget      per-round confidence -> a width per sequence
//!   oracle        GatedBudget on the accept-pattern bank (perfect foresight)
//!
//! The gaps are the findings: realizable - homogeneous is what the usable signal
//! buys today; ragged - realizable is what a ragged-verify kernel would be worth;
//! oracle - ragged is calibration headroom (how wrong the confidence is). The
//! confidence is the real draft-time signal (`*_conf_rounds.csv`, a_k = conf_k);
//! the oracle is the shipped accept-pattern bank (a_k = 1 iff depth k committed).
//!
//! Run: `cargo run --release --no-default-features --example spec_gating_ladder`

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
    fn conf_bank(&self) -> &'static str {
        match self {
            Drafter::Mtp => "data/banks/mtp_conf_rounds.csv",
            Drafter::Dflash => "data/banks/dflash_conf_rounds.csv",
        }
    }
    fn oracle_bank(&self) -> &'static str {
        match self {
            Drafter::Mtp => "data/banks/mtp_speedbench_rounds.csv",
            Drafter::Dflash => "data/banks/dflash_speedbench_rounds.csv",
        }
    }
    fn cost(&self) -> DrafterCost {
        match self {
            Drafter::Mtp => DrafterCost::Autoregressive {
                dense_params: 535_822_336.0,
                expert_params: 3_145_728.0,
                num_experts: 256,
                experts_per_tok: 8,
                shared_experts: 1,
            },
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
}

#[derive(Clone, Copy)]
enum Policy {
    NoSpec,
    Homogeneous(u32),
    RealizableGate(u32),
    RaggedGate(u32),
    Oracle(u32),
}

impl Policy {
    fn spec(&self, d: Drafter) -> Option<SpeculativeConfig> {
        let mk = |gamma: u32, policy: GammaPolicy, bank: &str| {
            Some(SpeculativeConfig {
                gamma,
                acceptance: AcceptanceModel::TraceRounds { path: bank.into() },
                policy,
                measured_cost: None,
                switch: Default::default(),
                drafter: Some(d.cost()),
            })
        };
        match *self {
            Policy::NoSpec => None,
            Policy::Homogeneous(g) => mk(g, GammaPolicy::GoodputBudget, d.conf_bank()),
            Policy::RealizableGate(g) => mk(g, GammaPolicy::GatedAggregate, d.conf_bank()),
            Policy::RaggedGate(g) => mk(g, GammaPolicy::GatedBudget, d.conf_bank()),
            Policy::Oracle(g) => mk(g, GammaPolicy::GatedBudget, d.oracle_bank()),
        }
    }
}

fn goodput(conc: usize, isl: u32, osl: u32, d: Drafter, p: Policy) -> f64 {
    let mut config = common::closed_loop_config(common::qwen36(), 16384, conc, isl, osl);
    config.speculative = p.spec(d);
    config.finalize();
    let mut sim = Simulator::new(config, None).expect("build sim");
    sim.run_with_callback(|_| {}).expect("run");
    sim.summary().throughput_metrics.output_tokens_per_sec
}

fn sweep(d: Drafter, isl: u32, osl: u32, concs: &[usize]) {
    let g = d.gamma_max();
    println!(
        "\n=== {}  (Qwen3.6-35B-A3B verifier, B200 TP1/EP1, decode-only) ===",
        d.label()
    );
    println!("Δ columns are vs the homogeneous priced policy (the envelope).");
    println!(
        "{:>6}  {:>8}  {:>8}  {:>9}  {:>9}  {:>9}",
        "conc", "nospec", "homog", "realiz Δ", "ragged Δ", "oracle Δ"
    );
    println!("{}", "-".repeat(60));
    for &conc in concs {
        let ns = goodput(conc, isl, osl, d, Policy::NoSpec);
        let homog = goodput(conc, isl, osl, d, Policy::Homogeneous(g));
        let realiz = goodput(conc, isl, osl, d, Policy::RealizableGate(g));
        let ragged = goodput(conc, isl, osl, d, Policy::RaggedGate(g));
        let oracle = goodput(conc, isl, osl, d, Policy::Oracle(g));
        let pct = |v: f64| 100.0 * (v - homog) / homog;
        println!(
            "{conc:>6}  {ns:>8.0}  {homog:>8.0}  {:>+8.1}%  {:>+8.1}%  {:>+8.1}%",
            pct(realiz),
            pct(ragged),
            pct(oracle)
        );
    }
}

fn main() {
    let isl: u32 = 1;
    let osl: u32 = 1024;
    let concs = [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
    println!("Gating ladder: homogeneous -> realizable gate -> ragged gate -> oracle.");
    println!("realiz: value of the usable confidence signal (engine-realizable, one width).");
    println!("ragged - realiz: value of per-sequence verify widths (ragged-verify kernel).");
    println!("oracle - ragged: calibration headroom (how wrong the confidence is).");
    sweep(Drafter::Mtp, isl, osl, &concs);
    sweep(Drafter::Dflash, isl, osl, &concs);
}
