//! Full (γ × batch) goodput+TPOT surface for MTP and DFlash, dumped as JSON.
//!
//! Companion to `spec_drafter_compare`, which only prints best-fixed/adaptive.
//! The blog Pareto (TPOT vs throughput) and throughput-vs-batch plots need the
//! WHOLE fixed-γ family — every γ swept over every batch — plus TPOT for each
//! point, and the adaptive (priced-budget) envelope with its γ*. Same verifier
//! (Qwen3.6-35B-A3B), same hardware, same acceptance banks, same ISL/OSL as the
//! compare sweep, so the numbers line up with PricingEnvelope.
//!
//! Run: `cargo run --release --no-default-features --example spec_drafter_grid > out.json`

mod common;

use inference_lab::config::{AcceptanceModel, DrafterCost, GammaPolicy, SpeculativeConfig};
use inference_lab::simulation::Simulator;

#[derive(Clone, Copy)]
enum Drafter {
    Mtp,
    Dflash,
}

impl Drafter {
    fn key(&self) -> &'static str {
        match self {
            Drafter::Mtp => "mtp",
            Drafter::Dflash => "dflash",
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

/// (goodput tok/s, tpot ms).
fn run_point(conc: usize, isl: u32, osl: u32, d: Drafter, p: Policy) -> (f64, f64) {
    let mut config = common::closed_loop_config(common::qwen36(), 16384, conc, isl, osl);
    config.speculative = p.spec(d);
    config.finalize();
    let mut sim = Simulator::new(config, None).expect("build sim");
    sim.run_with_callback(|_| {}).expect("run");
    let s = sim.summary();
    // per_token_mean is already in ms (collector multiplies by 1000).
    (
        s.throughput_metrics.output_tokens_per_sec,
        s.latency_metrics.per_token_ms.mean,
    )
}

fn arr_f(v: &[f64]) -> String {
    let items: Vec<String> = v.iter().map(|x| format!("{x:.1}")).collect();
    format!("[{}]", items.join(", "))
}
fn arr_u(v: &[u32]) -> String {
    let items: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    format!("[{}]", items.join(", "))
}

fn dump(d: Drafter, isl: u32, osl: u32, concs: &[usize]) -> String {
    let gmax = d.gamma_max();
    // nospec baseline
    let mut ns_g = Vec::new();
    let mut ns_t = Vec::new();
    for &c in concs {
        let (g, t) = run_point(c, isl, osl, d, Policy::NoSpec);
        ns_g.push(g);
        ns_t.push(t);
    }
    // fixed-γ family
    let mut fixed_blocks = Vec::new();
    for gamma in 1..=gmax {
        let mut gg = Vec::new();
        let mut tt = Vec::new();
        for &c in concs {
            let (g, t) = run_point(c, isl, osl, d, Policy::Fixed(gamma));
            gg.push(g);
            tt.push(t);
        }
        fixed_blocks.push(format!(
            "      {{ \"gamma\": {gamma}, \"goodput\": {}, \"tpot\": {} }}",
            arr_f(&gg),
            arr_f(&tt)
        ));
    }
    // adaptive (priced budget) + γ* (argmax over fixed, incl. nospec=0)
    let mut ad_g = Vec::new();
    let mut ad_t = Vec::new();
    let mut gstar = Vec::new();
    for (i, &c) in concs.iter().enumerate() {
        let (g, t) = run_point(c, isl, osl, d, Policy::Budget(gmax));
        ad_g.push(g);
        ad_t.push(t);
        // recover γ* by scanning the fixed family at this conc
        let mut best = ns_g[i];
        let mut bg = 0u32;
        for gamma in 1..=gmax {
            let (gp, _) = run_point(c, isl, osl, d, Policy::Fixed(gamma));
            if gp > best {
                best = gp;
                bg = gamma;
            }
        }
        gstar.push(bg);
    }
    format!(
        "    \"{}\": {{\n      \"gamma_max\": {gmax},\n      \"nospec\": {{ \"goodput\": {}, \"tpot\": {} }},\n      \"fixed\": [\n{}\n      ],\n      \"adaptive\": {{ \"goodput\": {}, \"tpot\": {}, \"gstar\": {} }}\n    }}",
        d.key(),
        arr_f(&ns_g),
        arr_f(&ns_t),
        fixed_blocks.join(",\n"),
        arr_f(&ad_g),
        arr_f(&ad_t),
        arr_u(&gstar),
    )
}

fn main() {
    let isl: u32 = 1;
    let osl: u32 = 1024;
    let concs = [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
    let conc_u: Vec<u32> = concs.iter().map(|&c| c as u32).collect();

    let mtp = dump(Drafter::Mtp, isl, osl, &concs);
    let dflash = dump(Drafter::Dflash, isl, osl, &concs);

    println!("{{");
    println!("  \"meta\": {{ \"model\": \"Qwen3.6-35B-A3B\", \"hw\": \"B200 TP1/EP1\", \"isl\": {isl}, \"osl\": {osl}, \"goodput_unit\": \"committed output tok/s\", \"tpot_unit\": \"ms\" }},");
    println!("  \"conc\": {},", arr_u(&conc_u));
    println!("  \"drafters\": {{");
    println!("{},", mtp);
    println!("{}", dflash);
    println!("  }}");
    println!("}}");
}
