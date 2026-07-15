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

use inference_lab::config::model::Qwen35Model;
use inference_lab::config::{
    AcceptanceModel, Config, DrafterCost, GammaPolicy, HardwareConfig, LengthDistribution,
    ModelConfig, ParallelConfig, Precision, SchedulerConfig, SimulationConfig, SpeculativeConfig,
    WorkloadConfig,
};
use inference_lab::simulation::Simulator;

fn b200_per_gpu() -> HardwareConfig {
    HardwareConfig {
        name: "B200".into(),
        flops_fp4: Some(9.0e15),
        flops_fp8: Some(4.5e15),
        flops_bf16: Some(2.25e15),
        flops_fp16: Some(2.25e15),
        memory_bandwidth: 8.0e12,
        memory_capacity: 206_158_430_208,
        kv_cache_capacity: 0,
        gpu_memory_utilization: 0.9,
        kv_tiers: Vec::new(),
    }
}

fn qwen36() -> ModelConfig {
    ModelConfig::Qwen35(Qwen35Model {
        name: "Qwen3.6-35B-A3B".into(),
        num_layers: 40,
        hidden_dim: 2048,
        max_seq_len: 262_144,
        num_attention_layers: 10,
        num_attention_heads: 16,
        num_kv_heads: 2,
        attn_head_dim: 256,
        linear_num_value_heads: 32,
        linear_num_key_heads: 16,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        linear_conv_kernel: 4,
        num_active_expert_params: 1_132_462_080,
        num_active_non_expert_params: 1_725_693_952,
        num_resident_expert_params: 32_338_083_840,
        num_resident_non_expert_params: 2_234_253_312,
        num_experts_per_tok: 8,
        num_routed_experts: 256,
        num_moe_layers: 40,
        expert_precision: Precision::Bf16,
        non_expert_precision: Precision::Bf16,
        kv_precision: Precision::Bf16,
    })
}

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
                draft_cost_frac: 0.0,
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

fn base_config(conc: usize, isl: u32, osl: u32) -> Config {
    Config {
        hardware: b200_per_gpu(),
        parallel: ParallelConfig {
            tp: 1,
            ep: 1,
            dp_attention: false,
        },
        model: qwen36(),
        scheduler: SchedulerConfig {
            max_num_batched_tokens: 16384,
            max_num_seqs: 32768,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
            max_num_partial_prefills: 1,
            block_size: 64,
            policy: "fcfs".into(),
            enable_preemption_free: true,
            enable_cascade_attention: false,
        },
        workload: WorkloadConfig {
            dataset_path: None,
            arrival_pattern: "closed_loop".into(),
            arrival_rate: 1.0,
            rate_schedule: None,
            num_concurrent_users: Some(conc),
            closed_loop_jitter_secs: Some(0.5e-3),
            input_len_dist: LengthDistribution::Fixed { value: isl },
            output_len_dist: LengthDistribution::Fixed { value: osl },
            num_requests: Some((conc * 20).max(2000)),
            duration_secs: None,
            seed: 7,
        },
        simulation: SimulationConfig::default(),
        speculative: None,
    }
}

/// (goodput tok/s, tpot ms).
fn run_point(conc: usize, isl: u32, osl: u32, d: Drafter, p: Policy) -> (f64, f64) {
    let mut config = base_config(conc, isl, osl);
    config.speculative = p.spec(d);
    config.finalize();
    let (mut sim, _cfg) = Simulator::new(config, None).expect("build sim");
    sim.run_with_callback(|_| {}).expect("run");
    let s = sim.get_metrics_summary();
    // per_token_mean is already in ms (collector multiplies by 1000).
    (s.output_tokens_per_sec, s.per_token_mean)
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
