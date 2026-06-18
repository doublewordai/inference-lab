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
    fn label(&self) -> &'static str {
        match self {
            Drafter::Mtp => "MTP (autoregressive, D=8)",
            Drafter::Dflash => "DFlash (block-parallel, D=16)",
        }
    }
    fn conf_bank(&self) -> &'static str {
        match self {
            Drafter::Mtp => "data/qwen36_nextn_conf_rounds.csv",
            Drafter::Dflash => "data/qwen36_dflash_conf_rounds.csv",
        }
    }
    fn oracle_bank(&self) -> &'static str {
        match self {
            Drafter::Mtp => "data/qwen36_nextn_speedbench_rounds.csv",
            Drafter::Dflash => "data/qwen36_dflash_speedbench_rounds.csv",
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
            Drafter::Dflash => DrafterCost::BlockParallel { params: 982_515_712.0, block: 16 },
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
                draft_cost_frac: 0.0,
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

fn base_config(conc: usize, isl: u32, osl: u32) -> Config {
    Config {
        hardware: b200_per_gpu(),
        parallel: ParallelConfig { tp: 1, ep: 1, dp_attention: false },
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

fn goodput(conc: usize, isl: u32, osl: u32, d: Drafter, p: Policy) -> f64 {
    let mut config = base_config(conc, isl, osl);
    config.speculative = p.spec(d);
    config.finalize();
    let (mut sim, _cfg) = Simulator::new(config, None).expect("build sim");
    sim.run_with_callback(|_| {}).expect("run");
    sim.get_metrics_summary().output_tokens_per_sec
}

fn sweep(d: Drafter, isl: u32, osl: u32, concs: &[usize]) {
    let g = d.gamma_max();
    println!("\n=== {}  (Qwen3.6-35B-A3B verifier, B200 TP1/EP1, decode-only) ===", d.label());
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
