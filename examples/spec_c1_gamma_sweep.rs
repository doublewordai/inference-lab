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

use inference_lab::config::{
    AcceptanceModel, Config, DeepseekV4Model, GammaPolicy, HardwareConfig, LengthDistribution,
    ModelConfig, ParallelConfig, Precision, SchedulerConfig, SimulationConfig, SpeculativeConfig,
    WorkloadConfig,
};
use inference_lab::simulation::Simulator;

fn b200_per_gpu() -> HardwareConfig {
    // B200 dense peaks ÷ 8 TB/s give the post's ridges: fp4 1125, fp8 563, bf16 281.
    HardwareConfig {
        name: "B200".into(),
        flops_fp4: Some(9.0e15),
        flops_fp8: Some(4.5e15),
        flops_bf16: Some(2.25e15),
        flops_fp16: Some(2.25e15),
        memory_bandwidth: 8.0e12,
        memory_capacity: 206_158_430_208, // 192 GiB HBM3e
        kv_cache_capacity: 0,
        gpu_memory_utilization: 0.9,
        kv_tiers: Vec::new(),
    }
}

fn deepseek_v4_flash() -> ModelConfig {
    // Architecture + param counts derived from the HF `deepseek-ai/DeepSeek-V4-Flash`
    // config.json and the actual safetensors weight shapes (not estimated). The
    // backbone is 43 MoE layers: 2 dense-attention (compress_ratio 0), 21 near
    // (4) + indexer, 20 far (128). Expert FFN 3·4096·2048 = 25.17M params each;
    // 256 routed + 1 shared. Non-expert (per-token GEMM) is attention QKVO
    // projections (106.9M/layer) + indexer + compressor + gate + head. MTP head
    // excluded (base-model decode), matching the Pro convention.
    ModelConfig::DeepseekV4(DeepseekV4Model {
        name: "DeepSeek-V4-Flash".into(),
        num_layers: 43,
        hidden_dim: 4096,
        num_heads: 64,
        max_seq_len: 1_048_576,
        kv_latent_dim: 512, // head_dim
        qk_rope_head_dim: 64,
        kv_precision: Precision::Fp8,
        num_active_expert_params: 7_574_913_024, // (6+1)·25.17M·43
        num_active_non_expert_params: 5_660_947_776, // attn+indexer+compressor+gate+head
        num_resident_expert_params: 278_107_521_024, // (256+1)·25.17M·43
        num_resident_non_expert_params: 6_225_000_000,
        expert_precision: Precision::Fp4,
        non_expert_precision: Precision::Fp8,
        window_size: 128,
        num_dense_layers: 2,
        num_near_layers: 21,
        num_far_layers: 20,
        near_compress_ratio: 4,
        far_compress_ratio: 128,
        index_topk: 512,
        index_n_heads: 64,
        index_head_dim: 128,
        index_kv_precision: None,
        num_experts_per_tok: 6,
        num_routed_experts: 256,
        num_moe_layers: 43,
    })
}

fn base_config(conc: usize, isl: u32, osl: u32) -> Config {
    Config {
        hardware: b200_per_gpu(),
        // Single B200, TP1/EP1 -- matches the post's per-GPU roofline (the
        // ~145GB of fp4 weights + small KV fit in 192GB). One pool, one device.
        parallel: ParallelConfig {
            tp: 1,
            ep: 1,
            dp_attention: false,
        },
        model: deepseek_v4_flash(),
        scheduler: SchedulerConfig {
            max_num_batched_tokens: 8192,
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
            // Break the synchronized-arrival regime so the decode batch is a
            // realistic fluctuating quantity, not a lockstep pulse.
            closed_loop_jitter_secs: Some(0.5e-3),
            input_len_dist: LengthDistribution::Fixed { value: isl },
            output_len_dist: LengthDistribution::Fixed { value: osl },
            // Long enough that steady state dominates the t=0 transient.
            num_requests: Some((conc * 20).max(2000)),
            duration_secs: None,
            seed: 7,
        },
        simulation: SimulationConfig::default(),
        speculative: None,
    }
}

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
                draft_cost_frac: c_draft,
                measured_cost: None,
                switch: Default::default(),
                drafter: None,
            }),
            Policy::Budget(g) => Some(SpeculativeConfig {
                gamma: g,
                acceptance,
                policy: GammaPolicy::GoodputBudget,
                draft_cost_frac: c_draft,
                measured_cost: None,
                switch: Default::default(),
                drafter: None,
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
    let mut config = base_config(conc, isl, osl);
    config.speculative = p.spec(alpha, c_draft);
    config.finalize();
    let (mut sim, _cfg) = Simulator::new(config, None).expect("build sim");
    sim.run_with_callback(|_| {}).expect("run");
    let s = sim.get_metrics_summary();
    (
        s.output_tokens_per_sec,
        s.per_token_mean * 1000.0,
        s.avg_bandwidth_util,
        s.avg_flops_util,
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
