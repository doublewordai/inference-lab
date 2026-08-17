//! Presets shared by the examples, resolved from the crate's catalog
//! (`inference_lab::catalog`), plus a closed-loop config builder.
#![allow(dead_code)]

use inference_lab::catalog;
use inference_lab::config::{
    ArrivalPattern, Config, HardwareConfig, LengthDistribution, ModelSpec, ParallelConfig,
    SchedulerConfig, WorkloadConfig,
};
use inference_lab::scheduler::SchedulingPolicy;

/// Per-GPU B200 at the datasheet memory figures (192 GB, 8 TB/s), dense
/// peaks — the spec the speculative-decoding post's rooflines use.
pub fn b200_per_gpu() -> HardwareConfig {
    catalog::hardware("b200").expect("catalog preset")
}

/// DeepSeek-V4-Flash.
pub fn deepseek_v4_flash() -> ModelSpec {
    catalog::model("deepseek-v4-flash").expect("catalog preset")
}

/// Qwen3.6-35B-A3B (bf16).
pub fn qwen36() -> ModelSpec {
    catalog::model("qwen3.6-35b-a3b").expect("catalog preset")
}

/// A single-B200, TP1/EP1, closed-loop config: `conc` users issuing
/// fixed-shape (`isl`, `osl`) requests, preemption-free scheduling with a
/// `max_num_batched_tokens` token budget, half-millisecond arrival jitter to
/// break lockstep, and enough requests that steady state dominates.
pub fn closed_loop_config(
    model: ModelSpec,
    max_num_batched_tokens: u32,
    conc: usize,
    isl: u32,
    osl: u32,
) -> Config {
    Config {
        hardware: b200_per_gpu(),
        parallel: ParallelConfig {
            tp: 1,
            ep: 1,
            dp_attention: false,
        },
        model,
        scheduler: SchedulerConfig {
            max_num_batched_tokens,
            max_num_seqs: 32768,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
            max_num_partial_prefills: 1,
            block_size: 64,
            gpu_memory_utilization: 0.9,
            kv_cache_capacity: 0,
            max_model_len: None,
            policy: SchedulingPolicy::FCFS,
            enable_preemption_free: true,
            enable_cascade_attention: false,
        },
        replicas: 1,
        router: Default::default(),
        decode_router: None,
        memory: Default::default(),
        time_correction: None,
        workload: WorkloadConfig {
            dataset_path: None,
            sessions_path: None,
            num_sessions: None,
            arrival_pattern: ArrivalPattern::ClosedLoop,
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
        speculative: None,
        fault: None,
    }
}
