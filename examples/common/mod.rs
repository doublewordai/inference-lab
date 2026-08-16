//! Hardware and model presets shared by the examples. Each example does
//! `mod common;` and uses what it needs; keeping the numbers here means a
//! preset changes in one place.
#![allow(dead_code)]

use inference_lab::config::{
    ArrivalPattern, Config, DeepseekV4Model, HardwareConfig, LengthDistribution, ModelConfig,
    ParallelConfig, Precision, Qwen35Model, SchedulerConfig, WorkloadConfig,
};
use inference_lab::scheduler::SchedulingPolicy;

/// Per-GPU B200 spec.
pub fn b200_per_gpu() -> HardwareConfig {
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

/// DeepSeek-V4-Flash.
pub fn deepseek_v4_flash() -> ModelConfig {
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
        indexer_retained_layers: None,
        index_kv_precision: None,
        num_experts_per_tok: 6,
        num_routed_experts: 256,
        num_moe_layers: 43,
    })
}

/// Qwen3.6-35B-A3B.
pub fn qwen36() -> ModelConfig {
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

/// A single-B200, TP1/EP1, closed-loop config: `conc` users issuing
/// fixed-shape (`isl`, `osl`) requests, preemption-free scheduling with a
/// `max_num_batched_tokens` token budget, half-millisecond arrival jitter to
/// break lockstep, and enough requests that steady state dominates.
pub fn closed_loop_config(
    model: ModelConfig,
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
            policy: SchedulingPolicy::FCFS,
            enable_preemption_free: true,
            enable_cascade_attention: false,
        },
        workload: WorkloadConfig {
            dataset_path: None,
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
    }
}
