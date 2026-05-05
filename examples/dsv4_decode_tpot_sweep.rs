//! TPOT roofline vs InferenceX measurement across the 6 GB300 dynamo-sglang
//! pareto points. Tests the launch-tax hypothesis: if most of the batch=1 gap
//! is fixed per-iteration overhead, the gap should shrink as batch grows.
//!
//! Run with: `cargo run --example dsv4_decode_tpot_sweep --no-default-features`

use inference_lab::config::{
    ClusterSpec, DeepseekV4Model, HardwareConfig, ModelConfig, SchedulerConfig,
};
use inference_lab::simulation::predict_decode_tpot;

fn b300_per_gpu(tp: u32) -> HardwareConfig {
    HardwareConfig {
        name: "B300".into(),
        compute_flops: 1.5e16,
        memory_bandwidth: 8.0e12,
        memory_capacity: 309_237_645_312,
        tp,
        kv_cache_capacity: 0,
        gpu_memory_utilization: 0.9,
        bytes_per_param: 1,
        kv_tiers: Vec::new(),
    }
}

fn dsv4_pro() -> ModelConfig {
    ModelConfig::DeepseekV4(DeepseekV4Model {
        name: "DeepSeek-V4-Pro".into(),
        num_parameters: 1_600_000_000_000,
        num_active_parameters: 49_000_000_000,
        num_layers: 61,
        hidden_dim: 7168,
        num_heads: 128,
        max_seq_len: 1_048_576,
        kv_latent_dim: 512,
        kv_bytes_per_value: 1,
        effective_bytes_per_active_param: 0.625,
        effective_bytes_per_resident_param: 0.625,
        window_size: 128,
        num_dense_layers: 1,
        num_near_layers: 30,
        num_far_layers: 30,
        near_compress_ratio: 4,
        far_compress_ratio: 128,
        index_topk: 1024,
        index_n_heads: 64,
        index_head_dim: 128,
        index_kv_bytes_per_value: None,
    })
}

fn cluster(decode_tp: u32) -> ClusterSpec {
    ClusterSpec {
        hardware: b300_per_gpu(decode_tp),
        num_workers: 1,
        node: 0,
    }
}

struct Point {
    label: &'static str,
    conc: u32,
    decode_tp: u32,
    real_tpot_ms: f64,
}

fn main() {
    let model = dsv4_pro();
    let _scheduler = SchedulerConfig {
        max_num_batched_tokens: 65536,
        max_num_seqs: 65536,
        enable_chunked_prefill: false,
        long_prefill_token_threshold: 0,
        max_num_partial_prefills: 1,
        block_size: 64,
        policy: "fcfs".into(),
        enable_preemption_free: true,
        enable_cascade_attention: false,
    };

    // 8k/1k workload, average seq len during decode = ISL + OSL/2.
    let avg_seq_len = 8192 + 512;

    let points = vec![
        Point { label: "conc=1     1×D TP4   batch=1     ", conc: 1,     decode_tp: 4,  real_tpot_ms: 11.92 },
        Point { label: "conc=512   1×D TP16  batch=512   ", conc: 512,   decode_tp: 16, real_tpot_ms: 22.53 },
        Point { label: "conc=512   1×D TP8   batch=512   ", conc: 512,   decode_tp: 8,  real_tpot_ms: 27.90 },
        Point { label: "conc=1024  1×D TP8   batch=1024  ", conc: 1024,  decode_tp: 8,  real_tpot_ms: 35.67 },
        Point { label: "conc=2048  1×D TP8   batch=2048  ", conc: 2048,  decode_tp: 8,  real_tpot_ms: 45.96 },
        Point { label: "conc=16384 1×D TP16  batch=16384 ", conc: 16384, decode_tp: 16, real_tpot_ms: 69.69 },
    ];

    println!("DSv4-Pro decode TPOT roofline vs InferenceX (GB300, dynamo-sglang, 8k/1k FP4)\n");
    println!(
        "{:<42}  {:>10}  {:>10}  {:>10}  {:>10}",
        "operating point", "real (ms)", "roof (ms)", "gap×", "gap (ms)"
    );
    println!("{}", "-".repeat(92));
    for p in &points {
        let roofline_s = predict_decode_tpot(&cluster(p.decode_tp), &model, p.conc, avg_seq_len);
        let roofline_ms = roofline_s * 1000.0;
        let gap = p.real_tpot_ms / roofline_ms;
        let gap_abs = p.real_tpot_ms - roofline_ms;
        println!(
            "{:<42}  {:>10.3}  {:>10.3}  {:>9.1}×  {:>10.3}",
            p.label, p.real_tpot_ms, roofline_ms, gap, gap_abs
        );
    }
    println!();
    println!("Hypothesis: a fixed per-iteration overhead (kernel launches, sparse-attn");
    println!("scoring, MLA bookkeeping) of a few ms should be roughly constant. If so,");
    println!("the absolute gap (last column) should be roughly flat across rows, and");
    println!("the multiplicative gap (4th column) should shrink as batch grows.");
}
