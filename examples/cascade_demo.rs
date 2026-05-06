//! Cascade attention demo: compares per-iteration time and memory traffic
//! for a decode batch where every request shares the same long prompt prefix,
//! with cascade attention on vs off.
//!
//! Run with: `cargo run --example cascade_demo --no-default-features`
//! (the example doesn't need the CLI feature)

use inference_lab::compute::ComputeEngine;
use inference_lab::config::{DenseModel, HardwareConfig, ModelConfig, ParallelConfig, Precision};
use inference_lab::request::Request;

fn main() {
    // H100 FP8, Llama-3-70B.
    let hardware = HardwareConfig {
        name: "H100".into(),
        flops_fp4: None,
        flops_fp8: Some(1.979e15),
        flops_bf16: Some(0.989e15),
        flops_fp16: Some(0.989e15),
        memory_bandwidth: 3.35e12,
        memory_capacity: 85_899_345_920,
        kv_cache_capacity: 0,
        gpu_memory_utilization: 0.9,
        kv_tiers: Vec::new(),
    };
    let parallel = ParallelConfig::default();

    let model = ModelConfig::Dense(DenseModel {
        name: "Llama-3-70B".into(),
        num_parameters: 70_000_000_000,
        num_active_parameters: None,
        num_layers: 80,
        hidden_dim: 8192,
        num_heads: 64,
        num_kv_heads: Some(8),
        max_seq_len: 8192,
        precision: Precision::Fp8,
    });

    let block_size: u32 = 16;
    let shared_prefix_tokens: u32 = 2048; // shared system prompt
    let suffix_tokens: u32 = 64;          // per-request suffix already prefilled
    let total_prompt = shared_prefix_tokens + suffix_tokens;

    let plain = ComputeEngine::new(hardware.clone(), parallel.clone(), model.clone());
    let cascade = ComputeEngine::new(hardware.clone(), parallel.clone(), model.clone())
        .with_cascade_attention(true, block_size);

    println!(
        "Decode-step iteration on H100 + Llama-3-70B (FP8), shared prefix = {} tokens, suffix = {} tokens",
        shared_prefix_tokens, suffix_tokens
    );
    println!();
    println!(
        "{:>6}  {:>14}  {:>14}  {:>14}  {:>14}  {:>10}",
        "batch", "plain_ms", "cascade_ms", "plain_GB", "cascade_GB", "speedup"
    );
    println!("{}", "-".repeat(80));

    for &batch_size in &[1usize, 2, 4, 8, 16, 32, 64, 128] {
        // Build a batch of decode-step requests sharing the first
        // `shared_prefix_tokens` of their prompts.
        let shared_hashes: Vec<u64> = (0..(shared_prefix_tokens / block_size) as u64)
            .map(|i| 0xDEAD_0000 + i)
            .collect();
        let suffix_blocks = (suffix_tokens / block_size) as u64;

        let requests: Vec<Request> = (0..batch_size)
            .map(|i| {
                let mut req = Request::new(
                    format!("req-{i}"),
                    0,
                    0.0,
                    total_prompt,
                    1,
                );
                // Already prefilled.
                req.num_computed_tokens = total_prompt;
                req.prompt_block_hashes = shared_hashes
                    .iter()
                    .copied()
                    .chain(
                        (0..suffix_blocks).map(|b| 0xBEEF_0000 + (i as u64) * 1024 + b),
                    )
                    .collect();
                req
            })
            .collect();
        let req_refs: Vec<&Request> = requests.iter().collect();
        let tokens = vec![1u32; batch_size];

        let t_plain = plain.calculate_iteration_time(&req_refs, &tokens);
        let t_cascade = cascade.calculate_iteration_time(&req_refs, &tokens);
        let b_plain = plain.calculate_bytes_transferred(&req_refs, &tokens);
        let b_cascade = cascade.calculate_bytes_transferred(&req_refs, &tokens);

        println!(
            "{:>6}  {:>14.3}  {:>14.3}  {:>14.3}  {:>14.3}  {:>9.2}x",
            batch_size,
            t_plain * 1000.0,
            t_cascade * 1000.0,
            b_plain / 1e9,
            b_cascade / 1e9,
            t_plain / t_cascade
        );
    }
}
