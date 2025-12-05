/// Core inference arithmetic formulas based on the inference-arithmetic.mdx blog post

use crate::config::{HardwareConfig, ModelConfig};

/// Calculate the compute-bound threshold (number of tokens at which inference becomes compute-bound)
/// Formula: threshold = (bytes_per_param * compute_flops) / memory_bandwidth
pub fn compute_bound_threshold(hardware: &HardwareConfig) -> u32 {
    (hardware.bytes_per_param as f64 * hardware.compute_flops / hardware.memory_bandwidth) as u32
}

/// Calculate FLOPS for a given number of tokens
/// Formula: FLOPS = 2 * num_tokens * active_parameters + attention_flops
/// For MoE models, uses active_parameters (not total) since only some experts are activated
/// Includes both matmul and attention FLOPs
pub fn flops_for_tokens(
    total_tokens: u32,
    model: &ModelConfig,
    requests: &[&crate::request::Request],
    tokens_per_request: &[u32],
) -> f64 {
    // MatMul FLOPs: 2 * num_tokens * active_parameters
    // For MoE: only counts activated expert parameters, not all experts
    let matmul_flops = 2.0 * total_tokens as f64 * model.active_parameters() as f64;

    // Attention FLOPs: 4 * L * B * S * T * D
    // where L = num_layers, B = batch size, S = new tokens, T = attended tokens, D = hidden_dim
    let mut attention_flops = 0.0;
    for (req, &num_new_tokens) in requests.iter().zip(tokens_per_request) {
        let batch_size = 1.0; // Each request is one sequence
        let s = num_new_tokens as f64; // New tokens being processed
        let t = (req.num_computed_tokens + num_new_tokens) as f64; // Total attended tokens
        let d = model.hidden_dim as f64;
        let l = model.num_layers as f64;

        // 4LBSTD FLOPs for attention across all layers
        // Note: Causal masking zeros out some values, but the matmul still computes them
        attention_flops += 4.0 * l * batch_size * s * t * d;
    }

    matmul_flops + attention_flops
}

/// Calculate memory transfer bytes for model weights
/// Formula: weight_bytes = num_parameters * bytes_per_param
pub fn model_weight_bytes(model: &ModelConfig, hardware: &HardwareConfig) -> f64 {
    model.num_parameters as f64 * hardware.bytes_per_param as f64
}

/// Calculate memory transfer bytes for KV cache for a given sequence length
/// Formula: kv_bytes = kv_cache_bytes_per_token * seq_len
pub fn kv_cache_bytes(seq_len: u32, model: &ModelConfig) -> f64 {
    model.kv_cache_bytes_per_token as f64 * seq_len as f64
}

/// Calculate total memory transfer bytes for an iteration
/// Formula: total_bytes = model_weights + sum(kv_cache for each request)
pub fn total_memory_transfer(
    model: &ModelConfig,
    hardware: &HardwareConfig,
    request_seq_lens: &[u32],
) -> f64 {
    let weight_bytes = model_weight_bytes(model, hardware);
    let kv_bytes: f64 = request_seq_lens
        .iter()
        .map(|&seq_len| kv_cache_bytes(seq_len, model))
        .sum();
    weight_bytes + kv_bytes
}

/// Check if a workload is compute-bound
/// A workload is compute-bound if the number of tokens >= compute-bound threshold
pub fn is_compute_bound(num_tokens: u32, hardware: &HardwareConfig) -> bool {
    num_tokens >= hardware.compute_bound_threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn test_compute_bound_threshold_h100() {
        let mut hardware = crate::config::HardwareConfig {
            name: "H100".to_string(),
            compute_flops: 1.513e15,      // 1513 TFLOPS bf16
            memory_bandwidth: 3.35e12,    // 3.35 TB/s
            memory_capacity: 80_000_000_000,
            kv_cache_capacity: 60_000_000_000,
            gpu_memory_utilization: 0.9,
            bytes_per_param: 2,           // bf16
            compute_bound_threshold: 0,
        };
        hardware.compute_threshold();

        let threshold = compute_bound_threshold(&hardware);

        // Should be approximately 903 for H100 bf16
        assert!(threshold > 900 && threshold < 910);
        assert_eq!(threshold, hardware.compute_bound_threshold);
    }

    #[test]
    fn test_flops_calculation() {
        let config = Config::test_default();

        // Create test request for 100 tokens
        let mut req = crate::request::Request::new("test".to_string(), 0, 0.0, 100, 50);
        req.num_computed_tokens = 0;
        let requests = vec![&req];
        let tokens = vec![100];

        // For 100 tokens through a 7B model
        let flops = flops_for_tokens(100, &config.model, &requests, &tokens);

        // MatMul FLOPs: 2 * 100 * 7e9 = 1.4e12
        // Attention FLOPs: 4 * 1 * 100 * 100 * 4096 (assuming default hidden_dim)
        // Total should be matmul + attention
        assert!(flops >= 1.4e12); // At least the matmul FLOPs
    }

    #[test]
    fn test_model_weight_bytes() {
        let config = Config::test_default();

        let bytes = model_weight_bytes(&config.model, &config.hardware);

        // 7e9 parameters * 2 bytes = 14GB
        assert_eq!(bytes, 14_000_000_000.0);
    }

    #[test]
    fn test_kv_cache_bytes() {
        let mut config = Config::test_default();
        config.model.kv_cache_bytes_per_token = 524_288; // 512KB per token

        let bytes = kv_cache_bytes(100, &config.model);

        // 524_288 * 100 = 52,428,800
        assert_eq!(bytes, 52_428_800.0);
    }

    #[test]
    fn test_total_memory_transfer() {
        let mut config = Config::test_default();
        config.model.kv_cache_bytes_per_token = 524_288;

        // 3 requests with sequence lengths 50, 100, 150
        let seq_lens = vec![50, 100, 150];
        let total = total_memory_transfer(&config.model, &config.hardware, &seq_lens);

        let expected_weights = 14_000_000_000.0;
        let expected_kv = 524_288.0 * (50.0 + 100.0 + 150.0);
        assert_eq!(total, expected_weights + expected_kv);
    }

    #[test]
    fn test_is_compute_bound() {
        let mut hardware = crate::config::HardwareConfig {
            name: "Test".to_string(),
            compute_flops: 1e15,
            memory_bandwidth: 1e12,
            memory_capacity: 80_000_000_000,
            kv_cache_capacity: 60_000_000_000,
            gpu_memory_utilization: 0.9,
            bytes_per_param: 2,
            compute_bound_threshold: 0,
        };
        hardware.compute_threshold();

        // Threshold should be 2000
        assert_eq!(hardware.compute_bound_threshold, 2000);

        assert!(!is_compute_bound(1000, &hardware));
        assert!(!is_compute_bound(1999, &hardware));
        assert!(is_compute_bound(2000, &hardware));
        assert!(is_compute_bound(3000, &hardware));
    }
}
