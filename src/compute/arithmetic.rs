/// Core inference arithmetic. All architecture-specific behaviour is hidden
/// behind `ModelCosts`; this module just composes those costs across batches.
use crate::config::{HardwareConfig, ModelConfig, ModelCosts};

/// Total FLOPs for processing a batch — matmul share scales with total tokens
/// processed, attention share is per-request because it depends on each
/// request's attended-token count.
pub fn flops_for_tokens(
    total_tokens: u32,
    model: &ModelConfig,
    requests: &[&crate::request::Request],
    tokens_per_request: &[u32],
) -> f64 {
    let matmul_flops = total_tokens as f64 * model.matmul_flops_per_token() as f64;

    let mut attention_flops = 0.0;
    for (req, &num_new_tokens) in requests.iter().zip(tokens_per_request) {
        let attended = req.num_computed_tokens + num_new_tokens;
        attention_flops += model.attention_flops(num_new_tokens, attended) as f64;
    }

    matmul_flops + attention_flops
}

/// Bytes transferred for model weights in one forward pass. For MoE this is
/// active params * effective bytes/param; for dense it's all of them. The
/// `_hardware` argument is kept for API symmetry — bytes-per-param now lives
/// on the model.
pub fn model_weight_bytes(model: &ModelConfig, _hardware: &HardwareConfig) -> f64 {
    model.weight_transfer_bytes_per_step() as f64
}

/// Bytes of KV cache read per decode step for a sequence of `seq_len` tokens.
pub fn kv_cache_bytes(seq_len: u32, model: &ModelConfig) -> f64 {
    model.kv_bytes_read_per_decode_step(seq_len) as f64
}

/// Number of leading prompt blocks shared by every request in the batch.
/// Uses the incremental prompt block hashes as the equality check: hash N
/// covers tokens 0..N*block_size, so two requests share a prefix of K blocks
/// iff their first K block hashes are pairwise equal.
pub fn shared_prefix_blocks(batch_requests: &[&crate::request::Request]) -> u32 {
    if batch_requests.len() < 2 {
        return 0;
    }
    let first = batch_requests[0].get_prompt_block_hashes();
    let mut shared = first.len();
    for req in &batch_requests[1..] {
        let other = req.get_prompt_block_hashes();
        let mut i = 0;
        while i < shared && i < other.len() && first[i] == other[i] {
            i += 1;
        }
        shared = i;
        if shared == 0 {
            return 0;
        }
    }
    shared as u32
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

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
        // Default test config: 32 layers * 32 kv_heads * 128 head_dim * 2 bytes * 2 (K+V)
        // = 524,288 bytes per token.
        let config = Config::test_default();
        let bytes = kv_cache_bytes(100, &config.model);
        assert_eq!(bytes, 52_428_800.0);
    }

    #[test]
    fn test_total_memory_transfer() {
        let config = Config::test_default();

        let seq_lens = vec![50, 100, 150];
        let total = total_memory_transfer(&config.model, &config.hardware, &seq_lens);

        let expected_weights = 14_000_000_000.0;
        let expected_kv = 524_288.0 * (50.0 + 100.0 + 150.0);
        assert_eq!(total, expected_weights + expected_kv);
    }
}
