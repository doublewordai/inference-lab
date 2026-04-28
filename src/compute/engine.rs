/// Compute engine for calculating inference timing
use super::arithmetic;
use crate::config::{HardwareConfig, ModelConfig};
use crate::request::Request;

pub struct ComputeEngine {
    hardware: HardwareConfig,
    model: ModelConfig,
    block_size: u32,
    enable_cascade_attention: bool,
}

impl ComputeEngine {
    pub fn new(hardware: HardwareConfig, model: ModelConfig) -> Self {
        Self {
            hardware,
            model,
            block_size: 0,
            enable_cascade_attention: false,
        }
    }

    /// Enable cascade attention modeling. When a scheduled batch shares a
    /// prompt prefix, the shared KV is counted once per iteration rather than
    /// once per request. `block_size` is the KV cache block size in tokens
    /// (matches the scheduler's block size).
    pub fn with_cascade_attention(mut self, enabled: bool, block_size: u32) -> Self {
        self.enable_cascade_attention = enabled;
        self.block_size = block_size;
        self
    }

    /// Calculate time to process an iteration (in seconds)
    /// Takes batch of requests and number of tokens to process for each
    /// Returns max(compute_time, memory_time) since they happen in parallel
    pub fn calculate_iteration_time(
        &self,
        batch_requests: &[&Request],
        tokens_per_request: &[u32],
    ) -> f64 {
        if batch_requests.is_empty() {
            return 0.0;
        }

        let total_tokens: u32 = tokens_per_request.iter().sum();

        // Calculate compute time: FLOPs / compute throughput
        let flops = arithmetic::flops_for_tokens(
            total_tokens,
            &self.model,
            batch_requests,
            tokens_per_request,
        );
        let compute_time = flops / self.hardware.compute_flops;

        // Calculate memory time: bytes transferred / memory bandwidth
        let bytes = self.calculate_bytes_transferred(batch_requests, tokens_per_request);
        let memory_time = bytes / self.hardware.memory_bandwidth;

        // We're limited by whichever takes longer
        compute_time.max(memory_time)
    }

    /// Calculate FLOPS utilization for this iteration (0.0 to 1.0)
    pub fn calculate_flops_utilization(
        &self,
        batch_requests: &[&Request],
        tokens_per_request: &[u32],
        actual_time: f64,
    ) -> f64 {
        if actual_time == 0.0 {
            return 0.0;
        }

        let total_tokens: u32 = tokens_per_request.iter().sum();
        let flops = arithmetic::flops_for_tokens(
            total_tokens,
            &self.model,
            batch_requests,
            tokens_per_request,
        );
        let theoretical_time = flops / self.hardware.compute_flops;
        (theoretical_time / actual_time).min(1.0)
    }

    /// Calculate memory bandwidth utilization for this iteration (0.0 to 1.0)
    pub fn calculate_bandwidth_utilization(&self, bytes_transferred: f64, actual_time: f64) -> f64 {
        if actual_time == 0.0 {
            return 0.0;
        }

        let theoretical_time = bytes_transferred / self.hardware.memory_bandwidth;
        (theoretical_time / actual_time).min(1.0)
    }

    /// Calculate total bytes transferred for a batch of requests
    pub fn calculate_bytes_transferred(
        &self,
        batch_requests: &[&Request],
        tokens_per_request: &[u32],
    ) -> f64 {
        // Model weights (constant per iteration)
        let weight_bytes = arithmetic::model_weight_bytes(&self.model, &self.hardware);

        // With cascade attention, the KV bytes for the shared prompt prefix
        // are loaded once per iteration instead of once per request.
        let shared_prefix_tokens = if self.enable_cascade_attention && self.block_size > 0 {
            arithmetic::shared_prefix_blocks(batch_requests) * self.block_size
        } else {
            0
        };
        let shared_kv_bytes = arithmetic::kv_cache_bytes(shared_prefix_tokens, &self.model);

        // KV cache bytes (depends on sequence lengths)
        let mut kv_cache_bytes = 0.0;
        for (req, &tokens) in batch_requests.iter().zip(tokens_per_request) {
            // Average sequence length during this iteration
            let avg_seq_len = req.num_computed_tokens + tokens / 2;
            // Subtract the shared portion that's accounted for once at batch level.
            let unshared = avg_seq_len.saturating_sub(shared_prefix_tokens);
            kv_cache_bytes += arithmetic::kv_cache_bytes(unshared, &self.model);
        }

        weight_bytes + shared_kv_bytes + kv_cache_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::request::Request;

    fn create_test_engine() -> ComputeEngine {
        let config = Config::test_default();
        ComputeEngine::new(config.hardware, config.model)
    }

    fn create_test_request(id: &str, computed: u32, prompt: u32) -> Request {
        let mut req = Request::new(id.to_string(), 0, 0.0, prompt, 50);
        req.num_computed_tokens = computed;
        req
    }

    #[test]
    fn test_high_token_time() {
        let engine = create_test_engine();

        // For 2000+ tokens, likely compute-bound
        let req1 = create_test_request("req-1", 0, 1000);
        let req2 = create_test_request("req-2", 0, 1000);

        let requests = vec![&req1, &req2];
        let tokens = vec![1000, 1000];

        let time = engine.calculate_iteration_time(&requests, &tokens);

        // Time should be max(compute_time, memory_time)
        // With 2000 tokens, likely compute-bound
        assert!(time > 0.0);
    }

    #[test]
    fn test_low_token_time() {
        let engine = create_test_engine();

        // For few tokens, likely memory-bound
        let req1 = create_test_request("req-1", 0, 100);

        let requests = vec![&req1];
        let tokens = vec![50]; // Only 50 tokens

        let time = engine.calculate_iteration_time(&requests, &tokens);

        // Time should be max(compute_time, memory_time)
        // With few tokens, likely memory-bound
        assert!(time > 0.0);
    }

    #[test]
    fn test_empty_batch() {
        let engine = create_test_engine();

        let requests: Vec<&Request> = vec![];
        let tokens: Vec<u32> = vec![];

        let time = engine.calculate_iteration_time(&requests, &tokens);
        assert_eq!(time, 0.0);
    }

    #[test]
    fn test_flops_utilization() {
        let engine = create_test_engine();

        // Test with 1000 tokens
        let req = create_test_request("req-1", 0, 1000);
        let requests = vec![&req];
        let tokens = vec![1000];

        let flops = arithmetic::flops_for_tokens(1000, &engine.model, &requests, &tokens);
        let theoretical_time = flops / engine.hardware.compute_flops;

        // If actual time equals theoretical, utilization should be 100%
        let util = engine.calculate_flops_utilization(&requests, &tokens, theoretical_time);
        assert!((util - 1.0).abs() < 1e-10);

        // If actual time is 2x theoretical, utilization should be 50%
        let util = engine.calculate_flops_utilization(&requests, &tokens, theoretical_time * 2.0);
        assert!((util - 0.5).abs() < 1e-10);

        // Test with zero time
        let util = engine.calculate_flops_utilization(&requests, &tokens, 0.0);
        assert_eq!(util, 0.0);
    }

    #[test]
    fn test_cascade_attention_reduces_bytes_transferred() {
        let config = Config::test_default();
        let block_size = config.scheduler.block_size;

        let plain = ComputeEngine::new(config.hardware.clone(), config.model.clone());
        let cascade = ComputeEngine::new(config.hardware.clone(), config.model.clone())
            .with_cascade_attention(true, block_size);

        // Two decode-step requests sharing the first 8 prompt blocks.
        let mut req_a = create_test_request("a", 200, 200);
        let mut req_b = create_test_request("b", 200, 200);
        let shared: Vec<u64> = (0..8).map(|i| 1000 + i as u64).collect();
        req_a.prompt_block_hashes = shared
            .iter()
            .copied()
            .chain(std::iter::once(99_001))
            .collect();
        req_b.prompt_block_hashes = shared
            .iter()
            .copied()
            .chain(std::iter::once(99_002))
            .collect();

        let requests = vec![&req_a, &req_b];
        let tokens = vec![1, 1]; // single decode step each

        let bytes_plain = plain.calculate_bytes_transferred(&requests, &tokens);
        let bytes_cascade = cascade.calculate_bytes_transferred(&requests, &tokens);

        // Cascade should load the shared 8*block_size tokens of KV once
        // instead of twice; expected saving is exactly that.
        let expected_saving =
            arithmetic::kv_cache_bytes(8 * block_size, &cascade.model);
        let actual_saving = bytes_plain - bytes_cascade;
        assert!(
            (actual_saving - expected_saving).abs() < 1e-6,
            "expected saving {expected_saving}, got {actual_saving}"
        );
    }

    #[test]
    fn test_cascade_attention_no_shared_prefix_no_change() {
        let config = Config::test_default();
        let block_size = config.scheduler.block_size;

        let plain = ComputeEngine::new(config.hardware.clone(), config.model.clone());
        let cascade = ComputeEngine::new(config.hardware.clone(), config.model.clone())
            .with_cascade_attention(true, block_size);

        let mut req_a = create_test_request("a", 200, 200);
        let mut req_b = create_test_request("b", 200, 200);
        req_a.prompt_block_hashes = vec![1, 2, 3];
        req_b.prompt_block_hashes = vec![4, 5, 6];

        let requests = vec![&req_a, &req_b];
        let tokens = vec![1, 1];

        let bytes_plain = plain.calculate_bytes_transferred(&requests, &tokens);
        let bytes_cascade = cascade.calculate_bytes_transferred(&requests, &tokens);
        assert!((bytes_plain - bytes_cascade).abs() < 1e-6);
    }

    #[test]
    fn test_bandwidth_utilization() {
        let engine = create_test_engine();

        let bytes = 1e12; // 1 TB
        let theoretical_time = bytes / engine.hardware.memory_bandwidth;

        // If actual time equals theoretical, utilization should be 100%
        let util = engine.calculate_bandwidth_utilization(bytes, theoretical_time);
        assert!((util - 1.0).abs() < 1e-10);

        // If actual time is 2x theoretical, utilization should be 50%
        let util = engine.calculate_bandwidth_utilization(bytes, theoretical_time * 2.0);
        assert!((util - 0.5).abs() < 1e-10);

        // Test with zero time
        let util = engine.calculate_bandwidth_utilization(bytes, 0.0);
        assert_eq!(util, 0.0);
    }
}
