/// Compute engine for calculating inference timing

use super::arithmetic;
use crate::config::{HardwareConfig, ModelConfig};
use crate::request::Request;

pub struct ComputeEngine {
    hardware: HardwareConfig,
    model: ModelConfig,
}

impl ComputeEngine {
    pub fn new(hardware: HardwareConfig, model: ModelConfig) -> Self {
        Self { hardware, model }
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
        let flops = arithmetic::flops_for_tokens(total_tokens, &self.model, batch_requests, tokens_per_request);
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
        let flops = arithmetic::flops_for_tokens(total_tokens, &self.model, batch_requests, tokens_per_request);
        let theoretical_time = flops / self.hardware.compute_flops;
        (theoretical_time / actual_time).min(1.0)
    }

    /// Calculate memory bandwidth utilization for this iteration (0.0 to 1.0)
    pub fn calculate_bandwidth_utilization(
        &self,
        bytes_transferred: f64,
        actual_time: f64,
    ) -> f64 {
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

        // KV cache bytes (depends on sequence lengths)
        let mut kv_cache_bytes = 0.0;
        for (req, &tokens) in batch_requests.iter().zip(tokens_per_request) {
            // Average sequence length during this iteration
            let avg_seq_len = req.num_computed_tokens + tokens / 2;
            kv_cache_bytes += arithmetic::kv_cache_bytes(avg_seq_len, &self.model);
        }

        weight_bytes + kv_cache_bytes
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
