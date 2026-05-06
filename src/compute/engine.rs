/// Compute engine for calculating inference timing
use super::arithmetic;
use crate::config::{CommsConfig, HardwareConfig, ModelConfig, ModelCosts, ParallelConfig};
use crate::request::Request;

pub struct ComputeEngine {
    hardware: HardwareConfig,
    parallel: ParallelConfig,
    model: ModelConfig,
    comms: Option<CommsConfig>,
    block_size: u32,
    enable_cascade_attention: bool,
}

impl ComputeEngine {
    pub fn new(hardware: HardwareConfig, parallel: ParallelConfig, model: ModelConfig) -> Self {
        Self {
            hardware,
            parallel,
            model,
            comms: None,
            block_size: 0,
            enable_cascade_attention: false,
        }
    }

    /// Enable the collective-comms time term. Without this, `calculate_iteration_time`
    /// contributes zero for TP all-reduce / EP all-to-all (the previous default).
    pub fn with_comms(mut self, comms: CommsConfig) -> Self {
        self.comms = Some(comms);
        self
    }

    fn aggregate_compute_flops(&self) -> f64 {
        self.hardware.compute_flops * self.parallel.tp as f64
    }

    fn aggregate_memory_bandwidth(&self) -> f64 {
        self.hardware.memory_bandwidth * self.parallel.tp as f64
    }

    /// Estimated time spent in TP all-reduce + EP all-to-all collectives for a
    /// batch of `total_tokens` tokens. Returns 0 if no `CommsConfig` is set or
    /// `link_bw` is non-positive. Each collective is modelled as
    /// `latency + bytes / link_bw`, summed over all calls in a forward pass.
    /// The result is added to `max(compute_time, memory_time)` — i.e., we
    /// assume collectives do not overlap with compute/memory work.
    fn collective_time(&self, total_tokens: u32) -> f64 {
        let Some(comms) = &self.comms else { return 0.0 };
        if comms.link_bw <= 0.0 {
            return 0.0;
        }

        let tokens = total_tokens as f64;
        let mut total = 0.0;

        // TP all-reduce: per-rank ring traffic = 2(tp-1)/tp × volume.
        // Skipped under DP-attention — each rank owns its sequence shard and
        // needs no per-layer allreduce.
        let tp = self.parallel.tp;
        if tp > 1 && !self.parallel.dp_attention {
            let ring_factor = 2.0 * (tp - 1) as f64 / tp as f64;
            let bytes_per_call =
                ring_factor * tokens * self.model.allreduce_bytes_per_token() as f64;
            let calls = self.model.num_tp_allreduces_per_pass() as f64;
            total += calls * (comms.allreduce_latency + bytes_per_call / comms.link_bw);
        }

        // EP all-to-all: per-rank send = (ep-1)/ep × per-rank-data, where
        // per-rank-data = V_global / ep (tokens distributed across ranks).
        // So per-rank send = (ep-1)/ep² × V_global. Without the extra /ep we'd
        // be charging the global cross-rank volume at the per-rank link rate.
        let ep = self.parallel.ep;
        if ep > 1 {
            let factor = (ep - 1) as f64 / (ep as f64 * ep as f64);
            let bytes_per_call =
                factor * tokens * self.model.alltoall_bytes_per_token() as f64;
            let calls = self.model.num_ep_alltoalls_per_pass() as f64;
            total += calls * (comms.alltoall_latency + bytes_per_call / comms.link_bw);
        }

        total
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
        let compute_time = flops / self.aggregate_compute_flops();

        // Calculate memory time: bytes transferred / memory bandwidth
        let bytes = self.calculate_bytes_transferred(batch_requests, tokens_per_request);
        let memory_time = bytes / self.aggregate_memory_bandwidth();

        // Compute and memory overlap; collectives are added on top
        // (assumed serial with the kernel timeline).
        compute_time.max(memory_time) + self.collective_time(total_tokens)
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
        let theoretical_time = flops / self.aggregate_compute_flops();
        (theoretical_time / actual_time).min(1.0)
    }

    /// Calculate memory bandwidth utilization for this iteration (0.0 to 1.0)
    pub fn calculate_bandwidth_utilization(&self, bytes_transferred: f64, actual_time: f64) -> f64 {
        if actual_time == 0.0 {
            return 0.0;
        }

        let theoretical_time = bytes_transferred / self.aggregate_memory_bandwidth();
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
        ComputeEngine::new(config.hardware, config.parallel, config.model)
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
        let theoretical_time = flops / engine.aggregate_compute_flops();

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

        let plain = ComputeEngine::new(config.hardware.clone(), config.parallel.clone(), config.model.clone());
        let cascade = ComputeEngine::new(config.hardware.clone(), config.parallel.clone(), config.model.clone())
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

        let plain = ComputeEngine::new(config.hardware.clone(), config.parallel.clone(), config.model.clone());
        let cascade = ComputeEngine::new(config.hardware.clone(), config.parallel.clone(), config.model.clone())
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
        let theoretical_time = bytes / engine.aggregate_memory_bandwidth();

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
