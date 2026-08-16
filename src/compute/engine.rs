//! Per-iteration roofline. Compute is split into precision-homogeneous
//! streams (e.g. FP4 expert GEMMs and FP8 attention/dense in DeepSeek-V4-Pro).
//! Each stream runs at its own FLOP rate, and we sum stream times since
//! kernels of different precisions are launched serially on the GPU. Memory
//! traffic flows through one HBM, so weight bytes for each precision plus KV
//! bytes (charged to the attention stream) all share the same bandwidth.

use crate::config::{
    CommsConfig, HardwareConfig, ModelConfig, ModelCosts, ParallelConfig, Precision,
};
use crate::request::Request;

pub struct ComputeEngine {
    hardware: HardwareConfig,
    parallel: ParallelConfig,
    model: ModelConfig,
    comms: Option<CommsConfig>,
    block_size: u32,
    enable_cascade_attention: bool,
}

/// Work accumulated on one precision stream.
#[derive(Default, Clone, Copy)]
struct StreamAcc {
    flops: f64,
    bytes: f64,
}

impl StreamAcc {
    fn is_empty(&self) -> bool {
        self.flops == 0.0 && self.bytes == 0.0
    }
}

/// Per-precision work for one step, indexed by [`Precision::index`].
type Streams = [StreamAcc; Precision::COUNT];

/// Roofline cost of one step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepCost {
    /// Wall time: per-precision `max(compute, memory)` summed, plus
    /// collectives.
    pub time: f64,
    /// Bytes moved HBM -> SM (weights of every stream plus KV reads).
    pub bytes: f64,
    /// FLOPs across all streams.
    pub flops: f64,
    /// Time the FLOPs alone would take at each stream's peak rate, summed.
    pub compute_time: f64,
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

    fn aggregate_flop_rate(&self, prec: Precision) -> Option<f64> {
        self.hardware
            .flop_rate(prec)
            .map(|r| r * self.parallel.tp as f64)
    }

    fn aggregate_memory_bandwidth(&self) -> f64 {
        self.hardware.memory_bandwidth * self.parallel.tp as f64
    }

    /// Aggregate bf16 FLOP rate (the precision the drafter heads stream at),
    /// falling back to fp16 if bf16 is unset. Exposed to price the drafter
    /// roofline alongside the verify cost.
    pub fn bf16_peak_flops(&self) -> f64 {
        self.aggregate_flop_rate(Precision::Bf16)
            .or_else(|| self.aggregate_flop_rate(Precision::Fp16))
            .unwrap_or(f64::INFINITY)
    }

    /// Aggregate memory bandwidth, exposed for the drafter roofline.
    pub fn mem_bandwidth(&self) -> f64 {
        self.aggregate_memory_bandwidth()
    }

    /// Estimated time spent in TP all-reduce + EP all-to-all collectives for a
    /// batch of `total_tokens` tokens. Returns 0 if no `CommsConfig` is set or
    /// `link_bw` is non-positive. Each collective is modelled as
    /// `latency + bytes / link_bw`, summed over all calls in a forward pass.
    /// The result is added to the per-stream max(compute, memory) sum — i.e.,
    /// we assume collectives do not overlap with compute/memory work.
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
            let bytes_per_call = factor * tokens * self.model.alltoall_bytes_per_token() as f64;
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

    /// Accumulate the batch's work per precision stream. Matmul FLOPs and
    /// weight bytes come from the model's per-precision splits; attention
    /// FLOPs, KV reads and per-sequence state reads attach to
    /// `attention_precision`.
    fn assemble_streams(&self, batch_requests: &[&Request], tokens_per_request: &[u32]) -> Streams {
        let mut streams: Streams = [StreamAcc::default(); Precision::COUNT];
        let total_tokens: u32 = tokens_per_request.iter().sum();

        for (prec, fpt) in self.model.matmul_flops_per_token_by_prec() {
            streams[prec.index()].flops += total_tokens as f64 * fpt as f64;
        }
        // Weight bytes per forward pass, also per precision. For MoE this grows
        // with the step's token count (coupon-collector expert loading).
        for (prec, b) in self.model.weight_bytes_per_step_by_prec(total_tokens) {
            streams[prec.index()].bytes += b as f64;
        }

        let attn = &mut streams[self.model.attention_precision().index()];

        // With cascade attention, the KV bytes for the shared prompt prefix
        // are loaded once per iteration instead of once per request.
        let shared_prefix_tokens = if self.enable_cascade_attention && self.block_size > 0 {
            Request::shared_prefix_blocks(batch_requests) * self.block_size
        } else {
            0
        };
        attn.bytes += self
            .model
            .kv_bytes_read_per_decode_step(shared_prefix_tokens) as f64;

        // Fixed per-sequence state (Mamba / GatedDeltaNet) is read once per
        // sequence per step, independent of context length.
        let state_bytes = self.model.per_sequence_state_bytes() as f64;
        for (req, &num_new) in batch_requests.iter().zip(tokens_per_request) {
            let attended = req.num_computed_tokens + num_new;
            attn.flops += self.model.attention_flops(num_new, attended) as f64;

            let avg_seq_len = req.num_computed_tokens + num_new / 2;
            let unshared = avg_seq_len.saturating_sub(shared_prefix_tokens);
            attn.bytes += self.model.kv_bytes_read_per_decode_step(unshared) as f64 + state_bytes;
        }

        streams
    }

    /// Roofline cost of processing `tokens_per_request[i]` positions for
    /// each `batch_requests[i]` in one step. Per-precision stream times are
    /// summed (kernels of different precisions are serial), then collectives
    /// are added on top.
    pub fn step_cost(&self, batch_requests: &[&Request], tokens_per_request: &[u32]) -> StepCost {
        if batch_requests.is_empty() {
            return StepCost {
                time: 0.0,
                bytes: 0.0,
                flops: 0.0,
                compute_time: 0.0,
            };
        }
        let total_tokens: u32 = tokens_per_request.iter().sum();
        let streams = self.assemble_streams(batch_requests, tokens_per_request);
        let bw = self.aggregate_memory_bandwidth();

        let mut cost = StepCost {
            time: 0.0,
            bytes: 0.0,
            flops: 0.0,
            compute_time: 0.0,
        };
        for (i, acc) in streams.iter().enumerate() {
            if acc.is_empty() {
                continue;
            }
            let prec = Precision::ALL[i];
            let rate = self.aggregate_flop_rate(prec).unwrap_or_else(|| {
                panic!(
                    "model declares a {prec:?} stream but hardware {} has no FLOP rate for {prec:?}",
                    self.hardware.name
                )
            });
            let compute_time = acc.flops / rate;
            let memory_time = acc.bytes / bw;
            cost.time += compute_time.max(memory_time);
            cost.compute_time += compute_time;
            cost.bytes += acc.bytes;
            cost.flops += acc.flops;
        }
        cost.time += self.collective_time(total_tokens);
        cost
    }

    /// Wall time of a step (see [`ComputeEngine::step_cost`]).
    pub fn calculate_iteration_time(
        &self,
        batch_requests: &[&Request],
        tokens_per_request: &[u32],
    ) -> f64 {
        self.step_cost(batch_requests, tokens_per_request).time
    }

    /// Fraction of `actual_time` the step's FLOPs would fill at peak rate,
    /// per-precision-weighted (FP4-heavy work is compared against FP4 peak).
    pub fn flops_utilization(&self, cost: &StepCost, actual_time: f64) -> f64 {
        if actual_time <= 0.0 {
            return 0.0;
        }
        (cost.compute_time / actual_time).min(1.0)
    }

    /// Fraction of `actual_time` the step's bytes would fill at peak
    /// bandwidth.
    pub fn bandwidth_utilization(&self, cost: &StepCost, actual_time: f64) -> f64 {
        if actual_time <= 0.0 {
            return 0.0;
        }
        (cost.bytes / self.aggregate_memory_bandwidth() / actual_time).min(1.0)
    }

    /// Bandwidth-roofline KV-read time delta between the batch's actual KV
    /// lengths and a hypothetical batch where every sequence holds
    /// `ref_seq_len` tokens of context: `sum_i (kv_bytes(L_i) -
    /// kv_bytes(ref)) / mem_bw`. Used to recontextualise a measured
    /// step-cost table benchmarked at a different sequence length (see
    /// [`crate::config::MeasuredCostConfig::ref_seq_len`]). Negative when
    /// the live batch's sequences are shorter than the reference. This is a
    /// conservative lower-bound correction: it prices only the bandwidth of
    /// the KV delta at peak, not attention-kernel shape effects.
    pub fn kv_read_seq_delta_seconds(&self, batch_requests: &[&Request], ref_seq_len: u32) -> f64 {
        let bw = self.aggregate_memory_bandwidth();
        let ref_bytes = self.model.kv_bytes_read_per_decode_step(ref_seq_len) as f64;
        batch_requests
            .iter()
            .map(|r| {
                self.model
                    .kv_bytes_read_per_decode_step(r.num_computed_tokens) as f64
                    - ref_bytes
            })
            .sum::<f64>()
            / bw
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
        let req1 = create_test_request("req-1", 0, 1000);
        let req2 = create_test_request("req-2", 0, 1000);
        let requests = vec![&req1, &req2];
        let tokens = vec![1000, 1000];
        let time = engine.calculate_iteration_time(&requests, &tokens);
        assert!(time > 0.0);
    }

    #[test]
    fn test_low_token_time() {
        let engine = create_test_engine();
        let req1 = create_test_request("req-1", 0, 100);
        let requests = vec![&req1];
        let tokens = vec![50];
        let time = engine.calculate_iteration_time(&requests, &tokens);
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
        let req = create_test_request("req-1", 0, 1000);
        let requests = vec![&req];
        let tokens = vec![1000];
        let cost = engine.step_cost(&requests, &tokens);
        // Single-precision test model: theoretical time = total_flops / aggregate_rate.
        let prec = engine.model.matmul_flops_per_token_by_prec()[0].0;
        let rate = engine.aggregate_flop_rate(prec).unwrap();
        let theoretical_time = cost.flops / rate;
        assert!((cost.compute_time - theoretical_time).abs() < 1e-12);
        let util = engine.flops_utilization(&cost, theoretical_time);
        assert!((util - 1.0).abs() < 1e-10);
        let util = engine.flops_utilization(&cost, theoretical_time * 2.0);
        assert!((util - 0.5).abs() < 1e-10);
        let util = engine.flops_utilization(&cost, 0.0);
        assert_eq!(util, 0.0);
    }

    #[test]
    fn test_cascade_attention_reduces_bytes_transferred() {
        let config = Config::test_default();
        let block_size = config.scheduler.block_size;

        let plain = ComputeEngine::new(
            config.hardware.clone(),
            config.parallel.clone(),
            config.model.clone(),
        );
        let cascade = ComputeEngine::new(
            config.hardware.clone(),
            config.parallel.clone(),
            config.model.clone(),
        )
        .with_cascade_attention(true, block_size);

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
        let tokens = vec![1, 1];

        let bytes_plain = plain.step_cost(&requests, &tokens).bytes;
        let bytes_cascade = cascade.step_cost(&requests, &tokens).bytes;

        // Cascade should load the shared 8*block_size tokens of KV once
        // instead of twice; expected saving is exactly that.
        let expected_saving = config.model.kv_bytes_read_per_decode_step(8 * block_size) as f64;
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

        let plain = ComputeEngine::new(
            config.hardware.clone(),
            config.parallel.clone(),
            config.model.clone(),
        );
        let cascade = ComputeEngine::new(
            config.hardware.clone(),
            config.parallel.clone(),
            config.model.clone(),
        )
        .with_cascade_attention(true, block_size);

        let mut req_a = create_test_request("a", 200, 200);
        let mut req_b = create_test_request("b", 200, 200);
        req_a.prompt_block_hashes = vec![1, 2, 3];
        req_b.prompt_block_hashes = vec![4, 5, 6];

        let requests = vec![&req_a, &req_b];
        let tokens = vec![1, 1];

        let bytes_plain = plain.step_cost(&requests, &tokens).bytes;
        let bytes_cascade = cascade.step_cost(&requests, &tokens).bytes;
        assert!((bytes_plain - bytes_cascade).abs() < 1e-6);
    }

    #[test]
    fn test_bandwidth_utilization() {
        let engine = create_test_engine();
        let cost = StepCost {
            time: 0.0,
            bytes: 1e12,
            flops: 0.0,
            compute_time: 0.0,
        };
        let theoretical_time = cost.bytes / engine.aggregate_memory_bandwidth();
        let util = engine.bandwidth_utilization(&cost, theoretical_time);
        assert!((util - 1.0).abs() < 1e-10);
        let util = engine.bandwidth_utilization(&cost, theoretical_time * 2.0);
        assert!((util - 0.5).abs() < 1e-10);
        let util = engine.bandwidth_utilization(&cost, 0.0);
        assert_eq!(util, 0.0);
    }
}
