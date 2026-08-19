//! Per-iteration roofline. Compute is split into precision-homogeneous
//! streams (e.g. FP4 expert GEMMs and FP8 attention/dense in DeepSeek-V4-Pro).
//! Each stream runs at its own FLOP rate, and we sum stream times since
//! kernels of different precisions are launched serially on the GPU. Memory
//! traffic flows through one HBM, so weight bytes for each precision plus KV
//! bytes (charged to the attention stream) all share the same bandwidth.

use crate::config::{HardwareConfig, ModelSpec, ParallelConfig, Precision};
use crate::request::Request;

#[derive(Clone)]
pub struct ComputeEngine {
    hardware: HardwareConfig,
    parallel: ParallelConfig,
    model: ModelSpec,
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
    pub fn new(hardware: HardwareConfig, parallel: ParallelConfig, model: ModelSpec) -> Self {
        Self {
            hardware,
            parallel,
            model,
            block_size: 0,
            enable_cascade_attention: false,
        }
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

    /// Collective time for a step of `total_tokens` tokens, priced on the
    /// hardware's [`FabricConfig`] and added serially to the compute/memory
    /// time (no overlap). Per layer, over the `tp`-wide replica:
    ///
    /// * attention: one all-reduce of the hidden-wide activation, unless
    ///   `dp_attention` (each rank owns its sequences' attention outright);
    /// * dense FFN, or MoE without `dp_attention`: one all-reduce, whatever
    ///   `ep` is — with TP attention every rank already holds every token,
    ///   so EP only decides which expert GEMMs a rank runs; its partial
    ///   outputs are summed by the same all-reduce (vLLM's TP + EP path);
    /// * under `dp_attention`, dense FFN or MoE with `ep = 1`: an all-gather
    ///   in and a reduce-scatter out (the ranks' tokens meet the sharded
    ///   FFN);
    /// * under `dp_attention`, MoE with `ep > 1`: each token lives on one
    ///   rank, so dispatch and combine all-to-alls over the `ep` group, each
    ///   rank moving its `tokens / ep` share of the routed activations
    ///   (`experts_per_tok` copies of the hidden vector).
    fn collective_time(&self, total_tokens: u32) -> f64 {
        let Some(fabric) = &self.hardware.fabric else {
            return 0.0;
        };
        let tp = self.parallel.tp;
        if tp <= 1 {
            return 0.0;
        }
        let ep = self.parallel.ep;
        let dpa = self.parallel.dp_attention;
        let tokens = total_tokens as f64;
        let hidden = tokens * self.model.allreduce_bytes_per_token() as f64;
        let ffn_exchange = |layers: u32| {
            let t = if dpa {
                fabric.allgather_reducescatter_time(tp, hidden)
            } else {
                fabric.allreduce_time(tp, hidden)
            };
            layers as f64 * t
        };

        let mut total = 0.0;
        if !dpa {
            total += self.model.num_layers() as f64 * fabric.allreduce_time(tp, hidden);
        }
        total += ffn_exchange(self.model.dense_ffn_layers());
        if ep > 1 && dpa {
            for r in self.model.weights.iter().filter_map(|w| w.routing) {
                let per_rank = tokens / ep as f64 * r.experts_per_tok as f64 * hidden / tokens;
                total += 2.0 * r.moe_layers as f64 * fabric.alltoall_time(ep, per_rank);
            }
        } else {
            total += ffn_exchange(self.model.moe_layers());
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
        // DP-attention: every rank holds and reads the full attention
        // projections, so the replica reads them tp times per step.
        if self.parallel.dp_attention && self.parallel.tp > 1 {
            streams[self.model.projection_precision().index()].bytes +=
                (self.parallel.tp - 1) as f64 * self.model.attention_weight_bytes() as f64;
        }

        let attn_idx = self.model.attention_precision.index();

        // With cascade attention, the KV bytes for the shared prompt prefix
        // are loaded once per iteration instead of once per request.
        let shared_prefix_tokens = if self.enable_cascade_attention && self.block_size > 0 {
            Request::shared_prefix_blocks(batch_requests) * self.block_size
        } else {
            0
        };
        streams[attn_idx].bytes +=
            self.model
                .kv_bytes_read_per_decode_step(shared_prefix_tokens) as f64;

        // Fixed per-sequence state (Mamba / GatedDeltaNet) is read once per
        // sequence per step, independent of context length.
        let state_bytes = self.model.per_sequence_state_bytes() as f64;
        for (req, &num_new) in batch_requests.iter().zip(tokens_per_request) {
            // Causal attention: the k-th new position attends
            // `computed + k` positions, so the chunk's score/AV work and KV
            // reads are those of `num_new` queries against the mean context
            // `computed + (num_new+1)/2` (a fresh s-token prompt costs
            // ~s²/2 pairs, a decode token attends `computed + 1`).
            let mean_context = req.num_computed_tokens + num_new.div_ceil(2);
            // Score/AV at the attention precision; an indexer's scoring GEMM
            // at its own (an fp8 indexer runs at the fp8 rate).
            for (prec, flops) in
                self.model
                    .attention_flops_by_prec(num_new, mean_context, !req.is_prefill())
            {
                streams[prec.index()].flops += flops;
            }

            let unshared = mean_context.saturating_sub(shared_prefix_tokens);
            streams[attn_idx].bytes +=
                self.model.kv_bytes_read_per_decode_step(unshared) as f64 + state_bytes;
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

    /// Roofline time of the batch's attention alone (score/AV FLOPs, KV and
    /// per-sequence state reads) on ONE GPU of the replica: what a
    /// DP-attention rank spends on its own sequences before the ranks meet
    /// at the FFN collective. `step_cost` over the union batch spreads the
    /// same work over all `tp` GPUs, i.e. prices the mean rank; the max over
    /// ranks minus that mean is the skew the step waits for.
    pub fn attention_seconds_on_one_gpu(
        &self,
        batch_requests: &[&Request],
        tokens_per_request: &[u32],
    ) -> f64 {
        if batch_requests.is_empty() {
            return 0.0;
        }
        let state_bytes = self.model.per_sequence_state_bytes() as f64;
        let mut compute = 0.0;
        let mut bytes = 0.0;
        for (req, &num_new) in batch_requests.iter().zip(tokens_per_request) {
            // Same causal accounting as `assemble_streams`, each precision's
            // FLOPs at its own rate.
            let mean_context = req.num_computed_tokens + num_new.div_ceil(2);
            for (prec, flops) in
                self.model
                    .attention_flops_by_prec(num_new, mean_context, !req.is_prefill())
            {
                compute += flops / self.hardware.flop_rate(prec).unwrap_or(f64::INFINITY);
            }
            bytes += self.model.kv_bytes_read_per_decode_step(mean_context) as f64 + state_bytes;
        }
        compute.max(bytes / self.hardware.memory_bandwidth)
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

    /// TP all-reduces are priced on the hardware fabric and added serially:
    /// tp=2 with a fabric costs the tp=1 step (at twice the resources)
    /// plus 2 × layers calls of `latency + factor × bytes / bandwidth`.
    #[test]
    fn tp_collectives_add_fabric_time() {
        use crate::config::{FabricConfig, FabricLink};
        let base = Config::test_default();
        let mut hw = base.hardware.clone();
        hw.fabric = Some(FabricConfig {
            gpus_per_node: 8,
            scale_up: FabricLink {
                bandwidth: 1e12,
                latency: 1e-6,
                in_network_reduction: false,
            },
            scale_out: None,
        });
        let mut par = base.parallel.clone();
        par.tp = 2;
        let tp2 = ComputeEngine::new(hw.clone(), par.clone(), base.model.clone());
        let mut hw_silent = hw.clone();
        hw_silent.fabric = None;
        let tp2_silent = ComputeEngine::new(hw_silent, par.clone(), base.model.clone());

        let req = create_test_request("r", 0, 1024);
        let with = tp2.calculate_iteration_time(&[&req], &[1024]);
        let without = tp2_silent.calculate_iteration_time(&[&req], &[1024]);
        // 1024 tokens × 4096 hidden × 2 B = 8 MiB per call; ring 2(1/2) = 1×;
        // 64 calls (2 per layer × 32 layers).
        let per_call = 1e-6 + 1024.0 * 4096.0 * 2.0 / 1e12;
        assert!(
            (with - without - 64.0 * per_call).abs() < 1e-9,
            "{with} {without}"
        );

        // DP-attention: no attention all-reduce; the 32 dense FFNs each
        // all-gather + reduce-scatter (ring bytes, two latencies), and every
        // rank reads the full attention projections.
        par.dp_attention = true;
        let dpa = ComputeEngine::new(hw, par, base.model.clone());
        let expected = 32.0 * (2e-6 + 1024.0 * 4096.0 * 2.0 / 1e12);
        assert!((dpa.collective_time(1024) - expected).abs() < 1e-12);
        let extra = dpa.step_cost(&[&req], &[1024]).bytes - tp2.step_cost(&[&req], &[1024]).bytes;
        assert_eq!(extra, base.model.attention_weight_bytes() as f64);
    }

    /// Under DP-attention, EP replaces the MoE layers' FFN exchange with
    /// dispatch + combine all-to-alls over the ep group; dense layers keep
    /// the all-gather + reduce-scatter. With TP attention, tokens are
    /// replicated and EP prices exactly like ep = 1.
    #[test]
    fn ep_prices_alltoalls_on_moe_layers() {
        use crate::config::{FabricConfig, FabricLink, Routing, WeightStream};
        let base = Config::test_default();
        let mut hw = base.hardware.clone();
        hw.fabric = Some(FabricConfig {
            gpus_per_node: 8,
            scale_up: FabricLink {
                bandwidth: 1e12,
                latency: 1e-6,
                in_network_reduction: false,
            },
            scale_out: None,
        });
        let mut model = base.model.clone();
        model.weights.push(WeightStream {
            precision: Precision::Bf16,
            active_params: 1_000_000,
            resident_params: 64_000_000,
            routing: Some(Routing {
                routed_experts: 64,
                experts_per_tok: 4,
                moe_layers: 24,
            }),
        });
        let mut par = base.parallel.clone();
        par.tp = 4;
        par.ep = 4;
        par.dp_attention = true;
        let eng = ComputeEngine::new(hw.clone(), par.clone(), model.clone());
        let tokens = 1024.0;
        let hidden = tokens * 4096.0 * 2.0;
        // Dense layers: all-gather + reduce-scatter, ring bytes, two calls.
        let ag_rs = 2e-6 + 2.0 * 3.0 / 4.0 * hidden / 1e12;
        // Each rank dispatches 1024/4 tokens × 4 experts × hidden bytes,
        // 3/4 of it to other ranks.
        let per_rank = tokens / 4.0 * 4.0 * 4096.0 * 2.0;
        let a2a = 1e-6 + 3.0 / 4.0 * per_rank / 1e12;
        let expected = 8.0 * ag_rs + 2.0 * 24.0 * a2a;
        assert!((eng.collective_time(1024) - expected).abs() < 1e-12);

        // TP attention: no dispatch — ep = 4 costs what ep = 1 costs.
        par.dp_attention = false;
        let ep4 = ComputeEngine::new(hw.clone(), par.clone(), model.clone());
        par.ep = 1;
        let ep1 = ComputeEngine::new(hw, par, model);
        let allreduce = 1e-6 + 2.0 * 3.0 / 4.0 * hidden / 1e12;
        assert_eq!(ep4.collective_time(1024), ep1.collective_time(1024));
        assert!((ep4.collective_time(1024) - 64.0 * allreduce).abs() < 1e-12);
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
