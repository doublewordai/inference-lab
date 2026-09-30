//! Per-iteration roofline. Compute is split into precision-homogeneous
//! streams (e.g. FP4 expert GEMMs and FP8 attention/dense in DeepSeek-V4-Pro).
//! Each stream runs at its own FLOP rate, and we sum stream times since
//! kernels of different precisions are launched serially on the GPU. Memory
//! traffic flows through one HBM, so weight bytes for each precision plus KV
//! bytes (charged to the attention stream) all share the same bandwidth.

use crate::config::{HardwareConfig, ModelSpec, MoeOverlap, ParallelConfig, Precision};
use crate::request::Request;

#[derive(Clone)]
pub struct ComputeEngine {
    hardware: HardwareConfig,
    parallel: ParallelConfig,
    model: ModelSpec,
    /// Per-GPU bandwidth of an active-KV store. `None` means KV is in HBM.
    kv_memory_bandwidth: Option<f64>,
    block_size: u32,
    enable_cascade_attention: bool,
}

/// Work accumulated on one precision stream.
#[derive(Default, Clone, Copy)]
struct StreamAcc {
    flops: f64,
    /// Total bytes in the historical accumulator. Keeping this alongside the
    /// split preserves bit-for-bit default arithmetic when KV remains in HBM.
    bytes: f64,
    /// Bytes read through HBM (weights and fixed per-sequence state).
    hbm_bytes: f64,
    /// Attention KV bytes, read through HBM unless an active tier is set.
    kv_bytes: f64,
    /// Routed-expert subset of `flops` / `hbm_bytes`, split out only when
    /// hidden overlap is selected.
    routed_flops: f64,
    routed_hbm_bytes: f64,
}

impl StreamAcc {
    fn is_empty(&self) -> bool {
        self.flops == 0.0 && self.bytes == 0.0
    }
}

#[derive(Default, Clone, Copy)]
struct CollectiveCost {
    /// Historical single serial accumulator, retained so the default path's
    /// floating-point operation order stays byte-identical.
    serial_total: f64,
    /// Collectives other than routed MoE dispatch/combine.
    other: f64,
    /// Existing serial time for routed MoE dispatch/combine.
    moe_serial: f64,
    /// Bandwidth-only part eligible to overlap with expert execution.
    moe_wire: f64,
    /// Dispatch/combine call floors that remain exposed.
    moe_latency: f64,
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
    /// Non-KV portion of `bytes` read through HBM.
    pub hbm_bytes: f64,
    /// Portion of `bytes` belonging to attention KV. This additionally uses
    /// HBM unless an active-KV tier is configured.
    pub kv_bytes: f64,
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
            kv_memory_bandwidth: None,
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

    fn aggregate_kv_memory_bandwidth(&self) -> f64 {
        self.kv_memory_bandwidth
            .unwrap_or(self.hardware.memory_bandwidth)
            * self.parallel.tp as f64
    }

    /// Place active KV outside HBM and price its reads at `bandwidth` per
    /// GPU. Cluster configuration derives this from the active store's
    /// direct GPU link.
    pub fn with_kv_memory_bandwidth(mut self, bandwidth: f64) -> Self {
        assert!(
            bandwidth.is_finite() && bandwidth > 0.0,
            "active-KV bandwidth must be finite and positive"
        );
        self.kv_memory_bandwidth = Some(bandwidth);
        self
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
    fn collective_cost(&self, total_tokens: u32) -> CollectiveCost {
        let Some(fabric) = &self.hardware.fabric else {
            return CollectiveCost::default();
        };
        let tp = self.parallel.tp;
        if tp <= 1 {
            return CollectiveCost::default();
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

        let mut cost = CollectiveCost::default();
        if !dpa {
            let t = self.model.num_layers() as f64 * fabric.allreduce_time(tp, hidden);
            cost.other += t;
            cost.serial_total += t;
        }
        let dense = ffn_exchange(self.model.dense_ffn_layers());
        cost.other += dense;
        cost.serial_total += dense;
        if ep > 1 && dpa {
            for r in self.model.weights.iter().filter_map(|w| w.routing) {
                let per_rank = if tokens == 0.0 {
                    0.0
                } else {
                    tokens / ep as f64 * r.experts_per_tok as f64 * hidden / tokens
                };
                let calls = 2.0 * r.moe_layers as f64;
                let serial = 2.0 * r.moe_layers as f64 * fabric.alltoall_time(ep, per_rank);
                cost.moe_serial += serial;
                cost.serial_total += serial;
                let (latency, wire) = fabric.alltoall_latency_and_wire_time(ep, per_rank);
                cost.moe_latency += calls * latency;
                cost.moe_wire += calls * wire;
            }
        } else {
            let moe = ffn_exchange(self.model.moe_layers());
            cost.other += moe;
            cost.serial_total += moe;
        }
        cost
    }

    #[cfg(test)]
    fn collective_time(&self, total_tokens: u32) -> f64 {
        self.collective_cost(total_tokens).serial_total
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

        // Preserve the historical per-precision accumulation order for the
        // default serial/HBM path. Routed work is then marked separately so
        // hidden-overlap pricing can isolate the expert kernel.
        for (prec, fpt) in self.model.matmul_flops_per_token_by_prec() {
            streams[prec.index()].flops += total_tokens as f64 * fpt as f64;
        }
        for (prec, bytes) in self.model.weight_bytes_per_step_by_prec(total_tokens) {
            streams[prec.index()].bytes += bytes as f64;
            streams[prec.index()].hbm_bytes += bytes as f64;
        }
        for weight in self
            .model
            .weights
            .iter()
            .filter(|weight| weight.routing.is_some())
        {
            let idx = weight.precision.index();
            let flops = total_tokens as f64 * (2 * weight.active_params) as f64;
            let bytes = (weight.params_read_per_step(total_tokens) as f64
                * weight.bytes_per_value()) as u64 as f64;
            streams[idx].routed_flops += flops;
            streams[idx].routed_hbm_bytes += bytes;
        }
        // DP-attention: every rank holds and reads the full attention
        // projections, so the replica reads them tp times per step.
        if self.parallel.dp_attention && self.parallel.tp > 1 {
            let bytes = (self.parallel.tp - 1) as f64 * self.model.attention_weight_bytes() as f64;
            let stream = &mut streams[self.model.projection_precision().index()];
            stream.bytes += bytes;
            stream.hbm_bytes += bytes;
        }

        let attn_idx = self.model.attention_precision.index();

        // With cascade attention, the KV bytes for the shared prompt prefix
        // are loaded once per iteration instead of once per request.
        let shared_prefix_tokens = if self.enable_cascade_attention && self.block_size > 0 {
            Request::shared_prefix_blocks(batch_requests) * self.block_size
        } else {
            0
        };
        let shared_kv = self
            .model
            .kv_bytes_read_per_decode_step(shared_prefix_tokens) as f64;
        streams[attn_idx].bytes += shared_kv;
        streams[attn_idx].kv_bytes += shared_kv;

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
            let kv_bytes = self.model.kv_bytes_read_per_decode_step(unshared) as f64;
            // This is deliberately one addition, matching the pre-split
            // accumulator and therefore its floating-point result.
            streams[attn_idx].bytes += kv_bytes + state_bytes;
            streams[attn_idx].kv_bytes += kv_bytes;
            streams[attn_idx].hbm_bytes += state_bytes;
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
                hbm_bytes: 0.0,
                kv_bytes: 0.0,
                flops: 0.0,
                compute_time: 0.0,
            };
        }
        let total_tokens: u32 = tokens_per_request.iter().sum();
        let streams = self.assemble_streams(batch_requests, tokens_per_request);
        let bw = self.aggregate_memory_bandwidth();
        let kv_bw = self.aggregate_kv_memory_bandwidth();
        let collectives = self.collective_cost(total_tokens);
        // Both overlap modes hide the dispatch/combine wire behind the expert
        // kernel; they differ only in the exposed per-layer floor added below.
        let overlap_moe = matches!(
            self.parallel.moe_overlap,
            MoeOverlap::Hidden | MoeOverlap::Megakernel
        ) && collectives.moe_serial > 0.0;
        // A megakernel steals SMs from the expert GEMM (comm-CTA fraction) and
        // replaces the collective call floor with a fill/drain + epilogue.
        let mega = if self.parallel.moe_overlap == MoeOverlap::Megakernel {
            Some(self.parallel.megakernel.expect(
                "moe_overlap = megakernel requires megakernel params \
                 (validated at config load)",
            ))
        } else {
            None
        };
        let gemm_rate_scale = mega.map_or(1.0, |p| 1.0 - p.comm_sm_fraction);

        let mut cost = StepCost {
            time: 0.0,
            bytes: 0.0,
            hbm_bytes: 0.0,
            kv_bytes: 0.0,
            flops: 0.0,
            compute_time: 0.0,
        };
        let mut routed_kernel_time = 0.0;
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
            let memory_time = if self.kv_memory_bandwidth.is_some() {
                acc.hbm_bytes / bw + acc.kv_bytes / kv_bw
            } else {
                acc.bytes / bw
            };
            if overlap_moe {
                let base_compute = (acc.flops - acc.routed_flops).max(0.0) / rate;
                let base_memory =
                    (acc.hbm_bytes - acc.routed_hbm_bytes).max(0.0) / bw + acc.kv_bytes / kv_bw;
                cost.time += base_compute.max(base_memory);
                routed_kernel_time +=
                    (acc.routed_flops / (rate * gemm_rate_scale)).max(acc.routed_hbm_bytes / bw);
            } else {
                cost.time += compute_time.max(memory_time);
            }
            cost.compute_time += compute_time;
            cost.hbm_bytes += acc.hbm_bytes;
            cost.kv_bytes += acc.kv_bytes;
            cost.bytes += acc.bytes;
            cost.flops += acc.flops;
        }
        if overlap_moe {
            let moe_floor = match mega {
                // Fill + drain (two signals) plus an epilogue per fused MoE
                // layer, in place of the exposed collective call floors.
                Some(p) => self.model.moe_layers() as f64 * (2.0 * p.signal_latency + p.epilogue),
                None => collectives.moe_latency,
            };
            cost.time +=
                collectives.other + routed_kernel_time.max(collectives.moe_wire) + moe_floor;
        } else {
            cost.time += collectives.serial_total;
        }
        cost
    }

    /// The routed-MoE dispatch/combine collective components for a step of
    /// `total_tokens` tokens: `(serial_total, wire, exposed_call_latency)` in
    /// seconds, summed over the routed layers. `serial_total` is the whole
    /// dispatch+combine time on the critical path in `Serial`; `wire` is the
    /// bandwidth part that hides behind the expert kernel in `Hidden` /
    /// `Megakernel`; `exposed_call_latency` is the collective call floor that
    /// stays exposed in `Hidden`. All zero without a routed stream and fabric.
    /// Exposed so lane-scale analyses can price a fused layer per MoE layer.
    pub fn moe_collective_seconds(&self, total_tokens: u32) -> (f64, f64, f64) {
        let c = self.collective_cost(total_tokens);
        (c.moe_serial, c.moe_wire, c.moe_latency)
    }

    /// Routed-expert kernel time for the batch: `max(expert GEMM FLOPs /
    /// (rate·(1−comm_sm_fraction)), expert weight-read bytes / bandwidth)`
    /// summed over precision streams — the term dispatch/combine overlaps in
    /// `Hidden` / `Megakernel`. At the decode knee this is the weight read
    /// (batch-independent, the megakernel's floor). `comm_sm_fraction` (in
    /// `[0, 1)`) models the GEMM SMs a comm-CTA partition removes; pass 0 for
    /// the un-partitioned rate. Zero without a routed stream.
    pub fn routed_kernel_seconds(
        &self,
        batch_requests: &[&Request],
        tokens_per_request: &[u32],
        comm_sm_fraction: f64,
    ) -> f64 {
        assert!(
            (0.0..1.0).contains(&comm_sm_fraction),
            "comm_sm_fraction must be in [0, 1)"
        );
        let bw = self.aggregate_memory_bandwidth();
        let scale = 1.0 - comm_sm_fraction;
        let streams = self.assemble_streams(batch_requests, tokens_per_request);
        streams
            .iter()
            .enumerate()
            .filter(|(_, acc)| acc.routed_flops != 0.0 || acc.routed_hbm_bytes != 0.0)
            .map(|(i, acc)| {
                let rate = self.aggregate_flop_rate(Precision::ALL[i]).unwrap();
                (acc.routed_flops / (rate * scale)).max(acc.routed_hbm_bytes / bw)
            })
            .sum()
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
        let mut kv_bytes = 0.0;
        let mut hbm_bytes = 0.0;
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
            kv_bytes += self.model.kv_bytes_read_per_decode_step(mean_context) as f64;
            hbm_bytes += state_bytes;
        }
        let memory = match self.kv_memory_bandwidth {
            Some(kv_bw) => hbm_bytes / self.hardware.memory_bandwidth + kv_bytes / kv_bw,
            None => (hbm_bytes + kv_bytes) / self.hardware.memory_bandwidth,
        };
        compute.max(memory)
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
        let memory_time = if self.kv_memory_bandwidth.is_some() {
            cost.hbm_bytes / self.aggregate_memory_bandwidth()
                + cost.kv_bytes / self.aggregate_kv_memory_bandwidth()
        } else {
            cost.bytes / self.aggregate_memory_bandwidth()
        };
        (memory_time / actual_time).min(1.0)
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
        let bw = self.aggregate_kv_memory_bandwidth();
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

    #[test]
    fn hidden_moe_overlap_hides_wire_but_not_call_latency() {
        use crate::config::{FabricConfig, FabricLink, LayerClass, Routing, WeightStream};

        let hardware = HardwareConfig {
            name: "test".into(),
            flops_fp4: None,
            flops_fp8: Some(1e30),
            flops_bf16: Some(1e30),
            flops_fp16: None,
            memory_bandwidth: 1e9,
            memory_capacity: 1_000_000,
            memory: None,
            fabric: Some(FabricConfig {
                gpus_per_node: 4,
                scale_up: FabricLink {
                    bandwidth: 1e12,
                    latency: 5e-6,
                    in_network_reduction: false,
                },
                scale_out: Some(FabricLink {
                    bandwidth: 1e5,
                    latency: 50e-6,
                    in_network_reduction: false,
                }),
            }),
        };
        let model = ModelSpec {
            name: "moe".into(),
            hidden_dim: 16,
            max_seq_len: 1024,
            attention_precision: Precision::Bf16,
            activation_bytes: 2,
            weights: vec![WeightStream {
                precision: Precision::Fp8,
                active_params: 1_000,
                resident_params: 8_000,
                routing: Some(Routing {
                    routed_experts: 8,
                    experts_per_tok: 1,
                    moe_layers: 1,
                }),
            }],
            layers: vec![LayerClass::Linear {
                count: 1,
                state_bytes: 0,
            }],
        };
        let parallel = ParallelConfig {
            tp: 8,
            ep: 8,
            dp_attention: true,
            moe_overlap: MoeOverlap::Serial,
            megakernel: None,
        };
        let request = Request::new("r".into(), 0, 0.0, 8, 1);
        let serial = ComputeEngine::new(hardware.clone(), parallel.clone(), model.clone())
            .step_cost(&[&request], &[8]);
        let mut hidden_parallel = parallel;
        hidden_parallel.moe_overlap = MoeOverlap::Hidden;
        let hidden = ComputeEngine::new(hardware, hidden_parallel, model.clone())
            .step_cost(&[&request], &[8]);

        let routed_kernel = model.weight_bytes_per_step_by_prec(8)[0].1 as f64 / (8.0 * 1e9);
        let per_rank_bytes = 8.0 / 8.0 * 16.0 * 2.0;
        let wire_per_call = (8.0 - 4.0) / 8.0 * per_rank_bytes / 1e5;
        let serial_expected = routed_kernel + 2.0 * (50e-6 + wire_per_call);
        let hidden_expected = routed_kernel.max(2.0 * wire_per_call) + 2.0 * 50e-6;
        assert!((serial.time - serial_expected).abs() < 1e-12);
        assert!((hidden.time - hidden_expected).abs() < 1e-12);
        assert!(hidden.time < serial.time);
    }

    /// The megakernel mode hides the wire like `Hidden` but replaces the
    /// exposed collective call floor with a per-MoE-layer fill/drain + epilogue
    /// (`moe_layers × (2·signal + epilogue)`). With `signal = scale_out.latency`
    /// and `epilogue = 0` it collapses onto `Hidden` (one MoE layer, two calls
    /// per layer); a smaller signal is strictly faster.
    #[test]
    fn megakernel_replaces_call_floor_with_fill_drain() {
        use crate::config::{
            FabricConfig, FabricLink, LayerClass, MegakernelParams, Routing, WeightStream,
        };

        let hardware = HardwareConfig {
            name: "test".into(),
            flops_fp4: None,
            flops_fp8: Some(1e30),
            flops_bf16: Some(1e30),
            flops_fp16: None,
            memory_bandwidth: 1e9,
            memory_capacity: 1_000_000,
            memory: None,
            fabric: Some(FabricConfig {
                gpus_per_node: 4,
                scale_up: FabricLink {
                    bandwidth: 1e12,
                    latency: 5e-6,
                    in_network_reduction: false,
                },
                scale_out: Some(FabricLink {
                    bandwidth: 1e5,
                    latency: 50e-6,
                    in_network_reduction: false,
                }),
            }),
        };
        let model = ModelSpec {
            name: "moe".into(),
            hidden_dim: 16,
            max_seq_len: 1024,
            attention_precision: Precision::Bf16,
            activation_bytes: 2,
            weights: vec![WeightStream {
                precision: Precision::Fp8,
                active_params: 1_000,
                resident_params: 8_000,
                routing: Some(Routing {
                    routed_experts: 8,
                    experts_per_tok: 1,
                    moe_layers: 1,
                }),
            }],
            layers: vec![LayerClass::Linear {
                count: 1,
                state_bytes: 0,
            }],
        };
        let base = ParallelConfig {
            tp: 8,
            ep: 8,
            dp_attention: true,
            moe_overlap: MoeOverlap::Hidden,
            megakernel: None,
        };
        let request = Request::new("r".into(), 0, 0.0, 8, 1);
        let hidden = ComputeEngine::new(hardware.clone(), base.clone(), model.clone())
            .step_cost(&[&request], &[8]);

        // signal = call floor, no epilogue → exactly Hidden.
        let matched = ParallelConfig {
            moe_overlap: MoeOverlap::Megakernel,
            megakernel: Some(MegakernelParams {
                signal_latency: 50e-6,
                epilogue: 0.0,
                comm_sm_fraction: 0.0,
            }),
            ..base.clone()
        };
        let mega_matched = ComputeEngine::new(hardware.clone(), matched, model.clone())
            .step_cost(&[&request], &[8]);
        assert!(
            (mega_matched.time - hidden.time).abs() < 1e-15,
            "{mega_matched:?} {hidden:?}"
        );

        // Smaller signal + small epilogue → cheaper than Hidden by the floor
        // delta: hidden floor 2×50µs=100µs vs mega 1×(2×1µs+2µs)=4µs.
        let fast = ParallelConfig {
            moe_overlap: MoeOverlap::Megakernel,
            megakernel: Some(MegakernelParams {
                signal_latency: 1e-6,
                epilogue: 2e-6,
                comm_sm_fraction: 0.0,
            }),
            ..base.clone()
        };
        let mega_fast = ComputeEngine::new(hardware, fast, model).step_cost(&[&request], &[8]);
        assert!((hidden.time - mega_fast.time - (100e-6 - 4e-6)).abs() < 1e-15);
        assert!(mega_fast.time < hidden.time);
    }

    /// The public MoE accessors reconstruct the closed form, and
    /// `comm_sm_fraction` reduces the routed kernel time only when the expert
    /// GEMM is compute-bound (weight-read-bound decode is unchanged).
    #[test]
    fn moe_accessors_and_comm_sm_fraction() {
        use crate::config::{FabricConfig, FabricLink, LayerClass, Routing, WeightStream};

        let fabric = FabricConfig {
            gpus_per_node: 4,
            scale_up: FabricLink {
                bandwidth: 1e12,
                latency: 5e-6,
                in_network_reduction: false,
            },
            scale_out: Some(FabricLink {
                bandwidth: 1e5,
                latency: 50e-6,
                in_network_reduction: false,
            }),
        };
        // Weight-read-bound routed stream (huge FLOP rate, small bandwidth).
        let mem_bound_hw = HardwareConfig {
            name: "mem".into(),
            flops_fp4: None,
            flops_fp8: Some(1e30),
            flops_bf16: Some(1e30),
            flops_fp16: None,
            memory_bandwidth: 1e9,
            memory_capacity: 1_000_000,
            memory: None,
            fabric: Some(fabric),
        };
        let model = ModelSpec {
            name: "moe".into(),
            hidden_dim: 16,
            max_seq_len: 1024,
            attention_precision: Precision::Bf16,
            activation_bytes: 2,
            weights: vec![WeightStream {
                precision: Precision::Fp8,
                active_params: 1_000,
                resident_params: 8_000,
                routing: Some(Routing {
                    routed_experts: 8,
                    experts_per_tok: 1,
                    moe_layers: 1,
                }),
            }],
            layers: vec![LayerClass::Linear {
                count: 1,
                state_bytes: 0,
            }],
        };
        let parallel = ParallelConfig {
            tp: 8,
            ep: 8,
            dp_attention: true,
            moe_overlap: MoeOverlap::Hidden,
            megakernel: None,
        };
        let req = Request::new("r".into(), 0, 0.0, 8, 1);
        let eng = ComputeEngine::new(mem_bound_hw, parallel.clone(), model.clone());

        // moe_collective_seconds: serial = wire + latency of two calls.
        let (serial, wire, latency) = eng.moe_collective_seconds(8);
        let per_rank_bytes = 8.0 / 8.0 * 16.0 * 2.0;
        let wire_per_call = (8.0 - 4.0) / 8.0 * per_rank_bytes / 1e5;
        assert!((wire - 2.0 * wire_per_call).abs() < 1e-15);
        assert!((latency - 2.0 * 50e-6).abs() < 1e-15);
        assert!((serial - (wire + latency)).abs() < 1e-15);

        // routed_kernel_seconds: weight read, unaffected by comm_sm_fraction.
        let wr = model.weight_bytes_per_step_by_prec(8)[0].1 as f64 / (8.0 * 1e9);
        assert!((eng.routed_kernel_seconds(&[&req], &[8], 0.0) - wr).abs() < 1e-18);
        assert!((eng.routed_kernel_seconds(&[&req], &[8], 0.5) - wr).abs() < 1e-18);

        // Compute-bound routed stream (tiny FLOP rate): halving the GEMM rate
        // via comm_sm_fraction doubles the routed kernel time.
        let compute_bound_hw = HardwareConfig {
            name: "cmp".into(),
            flops_fp4: None,
            flops_fp8: Some(1e3),
            flops_bf16: Some(1e3),
            flops_fp16: None,
            memory_bandwidth: 1e30,
            memory_capacity: 1_000_000,
            memory: None,
            fabric: Some(fabric),
        };
        let eng2 = ComputeEngine::new(compute_bound_hw, parallel, model);
        let full = eng2.routed_kernel_seconds(&[&req], &[8], 0.0);
        let half = eng2.routed_kernel_seconds(&[&req], &[8], 0.5);
        assert!(full > 0.0);
        assert!((half - 2.0 * full).abs() < 1e-18, "{half} {full}");
    }

    #[test]
    fn active_kv_bandwidth_prices_kv_reads_without_moving_weight_reads() {
        let model = crate::catalog::model("glm-5.2-fp8").unwrap();
        let hardware = crate::catalog::hardware("gh200").unwrap();
        let parallel = ParallelConfig::default();
        let hbm = ComputeEngine::new(hardware.clone(), parallel.clone(), model.clone());
        let grace =
            ComputeEngine::new(hardware.clone(), parallel, model).with_kv_memory_bandwidth(4.5e11);
        let request = create_test_request("r", 65_536, 1);
        let hbm_cost = hbm.step_cost(&[&request], &[1]);
        let grace_cost = grace.step_cost(&[&request], &[1]);

        assert_eq!(hbm_cost.hbm_bytes, grace_cost.hbm_bytes);
        assert_eq!(hbm_cost.kv_bytes, grace_cost.kv_bytes);
        assert!(grace_cost.time > hbm_cost.time);
        let hbm_delta = hbm.kv_read_seq_delta_seconds(&[&request], 0);
        let grace_delta = grace.kv_read_seq_delta_seconds(&[&request], 0);
        assert!((grace_delta / hbm_delta - hardware.memory_bandwidth / 4.5e11).abs() < 1e-9);
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
            hbm_bytes: 1e12,
            kv_bytes: 0.0,
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
