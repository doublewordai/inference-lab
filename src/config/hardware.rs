use serde::Deserialize;

use super::MemoryTemplate;

/// Numeric precision a kernel runs at. Hardware tracks one FLOP rate per
/// precision; models declare which precision each compute stream uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    Fp4,
    Fp8,
    Bf16,
    Fp16,
    Fp32,
}

impl Precision {
    /// Every precision, in `index()` order.
    pub const ALL: [Precision; 5] = [
        Precision::Fp4,
        Precision::Fp8,
        Precision::Bf16,
        Precision::Fp16,
        Precision::Fp32,
    ];
    pub const COUNT: usize = Self::ALL.len();

    /// Dense index for per-precision tables.
    pub fn index(self) -> usize {
        match self {
            Precision::Fp4 => 0,
            Precision::Fp8 => 1,
            Precision::Bf16 => 2,
            Precision::Fp16 => 3,
            Precision::Fp32 => 4,
        }
    }

    /// Bytes-per-value at this precision. FP4 is sub-byte; we model it as 0.5
    /// because fractional bytes only ever appear weighted by parameter counts
    /// of order 1e9+ — the rounding loss is negligible.
    pub fn bytes_per_value(&self) -> f64 {
        match self {
            Self::Fp4 => 0.5,
            Self::Fp8 => 1.0,
            Self::Bf16 | Self::Fp16 => 2.0,
            Self::Fp32 => 4.0,
        }
    }
}

/// Calibration of a deployment's step time against a measured engine:
/// `t = alpha × t_roofline + beta`. `alpha` is the kernel-efficiency gap to
/// the roofline (dominant at large batch), `beta` the fixed per-iteration
/// cost — scheduler, CPU, kernel launch — (dominant at small batch). Off by
/// default: without it every step is the datasheet roofline. Applied to the
/// executed step only, never to speculative policy candidate pricing, and
/// never on top of a measured step-cost table.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TimeCorrection {
    /// Multiplier on the roofline step time. Defaults to 1.
    #[serde(default = "one")]
    pub alpha: f64,
    /// Seconds added to every step. Defaults to 0.
    #[serde(default)]
    pub beta: f64,
}

fn one() -> f64 {
    1.0
}

impl TimeCorrection {
    pub fn validate(&self) -> Result<(), String> {
        if !(self.alpha.is_finite() && self.alpha > 0.0) {
            return Err(format!(
                "time_correction.alpha must be > 0 (got {})",
                self.alpha
            ));
        }
        if !(self.beta.is_finite() && self.beta >= 0.0) {
            return Err(format!(
                "time_correction.beta must be >= 0 (got {})",
                self.beta
            ));
        }
        Ok(())
    }
}

/// One tier of the collective fabric: what a GPU can inject into it and
/// what a collective call costs before any byte moves.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FabricLink {
    /// Per-GPU injection bandwidth, bytes/s per direction.
    pub bandwidth: f64,
    /// Fixed cost per collective call, seconds (launch + synchronisation).
    #[serde(default)]
    pub latency: f64,
    /// The fabric reduces in-network (NVLink SHARP / NVLS, IB SHARP): an
    /// all-reduce moves each rank's vector once in and once out, `bytes /
    /// bandwidth`, instead of the ring's `2(g-1)/g × bytes`.
    #[serde(default)]
    pub in_network_reduction: bool,
}

impl FabricLink {
    /// Bytes each rank moves per direction in an all-reduce of `bytes` over
    /// `g` ranks, as a multiple of `bytes`.
    fn allreduce_factor(&self, g: u32) -> f64 {
        if self.in_network_reduction {
            1.0
        } else {
            2.0 * (g - 1) as f64 / g as f64
        }
    }
}

/// The collective fabric a GPU sits in. `scale_up` is the switched fabric
/// inside a node (NVLink / NVSwitch: any GPU reaches any peer at its
/// injection rate); `scale_out` is the rail-optimised network across nodes
/// (GPU *i* drives NIC *i*, so cross-node traffic is bounded per GPU by its
/// NIC). Ranks of a parallel group are assumed packed node by node; a group
/// larger than `gpus_per_node` runs hierarchical collectives and needs
/// `scale_out`.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FabricConfig {
    /// GPUs sharing one scale-up domain.
    pub gpus_per_node: u32,
    pub scale_up: FabricLink,
    #[serde(default)]
    pub scale_out: Option<FabricLink>,
}

impl FabricConfig {
    /// Whether a group of `g` ranks can be priced: it fits one node, or
    /// `scale_out` is given.
    pub fn supports_group(&self, g: u32) -> bool {
        g <= self.gpus_per_node.max(1) || self.scale_out.is_some()
    }

    /// Time of one all-reduce of a `bytes`-long vector held by every one of
    /// `g` ranks. Inside a node: one call at the scale-up rate moving
    /// `allreduce_factor × bytes` per rank (ring `2(g-1)/g`, or 1 with
    /// in-network reduction). Across nodes: reduce-scatter and all-gather
    /// inside each node (`(k-1)/k × bytes` each) around an all-reduce of the
    /// `bytes/k` shard over the `n = ceil(g/k)` nodes on each rank's NIC.
    pub fn allreduce_time(&self, g: u32, bytes: f64) -> f64 {
        self.reduce_time(g, bytes, true, 1.0)
    }

    /// Time of an all-gather followed by a reduce-scatter of a `bytes`-long
    /// vector over `g` ranks (DP-attention feeding a TP-sharded FFN): the
    /// ring all-reduce's traffic, no in-network reduction, two calls'
    /// latency.
    pub fn allgather_reducescatter_time(&self, g: u32, bytes: f64) -> f64 {
        self.reduce_time(g, bytes, false, 2.0)
    }

    fn reduce_time(&self, g: u32, bytes: f64, in_network_ok: bool, calls: f64) -> f64 {
        if g <= 1 {
            return 0.0;
        }
        let k = self.gpus_per_node.max(1);
        let up = self.scale_up;
        let factor = |link: FabricLink, g: u32| {
            if in_network_ok {
                link.allreduce_factor(g)
            } else {
                2.0 * (g - 1) as f64 / g as f64
            }
        };
        if g <= k {
            return calls * up.latency + factor(up, g) * bytes / up.bandwidth;
        }
        let out = self
            .scale_out
            .expect("group spans nodes: validated by ClusterSpec::validate");
        let n = g.div_ceil(k);
        let kf = k as f64;
        2.0 * (calls * up.latency + (kf - 1.0) / kf * bytes / up.bandwidth)
            + calls * out.latency
            + factor(out, n) * (bytes / kf) / out.bandwidth
    }

    /// Time of one all-to-all where each of `g` ranks holds `per_rank_bytes`
    /// and sends an equal `1/g` share to every rank. Inside a node:
    /// `(g-1)/g × per_rank_bytes` at the scale-up rate. Across nodes the
    /// in-node share (`(k-1)/g`) and the cross-node share (`(g-k)/g`) move
    /// concurrently on their own links; the slower one bounds the call.
    pub fn alltoall_time(&self, g: u32, per_rank_bytes: f64) -> f64 {
        if g <= 1 {
            return 0.0;
        }
        let k = self.gpus_per_node.max(1);
        let up = self.scale_up;
        let gf = g as f64;
        if g <= k {
            return up.latency + (gf - 1.0) / gf * per_rank_bytes / up.bandwidth;
        }
        let out = self
            .scale_out
            .expect("group spans nodes: validated by ClusterSpec::validate");
        let kf = k as f64;
        let intra = up.latency + (kf - 1.0) / gf * per_rank_bytes / up.bandwidth;
        let inter = out.latency + (gf - kf) / gf * per_rank_bytes / out.bandwidth;
        intra.max(inter)
    }

    /// The fixed call latency and byte-transfer portion of a cross-rank
    /// all-to-all. This decomposition is used by overlap-aware MoE pricing:
    /// the wire portion may hide behind expert execution, but the call floor
    /// remains exposed. For a group spanning nodes, `scale_out.latency` is
    /// the per-op floor and the slower of the scale-up and scale-out byte
    /// paths is the wire portion.
    pub fn alltoall_latency_and_wire_time(&self, g: u32, per_rank_bytes: f64) -> (f64, f64) {
        if g <= 1 {
            return (0.0, 0.0);
        }
        let k = self.gpus_per_node.max(1);
        let up = self.scale_up;
        let gf = g as f64;
        if g <= k {
            return (up.latency, (gf - 1.0) / gf * per_rank_bytes / up.bandwidth);
        }
        let out = self
            .scale_out
            .expect("group spans nodes: validated by ClusterSpec::validate");
        let kf = k as f64;
        // Keep the scale-out call floor exposed. If an unusually slow
        // scale-up path would dominate despite its lower call floor, fold
        // only that residual into the wire portion so latency + wire stays
        // exactly equal to `alltoall_time`.
        let intra =
            (up.latency + (kf - 1.0) / gf * per_rank_bytes / up.bandwidth - out.latency).max(0.0);
        let inter = (gf - kf) / gf * per_rank_bytes / out.bandwidth;
        (out.latency, intra.max(inter))
    }
}

/// Per-GPU physical spec: what the accelerator can do, independent of how
/// it is deployed. Parallelism (TP / EP group sizes) lives on
/// `ParallelConfig`, memory-utilisation settings on `SchedulerConfig`, and
/// aggregate cluster figures are computed by `ClusterSpec`'s helpers. Named
/// presets ship in the crate's catalog (`inference_lab::catalog`).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareConfig {
    /// Accelerator name (e.g., "H100", "A100").
    pub name: String,

    /// Per-GPU dense FLOPS at FP4. Optional — `None` means the hardware does
    /// not natively support that precision; any model that declares an FP4
    /// stream against this hardware will fail at config time.
    #[serde(default)]
    pub flops_fp4: Option<f64>,
    /// Per-GPU dense FLOPS at FP8.
    #[serde(default)]
    pub flops_fp8: Option<f64>,
    /// Per-GPU dense FLOPS at BF16.
    #[serde(default)]
    pub flops_bf16: Option<f64>,
    /// Per-GPU dense FLOPS at FP16.
    #[serde(default)]
    pub flops_fp16: Option<f64>,

    /// Per-GPU memory bandwidth in bytes/sec.
    pub memory_bandwidth: f64,

    /// Per-GPU memory capacity in bytes.
    pub memory_capacity: u64,

    /// KV memory beyond HBM this node class offers — stores (host memory,
    /// NVMe, …) and the links that reach them (see [`MemoryTemplate`]).
    /// Which of them a deployment uses is the deployment's `[memory]`.
    #[serde(default)]
    pub memory: Option<MemoryTemplate>,

    /// Collective fabric (see [`FabricConfig`]). Required to price TP / EP
    /// collectives: a deployment with `tp > 1` or `ep > 1` on hardware
    /// without one is rejected.
    #[serde(default)]
    pub fabric: Option<FabricConfig>,
}

impl HardwareConfig {
    /// GPUs sharing a node's `per = "node"` memory: the memory template's
    /// `gpus_per_node`, else the fabric's, else 1.
    pub fn gpus_per_node(&self) -> u32 {
        self.memory
            .as_ref()
            .and_then(|m| m.gpus_per_node)
            .or_else(|| self.fabric.as_ref().map(|f| f.gpus_per_node))
            .unwrap_or(1)
            .max(1)
    }

    /// Per-GPU FLOP rate at the given precision, or `None` if the hardware
    /// does not declare a rate for it.
    pub fn flop_rate(&self, prec: Precision) -> Option<f64> {
        match prec {
            Precision::Fp4 => self.flops_fp4,
            Precision::Fp8 => self.flops_fp8,
            Precision::Bf16 => self.flops_bf16,
            Precision::Fp16 => self.flops_fp16,
            // Treat FP32 as 1/2 of FP16 if FP16 is set (typical Tensor Core
            // ratio); otherwise unknown.
            Precision::Fp32 => self.flops_fp16.map(|x| x / 2.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fabric(in_network: bool) -> FabricConfig {
        FabricConfig {
            gpus_per_node: 8,
            scale_up: FabricLink {
                bandwidth: 1e12,
                latency: 1e-6,
                in_network_reduction: in_network,
            },
            scale_out: Some(FabricLink {
                bandwidth: 1e11,
                latency: 1e-5,
                in_network_reduction: false,
            }),
        }
    }

    #[test]
    fn allreduce_inside_a_node_is_one_call() {
        // Ring: 2(g-1)/g × 8 MB at 1 TB/s = 14 µs, plus 1 µs latency.
        let t = fabric(false).allreduce_time(8, 8e6);
        assert!((t - 15e-6).abs() < 1e-12, "{t}");
        // In-network reduction: the vector moves once, 8 µs + 1 µs.
        let t = fabric(true).allreduce_time(8, 8e6);
        assert!((t - 9e-6).abs() < 1e-12, "{t}");
        assert_eq!(fabric(true).allreduce_time(1, 8e6), 0.0);
    }

    #[test]
    fn allreduce_across_nodes_is_hierarchical() {
        // 16 ranks on 8-GPU nodes: two in-node phases of 7/8 × 8 MB at 1 TB/s
        // (7 µs + 1 µs each) around a 2-node ring of the 1 MB shard on the
        // NIC: 2(1/2) × 1 MB / 100 GB/s = 10 µs + 10 µs latency.
        let t = fabric(false).allreduce_time(16, 8e6);
        assert!((t - (2.0 * 8e-6 + 20e-6)).abs() < 1e-12, "{t}");
    }

    #[test]
    fn alltoall_splits_between_fabrics() {
        // In node: 7/8 × 8 MB per rank at 1 TB/s = 7 µs + 1 µs.
        let t = fabric(false).alltoall_time(8, 8e6);
        assert!((t - 8e-6).abs() < 1e-12, "{t}");
        // 16 ranks: 7/16 of 8 MB stays in-node (3.5 µs + 1), 8/16 crosses
        // (4 MB / 100 GB/s = 40 µs + 10); the NIC bounds the call.
        let t = fabric(false).alltoall_time(16, 8e6);
        assert!((t - 50e-6).abs() < 1e-12, "{t}");
        let (latency, wire) = fabric(false).alltoall_latency_and_wire_time(16, 8e6);
        assert_eq!(latency, 1e-5);
        assert!((wire - 40e-6).abs() < 1e-12, "{wire}");
    }

    #[test]
    fn groups_beyond_a_node_need_scale_out() {
        let mut f = fabric(false);
        assert!(f.supports_group(8));
        assert!(f.supports_group(16));
        f.scale_out = None;
        assert!(f.supports_group(8));
        assert!(!f.supports_group(9));
    }
}
