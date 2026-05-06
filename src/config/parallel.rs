use serde::Deserialize;

fn default_dim() -> u32 {
    1
}

/// Parallelism configuration for a worker pool. Describes how the model is
/// laid out across the GPUs in a TP / EP group; aggregation helpers on
/// `ClusterSpec` use these to scale per-GPU hardware figures up to cluster
/// totals.
#[derive(Debug, Clone, Deserialize)]
pub struct ParallelConfig {
    /// Tensor-parallel group size. Defaults to 1.
    #[serde(default = "default_dim")]
    pub tp: u32,
    /// Expert-parallel group size for MoE layers. Defaults to 1 (experts
    /// replicated across the TP group rather than sharded).
    #[serde(default = "default_dim")]
    pub ep: u32,
    /// DP-attention layout (sglang's `--enable-dp-attention`). When true, the
    /// `tp` ranks run attention in data-parallel mode — each rank holds full
    /// attention weights and a 1/tp shard of sequences, so there is no TP
    /// all-reduce in the per-layer hot path. The `tp` value is then really a
    /// world-size knob, not a tensor-parallel group. EP collectives are
    /// unaffected. Defaults to false (classic TP).
    #[serde(default)]
    pub dp_attention: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self { tp: 1, ep: 1, dp_attention: false }
    }
}

/// Collective-communications cost model. Describes the fabric and per-call
/// fixed overheads used to estimate TP all-reduce and EP all-to-all time per
/// iteration. Optional on `ClusterSpec` and top-level `Config`; if absent
/// the simulator contributes zero collective time (current behaviour).
#[derive(Debug, Clone, Deserialize, Default)]
pub struct CommsConfig {
    /// Per-GPU bandwidth available for collective operations, in bytes/sec.
    /// On NVL72 this is the NVLink5 bidirectional limit (~900 GB/s).
    pub link_bw: f64,
    /// Per-call fixed latency for an all-reduce, in seconds. Captures
    /// kernel launch + NCCL setup overhead. ~5–10 μs is typical on NVLink.
    #[serde(default)]
    pub allreduce_latency: f64,
    /// Per-call fixed latency for an all-to-all, in seconds. Slightly higher
    /// than all-reduce in practice; ~8–15 μs on NVLink.
    #[serde(default)]
    pub alltoall_latency: f64,
}
