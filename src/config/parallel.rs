use serde::Deserialize;

fn default_dim() -> u32 {
    1
}

/// Parallelism configuration for a worker pool. Describes how the model is
/// laid out across the GPUs in a TP / EP group; aggregation helpers on
/// `ClusterSpec` use these to scale per-GPU hardware figures up to cluster
/// totals.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
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
        Self {
            tp: 1,
            ep: 1,
            dp_attention: false,
        }
    }
}
