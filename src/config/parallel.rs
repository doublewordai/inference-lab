use serde::Deserialize;

fn default_dim() -> u32 {
    1
}

/// How one replica of the model is laid out across its GPUs.
///
/// * `tp` — the replica's world size. Its GPUs pool FLOP rate, HBM bandwidth
///   and memory; weights are sharded across them (each expert's matrices
///   included, unless `ep` says otherwise). Every layer's sharded output is
///   all-reduced: once after attention, once after the FFN.
/// * `ep` — experts sharded across `ep` of the ranks (must divide `tp`);
///   expert reads and FLOPs are taken as balanced across the ranks. With TP
///   attention every rank holds every token, so the MoE output is still
///   combined by the FFN all-reduce (vLLM `--enable-expert-parallel`); the
///   traffic changes only under `dp_attention`.
/// * `dp_attention` — attention runs data-parallel over the `tp` ranks
///   (sglang `--enable-dp-attention`): each rank holds the full attention
///   projections (replicated: `tp×` resident, `tp×` read per step) and its
///   own sequences' KV, and needs no attention all-reduce; the FFN, still
///   TP-sharded, gathers the ranks' tokens with an all-gather and returns
///   them with a reduce-scatter; with `ep > 1` the MoE layers instead
///   dispatch each rank's tokens to the expert-owning ranks and combine
///   them back with all-to-alls over the `ep` group (DeepEP-style).
///
/// Collectives are priced on the hardware's [`super::FabricConfig`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParallelConfig {
    /// Replica world size. Defaults to 1.
    #[serde(default = "default_dim")]
    pub tp: u32,
    /// Expert-parallel group size (divides `tp`). Defaults to 1: experts
    /// TP-sharded like every other weight.
    #[serde(default = "default_dim")]
    pub ep: u32,
    /// Data-parallel attention over the `tp` ranks. Defaults to false.
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
