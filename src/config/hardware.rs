use serde::Deserialize;

fn default_gpu_memory_utilization() -> f64 {
    0.9
}

/// A spillover tier of the KV cache hierarchy. Tier ordering is implicit in
/// the enclosing `Vec<KVTier>` — earlier tiers are conceptually closer to
/// HBM. HBM itself is implicit (driven by `kv_cache_capacity`); tiers in
/// this list represent host RAM, NVMe, remote storage, and so on.
#[derive(Debug, Clone, Deserialize)]
pub struct KVTier {
    /// Human-readable tier name (e.g. "host_ram", "nvme").
    pub name: String,
    /// Capacity of this tier in bytes.
    pub capacity_bytes: u64,
    /// Promotion bandwidth from this tier to HBM, in bytes/sec.
    pub bandwidth_to_hbm: f64,
}

/// Per-GPU spec. Parallelism (TP / EP group sizes) lives on
/// `ParallelConfig`; aggregate cluster figures are computed by
/// `ClusterSpec`'s helpers.
#[derive(Debug, Clone, Deserialize)]
pub struct HardwareConfig {
    /// Accelerator name (e.g., "H100", "A100").
    pub name: String,

    /// Per-GPU compute capacity in FLOPS (for a specific precision, e.g. bf16).
    pub compute_flops: f64,

    /// Per-GPU memory bandwidth in bytes/sec.
    pub memory_bandwidth: f64,

    /// Per-GPU memory capacity in bytes.
    pub memory_capacity: u64,

    /// KV cache capacity in bytes. If left at 0, `ClusterSpec::compute_kv_cache_capacity`
    /// fills it in from `aggregate_memory_capacity * gpu_memory_utilization - model_size`.
    #[serde(default)]
    pub kv_cache_capacity: u64,

    /// Fraction of GPU memory to use (vLLM default: 0.9). Used to derive
    /// `kv_cache_capacity` if not explicitly set.
    #[serde(default = "default_gpu_memory_utilization")]
    pub gpu_memory_utilization: f64,

    /// Number of bytes per parameter (1 for fp8, 2 for bf16).
    pub bytes_per_param: u32,

    /// Spillover KV cache tiers below HBM, ordered by distance from HBM.
    /// Empty means single-tier (HBM only).
    #[serde(default)]
    pub kv_tiers: Vec<KVTier>,
}
