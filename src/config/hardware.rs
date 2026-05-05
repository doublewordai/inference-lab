use serde::Deserialize;

fn default_gpu_memory_utilization() -> f64 {
    0.9
}

fn default_tp() -> u32 {
    1
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

#[derive(Debug, Clone, Deserialize)]
pub struct HardwareConfig {
    /// Accelerator name (e.g., "H100", "A100")
    pub name: String,

    /// Per-GPU compute capacity in FLOPS (for specific precision, e.g., bf16).
    /// Aggregate cluster compute is `compute_flops * tp`.
    pub compute_flops: f64,

    /// Per-GPU memory bandwidth in bytes/sec. Aggregate cluster bandwidth is
    /// `memory_bandwidth * tp`.
    pub memory_bandwidth: f64,

    /// Per-GPU memory capacity in bytes. Aggregate cluster capacity is
    /// `memory_capacity * tp`.
    pub memory_capacity: u64,

    /// Tensor-parallel group size. Defaults to 1 (single GPU). The aggregate
    /// cluster numbers used for compute/bandwidth/capacity are the per-GPU
    /// fields times this.
    #[serde(default = "default_tp")]
    pub tp: u32,

    /// KV cache capacity in bytes (subset of memory_capacity)
    /// If not specified, calculated from gpu_memory_utilization
    #[serde(default)]
    pub kv_cache_capacity: u64,

    /// Fraction of GPU memory to use (vLLM default: 0.9)
    /// Used to calculate kv_cache_capacity if not explicitly set
    #[serde(default = "default_gpu_memory_utilization")]
    pub gpu_memory_utilization: f64,

    /// Number of bytes per parameter (1 for fp8, 2 for bf16)
    pub bytes_per_param: u32,

    /// Spillover KV cache tiers below HBM, ordered by distance from HBM.
    /// Empty means single-tier (HBM only) — current behaviour.
    #[serde(default)]
    pub kv_tiers: Vec<KVTier>,
}

impl HardwareConfig {
    /// Aggregate compute across the TP group, in FLOPS.
    pub fn aggregate_compute_flops(&self) -> f64 {
        self.compute_flops * self.tp as f64
    }

    /// Aggregate memory bandwidth across the TP group, in bytes/sec.
    pub fn aggregate_memory_bandwidth(&self) -> f64 {
        self.memory_bandwidth * self.tp as f64
    }

    /// Aggregate memory capacity across the TP group, in bytes.
    pub fn aggregate_memory_capacity(&self) -> u64 {
        self.memory_capacity.saturating_mul(self.tp as u64)
    }

    /// Calculate KV cache capacity if not explicitly set
    /// Formula: (aggregate_memory_capacity * gpu_memory_utilization) - model_size
    /// This matches vLLM's behavior: requested_memory - non_kv_cache_memory
    pub fn compute_kv_cache_capacity(&mut self, model_size_bytes: u64) {
        if self.kv_cache_capacity == 0 {
            let requested_memory =
                (self.aggregate_memory_capacity() as f64 * self.gpu_memory_utilization) as u64;
            // In vLLM, non_kv_cache_memory includes weights + activations + overhead
            // For simplicity, we approximate this as just the model weights
            self.kv_cache_capacity = requested_memory.saturating_sub(model_size_bytes);
        }
    }
}
