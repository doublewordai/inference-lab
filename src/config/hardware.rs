use serde::Deserialize;

fn default_gpu_memory_utilization() -> f64 {
    0.9
}

#[derive(Debug, Clone, Deserialize)]
pub struct HardwareConfig {
    /// Accelerator name (e.g., "H100", "A100")
    pub name: String,

    /// Compute capacity in FLOPS (for specific precision, e.g., bf16)
    pub compute_flops: f64,

    /// Memory bandwidth in bytes/sec
    pub memory_bandwidth: f64,

    /// Total memory capacity in bytes
    pub memory_capacity: u64,

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

    /// Compute-bound threshold (derived from flops/bandwidth ratio)
    /// This is calculated: bytes_per_param * compute_flops / memory_bandwidth
    #[serde(skip)]
    pub compute_bound_threshold: u32,
}

impl HardwareConfig {
    /// Calculate and set the compute-bound threshold
    pub fn compute_threshold(&mut self) {
        self.compute_bound_threshold =
            (self.bytes_per_param as f64 * self.compute_flops / self.memory_bandwidth) as u32;
    }

    /// Calculate KV cache capacity if not explicitly set
    /// Formula: (memory_capacity * gpu_memory_utilization) - model_size
    /// This matches vLLM's behavior: requested_memory - non_kv_cache_memory
    pub fn compute_kv_cache_capacity(&mut self, model_size_bytes: u64) {
        if self.kv_cache_capacity == 0 {
            let requested_memory = (self.memory_capacity as f64 * self.gpu_memory_utilization) as u64;
            // In vLLM, non_kv_cache_memory includes weights + activations + overhead
            // For simplicity, we approximate this as just the model weights
            self.kv_cache_capacity = requested_memory.saturating_sub(model_size_bytes);
        }
    }

    /// Initialize with threshold pre-computed
    pub fn with_threshold(mut self) -> Self {
        self.compute_threshold();
        self
    }
}
