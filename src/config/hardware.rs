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

/// Per-GPU spec. Parallelism (TP / EP group sizes) lives on
/// `ParallelConfig`; aggregate cluster figures are computed by
/// `ClusterSpec`'s helpers.
#[derive(Debug, Clone, Deserialize)]
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

    /// KV cache capacity in bytes. If left at 0, `ClusterSpec::compute_kv_cache_capacity`
    /// fills it in from `aggregate_memory_capacity * gpu_memory_utilization - model_size`.
    #[serde(default)]
    pub kv_cache_capacity: u64,

    /// Fraction of GPU memory to use (vLLM default: 0.9). Used to derive
    /// `kv_cache_capacity` if not explicitly set.
    #[serde(default = "default_gpu_memory_utilization")]
    pub gpu_memory_utilization: f64,

    /// Spillover KV cache tiers below HBM, ordered by distance from HBM.
    /// Empty means single-tier (HBM only).
    #[serde(default)]
    pub kv_tiers: Vec<KVTier>,
}

impl HardwareConfig {
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
