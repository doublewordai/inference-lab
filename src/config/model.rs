use crate::config::HardwareConfig;
use serde::Deserialize;

/// Architecture-agnostic cost model. Every consumer in the simulator goes
/// through this trait — there are no architecture knobs (`num_kv_heads`,
/// `sliding_window`, etc.) leaked into the simulator core. Adding a new
/// architecture means adding a new variant to `ModelConfig` and implementing
/// these methods; nothing else changes.
pub trait ModelCosts {
    fn name(&self) -> &str;
    fn max_seq_len(&self) -> u32;

    /// Matmul FLOPs for one forward pass over a single token.
    /// For MoE this counts only active expert params.
    fn matmul_flops_per_token(&self) -> u64;

    /// Self-attention compute (QK^T plus softmax-times-V) for processing
    /// `new_tokens` against `attended_tokens` of context, summed across
    /// layers. Architecture-specific caps (sliding window, sparse top-k)
    /// are applied internally.
    fn attention_flops(&self, new_tokens: u32, attended_tokens: u32) -> u64;

    /// Bytes transferred HBM -> SM for model weights in one forward pass.
    /// For MoE this is active params * effective bytes/param, not total.
    fn weight_transfer_bytes_per_step(&self) -> u64;

    /// Bytes of KV read per decode step for a sequence of `seq_len` tokens.
    /// Captures sliding window, sparse top-k, etc.
    fn kv_bytes_read_per_decode_step(&self, seq_len: u32) -> u64;

    /// Bytes of resident model weights in HBM (capacity accounting). For
    /// MoE this is total params, since all experts must reside in HBM
    /// even if only a subset are activated each step.
    fn weight_residency_bytes(&self) -> u64;

    /// Bytes of resident KV cache for a sequence of `seq_len` tokens.
    fn kv_storage_bytes(&self, seq_len: u32) -> u64;
}

/// Tagged-enum dispatch over architectures. The `type` field in TOML/JSON
/// chooses the variant: `dense`, `sliding`, or `deepseek_v4`.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelConfig {
    Dense(DenseModel),
    Sliding(SlidingWindowModel),
    DeepseekV4(DeepseekV4Model),
}

impl ModelConfig {
    /// Fill in any fields defaulted from hardware (notably `bytes_per_param`).
    /// Called once at config-load time.
    pub fn finalize(&mut self, hardware: &HardwareConfig) {
        match self {
            Self::Dense(m) => m.finalize(hardware),
            Self::Sliding(m) => m.finalize(hardware),
            Self::DeepseekV4(_) => {} // self-contained
        }
    }
}

impl ModelCosts for ModelConfig {
    fn name(&self) -> &str {
        match self {
            Self::Dense(m) => m.name(),
            Self::Sliding(m) => m.name(),
            Self::DeepseekV4(m) => m.name(),
        }
    }
    fn max_seq_len(&self) -> u32 {
        match self {
            Self::Dense(m) => m.max_seq_len(),
            Self::Sliding(m) => m.max_seq_len(),
            Self::DeepseekV4(m) => m.max_seq_len(),
        }
    }
    fn matmul_flops_per_token(&self) -> u64 {
        match self {
            Self::Dense(m) => m.matmul_flops_per_token(),
            Self::Sliding(m) => m.matmul_flops_per_token(),
            Self::DeepseekV4(m) => m.matmul_flops_per_token(),
        }
    }
    fn attention_flops(&self, s: u32, t: u32) -> u64 {
        match self {
            Self::Dense(m) => m.attention_flops(s, t),
            Self::Sliding(m) => m.attention_flops(s, t),
            Self::DeepseekV4(m) => m.attention_flops(s, t),
        }
    }
    fn weight_transfer_bytes_per_step(&self) -> u64 {
        match self {
            Self::Dense(m) => m.weight_transfer_bytes_per_step(),
            Self::Sliding(m) => m.weight_transfer_bytes_per_step(),
            Self::DeepseekV4(m) => m.weight_transfer_bytes_per_step(),
        }
    }
    fn kv_bytes_read_per_decode_step(&self, seq_len: u32) -> u64 {
        match self {
            Self::Dense(m) => m.kv_bytes_read_per_decode_step(seq_len),
            Self::Sliding(m) => m.kv_bytes_read_per_decode_step(seq_len),
            Self::DeepseekV4(m) => m.kv_bytes_read_per_decode_step(seq_len),
        }
    }
    fn weight_residency_bytes(&self) -> u64 {
        match self {
            Self::Dense(m) => m.weight_residency_bytes(),
            Self::Sliding(m) => m.weight_residency_bytes(),
            Self::DeepseekV4(m) => m.weight_residency_bytes(),
        }
    }
    fn kv_storage_bytes(&self, seq_len: u32) -> u64 {
        match self {
            Self::Dense(m) => m.kv_storage_bytes(seq_len),
            Self::Sliding(m) => m.kv_storage_bytes(seq_len),
            Self::DeepseekV4(m) => m.kv_storage_bytes(seq_len),
        }
    }
}

// ---------------------------------------------------------------------------
// Dense / GQA transformer (Llama-3, Qwen, etc.)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct DenseModel {
    pub name: String,
    pub num_parameters: u64,
    #[serde(default)]
    pub num_active_parameters: Option<u64>,
    pub num_layers: u32,
    pub hidden_dim: u32,
    pub num_heads: u32,
    /// GQA / MQA. Defaults to `num_heads` (vanilla MHA).
    #[serde(default)]
    pub num_kv_heads: Option<u32>,
    pub max_seq_len: u32,
    /// Bytes per stored weight; defaults to `hardware.bytes_per_param` at
    /// config-load time.
    #[serde(default)]
    pub bytes_per_param: Option<u32>,
}

impl DenseModel {
    fn finalize(&mut self, hardware: &HardwareConfig) {
        if self.bytes_per_param.is_none() {
            self.bytes_per_param = Some(hardware.bytes_per_param);
        }
    }
    fn bpp(&self) -> u64 {
        self.bytes_per_param.unwrap_or(2) as u64
    }
    fn active_params(&self) -> u64 {
        self.num_active_parameters.unwrap_or(self.num_parameters)
    }
    fn kv_heads(&self) -> u32 {
        self.num_kv_heads.unwrap_or(self.num_heads)
    }
    fn head_dim(&self) -> u32 {
        self.hidden_dim / self.num_heads
    }
    /// KV bytes per token for a single layer (K and V combined).
    fn kv_bytes_per_token_per_layer(&self) -> u64 {
        2 * self.kv_heads() as u64 * self.head_dim() as u64 * self.bpp()
    }
}

impl ModelCosts for DenseModel {
    fn name(&self) -> &str {
        &self.name
    }
    fn max_seq_len(&self) -> u32 {
        self.max_seq_len
    }
    fn matmul_flops_per_token(&self) -> u64 {
        2 * self.active_params()
    }
    fn attention_flops(&self, s: u32, t: u32) -> u64 {
        // 4 * L * S * T * D — both QK^T and softmax-V kernels.
        4u64 * self.num_layers as u64 * s as u64 * t as u64 * self.hidden_dim as u64
    }
    fn weight_transfer_bytes_per_step(&self) -> u64 {
        self.active_params() * self.bpp()
    }
    fn kv_bytes_read_per_decode_step(&self, seq_len: u32) -> u64 {
        self.kv_bytes_per_token_per_layer() * seq_len as u64 * self.num_layers as u64
    }
    fn weight_residency_bytes(&self) -> u64 {
        self.num_parameters * self.bpp()
    }
    fn kv_storage_bytes(&self, seq_len: u32) -> u64 {
        self.kv_bytes_per_token_per_layer() * seq_len as u64 * self.num_layers as u64
    }
}

// ---------------------------------------------------------------------------
// Sliding-window transformer (Mistral-style)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct SlidingWindowModel {
    pub name: String,
    pub num_parameters: u64,
    #[serde(default)]
    pub num_active_parameters: Option<u64>,
    pub num_layers: u32,
    pub hidden_dim: u32,
    pub num_heads: u32,
    #[serde(default)]
    pub num_kv_heads: Option<u32>,
    pub max_seq_len: u32,
    pub sliding_window: u32,
    /// Layers using sliding window; the rest use full attention.
    pub num_sliding_layers: u32,
    #[serde(default)]
    pub bytes_per_param: Option<u32>,
}

impl SlidingWindowModel {
    fn finalize(&mut self, hardware: &HardwareConfig) {
        if self.bytes_per_param.is_none() {
            self.bytes_per_param = Some(hardware.bytes_per_param);
        }
    }
    fn bpp(&self) -> u64 {
        self.bytes_per_param.unwrap_or(2) as u64
    }
    fn active_params(&self) -> u64 {
        self.num_active_parameters.unwrap_or(self.num_parameters)
    }
    fn kv_heads(&self) -> u32 {
        self.num_kv_heads.unwrap_or(self.num_heads)
    }
    fn head_dim(&self) -> u32 {
        self.hidden_dim / self.num_heads
    }
    fn kv_bytes_per_token_per_layer(&self) -> u64 {
        2 * self.kv_heads() as u64 * self.head_dim() as u64 * self.bpp()
    }
    fn num_full_layers(&self) -> u32 {
        self.num_layers.saturating_sub(self.num_sliding_layers)
    }
}

impl ModelCosts for SlidingWindowModel {
    fn name(&self) -> &str {
        &self.name
    }
    fn max_seq_len(&self) -> u32 {
        self.max_seq_len
    }
    fn matmul_flops_per_token(&self) -> u64 {
        2 * self.active_params()
    }
    fn attention_flops(&self, s: u32, t: u32) -> u64 {
        let d = self.hidden_dim as u64;
        let full = self.num_full_layers() as u64;
        let sliding = self.num_sliding_layers as u64;
        let t_sliding = t.min(self.sliding_window) as u64;
        let s = s as u64;
        let t = t as u64;
        4 * d * s * (full * t + sliding * t_sliding)
    }
    fn weight_transfer_bytes_per_step(&self) -> u64 {
        self.active_params() * self.bpp()
    }
    fn kv_bytes_read_per_decode_step(&self, seq_len: u32) -> u64 {
        let per = self.kv_bytes_per_token_per_layer();
        let full = self.num_full_layers() as u64;
        let sliding = self.num_sliding_layers as u64;
        let s_full = seq_len as u64;
        let s_slid = seq_len.min(self.sliding_window) as u64;
        per * (full * s_full + sliding * s_slid)
    }
    fn weight_residency_bytes(&self) -> u64 {
        self.num_parameters * self.bpp()
    }
    fn kv_storage_bytes(&self, seq_len: u32) -> u64 {
        let per = self.kv_bytes_per_token_per_layer();
        let full = self.num_full_layers() as u64;
        let sliding = self.num_sliding_layers as u64;
        let s_full = seq_len as u64;
        let s_slid = seq_len.min(self.sliding_window) as u64;
        per * (full * s_full + sliding * s_slid)
    }
}

// ---------------------------------------------------------------------------
// DeepSeek-V4-Pro: MoE + MLA + sparse / sliding hybrid attention
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct DeepseekV4Model {
    pub name: String,
    /// Total parameters resident in HBM.
    pub num_parameters: u64,
    /// Active parameters per token.
    pub num_active_parameters: u64,
    pub num_layers: u32,
    pub hidden_dim: u32,
    pub num_heads: u32,
    pub max_seq_len: u32,
    /// Width of the MLA latent vector stored per token per layer (K and V
    /// share this single vector in V4 — no factor of 2).
    pub kv_latent_dim: u32,
    /// Bytes per stored KV value (typically FP8 = 1).
    pub kv_bytes_per_value: u32,
    /// Effective bytes per *active* parameter (weighted average over the
    /// FP4/FP8 mix).
    pub effective_bytes_per_active_param: f32,
    /// Effective bytes per *resident* parameter (for HBM accounting, since
    /// total params dominate residency for MoE).
    pub effective_bytes_per_resident_param: f32,
    /// Sparse attention cap: number of historical tokens read per decode
    /// step in non-sliding layers. `None` = full attention.
    #[serde(default)]
    pub attention_topk: Option<u32>,
    /// Sliding window for sliding layers (`None` = no sliding layers).
    #[serde(default)]
    pub sliding_window: Option<u32>,
    /// Number of sliding layers; the remainder use sparse-or-full per
    /// `attention_topk`.
    #[serde(default)]
    pub num_sliding_layers: Option<u32>,
}

impl DeepseekV4Model {
    fn kv_bytes_per_token_per_layer(&self) -> u64 {
        // Single shared latent — no factor of 2 for K/V.
        self.kv_latent_dim as u64 * self.kv_bytes_per_value as u64
    }
    fn num_sliding(&self) -> u32 {
        self.num_sliding_layers.unwrap_or(0)
    }
    fn num_non_sliding(&self) -> u32 {
        self.num_layers.saturating_sub(self.num_sliding())
    }
    /// Tokens read per decode step in non-sliding layers (sparse top-k cap).
    fn non_sliding_attended(&self, seq_len: u32) -> u32 {
        match self.attention_topk {
            Some(k) => seq_len.min(k),
            None => seq_len,
        }
    }
    /// Tokens read per decode step in sliding layers.
    fn sliding_attended(&self, seq_len: u32) -> u32 {
        match self.sliding_window {
            Some(w) => seq_len.min(w),
            None => seq_len,
        }
    }
}

impl ModelCosts for DeepseekV4Model {
    fn name(&self) -> &str {
        &self.name
    }
    fn max_seq_len(&self) -> u32 {
        self.max_seq_len
    }
    fn matmul_flops_per_token(&self) -> u64 {
        2 * self.num_active_parameters
    }
    fn attention_flops(&self, s: u32, t: u32) -> u64 {
        let d = self.hidden_dim as u64;
        let s = s as u64;
        // Non-sliding layers attend over min(t, topk) tokens.
        let t_non = self.non_sliding_attended(t) as u64;
        let t_sld = self.sliding_attended(t) as u64;
        4 * d
            * s
            * (self.num_non_sliding() as u64 * t_non + self.num_sliding() as u64 * t_sld)
    }
    fn weight_transfer_bytes_per_step(&self) -> u64 {
        (self.num_active_parameters as f64 * self.effective_bytes_per_active_param as f64) as u64
    }
    fn kv_bytes_read_per_decode_step(&self, seq_len: u32) -> u64 {
        let per = self.kv_bytes_per_token_per_layer();
        let n_non = self.num_non_sliding() as u64;
        let n_sld = self.num_sliding() as u64;
        let t_non = self.non_sliding_attended(seq_len) as u64;
        let t_sld = self.sliding_attended(seq_len) as u64;
        per * (n_non * t_non + n_sld * t_sld)
    }
    fn weight_residency_bytes(&self) -> u64 {
        (self.num_parameters as f64 * self.effective_bytes_per_resident_param as f64) as u64
    }
    fn kv_storage_bytes(&self, seq_len: u32) -> u64 {
        // Storage isn't capped by topk — only the *read* per step is.
        // Sliding layers do cap storage at the window, since older tokens
        // are evicted.
        let per = self.kv_bytes_per_token_per_layer();
        let n_non = self.num_non_sliding() as u64;
        let n_sld = self.num_sliding() as u64;
        let s_non = seq_len as u64;
        let s_sld = self.sliding_attended(seq_len) as u64;
        per * (n_non * s_non + n_sld * s_sld)
    }
}
