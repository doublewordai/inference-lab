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

    /// Residual-stream dimension. Surfaced on the trait because the comms
    /// model needs it to size collective volumes.
    fn hidden_dim(&self) -> u32;

    /// Number of transformer layers. Needed by the comms model to count
    /// per-layer collectives.
    fn num_layers(&self) -> u32;

    /// Bytes per activation element on the wire. Defaults to 2 (bf16);
    /// override to 1 when collectives run at FP8.
    fn activation_bytes(&self) -> u32 {
        2
    }

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

    /// Activation bytes per token transferred in a single TP all-reduce.
    /// Default = `hidden_dim × activation_bytes`. The caller multiplies by
    /// the per-rank ring factor `2(tp-1)/tp` and divides by link bandwidth.
    fn allreduce_bytes_per_token(&self) -> u64 {
        self.hidden_dim() as u64 * self.activation_bytes() as u64
    }

    /// Number of TP all-reduces per forward pass. Default = `2 × num_layers`
    /// (post-attention + post-MLP per layer).
    fn num_tp_allreduces_per_pass(&self) -> u32 {
        2 * self.num_layers()
    }

    /// Activation bytes per token transferred in one direction of an EP
    /// all-to-all. Zero for dense / non-MoE models. The caller multiplies by
    /// the per-rank factor `(ep-1)/ep` and divides by link bandwidth.
    fn alltoall_bytes_per_token(&self) -> u64 {
        0
    }

    /// Number of EP all-to-alls per forward pass (= `2 × num_moe_layers`
    /// for standard MoE — dispatch and combine). Zero for dense models.
    fn num_ep_alltoalls_per_pass(&self) -> u32 {
        0
    }
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
    fn hidden_dim(&self) -> u32 {
        match self {
            Self::Dense(m) => m.hidden_dim(),
            Self::Sliding(m) => m.hidden_dim(),
            Self::DeepseekV4(m) => m.hidden_dim(),
        }
    }
    fn num_layers(&self) -> u32 {
        match self {
            Self::Dense(m) => m.num_layers(),
            Self::Sliding(m) => m.num_layers(),
            Self::DeepseekV4(m) => m.num_layers(),
        }
    }
    fn activation_bytes(&self) -> u32 {
        match self {
            Self::Dense(m) => m.activation_bytes(),
            Self::Sliding(m) => m.activation_bytes(),
            Self::DeepseekV4(m) => m.activation_bytes(),
        }
    }
    fn alltoall_bytes_per_token(&self) -> u64 {
        match self {
            Self::Dense(m) => m.alltoall_bytes_per_token(),
            Self::Sliding(m) => m.alltoall_bytes_per_token(),
            Self::DeepseekV4(m) => m.alltoall_bytes_per_token(),
        }
    }
    fn num_ep_alltoalls_per_pass(&self) -> u32 {
        match self {
            Self::Dense(m) => m.num_ep_alltoalls_per_pass(),
            Self::Sliding(m) => m.num_ep_alltoalls_per_pass(),
            Self::DeepseekV4(m) => m.num_ep_alltoalls_per_pass(),
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
    fn hidden_dim(&self) -> u32 {
        self.hidden_dim
    }
    fn num_layers(&self) -> u32 {
        self.num_layers
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
    fn hidden_dim(&self) -> u32 {
        self.hidden_dim
    }
    fn num_layers(&self) -> u32 {
        self.num_layers
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
// DeepSeek-V4-Pro: MoE + MLA + per-layer compressed-history attention
// ---------------------------------------------------------------------------
//
// Every layer reads a small `window_size` of recent tokens (always dense).
// On top of that, each layer is one of three classes:
//   - dense (`compress_ratio = 0`): window only, no compressed history
//   - near  (`compress_ratio = near_compress_ratio`, e.g. 4): window
//           plus an `Indexer` that picks `index_topk` of the
//           `seq_len / near_compress_ratio` compressed positions
//   - far   (`compress_ratio = far_compress_ratio`, e.g. 128): window
//           plus the entire stride-compressed history
// The Indexer is itself a small attention-like scoring module
// (`index_n_heads` × `index_head_dim`) running over every compressed
// candidate on each near layer, with its own KV cache. Its cost is
// non-negligible at high concurrency.

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
    /// share this single vector in V4 — no factor of 2). Matches `head_dim`
    /// in the upstream config.
    pub kv_latent_dim: u32,
    /// Bytes per stored KV value (typically FP8 = 1).
    pub kv_bytes_per_value: u32,
    /// Effective bytes per *active* parameter (weighted average over the
    /// FP4/FP8 mix).
    pub effective_bytes_per_active_param: f32,
    /// Effective bytes per *resident* parameter (for HBM accounting, since
    /// total params dominate residency for MoE).
    pub effective_bytes_per_resident_param: f32,
    /// Sliding window of recent tokens, present in every layer.
    pub window_size: u32,
    /// Layers with no compressed history (`compress_ratio = 0`).
    pub num_dense_layers: u32,
    /// Layers with `compress_ratio = near_compress_ratio` and an Indexer.
    pub num_near_layers: u32,
    /// Layers with `compress_ratio = far_compress_ratio` and no Indexer.
    pub num_far_layers: u32,
    /// Compression stride for near layers.
    pub near_compress_ratio: u32,
    /// Compression stride for far layers.
    pub far_compress_ratio: u32,
    /// Indexer top-k cap on near layers' compressed history.
    pub index_topk: u32,
    /// Indexer scoring head count (per near layer).
    pub index_n_heads: u32,
    /// Indexer scoring head dim (per near layer).
    pub index_head_dim: u32,
    /// Bytes per indexer KV value. Defaults to `kv_bytes_per_value` if
    /// unset.
    #[serde(default)]
    pub index_kv_bytes_per_value: Option<u32>,
    /// MoE: number of experts each token is routed to. Used by the EP
    /// all-to-all cost model.
    pub num_experts_per_tok: u32,
    /// MoE: number of layers that perform expert-parallel routing. For
    /// DSv4-Pro this is every layer (the dense compress_ratio=0 layer also
    /// runs MoE).
    pub num_moe_layers: u32,
}

impl DeepseekV4Model {
    fn kv_bytes_per_token_per_layer(&self) -> u64 {
        // Single shared latent — no factor of 2 for K/V.
        self.kv_latent_dim as u64 * self.kv_bytes_per_value as u64
    }
    fn index_kv_bpp(&self) -> u64 {
        self.index_kv_bytes_per_value.unwrap_or(self.kv_bytes_per_value) as u64
    }
    fn window_attended(&self, seq_len: u32) -> u32 {
        seq_len.min(self.window_size)
    }
    /// Compressed-history tokens attended per decode step in near layers.
    fn near_compressed_attended(&self, seq_len: u32) -> u32 {
        (seq_len / self.near_compress_ratio).min(self.index_topk)
    }
    /// Compressed-history tokens attended per decode step in far layers
    /// (no top-k cap; reads the entire stride-compressed history).
    fn far_compressed_attended(&self, seq_len: u32) -> u32 {
        seq_len / self.far_compress_ratio
    }
    /// Compressed positions held in a near layer's auxiliary KV cache
    /// (used both for storage and for the Indexer's scoring KV cache).
    fn near_compressed_positions(&self, seq_len: u32) -> u32 {
        seq_len / self.near_compress_ratio
    }
    fn far_compressed_positions(&self, seq_len: u32) -> u32 {
        seq_len / self.far_compress_ratio
    }
}

impl ModelCosts for DeepseekV4Model {
    fn name(&self) -> &str {
        &self.name
    }
    fn max_seq_len(&self) -> u32 {
        self.max_seq_len
    }
    fn hidden_dim(&self) -> u32 {
        self.hidden_dim
    }
    fn num_layers(&self) -> u32 {
        self.num_layers
    }
    fn alltoall_bytes_per_token(&self) -> u64 {
        // Each token is dispatched to `num_experts_per_tok` experts; each
        // dispatch sends one full hidden_dim-wide activation.
        self.num_experts_per_tok as u64
            * self.hidden_dim as u64
            * self.activation_bytes() as u64
    }
    fn num_ep_alltoalls_per_pass(&self) -> u32 {
        // Dispatch + combine per MoE layer.
        2 * self.num_moe_layers
    }
    fn matmul_flops_per_token(&self) -> u64 {
        2 * self.num_active_parameters
    }
    fn attention_flops(&self, s: u32, t: u32) -> u64 {
        let d = self.hidden_dim as u64;
        let s = s as u64;
        let win_t = self.window_attended(t) as u64;
        let near_t = self.near_compressed_attended(t) as u64;
        let far_t = self.far_compressed_attended(t) as u64;

        // Window attention: every layer reads min(t, window_size) recent tokens.
        let window_flops = 4 * d * s * (self.num_layers as u64) * win_t;
        // Near layers also attend top-k of compressed history (Indexer-selected).
        let near_flops = 4 * d * s * (self.num_near_layers as u64) * near_t;
        // Far layers attend the entire stride-compressed history.
        let far_flops = 4 * d * s * (self.num_far_layers as u64) * far_t;

        // Indexer scoring on each near layer: scores every compressed candidate
        // (t / near_ratio of them) with `index_n_heads` heads of `index_head_dim`.
        // Per (query, candidate, head): 2 × index_head_dim FLOPs.
        let indexer_candidates = self.near_compressed_positions(t) as u64;
        let indexer_flops = 2
            * (self.index_n_heads as u64)
            * (self.index_head_dim as u64)
            * s
            * (self.num_near_layers as u64)
            * indexer_candidates;

        window_flops + near_flops + far_flops + indexer_flops
    }
    fn weight_transfer_bytes_per_step(&self) -> u64 {
        (self.num_active_parameters as f64 * self.effective_bytes_per_active_param as f64) as u64
    }
    fn kv_bytes_read_per_decode_step(&self, seq_len: u32) -> u64 {
        let per = self.kv_bytes_per_token_per_layer();
        let win = self.window_attended(seq_len) as u64;
        let near_compressed = self.near_compressed_attended(seq_len) as u64;
        let far_compressed = self.far_compressed_attended(seq_len) as u64;

        // Sliding window in every layer.
        let window_total = (self.num_layers as u64) * win;
        // Near and far compressed history reads.
        let near_total = (self.num_near_layers as u64) * near_compressed;
        let far_total = (self.num_far_layers as u64) * far_compressed;

        // Indexer reads its full compressed-position KV cache on every near
        // layer to score candidates (head_dim_idx × bytes per position).
        let indexer_per_position = self.index_head_dim as u64 * self.index_kv_bpp();
        let indexer_total = (self.num_near_layers as u64)
            * (self.near_compressed_positions(seq_len) as u64)
            * indexer_per_position;

        per * (window_total + near_total + far_total) + indexer_total
    }
    fn weight_residency_bytes(&self) -> u64 {
        (self.num_parameters as f64 * self.effective_bytes_per_resident_param as f64) as u64
    }
    fn kv_storage_bytes(&self, seq_len: u32) -> u64 {
        let per = self.kv_bytes_per_token_per_layer();
        let win = self.window_attended(seq_len) as u64;

        // Dense layers store only the rolling window.
        let dense_total = (self.num_dense_layers as u64) * win;
        // Near layers store window + every compressed position.
        let near_total = (self.num_near_layers as u64)
            * (win + self.near_compressed_positions(seq_len) as u64);
        // Far layers store window + every (more aggressively) compressed position.
        let far_total = (self.num_far_layers as u64)
            * (win + self.far_compressed_positions(seq_len) as u64);

        // Indexer's auxiliary KV: one entry per compressed position, head_dim_idx wide.
        let indexer_per_position = self.index_head_dim as u64 * self.index_kv_bpp();
        let indexer_total = (self.num_near_layers as u64)
            * (self.near_compressed_positions(seq_len) as u64)
            * indexer_per_position;

        per * (dense_total + near_total + far_total) + indexer_total
    }
}
