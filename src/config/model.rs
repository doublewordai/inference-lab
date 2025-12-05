use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// Model name
    pub name: String,

    /// Total parameters in the model (all parameters, including inactive experts in MoE)
    pub num_parameters: u64,

    /// Active parameters used during inference (for MoE models with sparse activation)
    /// If not specified, defaults to num_parameters (dense models)
    #[serde(default)]
    pub num_active_parameters: Option<u64>,

    /// Number of transformer layers
    pub num_layers: u32,

    /// Hidden dimension
    pub hidden_dim: u32,

    /// Number of attention heads
    pub num_heads: u32,

    /// Number of KV heads (for GQA/MQA). If not specified, defaults to num_heads (MHA)
    #[serde(default)]
    pub num_kv_heads: Option<u32>,

    /// Maximum sequence length supported
    pub max_seq_len: u32,

    /// Sliding window size for sliding window attention layers (None = no sliding window)
    /// Only applies to layers marked as using sliding window attention
    #[serde(default)]
    pub sliding_window: Option<u32>,

    /// Number of layers using sliding window attention (rest use full attention)
    /// If not specified, defaults to 0 (all layers use full attention)
    #[serde(default)]
    pub num_sliding_layers: Option<u32>,

    /// KV cache size per token per layer (in bytes)
    /// For GQA: 2 * num_kv_heads * head_dim * bytes_per_param * num_layers
    /// For MHA: 2 * num_heads * head_dim * bytes_per_param * num_layers
    #[serde(skip)]
    pub kv_cache_bytes_per_token: u64,
}

impl ModelConfig {
    /// Get the number of active parameters (defaults to total parameters for dense models)
    pub fn active_parameters(&self) -> u64 {
        self.num_active_parameters.unwrap_or(self.num_parameters)
    }

    /// Calculate and set the KV cache size per token
    /// For models with sliding window attention, this calculates an average based on typical usage
    pub fn compute_kv_cache_size(&mut self, bytes_per_param: u32) {
        // Use num_kv_heads if specified (GQA/MQA), otherwise use num_heads (MHA)
        let kv_heads = self.num_kv_heads.unwrap_or(self.num_heads);
        let head_dim = self.hidden_dim / self.num_heads;

        // Bytes per token per layer
        let bytes_per_token_per_layer = 2 * kv_heads as u64 * head_dim as u64 * bytes_per_param as u64;

        // If no sliding window, all layers use full attention
        if self.sliding_window.is_none() || self.num_sliding_layers.is_none() {
            self.kv_cache_bytes_per_token = bytes_per_token_per_layer * self.num_layers as u64;
            return;
        }

        // With sliding window: some layers cap at window size, others grow with sequence
        let _num_sliding = self.num_sliding_layers.unwrap_or(0);

        // All layers contribute equally per token (sliding window just caps maximum)
        // At short sequences (< window), all layers grow at same rate
        // This is correct for the initial growth phase
        self.kv_cache_bytes_per_token = bytes_per_token_per_layer * self.num_layers as u64;
    }

    /// Initialize with KV cache size pre-computed
    pub fn with_kv_cache_size(mut self, bytes_per_param: u32) -> Self {
        self.compute_kv_cache_size(bytes_per_param);
        self
    }

    /// Calculate total KV cache size for a sequence, accounting for sliding window
    pub fn kv_cache_size_for_sequence(&self, seq_len: u32) -> u64 {
        // Use num_kv_heads if specified (GQA/MQA), otherwise use num_heads (MHA)
        let kv_heads = self.num_kv_heads.unwrap_or(self.num_heads);
        let head_dim = self.hidden_dim / self.num_heads;
        let bytes_per_token_per_layer = 2 * kv_heads as u64 * head_dim as u64 * 1; // Assuming bytes_per_param

        // No sliding window: simple linear growth
        if self.sliding_window.is_none() || self.num_sliding_layers.is_none() {
            return self.kv_cache_bytes_per_token * seq_len as u64;
        }

        let window = self.sliding_window.unwrap();
        let num_sliding = self.num_sliding_layers.unwrap_or(0);
        let num_full = self.num_layers.saturating_sub(num_sliding);

        // Full attention layers: grow linearly with sequence length
        let full_layers_kv = bytes_per_token_per_layer * num_full as u64 * seq_len as u64;

        // Sliding window layers: capped at window size
        let sliding_layers_kv = bytes_per_token_per_layer * num_sliding as u64 * seq_len.min(window) as u64;

        full_layers_kv + sliding_layers_kv
    }
}
