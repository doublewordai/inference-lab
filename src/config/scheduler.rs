use crate::scheduler::SchedulingPolicy;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerConfig {
    /// Maximum number of tokens processed in a single iteration
    pub max_num_batched_tokens: u32,

    /// Maximum number of sequences that can run concurrently
    pub max_num_seqs: u32,

    /// Scheduling policy: "fcfs", "priority", or a length-based variant.
    pub policy: SchedulingPolicy,

    /// Enable chunked prefilling
    pub enable_chunked_prefill: bool,

    /// Maximum tokens to prefill in a single iteration (vLLM's long_prefill_token_threshold)
    /// Defaults to 4% of max_model_len if not specified
    #[serde(default)]
    pub long_prefill_token_threshold: u32,

    /// vLLM's `max_num_partial_prefills`. Only its effect on the default
    /// `long_prefill_token_threshold` is modelled: as in vLLM's config
    /// post-init, a value above 1 with no explicit threshold sets the
    /// threshold to 4% of the model's max sequence length.
    #[serde(default = "default_max_num_partial_prefills")]
    pub max_num_partial_prefills: u32,

    /// Block size for KV cache (in tokens)
    pub block_size: u32,

    /// Fraction of GPU memory the engine may use (vLLM's
    /// `--gpu-memory-utilization`, default 0.9). The KV cache gets what is
    /// left after the model weights.
    #[serde(default = "default_gpu_memory_utilization")]
    pub gpu_memory_utilization: f64,

    /// Explicit KV cache capacity in bytes (across the TP group). 0 (default)
    /// derives it: `memory_capacity × tp × gpu_memory_utilization − weights`.
    #[serde(default)]
    pub kv_cache_capacity: u64,

    /// Serving-time context limit (vLLM's `--max-model-len`). Defaults to
    /// the model's `max_seq_len`; only the chunked-prefill threshold default
    /// depends on it.
    #[serde(default)]
    pub max_model_len: Option<u32>,

    /// Enable preemption-free scheduling mode
    /// When enabled, uses conservative admission control to guarantee zero preemptions
    #[serde(default)]
    pub enable_preemption_free: bool,

    /// Enable cascade attention. When a batch contains requests with a shared
    /// prompt prefix, the shared portion of the KV cache is loaded once per
    /// batch instead of once per request, reducing memory bandwidth.
    #[serde(default)]
    pub enable_cascade_attention: bool,
}

fn default_max_num_partial_prefills() -> u32 {
    1
}

fn default_gpu_memory_utilization() -> f64 {
    0.9
}

impl SchedulerConfig {
    /// Set default prefill threshold based on max model length (vLLM uses 4%)
    /// Only sets threshold if max_num_partial_prefills > 1 (matching vLLM behavior)
    pub fn set_default_prefill_threshold(&mut self, max_model_len: u32) {
        if self.enable_chunked_prefill
            && self.max_num_partial_prefills > 1
            && self.long_prefill_token_threshold == 0
        {
            self.long_prefill_token_threshold = (max_model_len as f64 * 0.04) as u32;
        }
    }
}
