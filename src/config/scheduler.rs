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
