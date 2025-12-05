use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct SchedulerConfig {
    /// Maximum number of tokens processed in a single iteration
    pub max_num_batched_tokens: u32,

    /// Maximum number of sequences that can run concurrently
    pub max_num_seqs: u32,

    /// Scheduling policy: "fcfs" or "priority"
    pub policy: String,

    /// Enable chunked prefilling
    pub enable_chunked_prefill: bool,

    /// Maximum tokens to prefill in a single iteration (vLLM's long_prefill_token_threshold)
    /// Defaults to 4% of max_model_len if not specified
    #[serde(default)]
    pub long_prefill_token_threshold: u32,

    /// Maximum number of sequences that can be partially prefilled concurrently (vLLM default: 1)
    /// This limits how many NEW waiting requests can start prefilling per iteration
    #[serde(default = "default_max_num_partial_prefills")]
    pub max_num_partial_prefills: u32,

    /// Block size for KV cache (in tokens)
    pub block_size: u32,
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
