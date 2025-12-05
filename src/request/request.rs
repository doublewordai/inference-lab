use super::status::RequestStatus;
use crate::config::ModelConfig;

pub type BlockId = u32;

/// Request represents a single inference request in the simulation
#[derive(Debug, Clone)]
pub struct Request {
    /// Unique request ID
    pub request_id: String,

    /// Client priority (lower = higher priority)
    pub priority: i32,

    /// Arrival time (simulated time)
    pub arrival_time: f64,

    /// Request status
    pub status: RequestStatus,

    /// Number of input tokens
    pub num_prompt_tokens: u32,

    /// Maximum number of output tokens to generate
    pub max_output_tokens: u32,

    /// Number of tokens computed so far
    pub num_computed_tokens: u32,

    /// Number of output tokens generated so far
    pub num_output_tokens: u32,

    /// Total tokens (prompt + output)
    pub num_tokens: u32,

    /// Number of prefix-cached tokens
    pub num_cached_tokens: u32,

    /// KV cache blocks allocated to this request
    pub kv_blocks: Vec<BlockId>,

    /// Number of times this request has been preempted
    pub num_preemptions: u32,

    /// Time when first token was generated (TTFT tracking)
    pub first_token_time: Option<f64>,

    /// Time when request completed
    pub completion_time: Option<f64>,

    /// Per-token generation times
    pub token_generation_times: Vec<f64>,

    /// Time spent preempted (not running)
    pub preempted_time: f64,

    /// Last preemption start time
    pub last_preempted_at: Option<f64>,
}

impl Request {
    /// Create a new request
    pub fn new(
        request_id: String,
        priority: i32,
        arrival_time: f64,
        num_prompt_tokens: u32,
        max_output_tokens: u32,
    ) -> Self {
        Self {
            request_id,
            priority,
            arrival_time,
            status: RequestStatus::Waiting,
            num_prompt_tokens,
            max_output_tokens,
            num_computed_tokens: 0,
            num_output_tokens: 0,
            num_tokens: num_prompt_tokens + max_output_tokens, // Total tokens to process
            num_cached_tokens: 0,
            kv_blocks: Vec::new(),
            num_preemptions: 0,
            first_token_time: None,
            completion_time: None,
            token_generation_times: Vec::new(),
            preempted_time: 0.0,
            last_preempted_at: None,
        }
    }

    /// Check if this is in prefill phase
    pub fn is_prefill(&self) -> bool {
        self.num_computed_tokens < self.num_prompt_tokens
    }

    /// Get number of tokens needed to process
    pub fn tokens_to_process(&self) -> u32 {
        if self.is_finished() {
            return 0; // Don't process more if we've generated all output
        }
        self.num_tokens - self.num_computed_tokens
    }

    /// Check if request is done
    pub fn is_finished(&self) -> bool {
        self.num_output_tokens >= self.max_output_tokens
    }

    /// Calculate KV cache requirement for this request
    pub fn kv_cache_size(&self, model: &ModelConfig) -> u64 {
        model.kv_cache_size_for_sequence(self.num_tokens)
    }

    /// Record that tokens were generated (update output token count and total)
    pub fn record_generated_tokens(&mut self, num_new_tokens: u32, current_time: f64) {
        // Update computed tokens
        self.num_computed_tokens += num_new_tokens;

        // If we've crossed into decode phase, update output tokens
        if self.num_computed_tokens > self.num_prompt_tokens {
            let new_output_tokens = (self.num_computed_tokens - self.num_prompt_tokens)
                .min(self.max_output_tokens); // Cap at max

            // Record first token time if this is the first output token
            if self.first_token_time.is_none() && new_output_tokens > 0 {
                self.first_token_time = Some(current_time);
            }

            self.num_output_tokens = new_output_tokens;
            // Note: num_tokens stays fixed at num_prompt_tokens + max_output_tokens

            // Record generation times for each decode token
            self.token_generation_times.push(current_time);
        }
    }

    /// Mark request as preempted
    pub fn mark_preempted(&mut self, current_time: f64) {
        self.status = RequestStatus::Preempted;
        self.num_preemptions += 1;
        self.last_preempted_at = Some(current_time);
    }

    /// Resume a preempted request
    pub fn resume(&mut self, current_time: f64) {
        if let Some(preempted_at) = self.last_preempted_at {
            self.preempted_time += current_time - preempted_at;
        }
        self.status = RequestStatus::Running;
        self.last_preempted_at = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_creation() {
        let req = Request::new("req-1".to_string(), 0, 0.0, 100, 50);

        assert_eq!(req.request_id, "req-1");
        assert_eq!(req.priority, 0);
        assert_eq!(req.arrival_time, 0.0);
        assert_eq!(req.status, RequestStatus::Waiting);
        assert_eq!(req.num_prompt_tokens, 100);
        assert_eq!(req.max_output_tokens, 50);
        assert_eq!(req.num_computed_tokens, 0);
        assert_eq!(req.num_output_tokens, 0);
        assert_eq!(req.num_tokens, 100);
    }

    #[test]
    fn test_is_prefill() {
        let mut req = Request::new("req-1".to_string(), 0, 0.0, 100, 50);

        assert!(req.is_prefill());

        req.num_computed_tokens = 50;
        assert!(req.is_prefill());

        req.num_computed_tokens = 100;
        assert!(!req.is_prefill());
    }

    #[test]
    fn test_tokens_to_process() {
        let mut req = Request::new("req-1".to_string(), 0, 0.0, 100, 50);

        assert_eq!(req.tokens_to_process(), 100);

        req.num_computed_tokens = 50;
        assert_eq!(req.tokens_to_process(), 50);

        req.num_computed_tokens = 100;
        assert_eq!(req.tokens_to_process(), 0);
    }

    #[test]
    fn test_is_finished() {
        let mut req = Request::new("req-1".to_string(), 0, 0.0, 100, 50);

        assert!(!req.is_finished());

        req.num_output_tokens = 25;
        assert!(!req.is_finished());

        req.num_output_tokens = 50;
        assert!(req.is_finished());

        req.num_output_tokens = 60;
        assert!(req.is_finished());
    }

    #[test]
    fn test_record_generated_tokens() {
        let mut req = Request::new("req-1".to_string(), 0, 0.0, 100, 50);

        // Prefill phase
        req.record_generated_tokens(50, 1.0);
        assert_eq!(req.num_computed_tokens, 50);
        assert_eq!(req.num_output_tokens, 0);
        assert_eq!(req.num_tokens, 100);
        assert!(req.first_token_time.is_none());

        // Complete prefill and start decode
        req.record_generated_tokens(51, 2.0);
        assert_eq!(req.num_computed_tokens, 101);
        assert_eq!(req.num_output_tokens, 1);
        assert_eq!(req.num_tokens, 101);
        assert_eq!(req.first_token_time, Some(2.0));
        assert_eq!(req.token_generation_times.len(), 1);

        // Continue decode
        req.record_generated_tokens(1, 3.0);
        assert_eq!(req.num_computed_tokens, 102);
        assert_eq!(req.num_output_tokens, 2);
        assert_eq!(req.num_tokens, 102);
        assert_eq!(req.first_token_time, Some(2.0)); // Doesn't change
        assert_eq!(req.token_generation_times.len(), 2);
    }

    #[test]
    fn test_preemption() {
        let mut req = Request::new("req-1".to_string(), 0, 0.0, 100, 50);
        req.status = RequestStatus::Running;

        // Preempt
        req.mark_preempted(5.0);
        assert_eq!(req.status, RequestStatus::Preempted);
        assert_eq!(req.num_preemptions, 1);
        assert_eq!(req.last_preempted_at, Some(5.0));

        // Resume
        req.resume(10.0);
        assert_eq!(req.status, RequestStatus::Running);
        assert_eq!(req.preempted_time, 5.0);
        assert!(req.last_preempted_at.is_none());

        // Preempt again
        req.mark_preempted(15.0);
        assert_eq!(req.num_preemptions, 2);

        // Resume again
        req.resume(20.0);
        assert_eq!(req.preempted_time, 10.0); // 5.0 + 5.0
    }
}
