use super::session::{Outlook, SessionStep};

pub type BlockId = u32;

/// One inference request as the scheduler and engine see it.
///
/// Token accounting follows the engine's forward passes: the prefill pass over
/// the prompt produces the first output token, and every decode pass over one
/// position produces one more. So a request with prompt length `P` and target
/// output `T` computes exactly `P + T - 1` positions, and its output count is
/// `num_computed_tokens - P + 1` once the prompt is done. Preemption discards
/// the resident KV (`num_computed_tokens = 0`) and keeps the tokens already
/// generated: on resume the request re-prefills the prompt plus the generated
/// tokens whose KV was lost, then continues decoding (vLLM v1 recompute
/// semantics).
#[derive(Debug, Clone)]
pub struct Request {
    /// Unique request ID.
    pub request_id: String,

    /// Client priority (lower = higher priority).
    pub priority: i32,

    /// Arrival time (simulated seconds).
    pub arrival_time: f64,

    /// Number of prompt tokens.
    pub num_prompt_tokens: u32,

    /// Maximum output tokens the client allows. The scheduler may use this
    /// (it is what a real scheduler can see).
    pub max_output_tokens: u32,

    /// Output tokens this request will actually produce (the sampled EOS
    /// point). Ground truth for the simulation; at most `max_output_tokens`.
    pub target_output_tokens: u32,

    /// Positions whose KV is resident: the context the next forward pass
    /// attends to. Reset to 0 by preemption.
    pub num_computed_tokens: u32,

    /// Output tokens generated so far. Monotone across preemptions.
    pub num_output_tokens: u32,

    /// Prefix tokens found in the KV cache at admission (set by the scheduler).
    pub num_cached_tokens: u32,

    /// Incremental content hashes of the sequence, one per `block_size`
    /// tokens: hash `n` covers all tokens up to the end of block `n`. Empty
    /// when the prompt has no content identity (synthetic workloads). Session
    /// workloads extend the hashes past the prompt over the tokens the request
    /// will generate, so the next step of the session can hit them.
    pub prompt_block_hashes: Vec<u64>,

    /// Session workloads: which step of which session this request is.
    /// Boxed: most requests carry none, and `Request` travels by value.
    pub session: Option<Box<SessionStep>>,

    /// KV cache blocks allocated to this request.
    pub kv_blocks: Vec<BlockId>,

    /// Number of times this request has been preempted.
    pub num_preemptions: u32,

    /// Time the first output token was produced (end of the prefill pass).
    pub first_token_time: Option<f64>,

    /// Disaggregated serving: time prefill finished on the prefill worker,
    /// and time the KV hand-off to the decode worker completed. `None` on an
    /// aggregated topology.
    pub prefill_done_time: Option<f64>,
    pub handoff_done_time: Option<f64>,

    /// Earliest simulation time at which this request becomes runnable while
    /// it is parked on an in-flight KV promotion from a slower tier.
    pub ready_at: Option<f64>,

    /// Speculative decoding: draft length to verify on this request's NEXT
    /// decode step. Decided at the end of the iteration that produced the last
    /// token (the simulator's analogue of vLLM proposing draft tokens at the
    /// end of a step). The scheduler reads this to reserve a
    /// `1 + pending_draft_len` verify pass in the token budget and KV, trimming
    /// it if capacity is tight. 0 means no speculation (and is always 0 when
    /// speculative decoding is disabled).
    pub pending_draft_len: u32,

    /// Speculative decoding with `TraceRounds` acceptance: the full-depth
    /// committed-draft count of the trace round drawn for this request's NEXT
    /// decode step. The verify realises `min(pending_round_commits, draft)`
    /// accepted tokens, so a scheduler-trimmed draft stays consistent. `None`
    /// when no round is pending (non-trace acceptance, or prefill).
    pub pending_round_commits: Option<u32>,
}

impl Request {
    /// The re-entry this request's session announces, if it is a session
    /// step with a successor: arrival at `completion_time` plus the harness
    /// gap, reusing `shared_tokens` of this context.
    pub fn outlook_at(&self, completion_time: f64) -> Option<Outlook> {
        self.session.as_ref().and_then(|s| s.outlook_at(completion_time))
    }

    /// Create a request that will produce `target_output_tokens` tokens (at
    /// least one: the prefill pass always yields a token) out of an allowed
    /// `max_output_tokens`.
    pub fn new_with_target(
        request_id: String,
        priority: i32,
        arrival_time: f64,
        num_prompt_tokens: u32,
        max_output_tokens: u32,
        target_output_tokens: u32,
    ) -> Self {
        let max_output_tokens = max_output_tokens.max(1);
        Self {
            request_id,
            priority,
            arrival_time,
            num_prompt_tokens,
            max_output_tokens,
            target_output_tokens: target_output_tokens.clamp(1, max_output_tokens),
            num_computed_tokens: 0,
            num_output_tokens: 0,
            num_cached_tokens: 0,
            prompt_block_hashes: Vec::new(),
            session: None,
            kv_blocks: Vec::new(),
            num_preemptions: 0,
            first_token_time: None,
            prefill_done_time: None,
            handoff_done_time: None,
            ready_at: None,
            pending_draft_len: 0,
            pending_round_commits: None,
        }
    }

    /// Create a request whose target output equals its maximum.
    pub fn new(
        request_id: String,
        priority: i32,
        arrival_time: f64,
        num_prompt_tokens: u32,
        max_output_tokens: u32,
    ) -> Self {
        Self::new_with_target(
            request_id,
            priority,
            arrival_time,
            num_prompt_tokens,
            max_output_tokens,
            max_output_tokens,
        )
    }

    /// Positions that must be resident before the request can decode: the
    /// prompt, plus (after a preemption) the already-generated tokens whose
    /// KV has to be rebuilt.
    pub fn prefill_len(&self) -> u32 {
        self.num_prompt_tokens + self.num_output_tokens.saturating_sub(1)
    }

    /// Whether the next pass is (re)prefill rather than decode.
    pub fn is_prefill(&self) -> bool {
        self.num_computed_tokens < self.prefill_len()
    }

    /// Positions this request will have computed when it finishes.
    pub fn planned_positions(&self) -> u32 {
        self.num_prompt_tokens + self.target_output_tokens - 1
    }

    /// Positions still to compute before the request finishes.
    pub fn tokens_to_process(&self) -> u32 {
        if self.is_finished() {
            return 0;
        }
        self.planned_positions()
            .saturating_sub(self.num_computed_tokens)
    }

    /// Whether every target output token has been generated.
    pub fn is_finished(&self) -> bool {
        self.num_output_tokens >= self.target_output_tokens
    }

    /// Prompt plus maximum output: the context a scheduler must plan for.
    pub fn total_tokens(&self) -> u32 {
        self.num_prompt_tokens + self.max_output_tokens
    }

    /// Positions left under the scheduler-visible bound (prompt + max output).
    pub fn remaining_tokens(&self) -> u32 {
        self.total_tokens().saturating_sub(self.num_computed_tokens)
    }

    /// Record that `num_new_tokens` positions were computed in a pass ending
    /// at `current_time`. Returns the number of output tokens the pass
    /// generated (1 when the pass completed prefill, the advance during
    /// decode, 0 mid-prefill or during recompute).
    pub fn record_generated_tokens(&mut self, num_new_tokens: u32, current_time: f64) -> u32 {
        self.num_computed_tokens += num_new_tokens;
        if self.num_computed_tokens < self.num_prompt_tokens {
            return 0;
        }
        let output =
            (self.num_computed_tokens - self.num_prompt_tokens + 1).min(self.max_output_tokens);
        if output <= self.num_output_tokens {
            return 0;
        }
        if self.num_output_tokens == 0 && self.first_token_time.is_none() {
            self.first_token_time = Some(current_time);
        }
        let generated = output - self.num_output_tokens;
        self.num_output_tokens = output;
        generated
    }

    /// Make the request arrive already prefilled at `time`: its prompt is
    /// resident and its first token produced, so it enters as pure decode
    /// work (a disaggregated decode pool seen in isolation).
    pub fn mark_prefilled(&mut self, time: f64) {
        self.num_computed_tokens = self.num_prompt_tokens;
        self.num_output_tokens = 1;
        self.first_token_time = Some(time);
    }

    /// Number of leading prompt blocks shared by every request in `batch`.
    /// Uses the incremental prompt block hashes as the equality check: hash N
    /// covers tokens 0..N*block_size, so two requests share a prefix of K
    /// blocks iff their first K block hashes are pairwise equal.
    pub fn shared_prefix_blocks(batch: &[&Request]) -> u32 {
        if batch.len() < 2 {
            return 0;
        }
        let first = &batch[0].prompt_block_hashes;
        let mut shared = first.len();
        for req in &batch[1..] {
            let other = &req.prompt_block_hashes;
            let mut i = 0;
            while i < shared && i < other.len() && first[i] == other[i] {
                i += 1;
            }
            shared = i;
            if shared == 0 {
                return 0;
            }
        }
        shared as u32
    }

    /// Preempt: the resident KV is discarded and must be recomputed on resume.
    /// The caller frees `kv_blocks`.
    pub fn preempt(&mut self) {
        self.num_preemptions += 1;
        self.num_computed_tokens = 0;
        self.num_cached_tokens = 0;
        self.pending_draft_len = 0;
        self.pending_round_commits = None;
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
        assert_eq!(req.num_prompt_tokens, 100);
        assert_eq!(req.max_output_tokens, 50);
        assert_eq!(req.num_computed_tokens, 0);
        assert_eq!(req.num_output_tokens, 0);
        // prompt(100) + target(50) - 1: the prefill pass yields token 1.
        assert_eq!(req.planned_positions(), 149);
        assert_eq!(req.tokens_to_process(), 149);
    }

    #[test]
    fn test_target_is_at_least_one_and_at_most_max() {
        let req = Request::new_with_target("r".into(), 0, 0.0, 10, 5, 0);
        assert_eq!(req.target_output_tokens, 1);
        let req = Request::new_with_target("r".into(), 0, 0.0, 10, 5, 9);
        assert_eq!(req.target_output_tokens, 5);
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
    fn test_prefill_pass_produces_first_token() {
        let mut req = Request::new("req-1".to_string(), 0, 0.0, 100, 50);

        // Chunked prefill: no output mid-prompt.
        assert_eq!(req.record_generated_tokens(50, 1.0), 0);
        assert_eq!(req.num_computed_tokens, 50);
        assert_eq!(req.num_output_tokens, 0);
        assert!(req.first_token_time.is_none());
        assert!(req.is_prefill());

        // Last prefill chunk: first token comes out of the prefill pass.
        assert_eq!(req.record_generated_tokens(50, 2.0), 1);
        assert_eq!(req.num_computed_tokens, 100);
        assert_eq!(req.num_output_tokens, 1);
        assert_eq!(req.first_token_time, Some(2.0));
        assert!(!req.is_prefill());

        // Decode: one position, one token.
        assert_eq!(req.record_generated_tokens(1, 3.0), 1);
        assert_eq!(req.num_computed_tokens, 101);
        assert_eq!(req.num_output_tokens, 2);
        assert_eq!(req.first_token_time, Some(2.0));

        // Speculative verify: three accepted + bonus advances by four.
        assert_eq!(req.record_generated_tokens(4, 4.0), 4);
        assert_eq!(req.num_output_tokens, 6);
    }

    #[test]
    fn test_finishes_after_planned_positions() {
        let mut req = Request::new("req-1".to_string(), 0, 0.0, 100, 50);
        req.record_generated_tokens(100, 1.0);
        for _ in 0..49 {
            assert!(!req.is_finished());
            req.record_generated_tokens(1, 2.0);
        }
        assert!(req.is_finished());
        assert_eq!(req.num_output_tokens, 50);
        assert_eq!(req.num_computed_tokens, req.planned_positions());
        assert_eq!(req.tokens_to_process(), 0);
    }

    #[test]
    fn test_single_token_request_finishes_at_prefill() {
        let mut req = Request::new("req-1".to_string(), 0, 0.0, 100, 1);
        assert_eq!(req.tokens_to_process(), 100);
        req.record_generated_tokens(100, 1.0);
        assert!(req.is_finished());
        assert_eq!(req.first_token_time, Some(1.0));
    }

    #[test]
    fn test_preemption_recomputes_prompt_and_generated_tokens() {
        let mut req = Request::new("req-1".to_string(), 0, 0.0, 100, 50);
        req.record_generated_tokens(100, 1.0);
        req.record_generated_tokens(1, 2.0);
        req.record_generated_tokens(1, 3.0);
        assert_eq!(req.num_output_tokens, 3);
        assert_eq!(req.num_computed_tokens, 102);

        req.preempt();
        assert_eq!(req.num_preemptions, 1);
        assert_eq!(req.num_computed_tokens, 0);
        // Tokens already generated survive; their KV must be rebuilt.
        assert_eq!(req.num_output_tokens, 3);
        assert!(req.is_prefill());
        assert_eq!(req.prefill_len(), 102);
        assert_eq!(req.first_token_time, Some(1.0));

        // Recompute pass produces no new output; decode then continues.
        assert_eq!(req.record_generated_tokens(102, 4.0), 0);
        assert_eq!(req.num_output_tokens, 3);
        assert!(!req.is_prefill());
        assert_eq!(req.record_generated_tokens(1, 5.0), 1);
        assert_eq!(req.num_output_tokens, 4);
        assert_eq!(req.tokens_to_process(), 149 - 103);
    }
}
