use crate::request::Request;

/// One running request scheduled into the step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledSeq {
    /// Index into the scheduler's `running` set.
    pub idx: usize,
    /// Positions this request computes in the step: a prefill chunk, or
    /// `1 + pending_draft_len` for decode (the verify width, i.e. the cost).
    pub num_tokens: u32,
}

/// Result of one `Scheduler::schedule` call.
#[derive(Debug, Default)]
pub struct ScheduleDecision {
    /// The batch that runs this step, in `running` order. Requests admitted
    /// this step come last.
    pub batch: Vec<ScheduledSeq>,

    /// Requests that finished at the end of the previous step, reaped and
    /// removed from `running` before anything else was scheduled.
    pub completed: Vec<Request>,

    /// Requests preempted this pass (returned to the head of the waiting
    /// queue; they recompute on resume). When non-zero, nothing new was
    /// admitted.
    pub num_preempted: usize,
}

impl ScheduleDecision {
    /// Total positions computed in this step.
    pub fn total_tokens(&self) -> u32 {
        self.batch.iter().map(|s| s.num_tokens).sum()
    }

    pub fn num_scheduled(&self) -> usize {
        self.batch.len()
    }
}
