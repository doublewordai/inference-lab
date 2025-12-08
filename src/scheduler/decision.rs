use crate::request::Request;

/// Result of a scheduling decision
#[derive(Debug)]
pub struct ScheduleDecision {
    /// Indices of newly scheduled requests (from waiting -> running)
    pub scheduled_new: Vec<usize>,

    /// Indices of continuing running requests
    pub scheduled_running: Vec<usize>,

    /// Indices of preempted requests
    pub preempted: Vec<usize>,

    /// Completed requests
    pub completed: Vec<Request>,

    /// Number of tokens for each newly scheduled request (parallel to scheduled_new)
    pub tokens_for_new: Vec<u32>,

    /// Number of tokens for each continuing request (parallel to scheduled_running)
    pub tokens_for_running: Vec<u32>,
}

impl ScheduleDecision {
    pub fn new() -> Self {
        Self {
            scheduled_new: Vec::new(),
            scheduled_running: Vec::new(),
            preempted: Vec::new(),
            completed: Vec::new(),
            tokens_for_new: Vec::new(),
            tokens_for_running: Vec::new(),
        }
    }

    /// Get total number of tokens scheduled in this iteration
    pub fn total_tokens(&self) -> u32 {
        self.tokens_for_new.iter().sum::<u32>() + self.tokens_for_running.iter().sum::<u32>()
    }

    /// Get number of scheduled requests (new + running)
    pub fn num_scheduled(&self) -> usize {
        self.scheduled_new.len() + self.scheduled_running.len()
    }
}
