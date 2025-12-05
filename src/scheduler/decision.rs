use crate::request::Request;
use std::collections::HashMap;

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

    /// Number of tokens to process for each scheduled request
    /// Key: index in running queue, Value: number of tokens
    pub tokens_per_request: HashMap<usize, u32>,
}

impl ScheduleDecision {
    pub fn new() -> Self {
        Self {
            scheduled_new: Vec::new(),
            scheduled_running: Vec::new(),
            preempted: Vec::new(),
            completed: Vec::new(),
            tokens_per_request: HashMap::new(),
        }
    }

    /// Get total number of tokens scheduled in this iteration
    pub fn total_tokens(&self) -> u32 {
        self.tokens_per_request.values().sum()
    }

    /// Get number of scheduled requests (new + running)
    pub fn num_scheduled(&self) -> usize {
        self.scheduled_new.len() + self.scheduled_running.len()
    }
}
