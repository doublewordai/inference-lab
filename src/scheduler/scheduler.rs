use super::{decision::ScheduleDecision, policy::SchedulingPolicy};
use crate::config::{HardwareConfig, ModelConfig, SchedulerConfig};
use crate::kv_cache::KVCacheManager;
use crate::request::{Request, RequestStatus};
use ordered_float::OrderedFloat;
use std::collections::VecDeque;

pub struct Scheduler {
    config: SchedulerConfig,
    _hardware: HardwareConfig,
    _model: ModelConfig,

    /// Waiting queue (FIFO)
    waiting: VecDeque<Request>,

    /// Running requests
    running: Vec<Request>,

    /// Scheduling policy
    policy: SchedulingPolicy,

    /// KV cache manager
    kv_cache_manager: KVCacheManager,

    /// Current iteration number
    iteration: u64,
}

impl Scheduler {
    pub fn new(
        config: SchedulerConfig,
        hardware: HardwareConfig,
        model: ModelConfig,
        kv_cache_manager: KVCacheManager,
    ) -> Result<Self, String> {
        let policy = SchedulingPolicy::from_str(&config.policy)?;

        Ok(Self {
            config,
            _hardware: hardware,
            _model: model,
            waiting: VecDeque::new(),
            running: Vec::new(),
            policy,
            kv_cache_manager,
            iteration: 0,
        })
    }

    /// Main scheduling function - called each iteration
    /// Returns a ScheduleDecision with all scheduling actions
    pub fn schedule(&mut self, current_time: f64) -> ScheduleDecision {
        self.iteration += 1;

        let mut decision = ScheduleDecision::new();
        let mut token_budget = self.config.max_num_batched_tokens;

        // Phase 1: Schedule RUNNING requests (with preemption if needed)
        let mut idx = 0;
        while idx < self.running.len() && token_budget > 0 {
            // Check if request is finished
            if self.running[idx].is_finished() {
                let mut req = self.running.remove(idx);
                req.status = RequestStatus::Completed;
                req.completion_time = Some(current_time);
                decision.completed.push(req);
                continue;
            }

            // Calculate tokens to schedule for this request
            let tokens_to_process = self.running[idx].tokens_to_process();
            let mut tokens_to_schedule = tokens_to_process.min(token_budget);

            // If in prefill phase, limit to only prefill tokens (don't cross into decode)
            if self.running[idx].is_prefill() {
                let remaining_prefill =
                    self.running[idx].num_prompt_tokens - self.running[idx].num_computed_tokens;
                tokens_to_schedule = tokens_to_schedule.min(remaining_prefill);

                // Apply chunked prefill limit if enabled (vLLM's long_prefill_token_threshold)
                if self.config.enable_chunked_prefill
                    && self.config.long_prefill_token_threshold > 0
                {
                    tokens_to_schedule =
                        tokens_to_schedule.min(self.config.long_prefill_token_threshold);
                }
            } else {
                // During decode phase, limit to 1 token per iteration (autoregressive generation)
                tokens_to_schedule = tokens_to_schedule.min(1);
            }

            // Try to allocate KV cache blocks if needed
            let blocks_needed =
                self.calculate_blocks_needed(&self.running[idx], tokens_to_schedule);

            if blocks_needed > 0 && self.kv_cache_manager.num_free_blocks() < blocks_needed {
                // Need to preempt
                if let Some(preempt_idx) = self.select_preemption_victim() {
                    let mut preempted_req = self.running.remove(preempt_idx);
                    self.preempt_request(&mut preempted_req, current_time);
                    self.waiting.push_back(preempted_req);
                    decision.preempted.push(preempt_idx);

                    // If we preempted ourselves, we can't schedule
                    if preempt_idx == idx {
                        continue;
                    }

                    // Adjust index if we preempted someone before us
                    if preempt_idx < idx {
                        idx -= 1;
                    }
                    continue;
                } else {
                    // Can't preempt anyone, skip this request
                    idx += 1;
                    continue;
                }
            }

            // Allocate blocks if needed
            if blocks_needed > 0 {
                if let Some(blocks) = self
                    .kv_cache_manager
                    .allocate_blocks(&self.running[idx], tokens_to_schedule)
                {
                    self.running[idx].kv_blocks.extend(blocks);
                } else {
                    // Shouldn't happen since we checked above
                    idx += 1;
                    continue;
                }
            }

            // Schedule this request
            decision.scheduled_running.push(idx);
            decision.tokens_per_request.insert(idx, tokens_to_schedule);
            token_budget -= tokens_to_schedule;
            idx += 1;
        }

        // Phase 2: Schedule WAITING requests (only if no preemptions occurred)
        if decision.preempted.is_empty() {
            while !self.waiting.is_empty() && token_budget > 0 {
                if self.running.len() >= self.config.max_num_seqs as usize {
                    break;
                }

                // Select next request based on scheduling policy
                let selected_idx = self.select_next_waiting_request();
                let mut request = self.waiting.get(selected_idx).unwrap().clone();

                // Check for prefix cache hits
                let cached_tokens = self.kv_cache_manager.check_prefix_cache(&request);
                request.num_cached_tokens = cached_tokens;
                request.num_computed_tokens += cached_tokens;

                // Calculate tokens to schedule
                let mut tokens_to_schedule = request.tokens_to_process().min(token_budget);

                // If in prefill phase, limit to only prefill tokens (don't cross into decode)
                if request.is_prefill() {
                    let remaining_prefill = request.num_prompt_tokens - request.num_computed_tokens;
                    tokens_to_schedule = tokens_to_schedule.min(remaining_prefill);

                    // Apply chunked prefill limit if enabled (vLLM's long_prefill_token_threshold)
                    if self.config.enable_chunked_prefill
                        && self.config.long_prefill_token_threshold > 0
                    {
                        tokens_to_schedule =
                            tokens_to_schedule.min(self.config.long_prefill_token_threshold);
                    }
                } else {
                    // During decode phase, limit to 1 token per iteration (autoregressive generation)
                    tokens_to_schedule = tokens_to_schedule.min(1);
                }

                if tokens_to_schedule == 0 {
                    break;
                }

                // Try to allocate KV cache
                let blocks_needed = self.calculate_blocks_needed(&request, tokens_to_schedule);
                if self.kv_cache_manager.num_free_blocks() < blocks_needed {
                    break; // Can't fit, stop scheduling new requests
                }

                // Allocate blocks
                if let Some(blocks) = self
                    .kv_cache_manager
                    .allocate_blocks(&request, tokens_to_schedule)
                {
                    request.kv_blocks.extend(blocks);
                } else {
                    break;
                }

                // Move to running
                request.status = RequestStatus::Running;
                let new_idx = self.running.len();

                decision.scheduled_new.push(new_idx);
                decision
                    .tokens_per_request
                    .insert(new_idx, tokens_to_schedule);
                token_budget -= tokens_to_schedule;

                self.running.push(request);
                self.waiting.remove(selected_idx);
            }
        }

        decision
    }

    /// Calculate how many new blocks are needed for a request
    fn calculate_blocks_needed(&self, request: &Request, num_new_tokens: u32) -> usize {
        let total_tokens = request.num_computed_tokens + num_new_tokens;
        let total_blocks_needed =
            ((total_tokens + self.config.block_size - 1) / self.config.block_size) as usize;
        total_blocks_needed.saturating_sub(request.kv_blocks.len())
    }

    /// Find the index of the best waiting request to schedule based on policy
    fn select_next_waiting_request(&self) -> usize {
        if self.waiting.is_empty() {
            return 0;
        }

        match self.policy {
            SchedulingPolicy::FCFS | SchedulingPolicy::Priority => {
                // FCFS and Priority: always take first (FIFO order)
                0
            }
            SchedulingPolicy::SJF => {
                // Shortest Job First: find request with smallest output length
                self.waiting
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, r)| r.max_output_tokens)
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            }
            SchedulingPolicy::SIF => {
                // Shortest Input First: find request with smallest input length
                self.waiting
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, r)| r.num_prompt_tokens)
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            }
            SchedulingPolicy::LIF => {
                // Longest Input First: find request with largest input length
                self.waiting
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, r)| r.num_prompt_tokens)
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            }
        }
    }

    /// Select a request to preempt based on policy
    fn select_preemption_victim(&self) -> Option<usize> {
        if self.running.is_empty() {
            return None;
        }

        match self.policy {
            SchedulingPolicy::FCFS => {
                // FCFS: preempt the last (most recent) request
                Some(self.running.len() - 1)
            }
            SchedulingPolicy::Priority => {
                // Priority: preempt the lowest priority (highest priority value) request
                self.running
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, r)| (r.priority, OrderedFloat(r.arrival_time)))
                    .map(|(idx, _)| idx)
            }
            SchedulingPolicy::SJF => {
                // SJF: preempt the longest remaining job
                self.running
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, r)| r.max_output_tokens - r.num_output_tokens)
                    .map(|(idx, _)| idx)
            }
            SchedulingPolicy::SIF => {
                // SIF: preempt the request with longest input
                self.running
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, r)| r.num_prompt_tokens)
                    .map(|(idx, _)| idx)
            }
            SchedulingPolicy::LIF => {
                // LIF: preempt the request with shortest input
                self.running
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, r)| r.num_prompt_tokens)
                    .map(|(idx, _)| idx)
            }
        }
    }

    /// Preempt a request (free KV blocks, update state)
    fn preempt_request(&mut self, request: &mut Request, current_time: f64) {
        request.mark_preempted(current_time);

        // Free KV cache blocks
        self.kv_cache_manager.free_blocks(&request.kv_blocks);
        request.kv_blocks.clear();

        // Note: num_computed_tokens is NOT reset - we can resume from where we left off
        // if prefix caching is enabled
    }

    /// Add a new request to the waiting queue
    pub fn add_request(&mut self, request: Request) {
        self.waiting.push_back(request);
    }

    /// Get number of running requests
    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    /// Get number of waiting requests
    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    /// Get reference to running requests
    pub fn running(&self) -> &Vec<Request> {
        &self.running
    }

    /// Get mutable reference to running requests
    pub fn running_mut(&mut self) -> &mut Vec<Request> {
        &mut self.running
    }

    /// Get reference to KV cache manager
    pub fn kv_cache_manager(&self) -> &KVCacheManager {
        &self.kv_cache_manager
    }

    /// Get mutable reference to KV cache manager
    pub fn kv_cache_manager_mut(&mut self) -> &mut KVCacheManager {
        &mut self.kv_cache_manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn create_test_scheduler() -> Scheduler {
        let config = Config::test_default();
        let kv_cache = KVCacheManager::new(
            config.hardware.kv_cache_capacity,
            config.scheduler.block_size,
            config.model.kv_cache_bytes_per_token,
            false,
        );
        Scheduler::new(config.scheduler, config.hardware, config.model, kv_cache).unwrap()
    }

    fn create_test_request(id: &str, prompt: u32, output: u32) -> Request {
        Request::new(id.to_string(), 0, 0.0, prompt, output)
    }

    #[test]
    fn test_scheduler_creation() {
        let scheduler = create_test_scheduler();
        assert_eq!(scheduler.num_running(), 0);
        assert_eq!(scheduler.num_waiting(), 0);
    }

    #[test]
    fn test_add_request() {
        let mut scheduler = create_test_scheduler();
        let req = create_test_request("req-1", 100, 50);

        scheduler.add_request(req);
        assert_eq!(scheduler.num_waiting(), 1);
    }

    #[test]
    fn test_schedule_single_request() {
        let mut scheduler = create_test_scheduler();
        let req = create_test_request("req-1", 16, 10);

        scheduler.add_request(req);

        let decision = scheduler.schedule(0.0);

        assert_eq!(decision.scheduled_new.len(), 1);
        assert_eq!(scheduler.num_running(), 1);
        assert_eq!(scheduler.num_waiting(), 0);
    }

    #[test]
    fn test_schedule_multiple_requests() {
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(create_test_request("req-1", 16, 10));
        scheduler.add_request(create_test_request("req-2", 16, 10));

        let decision = scheduler.schedule(0.0);

        // Both should be scheduled if token budget allows
        assert!(decision.scheduled_new.len() >= 1);
        assert!(scheduler.num_running() >= 1);
    }

    #[test]
    fn test_completion() {
        let mut scheduler = create_test_scheduler();
        let mut req = create_test_request("req-1", 16, 10);
        req.num_computed_tokens = 16; // Prefill done
        req.num_output_tokens = 10; // All output generated
        req.status = RequestStatus::Running;

        scheduler.running.push(req);

        let decision = scheduler.schedule(0.0);

        assert_eq!(decision.completed.len(), 1);
        assert_eq!(scheduler.num_running(), 0);
    }

    #[test]
    fn test_preemption_fcfs() {
        let mut scheduler = create_test_scheduler();

        // Add requests
        scheduler.add_request(create_test_request("req-1", 16, 10));
        scheduler.add_request(create_test_request("req-2", 16, 10));

        // Schedule them
        scheduler.schedule(0.0);

        // FCFS should preempt the last one
        if let Some(idx) = scheduler.select_preemption_victim() {
            assert_eq!(idx, scheduler.num_running() - 1);
        }
    }

    fn create_scheduler_with_policy(policy: &str) -> Scheduler {
        let mut config = Config::test_default();
        config.scheduler.policy = policy.to_string();
        let kv_cache = KVCacheManager::new(
            config.hardware.kv_cache_capacity,
            config.scheduler.block_size,
            config.model.kv_cache_bytes_per_token,
            false,
        );
        Scheduler::new(config.scheduler, config.hardware, config.model, kv_cache).unwrap()
    }

    #[test]
    fn test_sjf_selection() {
        let mut scheduler = create_scheduler_with_policy("sjf");

        // Add requests with different output lengths
        scheduler.add_request(create_test_request("req-long", 16, 100)); // longest
        scheduler.add_request(create_test_request("req-short", 16, 10)); // shortest
        scheduler.add_request(create_test_request("req-medium", 16, 50)); // medium

        // SJF should select the shortest job first (output=10)
        let idx = scheduler.select_next_waiting_request();
        assert_eq!(scheduler.waiting.get(idx).unwrap().max_output_tokens, 10);
    }

    #[test]
    fn test_sjf_preemption() {
        let mut scheduler = create_scheduler_with_policy("sjf");

        // Add running requests with different remaining outputs
        let mut req1 = create_test_request("req-1", 16, 100);
        req1.num_output_tokens = 50; // 50 remaining
        req1.status = RequestStatus::Running;

        let mut req2 = create_test_request("req-2", 16, 50);
        req2.num_output_tokens = 10; // 40 remaining
        req2.status = RequestStatus::Running;

        let mut req3 = create_test_request("req-3", 16, 30);
        req3.num_output_tokens = 5; // 25 remaining
        req3.status = RequestStatus::Running;

        scheduler.running.push(req1);
        scheduler.running.push(req2);
        scheduler.running.push(req3);

        // SJF should preempt the longest remaining job (req-1 with 50 remaining)
        let victim_idx = scheduler.select_preemption_victim().unwrap();
        assert_eq!(
            scheduler.running[victim_idx].max_output_tokens
                - scheduler.running[victim_idx].num_output_tokens,
            50
        );
    }

    #[test]
    fn test_sif_selection() {
        let mut scheduler = create_scheduler_with_policy("sif");

        // Add requests with different input lengths
        scheduler.add_request(create_test_request("req-long", 200, 10)); // longest input
        scheduler.add_request(create_test_request("req-short", 50, 10)); // shortest input
        scheduler.add_request(create_test_request("req-medium", 100, 10)); // medium input

        // SIF should select the shortest input first (prompt=50)
        let idx = scheduler.select_next_waiting_request();
        assert_eq!(scheduler.waiting.get(idx).unwrap().num_prompt_tokens, 50);
    }

    #[test]
    fn test_sif_preemption() {
        let mut scheduler = create_scheduler_with_policy("sif");

        // Add running requests with different input lengths
        let mut req1 = create_test_request("req-1", 200, 10);
        req1.status = RequestStatus::Running;

        let mut req2 = create_test_request("req-2", 50, 10);
        req2.status = RequestStatus::Running;

        let mut req3 = create_test_request("req-3", 100, 10);
        req3.status = RequestStatus::Running;

        scheduler.running.push(req1);
        scheduler.running.push(req2);
        scheduler.running.push(req3);

        // SIF should preempt the longest input (req-1 with 200 tokens)
        let victim_idx = scheduler.select_preemption_victim().unwrap();
        assert_eq!(scheduler.running[victim_idx].num_prompt_tokens, 200);
    }

    #[test]
    fn test_lif_selection() {
        let mut scheduler = create_scheduler_with_policy("lif");

        // Add requests with different input lengths
        scheduler.add_request(create_test_request("req-long", 200, 10)); // longest input
        scheduler.add_request(create_test_request("req-short", 50, 10)); // shortest input
        scheduler.add_request(create_test_request("req-medium", 100, 10)); // medium input

        // LIF should select the longest input first (prompt=200)
        let idx = scheduler.select_next_waiting_request();
        assert_eq!(scheduler.waiting.get(idx).unwrap().num_prompt_tokens, 200);
    }

    #[test]
    fn test_lif_preemption() {
        let mut scheduler = create_scheduler_with_policy("lif");

        // Add running requests with different input lengths
        let mut req1 = create_test_request("req-1", 200, 10);
        req1.status = RequestStatus::Running;

        let mut req2 = create_test_request("req-2", 50, 10);
        req2.status = RequestStatus::Running;

        let mut req3 = create_test_request("req-3", 100, 10);
        req3.status = RequestStatus::Running;

        scheduler.running.push(req1);
        scheduler.running.push(req2);
        scheduler.running.push(req3);

        // LIF should preempt the shortest input (req-2 with 50 tokens)
        let victim_idx = scheduler.select_preemption_victim().unwrap();
        assert_eq!(scheduler.running[victim_idx].num_prompt_tokens, 50);
    }

    #[test]
    fn test_priority_selection_fifo() {
        let mut scheduler = create_scheduler_with_policy("priority");

        // Priority uses FIFO for selection (takes first)
        scheduler.add_request(create_test_request("req-1", 100, 10));
        scheduler.add_request(create_test_request("req-2", 200, 20));
        scheduler.add_request(create_test_request("req-3", 50, 5));

        let idx = scheduler.select_next_waiting_request();
        assert_eq!(idx, 0); // Should always be first
    }

    #[test]
    fn test_priority_preemption() {
        let mut scheduler = create_scheduler_with_policy("priority");

        // Add running requests with different priorities (lower value = higher priority)
        let mut req1 = create_test_request("req-1", 100, 10);
        req1.priority = 0; // highest priority
        req1.arrival_time = 1.0;
        req1.status = RequestStatus::Running;

        let mut req2 = create_test_request("req-2", 100, 10);
        req2.priority = 5; // lowest priority
        req2.arrival_time = 2.0;
        req2.status = RequestStatus::Running;

        let mut req3 = create_test_request("req-3", 100, 10);
        req3.priority = 2; // medium priority
        req3.arrival_time = 3.0;
        req3.status = RequestStatus::Running;

        scheduler.running.push(req1);
        scheduler.running.push(req2);
        scheduler.running.push(req3);

        // Priority should preempt the lowest priority (highest value = 5)
        let victim_idx = scheduler.select_preemption_victim().unwrap();
        assert_eq!(scheduler.running[victim_idx].priority, 5);
    }

    #[test]
    fn test_fcfs_selection_fifo() {
        let scheduler = create_test_scheduler(); // default is FCFS

        let mut sched = scheduler;
        sched.add_request(create_test_request("req-1", 100, 10));
        sched.add_request(create_test_request("req-2", 200, 20));
        sched.add_request(create_test_request("req-3", 50, 5));

        // FCFS always selects first (FIFO)
        let idx = sched.select_next_waiting_request();
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_policy_selection_deterministic() {
        // Test that policies produce deterministic results for fixed inputs
        let mut scheduler = create_scheduler_with_policy("sjf");

        scheduler.add_request(create_test_request("req-1", 16, 100));
        scheduler.add_request(create_test_request("req-2", 16, 10));
        scheduler.add_request(create_test_request("req-3", 16, 50));

        let idx1 = scheduler.select_next_waiting_request();
        let idx2 = scheduler.select_next_waiting_request();
        let idx3 = scheduler.select_next_waiting_request();

        // Should always select same index for same state
        assert_eq!(idx1, idx2);
        assert_eq!(idx2, idx3);
    }
}
