use super::{decision::ScheduleDecision, decision::ScheduledSeq, policy::SchedulingPolicy};
use crate::config::SchedulerConfig;
use crate::kv_cache::KVCacheManager;
use crate::request::Request;
use ordered_float::OrderedFloat;
use std::collections::VecDeque;

/// vLLM-v1-style iteration scheduler for one worker: a waiting queue, a
/// running set, and the worker's KV cache manager. Each `schedule` call
/// reaps finished requests, grows the running set (preempting under KV
/// pressure), then admits waiting requests into the leftover token budget.
pub struct Scheduler {
    config: SchedulerConfig,

    /// Waiting queue. Preempted requests re-enter at the front.
    waiting: VecDeque<Request>,

    /// Running requests, in admission order.
    running: Vec<Request>,

    /// Requests holding HBM blocks while their KV cache is being promoted
    /// from a slower tier. They re-enter `waiting` once the transfer
    /// completes. Blocks reserved during the wait are not freed until the
    /// transfer completes.
    pending_transfers: Vec<Request>,

    policy: SchedulingPolicy,

    kv_cache_manager: KVCacheManager,

    /// Total preemptions performed so far.
    num_preemptions: u64,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig, kv_cache_manager: KVCacheManager) -> Result<Self, String> {
        let policy = config.policy.parse::<SchedulingPolicy>()?;

        Ok(Self {
            config,
            waiting: VecDeque::new(),
            running: Vec::new(),
            pending_transfers: Vec::new(),
            policy,
            kv_cache_manager,
            num_preemptions: 0,
        })
    }

    /// Promote any pending KV-transfer requests whose transfer has finished
    /// back to the waiting queue so they can be scheduled normally. On
    /// promotion the request's computed-token count is bumped to the cached
    /// prefix length, mirroring vLLM's behaviour (in v1's scheduler.py:653,
    /// resumed-from-remote-KVs requests have num_computed_tokens > 0).
    ///
    /// Also advances the bandwidth-shared transfer simulation: in-flight
    /// transfers split each tier's bandwidth equally, so adding more in-flight
    /// promotions slows everyone down.
    fn promote_finished_transfers(&mut self, current_time: f64) {
        let completed = self.kv_cache_manager.advance_transfers(current_time);
        let mut still_pending = Vec::with_capacity(self.pending_transfers.len());
        for mut req in self.pending_transfers.drain(..) {
            if completed.contains(&req.request_id) {
                // Publish the now-resident blocks to the HBM prefix cache so
                // subsequent same-prefix requests get a clean HBM hit.
                let cached_blocks = self
                    .kv_cache_manager
                    .content_blocks_for_tokens(req.num_cached_tokens);
                let hashes: Vec<u64> = req
                    .prompt_block_hashes
                    .iter()
                    .copied()
                    .take(cached_blocks)
                    .collect();
                let blocks: Vec<u32> = req.kv_blocks.iter().copied().take(cached_blocks).collect();
                self.kv_cache_manager
                    .publish_transferred_blocks(&hashes, &blocks);
                req.ready_at = None;
                req.num_computed_tokens = req.num_cached_tokens;
                self.waiting.push_back(req);
            } else {
                let remaining = self
                    .kv_cache_manager
                    .estimate_remaining_time(&req.request_id);
                req.ready_at = Some(current_time + remaining);
                still_pending.push(req);
            }
        }
        self.pending_transfers = still_pending;
    }

    /// Main scheduling function, called once per iteration.
    pub fn schedule(&mut self, current_time: f64) -> ScheduleDecision {
        // Promote any pending KV transfers whose deadline has passed; their
        // KV is now resident in HBM and they can be scheduled normally.
        self.promote_finished_transfers(current_time);

        let mut decision = ScheduleDecision::default();
        let mut token_budget = self.config.max_num_batched_tokens;

        // Phase 0: reap finished requests, freeing their blocks BEFORE any
        // allocation this pass. Within a step, retirement must precede
        // growth: a request later in `running` may have finished, and an
        // earlier request's allocation should be able to use those blocks.
        let mut idx = 0;
        while idx < self.running.len() {
            if self.running[idx].is_finished() {
                let mut req = self.running.remove(idx);
                self.kv_cache_manager.free_blocks(&req.kv_blocks);
                req.kv_blocks.clear();
                decision.completed.push(req);
            } else {
                idx += 1;
            }
        }

        // Phase 1: schedule RUNNING requests, preempting under KV pressure.
        let mut idx = 0;
        while idx < self.running.len() && token_budget > 0 {
            let tokens_to_schedule = self.tokens_to_schedule(&self.running[idx], token_budget);
            let blocks_needed = self
                .kv_cache_manager
                .blocks_needed(&self.running[idx], tokens_to_schedule);

            if blocks_needed > 0 && self.kv_cache_manager.num_free_blocks() < blocks_needed {
                if self.config.enable_preemption_free {
                    // Admission control guarantees every admitted request can
                    // grow to its bound without preemption; reaching here
                    // means that projection was violated. Fall through to
                    // preemption rather than wedge the worker.
                    log::error!(
                        "preemption-free admission violated at t={current_time}: \
                         request {} needs {blocks_needed} blocks, {} free",
                        self.running[idx].request_id,
                        self.kv_cache_manager.num_free_blocks()
                    );
                }
                // Victims are restricted to positions >= idx: earlier
                // positions are already recorded in the decision by index,
                // and removing one would invalidate those indices.
                if let Some(victim) = self.select_preemption_victim(idx) {
                    let mut preempted = self.running.remove(victim);
                    self.preempt_request(&mut preempted);
                    // vLLM puts preempted requests back at the head of the
                    // waiting queue; push_back would starve them behind the
                    // whole arrival backlog.
                    self.waiting.push_front(preempted);
                    decision.num_preempted += 1;
                    // Victim was at >= idx, so recorded indices stay valid.
                    // If we preempted ourselves, running[idx] is now the
                    // next request; either way, re-evaluate at idx.
                    continue;
                }
                // Nothing left to preempt; skip this request this step.
                idx += 1;
                continue;
            }

            if blocks_needed > 0 {
                let blocks = self
                    .kv_cache_manager
                    .allocate_blocks(&self.running[idx], tokens_to_schedule)
                    .expect("free-block check above guarantees allocation");
                self.running[idx].kv_blocks.extend(blocks);
            }

            decision.batch.push(ScheduledSeq {
                idx,
                num_tokens: tokens_to_schedule,
            });
            token_budget -= tokens_to_schedule;
            idx += 1;
        }

        // Phase 2: admit WAITING requests (only if nothing was preempted:
        // preemption means KV is full, so admitting would just be preempted
        // again).
        if decision.num_preempted == 0 {
            while !self.waiting.is_empty() && token_budget > 0 {
                if self.running.len() >= self.config.max_num_seqs as usize {
                    break;
                }

                let selected_idx = self.select_next_waiting_request();
                let request = &self.waiting[selected_idx];

                if self.config.enable_preemption_free && !self.can_admit_without_preemption(request)
                {
                    break; // Can't admit without risking future preemption need
                }

                let lookup = self.kv_cache_manager.peek_prefix_cache(request);
                let cached_tokens = self.usable_cached_tokens(request, lookup.total_cached_tokens);

                // If part of the prefix lives in a slower tier (or is in
                // flight for another request), kick off / join an async
                // promotion: reserve HBM blocks for the cached portion and
                // park the request until the transfer completes. The
                // running batch is unaffected (PCIe runs in parallel with
                // HBM).
                if (lookup.needs_promotion() || lookup.needs_join()) && cached_tokens > 0 {
                    // Each request reserves its own landing blocks; blocks
                    // already in flight for a leader are shared by
                    // reference. Bytes/PCIe cost is only paid for the
                    // spillover portion; the in-flight portion is joined.
                    let blocks_needed = self
                        .kv_cache_manager
                        .blocks_for_context(cached_tokens)
                        .saturating_sub(request.kv_blocks.len());
                    if self.kv_cache_manager.num_free_blocks() < blocks_needed {
                        break;
                    }
                    let mut request = self.waiting.remove(selected_idx).unwrap();
                    self.kv_cache_manager.record_prefix_lookup(&lookup);
                    request.num_cached_tokens = cached_tokens;
                    let allocated = self
                        .kv_cache_manager
                        .reserve_blocks_for_transfer(&request, cached_tokens)
                        .expect("blocks_needed already verified against capacity");
                    request.kv_blocks.extend(allocated);

                    // Two paths, possibly both: start a transfer for the
                    // spillover portion, and/or join an existing one for
                    // the in-flight portion.
                    if lookup.needs_promotion() {
                        self.kv_cache_manager.start_transfer(
                            request.request_id.clone(),
                            &lookup,
                            current_time,
                        );
                    } else {
                        self.kv_cache_manager
                            .join_transfer(request.request_id.clone(), &lookup);
                    }
                    let own_remaining = if lookup.needs_promotion() {
                        self.kv_cache_manager
                            .estimate_remaining_time(&request.request_id)
                    } else {
                        0.0
                    };
                    let join_remaining = lookup
                        .join_leader
                        .as_deref()
                        .map(|leader| self.kv_cache_manager.estimate_remaining_time(leader))
                        .unwrap_or(0.0);
                    request.ready_at = Some(current_time + own_remaining.max(join_remaining));
                    self.pending_transfers.push(request);
                    continue;
                }

                // Cached prefix already resident in HBM: skip its compute.
                // The blocks are still allocated (by reference) so the
                // request holds them for its lifetime.
                let tokens_to_schedule =
                    self.tokens_to_schedule_from(request, cached_tokens, token_budget);
                if tokens_to_schedule == 0 {
                    break;
                }
                // A promoted request already holds its cached-prefix blocks.
                let blocks_needed = self
                    .kv_cache_manager
                    .blocks_for_context(cached_tokens + tokens_to_schedule)
                    .saturating_sub(request.kv_blocks.len());
                if self.kv_cache_manager.num_free_blocks() < blocks_needed {
                    break; // Can't fit, stop scheduling new requests
                }

                let mut request = self.waiting.remove(selected_idx).unwrap();
                self.kv_cache_manager.record_prefix_lookup(&lookup);
                request.num_cached_tokens = cached_tokens;
                request.num_computed_tokens = cached_tokens;
                let blocks = self
                    .kv_cache_manager
                    .allocate_blocks(&request, tokens_to_schedule)
                    .expect("free-block check above guarantees allocation");
                request.kv_blocks.extend(blocks);

                decision.batch.push(ScheduledSeq {
                    idx: self.running.len(),
                    num_tokens: tokens_to_schedule,
                });
                token_budget -= tokens_to_schedule;
                self.running.push(request);
            }
        }

        decision
    }

    /// Prefix tokens whose compute can be skipped for `request`: block-aligned
    /// and short of the whole prompt, so at least the last block is computed
    /// (a fully cached prompt still needs a forward pass for its logits, as
    /// in vLLM).
    fn usable_cached_tokens(&self, request: &Request, cached: u32) -> u32 {
        let block_size = self.config.block_size.max(1);
        let cap = request.num_prompt_tokens.saturating_sub(1);
        (cached.min(cap) / block_size) * block_size
    }

    /// Positions to compute for `request` this step under `token_budget`,
    /// given that `computed` positions are already resident. Prefill is
    /// capped at the remaining prefill (never crossing into decode) and at
    /// the chunked-prefill threshold; decode processes one bonus position
    /// plus the pending speculative draft.
    fn tokens_to_schedule_from(&self, request: &Request, computed: u32, token_budget: u32) -> u32 {
        let planned = request.planned_positions();
        if request.is_finished() || computed >= planned {
            return 0;
        }
        let mut tokens = (planned - computed).min(token_budget);
        let prefill_len = request.prefill_len();
        if computed < prefill_len {
            tokens = tokens.min(prefill_len - computed);
            if self.config.enable_chunked_prefill && self.config.long_prefill_token_threshold > 0 {
                tokens = tokens.min(self.config.long_prefill_token_threshold);
            }
        } else {
            // `pending_draft_len` (decided last iteration; 0 if speculation
            // is off) makes the step process `1 + draft` positions, so the
            // token budget and KV are reserved for the verify cost. The
            // `.min(token_budget)` above already trims `draft` to fit.
            tokens = tokens.min(1 + request.pending_draft_len);
        }
        tokens
    }

    fn tokens_to_schedule(&self, request: &Request, token_budget: u32) -> u32 {
        self.tokens_to_schedule_from(request, request.num_computed_tokens, token_budget)
    }

    /// Whether admitting `request` keeps every running request able to grow
    /// to its bound (prompt + max output) without preemption. Conservative:
    /// assumes all sequences reach their peak simultaneously.
    fn can_admit_without_preemption(&self, request: &Request) -> bool {
        let running_peak: usize = self
            .running
            .iter()
            .map(|r| self.kv_cache_manager.blocks_for_context(r.total_tokens()))
            .sum();
        let new_peak = self
            .kv_cache_manager
            .blocks_for_context(request.total_tokens());
        running_peak + new_peak <= self.kv_cache_manager.total_blocks()
    }

    /// Index into `waiting` of the request the policy admits next.
    fn select_next_waiting_request(&self) -> usize {
        let pick = |key: &dyn Fn(&Request) -> u64, longest: bool| -> usize {
            let it = self.waiting.iter().enumerate();
            let best = if longest {
                it.max_by_key(|(_, r)| key(r))
            } else {
                it.min_by_key(|(_, r)| key(r))
            };
            best.map(|(i, _)| i).unwrap_or(0)
        };
        match self.policy {
            SchedulingPolicy::FCFS | SchedulingPolicy::Priority => 0,
            SchedulingPolicy::SIF => pick(&|r| r.num_prompt_tokens as u64, false),
            SchedulingPolicy::LIF => pick(&|r| r.num_prompt_tokens as u64, true),
            SchedulingPolicy::SOF => pick(&|r| r.max_output_tokens as u64, false),
            SchedulingPolicy::LOF => pick(&|r| r.max_output_tokens as u64, true),
            SchedulingPolicy::STF => pick(&|r| r.total_tokens() as u64, false),
            SchedulingPolicy::LTF => pick(&|r| r.total_tokens() as u64, true),
        }
    }

    /// Pick a preemption victim among running requests at positions
    /// `>= min_idx`. Positions before `min_idx` have already been scheduled
    /// this pass and recorded by index, so they must not be removed.
    fn select_preemption_victim(&self, min_idx: usize) -> Option<usize> {
        if min_idx >= self.running.len() {
            return None;
        }
        let candidates = || self.running.iter().enumerate().skip(min_idx);
        let longest = |key: &dyn Fn(&Request) -> u64| -> Option<usize> {
            candidates().max_by_key(|(_, r)| key(r)).map(|(i, _)| i)
        };
        match self.policy {
            // Preempt the most recently admitted request (vLLM's choice).
            SchedulingPolicy::FCFS => Some(self.running.len() - 1),
            // Lowest priority (highest value), latest arrival breaks ties.
            SchedulingPolicy::Priority => candidates()
                .max_by_key(|(_, r)| (r.priority, OrderedFloat(r.arrival_time)))
                .map(|(i, _)| i),
            // Length policies sacrifice the longest under memory pressure,
            // whichever end they prioritise on admission.
            SchedulingPolicy::SIF | SchedulingPolicy::LIF => {
                longest(&|r| r.num_prompt_tokens as u64)
            }
            SchedulingPolicy::SOF | SchedulingPolicy::LOF => {
                longest(&|r| (r.max_output_tokens - r.num_output_tokens) as u64)
            }
            SchedulingPolicy::STF | SchedulingPolicy::LTF => {
                longest(&|r| r.remaining_tokens() as u64)
            }
        }
    }

    /// Preempt a running request: free its KV; it recomputes on resume.
    fn preempt_request(&mut self, request: &mut Request) {
        self.num_preemptions += 1;
        self.kv_cache_manager.free_blocks(&request.kv_blocks);
        request.kv_blocks.clear();
        request.preempt();
    }

    /// Add a new request to the waiting queue.
    pub fn add_request(&mut self, request: Request) {
        self.waiting.push_back(request);
    }

    /// Record that running request `idx` computed `num_tokens` positions in
    /// the pass ending at `time`. Returns the output tokens it generated.
    pub fn record_progress(&mut self, idx: usize, num_tokens: u32, time: f64) -> u32 {
        self.running[idx].record_generated_tokens(num_tokens, time)
    }

    /// Set the speculative draft plan for the decode set (running requests
    /// that are neither prefilling nor finished), in `running` order.
    /// `plans` must have one entry per decode request; each entry is
    /// `(draft_len, round_commits)`.
    pub fn set_draft_plans(&mut self, plans: &[(u32, Option<u32>)]) {
        let mut k = 0usize;
        for req in self.running.iter_mut() {
            if !req.is_prefill() && !req.is_finished() {
                let (draft, commits) = plans.get(k).copied().unwrap_or((0, None));
                req.pending_draft_len = draft;
                req.pending_round_commits = commits;
                k += 1;
            }
        }
        debug_assert_eq!(k, plans.len(), "one draft plan per decode request");
    }

    /// Remove and return every running request whose prefill is complete and
    /// which still has output to generate, freeing its KV here. Used by a
    /// disaggregated prefill worker to hand requests to the decode pool.
    pub fn take_prefill_complete(&mut self) -> Vec<Request> {
        let mut handed = Vec::new();
        let mut keep = Vec::with_capacity(self.running.len());
        for mut r in self.running.drain(..) {
            if !r.is_prefill() && !r.is_finished() {
                self.kv_cache_manager.free_blocks(&r.kv_blocks);
                r.kv_blocks.clear();
                handed.push(r);
            } else {
                keep.push(r);
            }
        }
        self.running = keep;
        handed
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    /// Waiting requests, including requests parked on a KV transfer: they
    /// are still in the system, and the engine's idle check must see them.
    pub fn num_waiting(&self) -> usize {
        self.waiting.len() + self.pending_transfers.len()
    }

    /// Earliest `ready_at` among requests parked on a KV transfer.
    pub fn earliest_pending_ready(&self) -> Option<f64> {
        self.pending_transfers
            .iter()
            .filter_map(|r| r.ready_at)
            .min_by(f64::total_cmp)
    }

    pub fn num_preemptions(&self) -> u64 {
        self.num_preemptions
    }

    pub fn running(&self) -> &[Request] {
        &self.running
    }

    /// Requests parked on an in-flight KV promotion.
    pub fn pending_transfers(&self) -> &[Request] {
        &self.pending_transfers
    }

    pub fn kv_cache_manager(&self) -> &KVCacheManager {
        &self.kv_cache_manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, KVTier, ModelCosts};

    /// KV manager over the test model with `capacity` bytes.
    fn kv_manager(config: &Config, capacity: u64, prefix_caching: bool) -> KVCacheManager {
        let model = config.model.clone();
        KVCacheManager::new(
            capacity,
            config.scheduler.block_size,
            move |t| model.kv_storage_bytes(t),
            config.model.per_sequence_state_bytes(),
            prefix_caching,
        )
    }

    fn scheduler_from(config: Config, kv: KVCacheManager) -> Scheduler {
        Scheduler::new(config.scheduler, kv).unwrap()
    }

    fn create_test_scheduler() -> Scheduler {
        let config = Config::test_default();
        let kv = kv_manager(&config, config.hardware.kv_cache_capacity, false);
        scheduler_from(config, kv)
    }

    fn create_scheduler_with_policy(policy: &str) -> Scheduler {
        let mut config = Config::test_default();
        config.scheduler.policy = policy.to_string();
        let kv = kv_manager(&config, config.hardware.kv_cache_capacity, false);
        scheduler_from(config, kv)
    }

    /// Scheduler whose KV holds exactly `blocks` blocks (test model, no
    /// prefix caching).
    fn scheduler_with_blocks(policy: &str, blocks: u64) -> Scheduler {
        let mut config = Config::test_default();
        config.scheduler.policy = policy.to_string();
        let per_block = config.model.kv_storage_bytes(config.scheduler.block_size);
        let kv = kv_manager(&config, blocks * per_block, false);
        scheduler_from(config, kv)
    }

    fn create_test_request(id: &str, prompt: u32, output: u32) -> Request {
        Request::new(id.to_string(), 0, 0.0, prompt, output)
    }

    /// Apply a decision as the engine would: every batch entry computes its
    /// scheduled positions.
    fn apply(scheduler: &mut Scheduler, decision: &ScheduleDecision, time: f64) {
        for s in &decision.batch {
            scheduler.record_progress(s.idx, s.num_tokens, time);
        }
    }

    fn running_ids(scheduler: &Scheduler) -> Vec<&str> {
        scheduler
            .running()
            .iter()
            .map(|r| r.request_id.as_str())
            .collect()
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
        scheduler.add_request(create_test_request("req-1", 100, 50));
        assert_eq!(scheduler.num_waiting(), 1);
    }

    #[test]
    fn test_waiting_on_transfer_then_promoted() {
        let config = Config::test_default();
        let block_size = config.scheduler.block_size;
        let kv =
            kv_manager(&config, config.hardware.kv_cache_capacity, true).with_tiers(&[KVTier {
                name: "host_ram".into(),
                // Plenty of host RAM.
                capacity_bytes: 10 * 1024 * 1024 * 1024,
                // 1 GB/s, very slow on purpose so the transfer time is observable.
                bandwidth_to_hbm: 1e9,
            }]);
        let mut scheduler = scheduler_from(config, kv);

        // Seed the host-RAM tier: allocate then free a block carrying our
        // prefix hash, then recycle it with a different hash so `prefix_hash`
        // is demoted into host RAM.
        let prefix_hash = 0xCAFE_u64;
        let mgr = &mut scheduler.kv_cache_manager;
        let mut seed = create_test_request("seed", block_size, 1);
        seed.prompt_block_hashes = vec![prefix_hash];
        let blocks = mgr.allocate_blocks(&seed, block_size).unwrap();
        mgr.free_blocks(&blocks);
        let mut churn = create_test_request("churn", block_size, 1);
        churn.prompt_block_hashes = vec![0xDEAD_u64];
        mgr.allocate_blocks(&churn, block_size).unwrap();

        // A request whose prompt starts with the prefix hash.
        let mut req = create_test_request("req", block_size * 2, 1);
        req.prompt_block_hashes = vec![prefix_hash, 0xBEEF_u64];
        scheduler.add_request(req);

        // t=0: host-RAM hit; parked in pending_transfers, not running.
        let decision = scheduler.schedule(0.0);
        assert!(decision.batch.is_empty());
        assert_eq!(scheduler.num_waiting(), 1); // parked requests count as waiting
        assert_eq!(scheduler.pending_transfers.len(), 1);
        let ready_at = scheduler.pending_transfers[0].ready_at.unwrap();
        assert!(ready_at > 0.0);
        assert_eq!(scheduler.earliest_pending_ready(), Some(ready_at));

        // Before the transfer completes: still pending.
        let decision = scheduler.schedule(ready_at / 2.0);
        assert!(decision.batch.is_empty());
        assert_eq!(scheduler.pending_transfers.len(), 1);

        // After it completes, the request promotes back and runs from the
        // cached prefix.
        let decision = scheduler.schedule(ready_at + 1e-9);
        assert_eq!(scheduler.pending_transfers.len(), 0);
        assert_eq!(decision.batch.len(), 1);
        assert_eq!(scheduler.num_running(), 1);
        assert_eq!(scheduler.running()[0].num_computed_tokens, block_size);
        assert_eq!(decision.batch[0].num_tokens, block_size);
    }

    #[test]
    fn test_concurrent_same_prefix_join_one_transfer() {
        let config = Config::test_default();
        let block_size = config.scheduler.block_size;
        let per_block = config.model.kv_storage_bytes(block_size);
        // Constrained HBM so a model that didn't share the prefix block would
        // fail: one shared prefix block plus one private block per request.
        let kv = kv_manager(&config, 4 * per_block, true).with_tiers(&[KVTier {
            name: "host_ram".into(),
            capacity_bytes: 16 * per_block,
            bandwidth_to_hbm: 1e9,
        }]);
        let mut scheduler = scheduler_from(config, kv);

        // Pre-warm the prefix into host RAM.
        let prefix_hash = 0xABCDu64;
        {
            let mgr = &mut scheduler.kv_cache_manager;
            let mut seed = create_test_request("seed", block_size, 1);
            seed.prompt_block_hashes = vec![prefix_hash];
            let blocks = mgr.allocate_blocks(&seed, block_size).unwrap();
            mgr.free_blocks(&blocks);
            let mut churn = create_test_request("churn", block_size * 2, 1);
            churn.prompt_block_hashes = vec![0xDEAD, 0xBEEF];
            let cb = mgr.allocate_blocks(&churn, block_size * 2).unwrap();
            mgr.free_blocks(&cb);
        }

        for i in 0..3 {
            let mut req = create_test_request(&format!("req-{i}"), block_size * 2, 1);
            req.prompt_block_hashes = vec![prefix_hash, 0x1000 + i as u64];
            scheduler.add_request(req);
        }

        // First tick: leader starts the transfer; followers join it and
        // reference the same landing block.
        let _ = scheduler.schedule(0.0);
        assert_eq!(scheduler.pending_transfers.len(), 3);
        let leader_block = scheduler.pending_transfers[0].kv_blocks[0];
        for r in scheduler.pending_transfers.iter() {
            assert_eq!(r.kv_blocks[0], leader_block);
        }
        assert_eq!(
            scheduler.kv_cache_manager().block_ref_count(leader_block),
            3
        );

        let _ = scheduler.schedule(10.0);
        assert_eq!(scheduler.pending_transfers.len(), 0);
        assert_eq!(scheduler.num_running(), 3);
        // Each request holds the shared prefix block plus its own second block.
        let mut second: Vec<u32> = scheduler.running().iter().map(|r| r.kv_blocks[1]).collect();
        second.sort();
        second.dedup();
        assert_eq!(second.len(), 3);
        assert!(scheduler
            .running()
            .iter()
            .all(|r| r.kv_blocks[0] == leader_block));
        assert_eq!(scheduler.kv_cache_manager().num_free_blocks(), 0);
    }

    #[test]
    fn test_hbm_prefix_hit_skips_cached_compute() {
        let config = Config::test_default();
        let bs = config.scheduler.block_size;
        let kv = kv_manager(&config, config.hardware.kv_cache_capacity, true);
        let mut scheduler = scheduler_from(config, kv);

        let mut a = create_test_request("a", 4 * bs, 10);
        a.prompt_block_hashes = vec![1, 2, 3, 4];
        scheduler.add_request(a);
        let d = scheduler.schedule(0.0);
        assert_eq!(d.batch[0].num_tokens, 4 * bs);
        apply(&mut scheduler, &d, 1.0);

        // b shares a's first three blocks: it computes only its last block.
        let mut b = create_test_request("b", 4 * bs, 10);
        b.prompt_block_hashes = vec![1, 2, 3, 9];
        scheduler.add_request(b);
        let d = scheduler.schedule(1.0);
        let b_entry = d.batch.iter().find(|s| s.idx == 1).unwrap();
        assert_eq!(b_entry.num_tokens, bs);
        assert_eq!(scheduler.running()[1].num_computed_tokens, 3 * bs);
        assert_eq!(scheduler.running()[1].num_cached_tokens, 3 * bs);
        let stats = scheduler.kv_cache_manager().prefix_cache_stats();
        assert_eq!((stats.hits, stats.misses), (1, 1));

        // c is identical to a: a fully cached prompt still computes its
        // last block (the logits need a forward pass).
        let mut c = create_test_request("c", 4 * bs, 10);
        c.prompt_block_hashes = vec![1, 2, 3, 4];
        scheduler.add_request(c);
        let d = scheduler.schedule(2.0);
        let c_entry = d.batch.iter().find(|s| s.idx == 2).unwrap();
        assert_eq!(c_entry.num_tokens, bs);
        assert_eq!(scheduler.running()[2].num_computed_tokens, 3 * bs);
    }

    #[test]
    fn test_schedule_single_request() {
        let mut scheduler = create_test_scheduler();
        scheduler.add_request(create_test_request("req-1", 16, 10));
        let decision = scheduler.schedule(0.0);
        assert_eq!(decision.batch.len(), 1);
        assert_eq!(decision.batch[0].num_tokens, 16);
        assert_eq!(scheduler.num_running(), 1);
        assert_eq!(scheduler.num_waiting(), 0);
    }

    #[test]
    fn test_schedule_multiple_requests() {
        let mut scheduler = create_test_scheduler();
        scheduler.add_request(create_test_request("req-1", 16, 10));
        scheduler.add_request(create_test_request("req-2", 16, 10));
        let decision = scheduler.schedule(0.0);
        assert_eq!(decision.batch.len(), 2);
        assert_eq!(scheduler.num_running(), 2);
    }

    #[test]
    fn test_prefill_then_decode_then_completion() {
        let mut scheduler = create_test_scheduler();
        scheduler.add_request(create_test_request("req-1", 16, 3));
        // Prefill pass: 16 positions, yields token 1.
        let d = scheduler.schedule(0.0);
        assert_eq!(d.batch[0].num_tokens, 16);
        apply(&mut scheduler, &d, 1.0);
        assert_eq!(scheduler.running()[0].num_output_tokens, 1);
        // Two decode passes finish it (3 tokens total).
        let d = scheduler.schedule(1.0);
        assert_eq!(d.batch[0].num_tokens, 1);
        apply(&mut scheduler, &d, 2.0);
        let d = scheduler.schedule(2.0);
        assert_eq!(d.batch[0].num_tokens, 1);
        apply(&mut scheduler, &d, 3.0);
        assert!(scheduler.running()[0].is_finished());
        // Reaped at the next pass, blocks freed.
        let free_before = scheduler.kv_cache_manager().num_free_blocks();
        let d = scheduler.schedule(3.0);
        assert_eq!(d.completed.len(), 1);
        assert_eq!(scheduler.num_running(), 0);
        assert!(scheduler.kv_cache_manager().num_free_blocks() > free_before);
    }

    #[test]
    fn test_speculative_verify_width_and_trim() {
        let mut scheduler = create_test_scheduler();
        scheduler.add_request(create_test_request("req-1", 16, 4));
        let d = scheduler.schedule(0.0);
        apply(&mut scheduler, &d, 1.0);
        // Plan a 5-token draft: the verify width is 1 + 5, trimmed to the
        // three positions still needed.
        scheduler.set_draft_plans(&[(5, None)]);
        let d = scheduler.schedule(1.0);
        assert_eq!(d.batch[0].num_tokens, 3);
    }

    #[test]
    fn test_preemption_fcfs_takes_last() {
        let mut scheduler = create_test_scheduler();
        scheduler.add_request(create_test_request("req-1", 16, 10));
        scheduler.add_request(create_test_request("req-2", 16, 10));
        scheduler.schedule(0.0);
        assert_eq!(scheduler.select_preemption_victim(0), Some(1));
        assert_eq!(scheduler.select_preemption_victim(1), Some(1));
        assert_eq!(scheduler.select_preemption_victim(2), None);
    }

    #[test]
    fn test_preemption_recomputes_and_requeues_at_head() {
        // 6 blocks; block_size 16. a: 32-token prompt (2 blocks), b: 32
        // (2 blocks). Both at a block boundary after prefill+1 decode... use
        // 31-token prompts so the prefill pass fills two blocks exactly at
        // computed=32 after the first decode.
        let mut scheduler = scheduler_with_blocks("fcfs", 5);
        scheduler.add_request(create_test_request("a", 32, 100));
        scheduler.add_request(create_test_request("b", 32, 100));
        scheduler.add_request(create_test_request("late", 16, 100));
        let d = scheduler.schedule(0.0); // a, b prefill (4 blocks); late waits
        assert_eq!(d.batch.len(), 3); // 4 + 1 blocks fit
        apply(&mut scheduler, &d, 1.0);
        // Everyone at a block boundary needing a new block; 0 free.
        assert_eq!(scheduler.kv_cache_manager().num_free_blocks(), 0);
        let d = scheduler.schedule(1.0);
        // FCFS preempts from the end until the cursor fits: `late` (idx 2)
        // is preempted for a; then b (idx 1) needs a block, none free ->
        // preempts itself. a runs alone.
        assert_eq!(d.num_preempted, 2);
        assert_eq!(d.batch.len(), 1);
        assert_eq!(running_ids(&scheduler), vec!["a"]);
        assert_eq!(scheduler.num_preemptions(), 2);
        // Preempted requests are at the head of the waiting queue, in the
        // order they were preempted, with their KV gone but tokens kept.
        let head = scheduler.waiting.front().unwrap();
        assert_eq!(head.request_id, "b");
        assert_eq!(head.num_computed_tokens, 0);
        assert_eq!(head.num_output_tokens, 1);
        assert!(head.kv_blocks.is_empty());
        assert_eq!(head.num_preemptions, 1);
        assert!(head.is_prefill());
        // Nothing new is admitted in a pass that preempted.
        assert_eq!(scheduler.num_waiting(), 2);
    }

    #[test]
    fn test_preemption_victim_never_precedes_cursor() {
        // Regression: a length policy must not evict a request already
        // recorded in the batch (indices would shift). SIF's victim is the
        // longest prompt, which we admit first so it sits at index 0.
        let mut scheduler = scheduler_with_blocks("sif", 12);
        scheduler.add_request(create_test_request("A", 64, 100));
        let d = scheduler.schedule(0.0);
        apply(&mut scheduler, &d, 0.5);
        scheduler.add_request(create_test_request("B", 32, 100));
        scheduler.add_request(create_test_request("C", 32, 100));
        let d = scheduler.schedule(0.5);
        apply(&mut scheduler, &d, 1.0);
        assert_eq!(running_ids(&scheduler), vec!["A", "B", "C"]);
        let d = scheduler.schedule(1.0);
        apply(&mut scheduler, &d, 1.5);
        // A=66 (5 blocks), B=C=33 (3 blocks each): 11 used, 1 free.
        assert_eq!(scheduler.kv_cache_manager().num_free_blocks(), 1);
        for r in scheduler.running.iter_mut() {
            r.record_generated_tokens(15, 2.0);
        }
        // A=81, B=C=48: everyone's next position opens a new block. A takes
        // the last free one; B must preempt. SIF's longest prompt is A (idx
        // 0, already in the batch) — the victim must be chosen at >= idx 1.
        let d = scheduler.schedule(2.0);
        assert!(!running_ids(&scheduler).contains(&"C"));
        assert_eq!(running_ids(&scheduler), vec!["A", "B"]);
        let mut idxs: Vec<usize> = d.batch.iter().map(|s| s.idx).collect();
        idxs.dedup();
        assert_eq!(idxs.len(), d.batch.len(), "no duplicate batch indices");
        assert!(d.batch.iter().all(|s| s.idx < scheduler.num_running()));
        assert!(d.num_preempted >= 1);
    }

    #[test]
    fn test_sof_selection() {
        let mut scheduler = create_scheduler_with_policy("sof");
        scheduler.add_request(create_test_request("req-long", 16, 100));
        scheduler.add_request(create_test_request("req-short", 16, 10));
        scheduler.add_request(create_test_request("req-medium", 16, 50));
        let idx = scheduler.select_next_waiting_request();
        assert_eq!(scheduler.waiting[idx].max_output_tokens, 10);
    }

    #[test]
    fn test_sof_preemption() {
        let mut scheduler = create_scheduler_with_policy("sof");
        let mut req1 = create_test_request("req-1", 16, 100);
        req1.num_output_tokens = 50; // 50 remaining
        let mut req2 = create_test_request("req-2", 16, 50);
        req2.num_output_tokens = 10; // 40 remaining
        let mut req3 = create_test_request("req-3", 16, 30);
        req3.num_output_tokens = 5; // 25 remaining
        scheduler.running.extend([req1, req2, req3]);
        // Longest remaining output is preempted...
        assert_eq!(scheduler.select_preemption_victim(0), Some(0));
        // ...but only among positions at or after the cursor.
        assert_eq!(scheduler.select_preemption_victim(1), Some(1));
    }

    #[test]
    fn test_sif_selection() {
        let mut scheduler = create_scheduler_with_policy("sif");
        scheduler.add_request(create_test_request("req-long", 200, 10));
        scheduler.add_request(create_test_request("req-short", 50, 10));
        scheduler.add_request(create_test_request("req-medium", 100, 10));
        let idx = scheduler.select_next_waiting_request();
        assert_eq!(scheduler.waiting[idx].num_prompt_tokens, 50);
    }

    #[test]
    fn test_sif_preemption() {
        let mut scheduler = create_scheduler_with_policy("sif");
        scheduler.running.extend([
            create_test_request("req-1", 200, 10),
            create_test_request("req-2", 50, 10),
            create_test_request("req-3", 100, 10),
        ]);
        let victim = scheduler.select_preemption_victim(0).unwrap();
        assert_eq!(scheduler.running[victim].num_prompt_tokens, 200);
    }

    #[test]
    fn test_lif_selection() {
        let mut scheduler = create_scheduler_with_policy("lif");
        scheduler.add_request(create_test_request("req-long", 200, 10));
        scheduler.add_request(create_test_request("req-short", 50, 10));
        scheduler.add_request(create_test_request("req-medium", 100, 10));
        let idx = scheduler.select_next_waiting_request();
        assert_eq!(scheduler.waiting[idx].num_prompt_tokens, 200);
    }

    #[test]
    fn test_lif_preemption() {
        let mut scheduler = create_scheduler_with_policy("lif");
        scheduler.running.extend([
            create_test_request("req-1", 200, 10),
            create_test_request("req-2", 50, 10),
            create_test_request("req-3", 100, 10),
        ]);
        let victim = scheduler.select_preemption_victim(0).unwrap();
        assert_eq!(scheduler.running[victim].num_prompt_tokens, 200);
    }

    #[test]
    fn test_priority_selection_fifo() {
        let mut scheduler = create_scheduler_with_policy("priority");
        scheduler.add_request(create_test_request("req-1", 100, 10));
        scheduler.add_request(create_test_request("req-2", 200, 20));
        scheduler.add_request(create_test_request("req-3", 50, 5));
        assert_eq!(scheduler.select_next_waiting_request(), 0);
    }

    #[test]
    fn test_priority_preemption() {
        let mut scheduler = create_scheduler_with_policy("priority");
        let mut req1 = create_test_request("req-1", 100, 10);
        req1.priority = 0;
        req1.arrival_time = 1.0;
        let mut req2 = create_test_request("req-2", 100, 10);
        req2.priority = 5;
        req2.arrival_time = 2.0;
        let mut req3 = create_test_request("req-3", 100, 10);
        req3.priority = 2;
        req3.arrival_time = 3.0;
        scheduler.running.extend([req1, req2, req3]);
        let victim = scheduler.select_preemption_victim(0).unwrap();
        assert_eq!(scheduler.running[victim].priority, 5);
    }

    #[test]
    fn test_fcfs_selection_fifo() {
        let mut sched = create_test_scheduler();
        sched.add_request(create_test_request("req-1", 100, 10));
        sched.add_request(create_test_request("req-2", 200, 20));
        sched.add_request(create_test_request("req-3", 50, 5));
        assert_eq!(sched.select_next_waiting_request(), 0);
    }

    #[test]
    fn test_policy_selection_deterministic() {
        let mut scheduler = create_scheduler_with_policy("sjf");
        scheduler.add_request(create_test_request("req-1", 16, 100));
        scheduler.add_request(create_test_request("req-2", 16, 10));
        scheduler.add_request(create_test_request("req-3", 16, 50));
        let idx1 = scheduler.select_next_waiting_request();
        let idx2 = scheduler.select_next_waiting_request();
        assert_eq!(idx1, idx2);
        assert_eq!(scheduler.waiting[idx1].max_output_tokens, 10);
    }

    fn create_scheduler_with_preemption_free() -> Scheduler {
        let mut config = Config::test_default();
        config.scheduler.enable_preemption_free = true;
        let kv = kv_manager(&config, config.hardware.kv_cache_capacity, false);
        scheduler_from(config, kv)
    }

    #[test]
    fn test_preemption_free_admission_control() {
        let mut config = Config::test_default();
        config.scheduler.enable_preemption_free = true;
        config.hardware.kv_cache_capacity = 100_000_000; // Small cache
        let kv = kv_manager(&config, config.hardware.kv_cache_capacity, false);
        let mut scheduler = scheduler_from(config, kv);

        let total_blocks = scheduler.kv_cache_manager().total_blocks();
        // A request whose peak context takes 60% of the cache.
        let blocks_for_first = (total_blocks * 60) / 100;
        let tokens_for_first = blocks_for_first * 16;
        let half = tokens_for_first as u32 / 2;
        scheduler.add_request(create_test_request("req-large", half, half));
        let decision1 = scheduler.schedule(0.0);
        assert_eq!(
            decision1.batch.len(),
            1,
            "First request should be scheduled"
        );
        assert_eq!(scheduler.num_running(), 1);

        // Another 60% request is not admitted: 60% + 60% > 100%.
        scheduler.add_request(create_test_request("req-large2", half, half));
        let decision2 = scheduler.schedule(0.0);
        assert!(
            decision2.batch.iter().all(|s| s.idx == 0),
            "Second request should not be admitted"
        );
        assert_eq!(scheduler.num_running(), 1);
        assert_eq!(scheduler.num_waiting(), 1);
    }

    #[test]
    fn test_preemption_free_no_preemptions() {
        let mut scheduler = create_scheduler_with_preemption_free();
        scheduler.add_request(create_test_request("req-1", 100, 50));
        scheduler.add_request(create_test_request("req-2", 100, 50));
        scheduler.add_request(create_test_request("req-3", 100, 50));

        let mut total_preemptions = 0;
        let mut time = 0.0;
        for _ in 0..100 {
            let decision = scheduler.schedule(time);
            total_preemptions += decision.num_preempted;
            apply(&mut scheduler, &decision, time);
            time += 0.01;
            if scheduler.num_running() == 0 && scheduler.num_waiting() == 0 {
                break;
            }
        }
        assert_eq!(total_preemptions, 0);
        assert_eq!(scheduler.num_preemptions(), 0);
    }

    #[test]
    fn test_preemption_free_fcfs_ordering() {
        let mut scheduler = create_scheduler_with_preemption_free();
        let mut req1 = create_test_request("req-1", 16, 10);
        req1.arrival_time = 0.0;
        let mut req2 = create_test_request("req-2", 16, 10);
        req2.arrival_time = 1.0;
        let mut req3 = create_test_request("req-3", 16, 10);
        req3.arrival_time = 2.0;
        scheduler.add_request(req1);
        scheduler.add_request(req2);
        scheduler.add_request(req3);
        let decision = scheduler.schedule(0.0);
        assert!(!decision.batch.is_empty());
        assert_eq!(scheduler.running[0].request_id, "req-1");
    }

    #[test]
    fn test_take_prefill_complete_hands_off_and_frees_kv() {
        let mut scheduler = create_test_scheduler();
        scheduler.add_request(create_test_request("done", 16, 5));
        scheduler.add_request(create_test_request("mid", 64, 5));
        scheduler.add_request(create_test_request("one", 16, 1));
        let d = scheduler.schedule(0.0);
        // Give `mid` only part of its prefill.
        for s in &d.batch {
            let tokens = if s.idx == 1 { 16 } else { s.num_tokens };
            scheduler.record_progress(s.idx, tokens, 1.0);
        }
        let free_before = scheduler.kv_cache_manager().num_free_blocks();
        let handed = scheduler.take_prefill_complete();
        let ids: Vec<&str> = handed.iter().map(|r| r.request_id.as_str()).collect();
        // `done` finished prefill with output left; `one` is finished (stays
        // to be reaped); `mid` is still prefilling.
        assert_eq!(ids, vec!["done"]);
        assert!(handed[0].kv_blocks.is_empty());
        assert!(scheduler.kv_cache_manager().num_free_blocks() > free_before);
        assert_eq!(running_ids(&scheduler), vec!["mid", "one"]);
    }
}
