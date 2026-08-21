use super::{decision::ScheduleDecision, decision::ScheduledSeq, policy::SchedulingPolicy};
use crate::config::{
    LoadOverlap, PrefetchPolicy, PromoteFill, SchedulerConfig, SourcePolicy, StoragePrefetchPolicy,
};
use crate::kv_cache::{KVCacheManager, PrefixCacheLookup};
use crate::request::Request;
use ordered_float::OrderedFloat;
use std::collections::VecDeque;

/// A prefix to pull back into HBM ahead of its announced re-entry.
#[derive(Debug, Clone)]
struct PrefetchPlan {
    /// When to start it.
    fire_at: f64,
    /// The re-entry's block hashes (a prefix of the completed step's).
    hashes: Vec<u64>,
    tokens: u32,
    id: String,
}

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

    /// Requests whose external-tier hit is being staged into the next closer
    /// store. They hold store capacity but no HBM.
    pending_storage: Vec<Request>,

    policy: SchedulingPolicy,

    kv_cache_manager: KVCacheManager,

    /// Total preemptions performed so far (running requests evicted under
    /// KV pressure, and waiting requests whose promoted prefix was released
    /// to unblock admission — see `release_waiting_kv`).
    num_preemptions: u64,

    /// Where a tier-held prefix comes from at admission.
    source: SourcePolicy,

    /// Roofline seconds to recompute `tokens` positions of `request`
    /// starting at position `from`, alone on the worker; set by the worker
    /// from its compute engine. Needed by `source = min_time`.
    recompute_seconds: Option<RecomputeFn>,

    /// Whether demoted prefixes are pulled back ahead of their re-entry.
    prefetch: PrefetchPolicy,
    promote_fill: PromoteFill,
    storage_prefetch: StoragePrefetchPolicy,
    load_overlap: LoadOverlap,
    /// Prefetches planned, soonest first.
    prefetch_plans: Vec<PrefetchPlan>,
    /// Prefetches in flight: synthetic requests holding the landing
    /// blocks; on completion their blocks are published and freed
    /// (hittable) instead of scheduled.
    prefetches: Vec<Request>,
    /// Synthetic outlook prefetches still moving through external stores;
    /// like demand stages, these hold no HBM yet.
    pending_prefetch_storage: Vec<Request>,
    next_prefetch_seq: u64,

    /// Requests refused at submission, handed back as completed on the
    /// next pass.
    rejected: Vec<Request>,
}

/// `(request, from, tokens) -> seconds`.
pub type RecomputeFn = Box<dyn Fn(&Request, u32, u32) -> f64 + Send + Sync>;

impl Scheduler {
    pub fn new(config: SchedulerConfig, kv_cache_manager: KVCacheManager) -> Self {
        Self {
            policy: config.policy,
            config,
            waiting: VecDeque::new(),
            running: Vec::new(),
            pending_transfers: Vec::new(),
            pending_storage: Vec::new(),
            kv_cache_manager,
            num_preemptions: 0,
            source: SourcePolicy::Promote {},
            recompute_seconds: None,
            prefetch: PrefetchPolicy::None {},
            promote_fill: PromoteFill::Direct,
            storage_prefetch: StoragePrefetchPolicy::WaitComplete {},
            load_overlap: LoadOverlap::None,
            prefetch_plans: Vec::new(),
            prefetches: Vec::new(),
            pending_prefetch_storage: Vec::new(),
            next_prefetch_seq: 0,
            rejected: Vec::new(),
        }
    }

    /// Set the prefetch policy.
    pub fn with_prefetch(mut self, prefetch: PrefetchPolicy) -> Self {
        self.prefetch = prefetch;
        self
    }

    /// Configure HiCache-style staged reads and final-leg overlap.
    pub fn with_hicache(
        mut self,
        promote_fill: PromoteFill,
        storage_prefetch: StoragePrefetchPolicy,
        load_overlap: LoadOverlap,
    ) -> Self {
        self.promote_fill = promote_fill;
        self.storage_prefetch = storage_prefetch;
        self.load_overlap = load_overlap;
        self
    }

    /// A session step completed with an outlook: under `prefetch =
    /// outlook`, plan to pull its shared prefix back into HBM so it lands
    /// `lead` seconds before the re-entry, assuming the fetch path's
    /// current fair share. A plan whose start is already past is dropped
    /// (the prefix is either still in HBM or promotes on arrival).
    fn plan_prefetch(&mut self, req: &Request, now: f64) {
        let PrefetchPolicy::Outlook { lead } = self.prefetch else {
            return;
        };
        let Some(outlook) = req.outlook_at(now) else {
            return;
        };
        let block_size = self.config.block_size.max(1);
        let blocks = (outlook.shared_tokens / block_size) as usize;
        if blocks == 0 || blocks > req.prompt_block_hashes.len() {
            return;
        }
        let tokens = blocks as u32 * block_size;
        let transfer = self.kv_cache_manager.estimate_promotion_of(tokens);
        if !transfer.is_finite() {
            return;
        }
        let fire_at = outlook.next_arrival - lead - transfer;
        if fire_at <= now {
            return;
        }
        let id = format!("pf:{}:{}", req.request_id, self.next_prefetch_seq);
        self.next_prefetch_seq += 1;
        let plan = PrefetchPlan {
            fire_at,
            hashes: req.prompt_block_hashes[..blocks].to_vec(),
            tokens,
            id,
        };
        let at = self
            .prefetch_plans
            .partition_point(|p| p.fire_at <= plan.fire_at);
        self.prefetch_plans.insert(at, plan);
    }

    /// Start every prefetch plan that is due: whatever of its prefix a
    /// tier holds is promoted under a synthetic leader; a prefix still in
    /// HBM, already in flight, or gone from every tier needs nothing. A
    /// plan that can't reserve its landing blocks is dropped.
    fn fire_due_prefetches(&mut self, now: f64) {
        while self
            .prefetch_plans
            .first()
            .is_some_and(|p| p.fire_at <= now)
        {
            let plan = self.prefetch_plans.remove(0);
            // One token past the planned range so every planned block is
            // a usable prefix block (the last prompt block is never cached).
            let mut probe = Request::new(plan.id.clone(), 0, now, plan.tokens + 1, 1);
            probe.prompt_block_hashes = plan.hashes.clone();
            let lookup = self.kv_cache_manager.peek_prefix_cache(&probe);
            if !lookup.needs_promotion() {
                continue;
            }
            let promoted: u32 = lookup.promote_tokens_per_tier.iter().sum();
            if self.start_prefetch_leg(probe, lookup, now) {
                self.kv_cache_manager.record_prefetch(promoted);
            }
        }
    }

    /// Start the next leg of a synthetic outlook prefetch. External sources
    /// stage without HBM; the first source that no longer needs staging
    /// reserves the final landing blocks and loads into HBM.
    fn start_prefetch_leg(
        &mut self,
        mut probe: Request,
        lookup: PrefixCacheLookup,
        now: f64,
    ) -> bool {
        let cached = lookup
            .total_cached_tokens
            .min(probe.num_prompt_tokens.saturating_sub(1));
        if self.kv_cache_manager.needs_staging(&lookup) {
            if !self
                .kv_cache_manager
                .start_staging(probe.request_id.clone(), &lookup, now)
            {
                return false;
            }
            probe.num_cached_tokens = cached;
            self.pending_prefetch_storage.push(probe);
            return true;
        }
        let blocks_needed = self.kv_cache_manager.blocks_for_context(cached);
        if self.kv_cache_manager.num_free_blocks() < blocks_needed
            || self
                .kv_cache_manager
                .reserve_blocks_for_transfer(&mut probe, cached)
                .is_none()
        {
            return false;
        }
        probe.num_cached_tokens = cached;
        self.kv_cache_manager.start_transfer(
            probe.request_id.clone(),
            &lookup,
            &probe.prompt_block_hashes,
            now,
        );
        self.prefetches.push(probe);
        true
    }

    /// When the next prefetch plan is due, if any.
    pub fn next_prefetch_at(&self) -> Option<f64> {
        let planned = self.prefetch_plans.first().map(|p| p.fire_at);
        let deadline = self
            .pending_storage
            .iter()
            .filter_map(|r| r.ready_at)
            .min_by(f64::total_cmp);
        match (planned, deadline) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    }

    /// Whether calling [`Scheduler::schedule`] can change scheduler-local
    /// state at `now`. Transfer completions live on the shared memory graph,
    /// so a caller using this to skip an idle DP-attention rank must check
    /// that worker's completion queue separately.
    pub(crate) fn has_local_work_for_schedule(&self, now: f64) -> bool {
        !self.running.is_empty()
            || !self.waiting.is_empty()
            || !self.pending_transfers.is_empty()
            || !self.pending_storage.is_empty()
            || !self.rejected.is_empty()
            || self.next_prefetch_at().is_some_and(|at| at <= now)
    }

    /// Set where a tier-held prefix comes from at admission. `min_time`
    /// needs `recompute_seconds` (the worker's roofline for a prefill
    /// chunk); without it every prefix promotes.
    pub fn with_source(
        mut self,
        source: SourcePolicy,
        recompute_seconds: Option<RecomputeFn>,
    ) -> Self {
        self.source = source;
        self.recompute_seconds = recompute_seconds;
        self
    }

    /// Fetch or recompute the tier-held part of `lookup` for `request`:
    /// under `min_time`, recompute when the roofline prefill of those
    /// tokens (from position `resident`, the HBM + in-flight prefix)
    /// beats the promotion at the fetch path's current fair share.
    fn recompute_instead(
        &self,
        request: &Request,
        lookup: &PrefixCacheLookup,
        resident: u32,
    ) -> bool {
        let SourcePolicy::MinTime {} = self.source else {
            return false;
        };
        let Some(est) = &self.recompute_seconds else {
            return false;
        };
        let promote_tokens: u32 = lookup.promote_tokens_per_tier.iter().sum();
        if promote_tokens == 0 {
            return false;
        }
        let fetch = self.kv_cache_manager.estimate_fetch(lookup);
        let recompute = est(request, resident, promote_tokens);
        recompute <= fetch
    }

    fn park_on_storage_stage(
        &mut self,
        mut request: Request,
        lookup: &PrefixCacheLookup,
        cached_tokens: u32,
        current_time: f64,
    ) -> Option<Request> {
        if !self.promote_fill.stages()
            || !self.kv_cache_manager.needs_staging(lookup)
            || !self.kv_cache_manager.start_staging(
                request.request_id.clone(),
                lookup,
                current_time,
            )
        {
            return Some(request);
        }
        request.num_cached_tokens = cached_tokens;
        if let StoragePrefetchPolicy::Timeout { seconds } = self.storage_prefetch {
            request.ready_at.get_or_insert(current_time + seconds);
        } else {
            request.ready_at = None;
        }
        self.pending_storage.push(request);
        None
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
        // A timeout applies to the whole staged load, not separately to each
        // storage leg. Cancel expired flows before advancing the shared graph
        // so a completion observed after its deadline cannot become usable.
        let expired: Vec<String> = self
            .pending_storage
            .iter()
            .filter(|req| {
                req.ready_at
                    .is_some_and(|deadline| deadline <= current_time)
            })
            .map(|req| req.request_id.clone())
            .collect();
        for request_id in expired {
            self.kv_cache_manager.cancel_staging(&request_id);
        }
        let completed = self.kv_cache_manager.advance_transfers(current_time);
        let mut still_pending = Vec::with_capacity(self.pending_transfers.len());
        for mut req in self.pending_transfers.drain(..) {
            if let Some(landed_blocks) = completed
                .as_ref()
                .and_then(|completed| completed.hbm.get(&req.request_id))
                .copied()
            {
                // Publish the now-resident blocks to the HBM prefix cache so
                // subsequent same-prefix requests get a clean HBM hit.
                let cached_blocks = self
                    .kv_cache_manager
                    .content_blocks_for_tokens(req.num_cached_tokens)
                    .min(landed_blocks as usize);
                req.num_cached_tokens =
                    (cached_blocks as u32).saturating_mul(self.kv_cache_manager.block_size());
                self.kv_cache_manager
                    .publish_transferred_blocks(&mut req, cached_blocks);
                if self.load_overlap == LoadOverlap::Layerwise && self.promote_fill.stages() {
                    req.load_overlap_credit = req
                        .load_started_at
                        .map_or(0.0, |started| (current_time - started).max(0.0));
                }
                req.load_started_at = None;
                req.ready_at = None;
                req.num_computed_tokens = req.num_cached_tokens;
                // Its prefix is hot and its landing blocks are held: admit
                // it ahead of the queue rather than let it hold KV behind
                // requests that cannot fit (vLLM schedules a request whose
                // remote KV landed on the next pass).
                self.waiting.push_front(req);
            } else {
                // Its `ready_at` (the estimate made when it parked) is
                // refreshed only when the worker would otherwise idle — see
                // `earliest_pending_ready`.
                still_pending.push(req);
            }
        }
        self.pending_transfers = still_pending;

        let mut still_staging = Vec::with_capacity(self.pending_storage.len());
        for mut req in self.pending_storage.drain(..) {
            let staged = completed
                .as_ref()
                .is_some_and(|completed| completed.staged.contains(&req.request_id));
            let timed_out = matches!(self.storage_prefetch, StoragePrefetchPolicy::Timeout { .. })
                && req
                    .ready_at
                    .is_some_and(|deadline| deadline <= current_time);
            if timed_out {
                req.storage_prefetch_abandoned = true;
                req.ready_at = None;
                req.num_cached_tokens = 0;
                self.waiting.push_front(req);
            } else if staged {
                self.kv_cache_manager.accept_staging(&req.request_id);
                req.num_cached_tokens = 0;
                self.waiting.push_front(req);
            } else {
                still_staging.push(req);
            }
        }
        self.pending_storage = still_staging;

        // Outlook prefetches use the same staged path, but never become
        // runnable requests: after each store leg, start the next one or the
        // final HBM load.
        let pending_prefetch_storage = std::mem::take(&mut self.pending_prefetch_storage);
        for probe in pending_prefetch_storage {
            let staged = completed
                .as_ref()
                .is_some_and(|completed| completed.staged.contains(&probe.request_id));
            if !staged {
                self.pending_prefetch_storage.push(probe);
                continue;
            }
            self.kv_cache_manager.accept_staging(&probe.request_id);
            let lookup = self.kv_cache_manager.peek_prefix_cache(&probe);
            if !lookup.needs_promotion() {
                self.kv_cache_manager.cancel_staging(&probe.request_id);
                continue;
            }
            let request_id = probe.request_id.clone();
            if !self.start_prefetch_leg(probe, lookup, current_time) {
                self.kv_cache_manager.cancel_staging(&request_id);
            }
        }

        // Landed prefetches: publish and free (hittable, keeping their
        // outlook order); the rest wait on.
        let mut still_flying = Vec::with_capacity(self.prefetches.len());
        for mut req in self.prefetches.drain(..) {
            if let Some(landed_blocks) = completed
                .as_ref()
                .and_then(|completed| completed.hbm.get(&req.request_id))
                .copied()
            {
                let cached_blocks = self
                    .kv_cache_manager
                    .content_blocks_for_tokens(req.num_cached_tokens)
                    .min(landed_blocks as usize);
                req.num_cached_tokens =
                    (cached_blocks as u32).saturating_mul(self.kv_cache_manager.block_size());
                self.kv_cache_manager
                    .publish_transferred_blocks(&mut req, cached_blocks);
                self.kv_cache_manager.free_request(&mut req);
            } else {
                still_flying.push(req);
            }
        }
        self.prefetches = still_flying;
    }

    /// Main scheduling function, called once per iteration.
    pub fn schedule(&mut self, current_time: f64) -> ScheduleDecision {
        // Promote any pending KV transfers whose deadline has passed; their
        // KV is now resident in HBM and they can be scheduled normally.
        self.promote_finished_transfers(current_time);

        let mut decision = ScheduleDecision::default();
        let mut token_budget = self.config.max_num_batched_tokens;
        decision.completed.append(&mut self.rejected);

        // Phase 0: reap finished requests, freeing their blocks BEFORE any
        // allocation this pass. Within a step, retirement must precede
        // growth: a request later in `running` may have finished, and an
        // earlier request's allocation should be able to use those blocks.
        let mut idx = 0;
        while idx < self.running.len() {
            if self.running[idx].is_finished() {
                let mut req = self.running.remove(idx);
                // A session step announces its re-entry to the cache
                // before its blocks go back on the free list, so the
                // outlook policies see the marks as the blocks are freed.
                let outlook = req.outlook_at(current_time);
                self.kv_cache_manager.set_request_outlook(&mut req, outlook);
                self.plan_prefetch(&req, current_time);
                self.kv_cache_manager.free_request(&mut req);
                decision.completed.push(req);
            } else {
                idx += 1;
            }
        }

        // Prefetches due now start before admission competes for the
        // free blocks they land in.
        self.fire_due_prefetches(current_time);

        // Phase 1: schedule RUNNING requests, preempting under KV pressure.
        let mut idx = 0;
        while idx < self.running.len() && token_budget > 0 {
            let tokens_to_schedule = self.tokens_to_schedule(&self.running[idx], token_budget);
            let blocks_needed = self
                .kv_cache_manager
                .blocks_needed(&self.running[idx], tokens_to_schedule);

            // A running request outranks the prefixes waiting requests hold
            // from earlier promotions (their tier copies stay): give those
            // up before preempting anything that has computed state.
            if blocks_needed > 0 && self.kv_cache_manager.num_free_blocks() < blocks_needed {
                self.release_waiting_kv(usize::MAX, blocks_needed);
            }
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
                self.kv_cache_manager
                    .allocate_blocks(&mut self.running[idx], tokens_to_schedule)
                    .expect("free-block check above guarantees allocation");
            }

            decision.batch.push(ScheduledSeq {
                idx,
                num_tokens: tokens_to_schedule,
            });
            token_budget -= tokens_to_schedule;
            idx += 1;
        }

        // HiCache best-effort storage prefetching runs only while the request
        // is genuinely queued. Once an admission slot and token budget exist,
        // stop its unfinished stage and let the normal lookup use whatever
        // closer prefix has already landed.
        if decision.num_preempted == 0
            && token_budget > 0
            && self.running.len() < self.config.max_num_seqs as usize
            && matches!(self.storage_prefetch, StoragePrefetchPolicy::BestEffort {})
            && !self.pending_storage.is_empty()
        {
            let mut request = self.pending_storage.remove(0);
            self.kv_cache_manager.cancel_staging(&request.request_id);
            request.ready_at = None;
            request.num_cached_tokens = 0;
            request.storage_prefetch_abandoned = true;
            self.waiting.push_front(request);
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

                // KV for the first `num_computed_tokens` positions arrived
                // from outside this worker (a disaggregated hand-off): it is
                // resident, needs blocks, and needs no compute. A preempted
                // request never takes this path (preemption zeroes
                // `num_computed_tokens`); a request parked on a tier
                // promotion holds its reserved blocks so it doesn't either.
                if request.num_computed_tokens > 0 && request.kv_blocks.is_empty() {
                    let tokens_to_schedule = self.tokens_to_schedule(request, token_budget);
                    if tokens_to_schedule == 0 {
                        break;
                    }
                    if !self
                        .kv_cache_manager
                        .can_grow_to(request, request.num_computed_tokens + tokens_to_schedule)
                    {
                        break;
                    }
                    let mut request = self.waiting.remove(selected_idx).unwrap();
                    // Blocks for the transferred context plus this step.
                    // Prompt blocks this worker already holds are shared by
                    // reference (they were skipped by the transfer); the
                    // rest are fresh and publish the prompt's hashes here.
                    self.kv_cache_manager
                        .allocate_blocks(&mut request, tokens_to_schedule)
                        .expect("free-block check above guarantees allocation");
                    decision.batch.push(ScheduledSeq {
                        idx: self.running.len(),
                        num_tokens: tokens_to_schedule,
                    });
                    token_budget -= tokens_to_schedule;
                    self.running.push(request);
                    continue;
                }

                let mut lookup = self.kv_cache_manager.peek_prefix_cache(request);
                if request.storage_prefetch_abandoned {
                    self.kv_cache_manager.discard_unstaged_sources(&mut lookup);
                }
                // Fetch or recompute the tier-held part: recomputing
                // shrinks the lookup to the HBM + in-flight prefix (the
                // tier keeps its copy).
                let resident = lookup.hbm_tokens + lookup.in_flight_tokens;
                if self.recompute_instead(request, &lookup, resident) {
                    let recomputed: u32 = lookup.promote_tokens_per_tier.iter().sum();
                    self.kv_cache_manager.record_recompute(recomputed);
                    lookup.total_cached_tokens = resident;
                    lookup
                        .promote_tokens_per_tier
                        .iter_mut()
                        .for_each(|t| *t = 0);
                    lookup
                        .promote_bytes_per_tier
                        .iter_mut()
                        .for_each(|b| *b = 0);
                }
                let cached_tokens = self.usable_cached_tokens(request, lookup.total_cached_tokens);

                // HiCache stages an external-store hit into the next closer
                // store before allocating any destination HBM. Re-run the
                // lookup after the stage lands; a hierarchy with more than
                // three levels repeats this one level at a time.
                if lookup.needs_promotion() && self.kv_cache_manager.needs_staging(&lookup) {
                    let request = self.waiting.remove(selected_idx).unwrap();
                    match self.park_on_storage_stage(request, &lookup, cached_tokens, current_time)
                    {
                        None => continue,
                        Some(request) => {
                            self.waiting.insert(selected_idx, request);
                            // The source changed between lookup and submit.
                            // Re-run lookup rather than falling through with
                            // the stale result.
                            continue;
                        }
                    }
                }

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
                    // Promote only what can then run: a landing that fits
                    // but leaves no room for the rest of the prompt would
                    // sit on the KV while running requests starve (and be
                    // released and re-promoted every time one grows).
                    if !self
                        .kv_cache_manager
                        .can_grow_to(request, request.num_prompt_tokens)
                    {
                        break;
                    }
                    let mut request = self.waiting.remove(selected_idx).unwrap();
                    self.kv_cache_manager.record_prefix_lookup(&lookup);
                    request.num_cached_tokens = cached_tokens;
                    // A request back from an earlier promotion already holds
                    // (and has computed) that prefix; reserve only the
                    // positions beyond it — a sibling on a node-shared tier
                    // may have backed more of its prompt since.
                    let beyond_held = cached_tokens.saturating_sub(request.num_computed_tokens);
                    self.kv_cache_manager
                        .reserve_blocks_for_transfer(&mut request, beyond_held)
                        .expect("blocks_needed already verified against capacity");

                    // Two paths, possibly both: start a transfer for the
                    // spillover portion, and/or join an existing one for
                    // the in-flight portion.
                    if lookup.needs_promotion() {
                        self.kv_cache_manager.start_transfer(
                            request.request_id.clone(),
                            &lookup,
                            &request.prompt_block_hashes,
                            current_time,
                        );
                    } else {
                        self.kv_cache_manager
                            .join_transfer(request.request_id.clone(), &lookup);
                    }
                    // Its ready time is projected on demand (see
                    // `earliest_pending_ready`): the memory graph's drain
                    // event wakes the worker when the transfer completes.
                    request.load_started_at = (self.load_overlap == LoadOverlap::Layerwise
                        && self.promote_fill.stages())
                    .then_some(current_time);
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
                // Cached-prefix blocks held by running requests are shared by
                // reference and cost no free blocks; a promoted request already
                // holds its own. Same rule as the allocation below.
                if !self
                    .kv_cache_manager
                    .can_grow_to(request, cached_tokens + tokens_to_schedule)
                {
                    // Nothing runs and nothing is in flight, so no block will
                    // free itself: the KV is held by other waiting requests'
                    // promoted prefixes. Give those up (tier copies stay)
                    // from the back of the queue until this one fits.
                    let need = self
                        .kv_cache_manager
                        .blocks_for_context(cached_tokens + tokens_to_schedule)
                        .saturating_sub(request.kv_blocks.len() + request.aux_blocks.len());
                    if self.running.is_empty()
                        && self.pending_transfers.is_empty()
                        && self.release_waiting_kv(selected_idx, need)
                    {
                        continue;
                    }
                    break; // Can't fit, stop scheduling new requests
                }

                let mut request = self.waiting.remove(selected_idx).unwrap();
                // A request re-admitted after a tier promotion already holds
                // its landing blocks and recorded its lookup when it parked.
                if request.kv_blocks.is_empty() {
                    self.kv_cache_manager.record_prefix_lookup(&lookup);
                }
                request.num_cached_tokens = cached_tokens;
                request.num_computed_tokens = cached_tokens;
                self.kv_cache_manager
                    .allocate_blocks(&mut request, tokens_to_schedule)
                    .expect("free-block check above guarantees allocation");

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

    /// Release the KV held by waiting requests other than `keep` (their
    /// promoted prefixes), from the back of the queue, until at least
    /// `needed` blocks are free. Those requests recompute or re-promote
    /// later. Returns whether enough was freed.
    fn release_waiting_kv(&mut self, keep: usize, needed: usize) -> bool {
        let mut i = self.waiting.len();
        while self.kv_cache_manager.num_free_blocks() < needed && i > 0 {
            i -= 1;
            if i == keep || self.waiting[i].kv_blocks.is_empty() {
                continue;
            }
            let r = &mut self.waiting[i];
            self.kv_cache_manager.free_request(r);
            r.num_computed_tokens = 0;
            r.num_cached_tokens = 0;
            r.load_overlap_credit = 0.0;
            r.load_started_at = None;
            r.storage_prefetch_abandoned = false;
            self.num_preemptions += 1;
        }
        self.kv_cache_manager.num_free_blocks() >= needed
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
        self.kv_cache_manager.free_request(request);
        request.preempt();
        request.storage_prefetch_abandoned = false;
    }

    /// Add a new request to the waiting queue at the scheduler's current
    /// time. `request.arrival_time` remains the latency-accounting origin.
    pub fn add_request_at(&mut self, mut request: Request, current_time: f64) {
        // A context that can never fit in this worker's KV cache would
        // wait forever (admitted, preempted, re-queued): refuse it now.
        let needed = self
            .kv_cache_manager
            .blocks_for_context(request.planned_positions());
        if needed > self.kv_cache_manager.total_blocks() {
            request.rejected = true;
            self.rejected.push(request);
            return;
        }
        let lookup = self.kv_cache_manager.peek_prefix_cache(&request);
        let resident = lookup.hbm_tokens + lookup.in_flight_tokens;
        let cached = self.usable_cached_tokens(&request, lookup.total_cached_tokens);
        if !self.recompute_instead(&request, &lookup, resident)
            && self.kv_cache_manager.needs_staging(&lookup)
        {
            match self.park_on_storage_stage(request, &lookup, cached, current_time) {
                None => return,
                Some(req) => request = req,
            }
        }
        self.waiting.push_back(request);
    }

    /// Add a request whose delivery time is its declared arrival time.
    pub fn add_request(&mut self, request: Request) {
        let arrival = request.arrival_time;
        self.add_request_at(request, arrival);
    }

    /// Record that running request `idx` computed `num_tokens` positions in
    /// the pass ending at `time`. Returns the output tokens it generated.
    pub fn record_progress(&mut self, idx: usize, num_tokens: u32, time: f64) -> u32 {
        self.running[idx].load_overlap_credit = 0.0;
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
    /// which still has output to generate. Used by a disaggregated prefill
    /// worker to hand requests to the decode pool. The requests keep their
    /// KV blocks here: the transfer reads them, and the engine releases
    /// them with `release_handed_off` when it drains (as a P worker holds
    /// its blocks until the connector is done with them).
    pub fn take_prefill_complete(&mut self) -> Vec<Request> {
        let mut handed = Vec::new();
        let mut keep = Vec::with_capacity(self.running.len());
        for r in self.running.drain(..) {
            if !r.is_prefill() && !r.is_finished() {
                handed.push(r);
            } else {
                keep.push(r);
            }
        }
        self.running = keep;
        handed
    }

    /// A handed-off request's transfer has drained: release the KV it held
    /// on this worker (its hashes stay hittable until recycled).
    pub fn release_handed_off(&mut self, request: &mut Request) {
        self.kv_cache_manager.free_request(request);
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    /// Waiting requests, including requests parked on a KV transfer: they
    /// are still in the system, and the engine's idle check must see them.
    pub fn num_waiting(&self) -> usize {
        self.waiting.len() + self.pending_transfers.len() + self.pending_storage.len()
    }

    /// Prompt tokens still to be prefilled on this worker: every queued or
    /// parked request's remaining prefill plus the unfinished part of every
    /// in-progress prefill. The prefill work ahead of a new arrival, for
    /// routing.
    pub fn queued_prefill_tokens(&self) -> u64 {
        let remaining = |r: &Request| r.prefill_len().saturating_sub(r.num_computed_tokens) as u64;
        self.waiting.iter().map(remaining).sum::<u64>()
            + self.pending_transfers.iter().map(remaining).sum::<u64>()
            + self.pending_storage.iter().map(remaining).sum::<u64>()
            + self
                .running
                .iter()
                .filter(|r| r.is_prefill())
                .map(remaining)
                .sum::<u64>()
    }

    /// Earliest projected ready time among requests parked on a KV
    /// transfer, under the transfers' current rates at `now`.
    pub fn earliest_pending_ready(&self, now: f64) -> Option<f64> {
        self.pending_transfers
            .iter()
            .map(|r| now + self.kv_cache_manager.estimate_remaining_time(&r.request_id))
            .chain(self.pending_storage.iter().map(|r| {
                let transfer = now + self.kv_cache_manager.estimate_remaining_time(&r.request_id);
                r.ready_at
                    .map_or(transfer, |deadline| deadline.min(transfer))
            }))
            .min_by(f64::total_cmp)
    }

    pub fn num_preemptions(&self) -> u64 {
        self.num_preemptions
    }

    pub fn running(&self) -> &[Request] {
        &self.running
    }

    /// Requests parked on an in-flight KV promotion.
    /// Blocks held by requests in each of this scheduler's queues
    /// `(running, waiting, parked, prefetches)` — for stall diagnostics.
    /// Storage-staged requests are deliberately absent: they hold no HBM.
    pub fn held_blocks_by_queue(&self) -> (usize, usize, usize, usize) {
        let n = |v: &[Request]| v.iter().map(|r| r.kv_blocks.len()).sum::<usize>();
        (
            n(&self.running),
            self.waiting.iter().map(|r| r.kv_blocks.len()).sum(),
            n(&self.pending_transfers),
            n(&self.prefetches),
        )
    }

    /// The waiting queue (admission order), excluding parked requests.
    pub fn waiting(&self) -> &VecDeque<Request> {
        &self.waiting
    }

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
    use crate::config::Config;

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
        Scheduler::new(config.scheduler, kv)
    }

    fn create_test_scheduler() -> Scheduler {
        let config = Config::test_default();
        let kv = kv_manager(&config, config.scheduler.kv_cache_capacity, false);
        scheduler_from(config, kv)
    }

    fn create_scheduler_with_policy(policy: &str) -> Scheduler {
        let mut config = Config::test_default();
        config.scheduler.policy = policy.parse().unwrap();
        let kv = kv_manager(&config, config.scheduler.kv_cache_capacity, false);
        scheduler_from(config, kv)
    }

    /// Scheduler whose KV holds exactly `blocks` blocks (test model, no
    /// prefix caching).
    fn scheduler_with_blocks(policy: &str, blocks: u64) -> Scheduler {
        let mut config = Config::test_default();
        config.scheduler.policy = policy.parse().unwrap();
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
    fn schedule_skip_guard_covers_every_local_mutation_source() {
        let mut scheduler = create_test_scheduler();
        assert!(!scheduler.has_local_work_for_schedule(1.0));

        scheduler
            .waiting
            .push_back(create_test_request("waiting", 16, 1));
        assert!(scheduler.has_local_work_for_schedule(1.0));
        scheduler.waiting.clear();

        scheduler
            .running
            .push(create_test_request("running", 16, 1));
        assert!(scheduler.has_local_work_for_schedule(1.0));
        scheduler.running.clear();

        scheduler
            .pending_transfers
            .push(create_test_request("parked", 16, 1));
        assert!(scheduler.has_local_work_for_schedule(1.0));
        scheduler.pending_transfers.clear();

        scheduler
            .pending_storage
            .push(create_test_request("storage", 16, 1));
        assert!(scheduler.has_local_work_for_schedule(1.0));
        scheduler.pending_storage.clear();

        scheduler
            .rejected
            .push(create_test_request("rejected", 16, 1));
        assert!(scheduler.has_local_work_for_schedule(1.0));
        scheduler.rejected.clear();

        scheduler.prefetch_plans.push(PrefetchPlan {
            fire_at: 2.0,
            hashes: vec![1],
            tokens: 16,
            id: "prefetch".into(),
        });
        assert!(!scheduler.has_local_work_for_schedule(1.0));
        assert!(scheduler.has_local_work_for_schedule(2.0));
    }

    #[test]
    fn test_add_request() {
        let mut scheduler = create_test_scheduler();
        scheduler.add_request(create_test_request("req-1", 100, 50));
        assert_eq!(scheduler.num_waiting(), 1);
    }

    /// Scheduler with one private tier at `bw` bytes/s holding `prefix_hash`
    /// (demoted out of a 4-block HBM), under `source`, pricing recompute at
    /// `recompute_seconds_per_token`.
    fn tiered_scheduler(
        bw: f64,
        source: SourcePolicy,
        recompute_per_token: f64,
    ) -> (Scheduler, u64) {
        let config = Config::test_default();
        let block_size = config.scheduler.block_size;
        let per_block = config.model.kv_storage_bytes(block_size);
        let kv = kv_manager(&config, 4 * per_block, true).with_private_tiers(&[(
            "host_ram",
            10 * 1024 * 1024 * 1024,
            bw,
        )]);
        let est: RecomputeFn = Box::new(move |_r: &Request, _from: u32, tokens: u32| {
            tokens as f64 * recompute_per_token
        });
        let mut scheduler = scheduler_from(config, kv).with_source(source, Some(est));
        let prefix_hash = 0xCAFE_u64;
        let mgr = &mut scheduler.kv_cache_manager;
        let mut seed = create_test_request("seed", block_size, 1);
        seed.prompt_block_hashes = vec![prefix_hash];
        mgr.allocate_blocks(&mut seed, block_size).unwrap();
        mgr.free_request(&mut seed);
        let mut churn = create_test_request("churn", block_size * 4, 1);
        churn.prompt_block_hashes = vec![0xDEAD_u64, 0xDEAE, 0xDEAF, 0xDEB0];
        mgr.allocate_blocks(&mut churn, block_size * 4).unwrap();
        mgr.free_request(&mut churn);
        let mut probe = create_test_request("probe", block_size + 1, 1);
        probe.prompt_block_hashes = vec![prefix_hash];
        assert!(mgr.num_tiers() == 1 && mgr.peek_prefix_cache(&probe).needs_promotion());
        (scheduler, prefix_hash)
    }

    /// Two external tiers with one cached block in the slower tier. The
    /// stage takes one second and the final host-to-HBM leg half a second.
    fn hicache_scheduler(policy: StoragePrefetchPolicy) -> (Scheduler, u32, u64) {
        let config = Config::test_default();
        let block_size = config.scheduler.block_size;
        let per_block = config.model.kv_storage_bytes(block_size);
        let kv = kv_manager(&config, 4 * per_block, true).with_private_tiers(&[
            ("host_ram", 16 * per_block, 2.0 * per_block as f64),
            ("nvme", 16 * per_block, per_block as f64),
        ]);
        let scheduler = scheduler_from(config, kv).with_hicache(
            PromoteFill::Through,
            policy,
            LoadOverlap::None,
        );
        let prefix = 0xCAFE_u64;
        scheduler.kv_cache_manager.plant_in_tier_path(1, &[prefix]);
        (scheduler, block_size, prefix)
    }

    /// Three external tiers, so a deepest hit takes two store-to-store legs
    /// before its final HBM load.
    fn deep_hicache_scheduler(policy: StoragePrefetchPolicy) -> (Scheduler, u32, u64) {
        let config = Config::test_default();
        let block_size = config.scheduler.block_size;
        let per_block = config.model.kv_storage_bytes(block_size);
        let kv = kv_manager(&config, 4 * per_block, true).with_private_tiers(&[
            ("host_ram", 16 * per_block, 4.0 * per_block as f64),
            ("local_ssd", 16 * per_block, 2.0 * per_block as f64),
            ("remote_store", 16 * per_block, per_block as f64),
        ]);
        let scheduler = scheduler_from(config, kv).with_hicache(
            PromoteFill::Through,
            policy,
            LoadOverlap::None,
        );
        let prefix = 0xD00D_u64;
        scheduler.kv_cache_manager.plant_in_tier_path(2, &[prefix]);
        (scheduler, block_size, prefix)
    }

    #[test]
    fn hicache_stages_storage_without_reserving_hbm_then_loads_from_host() {
        let (mut scheduler, block_size, prefix) =
            hicache_scheduler(StoragePrefetchPolicy::WaitComplete {});
        let free = scheduler.kv_cache_manager.num_free_blocks();
        let mut req = create_test_request("req", 2 * block_size, 1);
        req.prompt_block_hashes = vec![prefix, 0xBEEF];
        scheduler.add_request(req);

        assert_eq!(scheduler.pending_storage.len(), 1);
        assert!(scheduler.pending_transfers.is_empty());
        assert_eq!(scheduler.kv_cache_manager.num_free_blocks(), free);
        assert_eq!(scheduler.held_blocks_by_queue(), (0, 0, 0, 0));

        let stage_done = scheduler.earliest_pending_ready(0.0).unwrap();
        assert!((stage_done - 1.0).abs() < 1e-9);
        assert!(scheduler.schedule(stage_done / 2.0).batch.is_empty());
        let decision = scheduler.schedule(stage_done + 1e-9);
        assert!(decision.batch.is_empty());
        assert!(scheduler.pending_storage.is_empty());
        assert_eq!(scheduler.pending_transfers.len(), 1);
        assert_eq!(scheduler.kv_cache_manager.num_free_blocks(), free - 1);

        let hbm_done = scheduler.earliest_pending_ready(stage_done + 1e-9).unwrap();
        assert!((hbm_done - 1.5).abs() < 1e-6);
        let decision = scheduler.schedule(hbm_done + 1e-9);
        assert_eq!(decision.batch.len(), 1);
        assert_eq!(decision.batch[0].num_tokens, block_size);
        assert_eq!(scheduler.running[0].num_computed_tokens, block_size);
    }

    #[test]
    fn hicache_best_effort_and_timeout_abandon_deeper_storage() {
        for (policy, stop_at) in [
            (StoragePrefetchPolicy::BestEffort {}, 0.0),
            // Reaping after the storage flow would otherwise have landed
            // still rejects it because the deadline expired first.
            (StoragePrefetchPolicy::Timeout { seconds: 0.25 }, 1.0),
        ] {
            let (mut scheduler, block_size, prefix) = hicache_scheduler(policy);
            let mut req = create_test_request("req", 2 * block_size, 1);
            req.prompt_block_hashes = vec![prefix, 0xBEEF];
            scheduler.add_request(req);
            assert_eq!(scheduler.pending_storage.len(), 1);

            let decision = scheduler.schedule(stop_at);
            assert!(scheduler.pending_storage.is_empty());
            assert!(scheduler.pending_transfers.is_empty());
            assert_eq!(decision.batch.len(), 1);
            assert_eq!(
                decision.batch[0].num_tokens,
                2 * block_size,
                "the incomplete storage suffix is recomputed"
            );
        }
    }

    #[test]
    fn hicache_timeout_deadline_covers_the_whole_staged_load() {
        let (mut scheduler, block_size, prefix) =
            deep_hicache_scheduler(StoragePrefetchPolicy::Timeout { seconds: 1.25 });
        let mut req = create_test_request("req", 2 * block_size, 1);
        req.prompt_block_hashes = vec![prefix, 0xBEEF];
        scheduler.add_request(req);

        scheduler.schedule(1.0 + 1e-9);
        assert_eq!(scheduler.pending_storage.len(), 1, "second leg started");
        assert_eq!(scheduler.pending_storage[0].ready_at, Some(1.25));

        let decision = scheduler.schedule(1.25);
        assert!(scheduler.pending_storage.is_empty());
        assert_eq!(decision.batch.len(), 1);
        assert_eq!(decision.batch[0].num_tokens, 2 * block_size);
    }

    #[test]
    fn hicache_late_delivery_starts_storage_at_delivery_time() {
        let (mut scheduler, block_size, prefix) =
            hicache_scheduler(StoragePrefetchPolicy::WaitComplete {});
        let mut req = create_test_request("late", 2 * block_size, 1);
        req.prompt_block_hashes = vec![prefix, 0xBEEF];
        scheduler.add_request_at(req, 10.0);

        assert_eq!(scheduler.pending_storage.len(), 1);
        assert!((scheduler.earliest_pending_ready(10.0).unwrap() - 11.0).abs() < 1e-9);
    }

    #[test]
    fn outlook_prefetch_uses_the_same_staged_path() {
        let (mut scheduler, block_size, prefix) =
            hicache_scheduler(StoragePrefetchPolicy::WaitComplete {});
        let free = scheduler.kv_cache_manager.num_free_blocks();
        scheduler.prefetch_plans.push(PrefetchPlan {
            fire_at: 0.0,
            hashes: vec![prefix],
            tokens: block_size,
            id: "pf:staged".into(),
        });

        scheduler.schedule(0.0);
        assert_eq!(scheduler.pending_prefetch_storage.len(), 1);
        assert!(scheduler.prefetches.is_empty());
        assert_eq!(scheduler.kv_cache_manager.num_free_blocks(), free);

        scheduler.schedule(1.0 + 1e-9);
        assert!(scheduler.pending_prefetch_storage.is_empty());
        assert_eq!(scheduler.prefetches.len(), 1);
        assert_eq!(scheduler.kv_cache_manager.num_free_blocks(), free - 1);

        scheduler.schedule(1.5 + 2e-9);
        assert!(scheduler.prefetches.is_empty());
        assert!(scheduler.kv_cache_manager.hbm_contains(prefix));
        assert_eq!(scheduler.kv_cache_manager.num_free_blocks(), free);
        assert_eq!(
            scheduler.kv_cache_manager.prefix_cache_stats().prefetches,
            1
        );
    }

    #[test]
    fn outlook_prefetch_lands_a_demoted_prefix_before_its_re_entry() {
        use crate::config::PrefetchPolicy;
        use crate::request::SessionStep;
        let config = Config::test_default();
        let block_size = config.scheduler.block_size;
        let per_block = config.model.kv_storage_bytes(block_size);
        // 4 HBM blocks; a tier at 1 block/s so a 2-block prefix takes 2 s
        // to promote (private tier: write is free, blocks land instantly).
        let bw = per_block as f64;
        let build = |prefetch: PrefetchPolicy| {
            let kv = kv_manager(&config, 4 * per_block, true).with_private_tiers(&[(
                "host_ram",
                10 * 1024 * 1024 * 1024,
                bw,
            )]);
            scheduler_from(config.clone(), kv).with_prefetch(prefetch)
        };
        // A two-block session step whose successor arrives 10 s after it
        // completes and reuses both blocks.
        let step = |id: &str| {
            let mut r = create_test_request(id, block_size * 2, 1);
            r.prompt_block_hashes = vec![0xA1, 0xA2];
            r.session = Some(Box::new(SessionStep {
                session: 0,
                step: 0,
                gap: 0.0,
                shared_tokens: 0,
                kind: None,
                parent_bytes_written: None,
                reuse_distance_bytes: None,
                parent_bytes_touched: None,
                reuse_touched_bytes: None,
                next_gap: Some(10.0),
                next_shared_tokens: block_size * 2,
            }));
            r
        };
        // Run the step to completion at t=0..1: prefill both blocks, one
        // decode token, finished.
        let mut s = build(PrefetchPolicy::Outlook { lead: 0.0 });
        s.add_request(step("a0"));
        let d = s.schedule(0.0);
        assert_eq!(d.batch.len(), 1);
        apply(&mut s, &d, 0.5);
        let d = s.schedule(1.0);
        assert_eq!(d.completed.len(), 1, "a0 done at t=1");
        // Plan: re-entry at 11, transfer 2 s → fire at 9.
        assert_eq!(s.next_prefetch_at(), Some(9.0));
        // Churn pushes both blocks out to the tier (write_back).
        let mut churn = create_test_request("churn", block_size * 4, 1);
        churn.prompt_block_hashes = vec![0xC1, 0xC2, 0xC3, 0xC4];
        s.add_request(churn);
        let d = s.schedule(2.0);
        apply(&mut s, &d, 2.5);
        let d = s.schedule(3.0);
        assert_eq!(d.completed.len(), 1);
        let g = s.kv_cache_manager.memory().unwrap().0.clone();
        assert!(g.lock().unwrap().holds_hash(0, 0xA1) && g.lock().unwrap().holds_hash(0, 0xA2));
        assert!(!s.kv_cache_manager.hbm_contains(0xA1));
        // Before 9 nothing starts; at 9 the prefetch goes out.
        s.schedule(8.0);
        assert_eq!(s.kv_cache_manager.prefix_cache_stats().prefetches, 0);
        s.schedule(9.0);
        assert_eq!(s.kv_cache_manager.prefix_cache_stats().prefetches, 1);
        assert_eq!(
            s.kv_cache_manager.prefix_cache_stats().prefetch_tokens,
            u64::from(block_size) * 2
        );
        assert_eq!(s.prefetches.len(), 1);
        assert_eq!(s.next_prefetch_at(), None);
        // It lands at 11: both blocks back in HBM, free and hittable.
        s.schedule(11.0 + 1e-9);
        assert!(s.prefetches.is_empty());
        assert!(s.kv_cache_manager.hbm_contains(0xA1) && s.kv_cache_manager.hbm_contains(0xA2));
        assert_eq!(s.kv_cache_manager.num_free_blocks(), 4);
        // The re-entry hits HBM: admitted straight in with the prefix cached.
        let mut a1 = create_test_request("a1", block_size * 3, 1);
        a1.prompt_block_hashes = vec![0xA1, 0xA2, 0xA3];
        s.add_request(a1);
        let d = s.schedule(11.5);
        assert_eq!(d.batch.len(), 1);
        assert_eq!(
            d.batch[0].num_tokens, block_size,
            "only the novel block computes"
        );
        assert!(s.pending_transfers.is_empty());

        // Without prefetch the same re-entry parks on a promotion.
        let mut s = build(PrefetchPolicy::None {});
        s.add_request(step("a0"));
        let d = s.schedule(0.0);
        apply(&mut s, &d, 0.5);
        s.schedule(1.0);
        assert_eq!(s.next_prefetch_at(), None);
        let mut churn = create_test_request("churn", block_size * 4, 1);
        churn.prompt_block_hashes = vec![0xC1, 0xC2, 0xC3, 0xC4];
        s.add_request(churn);
        let d = s.schedule(2.0);
        apply(&mut s, &d, 2.5);
        s.schedule(3.0);
        s.schedule(9.0);
        let mut a1 = create_test_request("a1", block_size * 3, 1);
        a1.prompt_block_hashes = vec![0xA1, 0xA2, 0xA3];
        s.add_request(a1);
        let d = s.schedule(11.5);
        assert!(d.batch.is_empty());
        assert_eq!(s.pending_transfers.len(), 1);
    }

    #[test]
    fn a_context_larger_than_the_kv_cache_is_rejected_not_queued() {
        let config = Config::test_default();
        let block_size = config.scheduler.block_size;
        let mut s = scheduler_with_blocks("fcfs", 4);
        // 4 blocks: a prompt of 3 blocks + 1 output block fits exactly...
        s.add_request(create_test_request("fits", block_size * 3, block_size));
        assert_eq!(s.num_waiting(), 1);
        // ...one more output token would need a fifth block: refused.
        s.add_request(create_test_request(
            "too_big",
            block_size * 3,
            block_size + 2,
        ));
        assert_eq!(s.num_waiting(), 1);
        let d = s.schedule(0.0);
        assert_eq!(d.completed.len(), 1);
        assert!(d.completed[0].rejected);
        assert_eq!(d.completed[0].request_id, "too_big");
        assert_eq!(d.completed[0].num_output_tokens, 0);
        assert_eq!(d.batch.len(), 1, "the fitting request runs");
        assert!(s.schedule(1.0).completed.iter().all(|r| !r.rejected));
    }

    #[test]
    fn a_second_promotion_reserves_only_beyond_the_prefix_already_held() {
        // HBM of 4 blocks, one private tier. A request whose prompt is
        // [A, B, C] finds A in the tier, promotes it, and by the time it is
        // re-admitted B has been backed too (a sibling wrote it): the second
        // reservation must cover B only, not A again — with one other block
        // held, A's block plus B's landing block leave exactly one free.
        let config = Config::test_default();
        let bs = config.scheduler.block_size;
        let per_block = config.model.kv_storage_bytes(bs);
        let kv = kv_manager(&config, 4 * per_block, true).with_private_tiers(&[(
            "host_ram",
            10 * 1024 * 1024 * 1024,
            per_block as f64, // one block per second
        )]);
        let mut s = scheduler_from(config, kv);
        let (a, b, c) = (0xA_u64, 0xB_u64, 0xC_u64);
        {
            let mgr = &mut s.kv_cache_manager;
            let mut seed = create_test_request("seed", bs, 1);
            seed.prompt_block_hashes = vec![a];
            mgr.allocate_blocks(&mut seed, bs).unwrap();
            mgr.free_request(&mut seed);
            let mut churn = create_test_request("churn", bs * 4, 1);
            churn.prompt_block_hashes = vec![0xD1, 0xD2, 0xD3, 0xD4];
            mgr.allocate_blocks(&mut churn, bs * 4).unwrap();
            mgr.free_request(&mut churn);
            assert!(mgr.memory().unwrap().0.lock().unwrap().holds_hash(0, a));
        }
        // A one-block request that keeps decoding holds a block throughout.
        s.add_request(create_test_request("hold", 1, bs - 1));
        let mut req = create_test_request("req", bs * 3, 1);
        req.prompt_block_hashes = vec![a, b, c];
        s.add_request(req);
        let d = s.schedule(0.0);
        assert_eq!(d.batch.len(), 1, "hold runs; req parks on A's promotion");
        assert_eq!(s.pending_transfers.len(), 1);
        assert_eq!(s.pending_transfers[0].num_cached_tokens, bs);
        // B lands in the tier while A is in flight.
        s.kv_cache_manager.plant_in_tier_path(0, &[a, b]);
        // A lands: req is re-admitted, sees B in the tier, and parks again
        // — reserving one block for B (free: 4 − hold − A − B = 1).
        let ready = s.earliest_pending_ready(0.0).unwrap();
        apply(&mut s, &d, ready / 2.0);
        let d = s.schedule(ready + 1e-9);
        assert_eq!(s.pending_transfers.len(), 1);
        let r = &s.pending_transfers[0];
        assert_eq!(r.request_id, "req");
        assert_eq!(r.num_cached_tokens, 2 * bs);
        assert_eq!(r.kv_blocks.len(), 2, "A's block plus B's landing block");
        assert_eq!(s.kv_cache_manager.num_free_blocks(), 1);
        assert_eq!(d.batch.len(), 1);
    }

    #[test]
    fn a_promotion_starts_only_when_the_whole_prompt_would_fit() {
        // HBM of 3 blocks; two 3-block requests each find their first block
        // in the tier. Promoting both would park two landings that can
        // never both run; the second waits (no landing block held) until
        // the first has run.
        let config = Config::test_default();
        let bs = config.scheduler.block_size;
        let per_block = config.model.kv_storage_bytes(bs);
        let kv = kv_manager(&config, 3 * per_block, true).with_private_tiers(&[(
            "host_ram",
            10 * 1024 * 1024 * 1024,
            per_block as f64,
        )]);
        let mut s = scheduler_from(config, kv);
        s.kv_cache_manager.plant_in_tier(0, 0xA1);
        s.kv_cache_manager.plant_in_tier(0, 0xA2);
        let mut x1 = create_test_request("x1", bs * 3, 1);
        x1.prompt_block_hashes = vec![0xA1, 0xB1, 0xC1];
        let mut x2 = create_test_request("x2", bs * 3, 1);
        x2.prompt_block_hashes = vec![0xA2, 0xB2, 0xC2];
        s.add_request(x1);
        s.add_request(x2);
        let d = s.schedule(0.0);
        assert!(d.batch.is_empty());
        assert_eq!(
            s.pending_transfers.len(),
            1,
            "x1 promotes; x2 would not fit"
        );
        assert_eq!(s.pending_transfers[0].request_id, "x1");
        assert_eq!(s.waiting().len(), 1);
        assert!(s.waiting()[0].kv_blocks.is_empty());
        assert_eq!(s.kv_cache_manager.num_free_blocks(), 2);
        let ready = s.earliest_pending_ready(0.0).unwrap();
        let d = s.schedule(ready + 1e-9);
        assert_eq!(d.batch.len(), 1);
        assert_eq!(running_ids(&s), vec!["x1"]);
        assert_eq!(s.num_preemptions(), 0);
        // x1 finishes; x2 promotes its first block and runs in turn.
        apply(&mut s, &d, ready + 1.0);
        let d = s.schedule(ready + 1.5);
        assert_eq!(d.completed.len(), 1);
        assert_eq!(s.pending_transfers.len(), 1);
        assert_eq!(s.pending_transfers[0].request_id, "x2");
    }

    #[test]
    fn a_growing_request_takes_back_kv_held_by_waiting_promotions() {
        // HBM of 4 blocks. `run` (1-block prompt, long output) is running.
        // `held` (prompt [A, B, C], A in the tier) promotes A while the
        // whole prompt still fits, then `run` grows into the free blocks;
        // when `held` lands it cannot get B and C and waits holding A's
        // block. `run` needs one more block with none free: it takes back
        // `held`'s landing block (the tier copy stays) and is not preempted.
        let config = Config::test_default();
        let bs = config.scheduler.block_size;
        let per_block = config.model.kv_storage_bytes(bs);
        let kv = kv_manager(&config, 4 * per_block, true).with_private_tiers(&[(
            "host_ram",
            10 * 1024 * 1024 * 1024,
            per_block as f64,
        )]);
        let mut s = scheduler_from(config, kv);
        s.kv_cache_manager.plant_in_tier(0, 0xA);
        s.add_request(create_test_request("run", 1, 4 * bs));
        let d = s.schedule(0.0);
        assert_eq!(running_ids(&s), vec!["run"]);
        apply(&mut s, &d, 0.5);
        let mut held = create_test_request("held", bs * 3, 1);
        held.prompt_block_hashes = vec![0xA, 0xB, 0xC];
        s.add_request(held);
        let d = s.schedule(1.0);
        assert_eq!(
            s.pending_transfers.len(),
            1,
            "3 free ≥ 3-block prompt: promote"
        );
        assert_eq!(s.kv_cache_manager.num_free_blocks(), 2);
        // `run` decodes across two block boundaries before A lands.
        let ready = s.earliest_pending_ready(1.0).unwrap();
        let mut t = 1.0;
        let mut d = d;
        while s.kv_cache_manager.num_free_blocks() > 0 {
            apply(&mut s, &d, t);
            t += 1e-3;
            d = s.schedule(t);
            assert!(t < ready, "run fills the free blocks before A lands");
        }
        assert_eq!(s.pending_transfers.len(), 1);
        // A lands: `held` cannot take B and C; it waits holding A's block.
        apply(&mut s, &d, ready);
        let mut d = s.schedule(ready + 1e-9);
        assert_eq!(s.waiting().len(), 1);
        assert_eq!(s.waiting()[0].kv_blocks.len(), 1);
        assert_eq!(s.kv_cache_manager.num_free_blocks(), 0);
        // `run` keeps decoding until it needs another block: it takes it
        // from `held` rather than preempting itself.
        let mut t = ready + 1e-9;
        for _ in 0..(bs + 1) {
            apply(&mut s, &d, t);
            t += 1e-3;
            d = s.schedule(t);
            assert_eq!(running_ids(&s), vec!["run"], "run is never preempted");
        }
        assert!(
            s.waiting()[0].kv_blocks.is_empty(),
            "held gave up its landing block"
        );
        assert_eq!(s.waiting()[0].num_computed_tokens, 0);
        assert!(s.running()[0].kv_blocks.len() >= 4);
    }

    #[test]
    fn min_time_recomputes_a_slow_fetch_and_promotes_a_fast_one() {
        let config = Config::test_default();
        let block_size = config.scheduler.block_size;
        let per_block = config.model.kv_storage_bytes(block_size) as f64;
        // Recompute at 1 ms/token: a block takes block_size ms.
        let recompute = 1e-3;
        // Slow tier: the block's bytes take 10× longer than recomputing.
        let slow = per_block / (10.0 * recompute * block_size as f64);
        let (mut s, h) = tiered_scheduler(slow, SourcePolicy::MinTime {}, recompute);
        let mut req = create_test_request("req", block_size * 2, 1);
        req.prompt_block_hashes = vec![h, 0xBEEF_u64];
        s.add_request(req);
        let d = s.schedule(0.0);
        assert_eq!(d.batch.len(), 1, "admitted straight into the batch");
        assert!(
            s.pending_transfers.is_empty(),
            "nothing parked on a transfer"
        );
        assert_eq!(
            d.batch[0].num_tokens,
            block_size * 2,
            "whole prompt recomputed"
        );
        let st = s.kv_cache_manager.prefix_cache_stats();
        assert_eq!(
            (st.recomputed, st.recomputed_tokens),
            (1, block_size as u64)
        );
        assert!(s.kv_cache_manager.num_tiers() == 1);
        // The tier keeps its copy.
        assert!(s
            .kv_cache_manager
            .memory()
            .unwrap()
            .0
            .lock()
            .unwrap()
            .holds_hash(0, h));

        // Fast tier: 10× faster than recomputing → promote as before.
        let fast = per_block * 10.0 / (recompute * block_size as f64);
        let (mut s, h) = tiered_scheduler(fast, SourcePolicy::MinTime {}, recompute);
        let mut req = create_test_request("req", block_size * 2, 1);
        req.prompt_block_hashes = vec![h, 0xBEEF_u64];
        s.add_request(req);
        let d = s.schedule(0.0);
        assert!(d.batch.is_empty());
        assert_eq!(s.pending_transfers.len(), 1);
        assert_eq!(s.kv_cache_manager.prefix_cache_stats().recomputed, 0);

        // `promote` ignores the comparison: even the slow tier is fetched.
        let (mut s, h) = tiered_scheduler(slow, SourcePolicy::Promote {}, recompute);
        let mut req = create_test_request("req", block_size * 2, 1);
        req.prompt_block_hashes = vec![h, 0xBEEF_u64];
        s.add_request(req);
        let d = s.schedule(0.0);
        assert!(d.batch.is_empty());
        assert_eq!(s.pending_transfers.len(), 1);
    }

    #[test]
    fn test_waiting_on_transfer_then_promoted() {
        let config = Config::test_default();
        let block_size = config.scheduler.block_size;
        let per_block = config.model.kv_storage_bytes(block_size);
        // Four HBM blocks, so a churn allocation can push the seeded prefix
        // out to host RAM (recycling is LRU: only a full HBM evicts).
        let kv = kv_manager(&config, 4 * per_block, true).with_private_tiers(&[(
            "host_ram",
            10 * 1024 * 1024 * 1024,
            1e9,
        )]);
        let mut scheduler = scheduler_from(config, kv);

        // Seed the host-RAM tier: allocate then free a block carrying our
        // prefix hash, then fill HBM with other content so `prefix_hash` is
        // demoted into host RAM. Free the churn again so HBM has room.
        let prefix_hash = 0xCAFE_u64;
        let mgr = &mut scheduler.kv_cache_manager;
        let mut seed = create_test_request("seed", block_size, 1);
        seed.prompt_block_hashes = vec![prefix_hash];
        mgr.allocate_blocks(&mut seed, block_size).unwrap();
        mgr.free_request(&mut seed);
        let mut churn = create_test_request("churn", block_size * 4, 1);
        churn.prompt_block_hashes = vec![0xDEAD_u64, 0xDEAE, 0xDEAF, 0xDEB0];
        mgr.allocate_blocks(&mut churn, block_size * 4).unwrap();
        mgr.free_request(&mut churn);

        // A request whose prompt starts with the prefix hash.
        let mut req = create_test_request("req", block_size * 2, 1);
        req.prompt_block_hashes = vec![prefix_hash, 0xBEEF_u64];
        scheduler.add_request(req);

        // t=0: host-RAM hit; parked in pending_transfers, not running.
        let decision = scheduler.schedule(0.0);
        assert!(decision.batch.is_empty());
        assert_eq!(scheduler.num_waiting(), 1); // parked requests count as waiting
        assert_eq!(scheduler.pending_transfers.len(), 1);
        let ready_at = scheduler.earliest_pending_ready(0.0).unwrap();
        assert!(ready_at > 0.0);

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
        let kv = kv_manager(&config, 4 * per_block, true).with_private_tiers(&[(
            "host_ram",
            16 * per_block,
            1e9,
        )]);
        let mut scheduler = scheduler_from(config, kv);

        // Pre-warm the prefix into host RAM.
        let prefix_hash = 0xABCDu64;
        {
            let mgr = &mut scheduler.kv_cache_manager;
            let mut seed = create_test_request("seed", block_size, 1);
            seed.prompt_block_hashes = vec![prefix_hash];
            mgr.allocate_blocks(&mut seed, block_size).unwrap();
            mgr.free_request(&mut seed);
            // Fill all four HBM blocks so the seed is evicted (LRU recycles
            // the oldest free block, so only a full HBM demotes).
            let mut churn = create_test_request("churn", block_size * 4, 1);
            churn.prompt_block_hashes = vec![0xDEAD, 0xBEEF, 0xF00D, 0xFACE];
            mgr.allocate_blocks(&mut churn, block_size * 4).unwrap();
            mgr.free_request(&mut churn);
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
        // One landing block shared by reference: 4 − 1 = 3 free.
        for r in scheduler.pending_transfers.iter() {
            assert_eq!(r.kv_blocks.len(), 1);
        }
        assert_eq!(scheduler.kv_cache_manager().num_free_blocks(), 3);

        let _ = scheduler.schedule(10.0);
        assert_eq!(scheduler.pending_transfers.len(), 0);
        assert_eq!(scheduler.num_running(), 3);
        // Each request holds the shared prefix block plus its own second
        // block: 1 shared + 3 private = all four.
        assert!(scheduler.running().iter().all(|r| r.kv_blocks.len() == 2));
        assert_eq!(scheduler.kv_cache_manager().num_free_blocks(), 0);
    }

    #[test]
    fn test_hbm_prefix_hit_skips_cached_compute() {
        let config = Config::test_default();
        let bs = config.scheduler.block_size;
        let kv = kv_manager(&config, config.scheduler.kv_cache_capacity, true);
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
    fn test_admission_shares_held_prefix_blocks() {
        // KV holds exactly 6 blocks. `a` (4 prompt blocks + 1 for its first
        // decode step) is running; `b` shares a's first three blocks and
        // needs one fresh block. vLLM admits it: hits on blocks a holds cost
        // nothing, and one block is free.
        let config = Config::test_default();
        let bs = config.scheduler.block_size;
        let per_block = config.model.kv_storage_bytes(bs);
        let kv = kv_manager(&config, 6 * per_block, true);
        let mut scheduler = scheduler_from(config, kv);

        let mut a = create_test_request("a", 4 * bs, 10);
        a.prompt_block_hashes = vec![1, 2, 3, 4];
        scheduler.add_request(a);
        let d = scheduler.schedule(0.0);
        apply(&mut scheduler, &d, 1.0);
        assert_eq!(scheduler.kv_cache_manager().num_free_blocks(), 2);

        let mut b = create_test_request("b", 4 * bs, 10);
        b.prompt_block_hashes = vec![1, 2, 3, 9];
        scheduler.add_request(b);
        let d = scheduler.schedule(1.0);
        assert_eq!(running_ids(&scheduler), vec!["a", "b"]);
        let b_entry = d.batch.iter().find(|s| s.idx == 1).unwrap();
        assert_eq!(b_entry.num_tokens, bs);
        assert_eq!(scheduler.kv_cache_manager().num_free_blocks(), 0);
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
        scheduler.add_request(create_test_request("a", 32, 10));
        scheduler.add_request(create_test_request("b", 32, 10));
        scheduler.add_request(create_test_request("late", 16, 10));
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
        let kv = kv_manager(&config, config.scheduler.kv_cache_capacity, false);
        scheduler_from(config, kv)
    }

    #[test]
    fn test_preemption_free_admission_control() {
        let mut config = Config::test_default();
        config.scheduler.enable_preemption_free = true;
        config.scheduler.kv_cache_capacity = 100_000_000; // Small cache
        let kv = kv_manager(&config, config.scheduler.kv_cache_capacity, false);
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
    fn test_take_prefill_complete_hands_off_and_holds_kv_until_released() {
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
        // The blocks stay allocated (the transfer reads them) until the
        // engine releases them.
        assert!(!handed[0].kv_blocks.is_empty());
        assert_eq!(scheduler.kv_cache_manager().num_free_blocks(), free_before);
        assert_eq!(running_ids(&scheduler), vec!["mid", "one"]);
        let mut done = handed.into_iter().next().unwrap();
        scheduler.release_handed_off(&mut done);
        assert!(done.kv_blocks.is_empty());
        assert!(scheduler.kv_cache_manager().num_free_blocks() > free_before);
    }
}
