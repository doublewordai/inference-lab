//! Unified discrete-event engine. Pure state machine: knows about pools,
//! workers, the event heap, KV bookkeeping and time. Knows nothing about
//! request generation, metrics, real wall-clock, or how to render progress.
//! Drivers (batch [`super::sim::Simulator`], serve [`crate::serve::engine`])
//! pump it by alternating `next_event_time` / `submit` / `step`.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::compute::ComputeEngine;
use crate::config::{ClusterSpec, DisaggTopology, ModelConfig, ModelCosts, SchedulerConfig};
use crate::kv_cache::{KVCacheManager, Link};
use crate::request::Request;
use crate::scheduler::Scheduler;

pub type PoolId = usize;
pub type LinkId = usize;

/// Per-request timing breakdown produced when a request completes.
#[derive(Debug, Clone)]
pub struct RequestTiming {
    pub request_id: String,
    pub arrival_time: f64,
    /// Time the request's prefill phase finished on the prefill worker. For
    /// aggregated topologies, falls back to `first_token_time`.
    pub prefill_done_time: f64,
    /// Time the KV hand-off transfer completed and the request entered the
    /// decode worker. Equal to `prefill_done_time` for aggregated mode.
    pub handoff_done_time: f64,
    /// Time the first output token was produced (= TTFT relative to arrival).
    pub first_token_time: f64,
    pub completion_time: f64,
    pub num_prompt_tokens: u32,
    pub num_output_tokens: u32,
}

impl RequestTiming {
    pub fn ttft(&self) -> f64 {
        self.first_token_time - self.arrival_time
    }
    pub fn e2e(&self) -> f64 {
        self.completion_time - self.arrival_time
    }
    /// Mean time per output token, computed across the decode phase only
    /// (token 1 onwards).
    pub fn tpot(&self) -> Option<f64> {
        if self.num_output_tokens <= 1 {
            return None;
        }
        let decode_span = self.completion_time - self.first_token_time;
        Some(decode_span / (self.num_output_tokens - 1) as f64)
    }
}

/// One worker = one scheduler + one compute engine + (transitively) one KV
/// cache manager.
pub(crate) struct Worker {
    pub scheduler: Scheduler,
    pub compute_engine: ComputeEngine,
}

impl Worker {
    pub fn new(
        cluster: &ClusterSpec,
        model: ModelConfig,
        scheduler_config: SchedulerConfig,
    ) -> Result<Self, String> {
        let mut cluster = cluster.clone();
        let model_size_bytes = model.weight_residency_bytes();
        cluster.compute_kv_cache_capacity(model_size_bytes);

        let kv_cache_manager = KVCacheManager::new(
            cluster.hardware.kv_cache_capacity,
            scheduler_config.block_size,
            model.kv_storage_bytes(1),
            true,
        )
        .with_tiers(&cluster.hardware.kv_tiers);

        let scheduler = Scheduler::new(
            scheduler_config.clone(),
            cluster.hardware.clone(),
            model.clone(),
            kv_cache_manager,
        )?;
        let mut compute_engine =
            ComputeEngine::new(cluster.hardware.clone(), cluster.parallel.clone(), model)
                .with_cascade_attention(
                    scheduler_config.enable_cascade_attention,
                    scheduler_config.block_size,
                );
        if let Some(comms) = cluster.comms {
            compute_engine = compute_engine.with_comms(comms);
        }
        Ok(Self {
            scheduler,
            compute_engine,
        })
    }
}

pub(crate) struct WorkerPool {
    pub workers: Vec<Worker>,
    next_worker: usize,
}

impl WorkerPool {
    pub fn new(workers: Vec<Worker>) -> Self {
        Self {
            workers,
            next_worker: 0,
        }
    }

    fn pick_round_robin(&mut self) -> usize {
        let n = self.workers.len().max(1);
        let idx = self.next_worker % n;
        self.next_worker = (idx + 1) % n;
        idx
    }
}

#[derive(Debug, Clone, Copy)]
enum PoolRole {
    Aggregated,
    DisaggPrefill,
    DisaggDecode,
}

pub(crate) enum Roles {
    Aggregated {
        pool: PoolId,
    },
    Disagg {
        prefill: PoolId,
        decode: PoolId,
        handoff: LinkId,
    },
}

pub struct Topology {
    pub(crate) pools: Vec<WorkerPool>,
    pub(crate) links: Vec<Link>,
    pub(crate) roles: Roles,
    model: ModelConfig,
}

impl Topology {
    pub fn aggregated(
        cluster: ClusterSpec,
        model: ModelConfig,
        scheduler_config: SchedulerConfig,
    ) -> Result<Self, String> {
        let n = cluster.num_workers.max(1) as usize;
        let mut workers = Vec::with_capacity(n);
        for _ in 0..n {
            workers.push(Worker::new(&cluster, model.clone(), scheduler_config.clone())?);
        }
        Ok(Self {
            pools: vec![WorkerPool::new(workers)],
            links: vec![],
            roles: Roles::Aggregated { pool: 0 },
            model,
        })
    }

    pub fn from_disagg(
        topology: &DisaggTopology,
        model: ModelConfig,
        scheduler_config: SchedulerConfig,
    ) -> Result<Self, String> {
        let p_count = topology.prefill.num_workers.max(1) as usize;
        let d_count = topology.decode.num_workers.max(1) as usize;
        let mut p_workers = Vec::with_capacity(p_count);
        for _ in 0..p_count {
            p_workers.push(Worker::new(
                &topology.prefill,
                model.clone(),
                scheduler_config.clone(),
            )?);
        }
        let mut d_workers = Vec::with_capacity(d_count);
        for _ in 0..d_count {
            d_workers.push(Worker::new(
                &topology.decode,
                model.clone(),
                scheduler_config.clone(),
            )?);
        }
        Ok(Self {
            pools: vec![WorkerPool::new(p_workers), WorkerPool::new(d_workers)],
            links: vec![Link::new(topology.kv_link_bw)],
            roles: Roles::Disagg {
                prefill: 0,
                decode: 1,
                handoff: 0,
            },
            model,
        })
    }

    fn entry_pool(&self) -> PoolId {
        match self.roles {
            Roles::Aggregated { pool } => pool,
            Roles::Disagg { prefill, .. } => prefill,
        }
    }

    fn role_for_pool(&self, pool: PoolId) -> PoolRole {
        match self.roles {
            Roles::Aggregated { .. } => PoolRole::Aggregated,
            Roles::Disagg {
                prefill, decode, ..
            } => {
                if pool == prefill {
                    PoolRole::DisaggPrefill
                } else if pool == decode {
                    PoolRole::DisaggDecode
                } else {
                    PoolRole::Aggregated
                }
            }
        }
    }
}

#[derive(Debug)]
enum EventKind {
    Arrival(Request),
    WorkerReady {
        pool: PoolId,
        worker: usize,
    },
    LinkComplete {
        link: LinkId,
        request_id: String,
        then_pool: PoolId,
    },
}

#[derive(Debug)]
struct TimedEvent {
    time: f64,
    seq: u64,
    kind: EventKind,
}

impl PartialEq for TimedEvent {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time && self.seq == other.seq
    }
}
impl Eq for TimedEvent {}
impl PartialOrd for TimedEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TimedEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is max-heap; reverse so smaller time pops first.
        other
            .time
            .partial_cmp(&self.time)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

struct Bookkeeping {
    arrival_time: f64,
    num_prompt_tokens: u32,
    prefill_done_time: Option<f64>,
    handoff_done_time: Option<f64>,
}

/// Per-request progress yielded by a worker iteration. `num_tokens` is the
/// tokens generated *during this iter* for this request (one for decode, the
/// prefill chunk size for prefill).
#[derive(Debug, Clone)]
pub struct RequestProgress {
    pub request_id: String,
    /// Whether the request was in prefill phase BEFORE this iteration ran.
    pub was_prefill: bool,
    pub num_tokens: u32,
}

/// Information about the iteration that ran during a `step` call. Present
/// when the popped event was a `WorkerReady` that resulted in scheduled work.
#[derive(Debug, Clone)]
pub struct IterationInfo {
    pub pool: PoolId,
    pub worker: usize,
    pub start_time: f64,
    pub end_time: f64,
    pub iteration_time: f64,
    pub batch_size: usize,
    pub bandwidth_util: f64,
    pub flops_util: f64,
    pub progress: Vec<RequestProgress>,
}

/// What kind of event drove this step. Useful for drivers that want to filter
/// (e.g. throttle progress callbacks to iteration ends only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepKind {
    Arrival,
    Iteration,
    LinkComplete,
}

#[derive(Debug)]
pub struct StepOutcome {
    pub time: f64,
    pub kind: StepKind,
    pub iteration: Option<IterationInfo>,
    pub completions: Vec<RequestTiming>,
}

pub struct Engine {
    topology: Topology,
    events: BinaryHeap<TimedEvent>,
    bookkeeping: HashMap<String, Bookkeeping>,
    /// Requests that finished prefill on the P pool and are mid-handoff over
    /// the link.
    parked: HashMap<String, Request>,
    /// `worker_busy[pool][worker]` is true iff a `WorkerReady` for that
    /// worker is currently scheduled in the queue.
    worker_busy: Vec<Vec<bool>>,
    /// Per-pool (∑ batch·dt, ∑ dt) for time-weighted mean batch size.
    pool_batch_acc: Vec<(f64, f64)>,
    current_time: f64,
    seq_counter: u64,
}

impl Engine {
    pub fn new(topology: Topology) -> Self {
        let mut worker_busy = Vec::with_capacity(topology.pools.len());
        for pool in &topology.pools {
            worker_busy.push(vec![false; pool.workers.len()]);
        }
        let pool_batch_acc = vec![(0.0_f64, 0.0_f64); topology.pools.len()];
        Self {
            topology,
            events: BinaryHeap::new(),
            bookkeeping: HashMap::new(),
            parked: HashMap::new(),
            worker_busy,
            pool_batch_acc,
            current_time: 0.0,
            seq_counter: 0,
        }
    }

    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Push the clock forward to `at_time` if it's strictly larger. Useful for
    /// real-time drivers that want sim-time to track wall-time when there is
    /// no event to advance through.
    pub fn advance_to(&mut self, at_time: f64) {
        if at_time > self.current_time {
            self.current_time = at_time;
        }
    }

    pub fn next_event_time(&self) -> Option<f64> {
        self.events.peek().map(|e| e.time)
    }

    /// Submit a request for arrival at `req.arrival_time`. The request is
    /// enqueued as an `Arrival` event; it does not enter any worker until a
    /// `step()` call processes that event. If `arrival_time` is in the past
    /// (relative to the engine clock) the event fires immediately at `now`,
    /// but bookkeeping retains the original `arrival_time` so TTFT is still
    /// computed against the request's intended emission moment.
    pub fn submit(&mut self, req: Request) {
        let when = req.arrival_time.max(self.current_time);
        self.push(when, EventKind::Arrival(req));
    }

    fn push(&mut self, time: f64, kind: EventKind) {
        self.seq_counter += 1;
        self.events.push(TimedEvent {
            time,
            seq: self.seq_counter,
            kind,
        });
    }

    /// Time-weighted mean batch size per pool, across the wall-clock the
    /// engine has run so far. `None` for pools that never ran an iteration.
    pub fn pool_batch_means(&self) -> Vec<Option<f64>> {
        self.pool_batch_acc
            .iter()
            .map(|&(num, den)| if den > 0.0 { Some(num / den) } else { None })
            .collect()
    }

    /// Sum of `running` across all pools (for progress reporting). This
    /// counts requests currently held by any worker, including those mid-iter.
    pub fn aggregate_running(&self) -> usize {
        self.topology
            .pools
            .iter()
            .flat_map(|p| &p.workers)
            .map(|w| w.scheduler.num_running())
            .sum::<usize>()
            + self.parked.len()
    }

    /// Sum of `waiting` across all pools.
    pub fn aggregate_waiting(&self) -> usize {
        self.topology
            .pools
            .iter()
            .flat_map(|p| &p.workers)
            .map(|w| w.scheduler.num_waiting())
            .sum()
    }

    /// Number of running requests currently in the prefill phase, summed
    /// across every worker. Used by drivers that report prefill/decode
    /// breakdown in their progress streams.
    pub fn aggregate_prefilling(&self) -> usize {
        self.topology
            .pools
            .iter()
            .flat_map(|p| &p.workers)
            .map(|w| w.scheduler.running().iter().filter(|r| r.is_prefill()).count())
            .sum()
    }

    /// Aggregate prefix-cache stats across every worker's KV manager.
    /// Returned as `(hits, misses, hit_size_sum, hit_size_count)`.
    pub fn aggregate_prefix_cache(&self) -> (u64, u64, u64, u64) {
        let mut hits = 0u64;
        let mut misses = 0u64;
        let mut size_sum = 0u64;
        let mut size_count = 0u64;
        for pool in &self.topology.pools {
            for worker in &pool.workers {
                let mgr = worker.scheduler.kv_cache_manager();
                hits += mgr.num_prefix_cache_hits;
                misses += mgr.num_prefix_cache_misses;
                size_sum += mgr.hit_size_sum;
                size_count += mgr.hit_size_count;
            }
        }
        (hits, misses, size_sum, size_count)
    }

    /// Aggregate KV cache utilisation across pools, weighted by capacity.
    /// Returns 0.0 if no KV cache is configured anywhere.
    pub fn kv_cache_util(&self) -> f64 {
        let mut used = 0.0_f64;
        let mut total = 0.0_f64;
        for pool in &self.topology.pools {
            for worker in &pool.workers {
                let mgr = worker.scheduler.kv_cache_manager();
                let u = mgr.utilization();
                let cap = mgr.total_blocks() as f64;
                used += u * cap;
                total += cap;
            }
        }
        if total > 0.0 {
            used / total
        } else {
            0.0
        }
    }

    /// Whether any work is in flight. `false` when the heap is empty AND
    /// every pool has no running/waiting requests AND no parked handoffs.
    pub fn is_idle(&self) -> bool {
        self.events.is_empty()
            && self.parked.is_empty()
            && self.aggregate_running() == 0
            && self.aggregate_waiting() == 0
    }

    /// Pop the next event and process it. Returns information about what
    /// happened, including any completed requests.
    pub fn step(&mut self) -> Result<StepOutcome, String> {
        let ev = self
            .events
            .pop()
            .ok_or_else(|| "step called with empty event queue".to_string())?;
        if ev.time + 1e-9 < self.current_time {
            return Err(format!(
                "event at t={} earlier than clock t={}",
                ev.time, self.current_time
            ));
        }
        self.current_time = ev.time;

        match ev.kind {
            EventKind::Arrival(req) => {
                self.handle_arrival(req);
                Ok(StepOutcome {
                    time: self.current_time,
                    kind: StepKind::Arrival,
                    iteration: None,
                    completions: Vec::new(),
                })
            }
            EventKind::WorkerReady { pool, worker } => {
                let (iteration, completions) = self.handle_worker_ready(pool, worker);
                Ok(StepOutcome {
                    time: self.current_time,
                    kind: StepKind::Iteration,
                    iteration,
                    completions,
                })
            }
            EventKind::LinkComplete {
                link,
                request_id,
                then_pool,
            } => {
                self.handle_link_complete(link, request_id, then_pool)?;
                Ok(StepOutcome {
                    time: self.current_time,
                    kind: StepKind::LinkComplete,
                    iteration: None,
                    completions: Vec::new(),
                })
            }
        }
    }

    fn handle_arrival(&mut self, req: Request) {
        self.bookkeeping.insert(
            req.request_id.clone(),
            Bookkeeping {
                arrival_time: req.arrival_time,
                num_prompt_tokens: req.num_prompt_tokens,
                prefill_done_time: None,
                handoff_done_time: None,
            },
        );
        let entry = self.topology.entry_pool();
        self.route_into_pool(entry, req);
    }

    fn route_into_pool(&mut self, pool_id: PoolId, req: Request) {
        let worker_idx = self.topology.pools[pool_id].pick_round_robin();
        self.topology.pools[pool_id].workers[worker_idx]
            .scheduler
            .add_request(req);
        self.maybe_wake_worker(pool_id, worker_idx, self.current_time);
    }

    fn maybe_wake_worker(&mut self, pool: PoolId, worker: usize, when: f64) {
        if !self.worker_busy[pool][worker] {
            self.worker_busy[pool][worker] = true;
            self.push(when, EventKind::WorkerReady { pool, worker });
        }
    }

    fn handle_worker_ready(
        &mut self,
        pool: PoolId,
        worker: usize,
    ) -> (Option<IterationInfo>, Vec<RequestTiming>) {
        // `WorkerReady` fires at the END of the worker's prior iteration (or
        // at t=0 when the worker first wakes). Mark idle and re-evaluate.
        self.worker_busy[pool][worker] = false;
        let now = self.current_time;
        let role = self.topology.role_for_pool(pool);
        let outcome = self.run_iteration(pool, worker, role, now);

        // Completions from `schedule()` finished at the end of the *previous*
        // iteration, i.e. `now`. Handoffs finished prefill in the iter that
        // ran *during* this step, so they're stamped at `end_time`.
        let mut timings = Vec::with_capacity(outcome.completed.len());
        for req in outcome.completed {
            timings.push(self.finalise(req, now));
        }
        let handoff_time = outcome.iteration.as_ref().map(|i| i.end_time).unwrap_or(now);
        for req in outcome.handed_off {
            self.start_handoff(req, handoff_time);
        }
        if let Some(end) = outcome.iteration.as_ref().map(|i| i.end_time) {
            self.worker_busy[pool][worker] = true;
            self.push(end, EventKind::WorkerReady { pool, worker });
        }
        (outcome.iteration, timings)
    }

    fn run_iteration(
        &mut self,
        pool: PoolId,
        worker: usize,
        role: PoolRole,
        now: f64,
    ) -> RunIterationOutcome {
        let w = &mut self.topology.pools[pool].workers[worker];
        let decision = w.scheduler.schedule(now);
        let completed = decision.completed;

        let mut batch_indices: Vec<usize> = decision.scheduled_new.to_vec();
        batch_indices.extend(decision.scheduled_running.iter().copied());
        let mut tokens_per_request: Vec<u32> = decision.tokens_for_new.clone();
        tokens_per_request.extend(decision.tokens_for_running.iter().copied());

        if batch_indices.is_empty() {
            return RunIterationOutcome {
                iteration: None,
                completed,
                handed_off: Vec::new(),
            };
        }

        let batch_size = batch_indices.len();

        // Capture per-request progress (and was_prefill) before mutating.
        let mut progress = Vec::with_capacity(batch_size);
        {
            let running = w.scheduler.running();
            for (i, &idx) in batch_indices.iter().enumerate() {
                if let Some(req) = running.get(idx) {
                    progress.push(RequestProgress {
                        request_id: req.request_id.clone(),
                        was_prefill: req.is_prefill(),
                        num_tokens: tokens_per_request[i],
                    });
                }
            }
        }

        let (iter_time, bandwidth_util, flops_util) = {
            let running = w.scheduler.running();
            let batch_refs: Vec<&Request> = batch_indices.iter().map(|&i| &running[i]).collect();
            let iter_time = w
                .compute_engine
                .calculate_iteration_time(&batch_refs, &tokens_per_request);
            let bytes = w
                .compute_engine
                .calculate_bytes_transferred(&batch_refs, &tokens_per_request);
            let bw = w
                .compute_engine
                .calculate_bandwidth_utilization(bytes, iter_time);
            let flops = w.compute_engine.calculate_flops_utilization(
                &batch_refs,
                &tokens_per_request,
                iter_time,
            );
            (iter_time, bw, flops)
        };
        let end_time = now + iter_time;

        for (i, &idx) in decision.scheduled_new.iter().enumerate() {
            if let Some(req) = w.scheduler.running_mut().get_mut(idx) {
                req.record_generated_tokens(decision.tokens_for_new[i], end_time);
            }
        }
        for (i, &idx) in decision.scheduled_running.iter().enumerate() {
            if let Some(req) = w.scheduler.running_mut().get_mut(idx) {
                req.record_generated_tokens(decision.tokens_for_running[i], end_time);
            }
        }

        let handed_off = if matches!(role, PoolRole::DisaggPrefill) {
            // Pull out anything whose prefill is now complete; free its KV
            // (it's about to leave this worker via the link).
            let running = w.scheduler.running_mut();
            let mut keep = Vec::with_capacity(running.len());
            let mut handed = Vec::new();
            for r in running.drain(..) {
                if r.num_computed_tokens >= r.num_prompt_tokens {
                    handed.push(r);
                } else {
                    keep.push(r);
                }
            }
            *running = keep;
            for req in &handed {
                w.scheduler
                    .kv_cache_manager_mut()
                    .free_blocks(&req.kv_blocks);
            }
            handed
        } else {
            Vec::new()
        };

        // Time-weighted batch accumulator.
        let dt = (end_time - now).max(0.0);
        let acc = &mut self.pool_batch_acc[pool];
        acc.0 += batch_size as f64 * dt;
        acc.1 += dt;

        RunIterationOutcome {
            iteration: Some(IterationInfo {
                pool,
                worker,
                start_time: now,
                end_time,
                iteration_time: iter_time,
                batch_size,
                bandwidth_util,
                flops_util,
                progress,
            }),
            completed,
            handed_off,
        }
    }

    fn start_handoff(&mut self, mut req: Request, prefill_done_at: f64) {
        if let Some(b) = self.bookkeeping.get_mut(&req.request_id) {
            b.prefill_done_time = Some(prefill_done_at);
        }
        req.kv_blocks.clear();
        let kv_bytes = self.topology.model.kv_storage_bytes(req.num_prompt_tokens);
        let id = req.request_id.clone();
        let (link, then_pool) = match self.topology.roles {
            Roles::Disagg {
                handoff, decode, ..
            } => (handoff, decode),
            _ => return,
        };
        self.topology.links[link].submit(id.clone(), kv_bytes, prefill_done_at);
        let eta = self.topology.links[link].estimate_remaining(&id);
        let drain_time = prefill_done_at + eta;
        self.push(
            drain_time,
            EventKind::LinkComplete {
                link,
                request_id: id.clone(),
                then_pool,
            },
        );
        self.parked.insert(id, req);
    }

    fn handle_link_complete(
        &mut self,
        link: LinkId,
        request_id: String,
        then_pool: PoolId,
    ) -> Result<(), String> {
        let now = self.current_time;
        let _ = self.topology.links[link].advance(now);
        if let Some(b) = self.bookkeeping.get_mut(&request_id) {
            b.handoff_done_time = Some(now);
        }
        let req = self
            .parked
            .remove(&request_id)
            .ok_or_else(|| format!("link complete for unknown request {request_id}"))?;
        self.route_into_pool(then_pool, req);
        Ok(())
    }

    fn finalise(&mut self, req: Request, completion_time: f64) -> RequestTiming {
        let id = req.request_id.clone();
        let book = self.bookkeeping.remove(&id);
        let arrival = book
            .as_ref()
            .map(|b| b.arrival_time)
            .unwrap_or(req.arrival_time);
        let first_token = req.first_token_time.unwrap_or(completion_time);
        // Aggregated mode doesn't track an explicit prefill→decode boundary
        // (no handoff to time-stamp). Fall back to first_token_time so the
        // breakdown reads as "prefill ends when the first decode token is
        // produced," matching the previous hand-rolled behaviour.
        let prefill_done = book
            .as_ref()
            .and_then(|b| b.prefill_done_time)
            .unwrap_or(first_token);
        let handoff_done = book
            .as_ref()
            .and_then(|b| b.handoff_done_time)
            .unwrap_or(prefill_done);
        RequestTiming {
            request_id: id,
            arrival_time: arrival,
            prefill_done_time: prefill_done,
            handoff_done_time: handoff_done,
            first_token_time: first_token,
            completion_time,
            num_prompt_tokens: book
                .as_ref()
                .map(|b| b.num_prompt_tokens)
                .unwrap_or(req.num_prompt_tokens),
            num_output_tokens: req.num_output_tokens,
        }
    }
}

struct RunIterationOutcome {
    iteration: Option<IterationInfo>,
    completed: Vec<Request>,
    handed_off: Vec<Request>,
}
