//! Unified discrete-event engine. Pure state machine: knows about pools,
//! workers, the event heap, KV bookkeeping and time. Knows nothing about
//! request generation, metrics, real wall-clock, or how to render progress.
//! Drivers (batch [`super::simulator::Simulator`], serve `crate::serve::engine`)
//! pump it by alternating `next_event_time` / `submit` / `step`.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use super::spec::{DepthSample, PlanCosts, SpecPlanner};
use crate::compute::ComputeEngine;
use crate::config::{ClusterSpec, DisaggTopology, ModelSpec, SchedulerConfig, SpeculativeConfig};
use crate::kv_cache::{KVCacheManager, Link, PrefixCacheStats};
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
    /// Times the request was preempted (and recomputed) before completing.
    pub num_preemptions: u32,
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
        model: ModelSpec,
        scheduler_config: SchedulerConfig,
    ) -> Result<Self, String> {
        let kv_capacity =
            cluster.kv_cache_capacity(&scheduler_config, cluster.resident_weight_bytes(&model));

        // Blocks are charged from the model's exact KV curve: linear-KV
        // models get ceil(t / block_size), models whose footprint is
        // nonlinear in position (sliding window, DeepSeek-V4's window +
        // compressed history) get their real bytes.
        let kv_model = model.clone();
        let kv_cache_manager = KVCacheManager::new(
            kv_capacity,
            scheduler_config.block_size,
            move |t| kv_model.kv_storage_bytes(t),
            model.per_sequence_state_bytes(),
            true,
        )
        .with_tiers(&cluster.hardware.kv_tiers);
        if kv_cache_manager.total_blocks() == 0 {
            return Err(format!(
                "KV cache capacity ({} bytes) holds less than one {}-token block ({} bytes) of {}",
                kv_capacity,
                scheduler_config.block_size,
                kv_cache_manager.bytes_per_block(),
                model.name
            ));
        }

        let scheduler = Scheduler::new(scheduler_config.clone(), kv_cache_manager);
        let compute_engine = cluster.compute_engine(model).with_cascade_attention(
            scheduler_config.enable_cascade_attention,
            scheduler_config.block_size,
        );
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
    model: ModelSpec,
}

impl Topology {
    pub fn aggregated(
        cluster: ClusterSpec,
        model: ModelSpec,
        scheduler_config: SchedulerConfig,
    ) -> Result<Self, String> {
        let n = cluster.num_workers.max(1) as usize;
        let mut workers = Vec::with_capacity(n);
        for _ in 0..n {
            workers.push(Worker::new(
                &cluster,
                model.clone(),
                scheduler_config.clone(),
            )?);
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
        model: ModelSpec,
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
    /// The next hand-off on `link` is due to complete (under the contention
    /// in force when it was scheduled). `generation` invalidates events made
    /// stale by a later change to the link's in-flight set.
    LinkDrain {
        link: LinkId,
        generation: u64,
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

/// Per-request progress yielded by a worker iteration.
#[derive(Debug, Clone)]
pub struct RequestProgress {
    pub request_id: String,
    /// Whether the request was in prefill phase BEFORE this iteration ran.
    pub was_prefill: bool,
    /// Positions computed for this request in the iteration: the prefill
    /// chunk, or the decode verify width (`1 + draft`).
    pub num_tokens: u32,
    /// Output tokens the iteration produced for this request: 1 when it
    /// completed prefill, `1 + accepted` for a decode step, 0 otherwise.
    pub num_output: u32,
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
    /// Requests that finished prefill on the P pool and are mid-handoff over
    /// the link.
    parked: HashMap<String, Request>,
    /// Per-link generation of the scheduled `LinkDrain` event; only the
    /// event carrying the current generation is acted on.
    link_generation: Vec<u64>,
    /// `worker_busy[pool][worker]` is true iff a `WorkerReady` for that
    /// worker is currently scheduled in the queue.
    worker_busy: Vec<Vec<bool>>,
    /// Per-pool (∑ batch·dt, ∑ dt) for time-weighted mean batch size.
    pool_batch_acc: Vec<(f64, f64)>,
    current_time: f64,
    seq_counter: u64,
    /// Speculative decoding planner. `None` = no speculation.
    spec: Option<SpecPlanner>,
    /// Optional affine step-time correction (alpha, beta seconds).
    time_correction: Option<(f64, f64)>,
}

impl Engine {
    pub fn new(topology: Topology) -> Self {
        let mut worker_busy = Vec::with_capacity(topology.pools.len());
        for pool in &topology.pools {
            worker_busy.push(vec![false; pool.workers.len()]);
        }
        let pool_batch_acc = vec![(0.0_f64, 0.0_f64); topology.pools.len()];
        let link_generation = vec![0u64; topology.links.len()];
        Self {
            topology,
            events: BinaryHeap::new(),
            parked: HashMap::new(),
            link_generation,
            worker_busy,
            pool_batch_acc,
            current_time: 0.0,
            seq_counter: 0,
            spec: None,
            time_correction: None,
        }
    }

    /// Apply an affine empirical correction to every executed iteration time:
    /// `t = alpha * t_model + beta`. `alpha` captures the kernel-efficiency
    /// gap to the roofline (dominant at large batch); `beta` captures fixed
    /// per-iteration overhead — scheduler, CPU, launch latency (dominant at
    /// small batch). Applied to the actual step time only, never to policy
    /// candidate pricing, so speculative width choices stay model-relative.
    pub fn set_time_correction(&mut self, alpha: f64, beta: f64) {
        self.time_correction = Some((alpha, beta));
    }

    /// Enable speculative decoding. Decode steps then verify `gamma + 1` tokens
    /// (cost) and advance by `accepted + 1` (progress) per the acceptance model.
    /// Loads the trace bank / measured cost table the config names; a bad
    /// path is returned as an error.
    pub fn enable_speculative(&mut self, cfg: SpeculativeConfig, seed: u64) -> Result<(), String> {
        self.spec = Some(SpecPlanner::new(cfg, seed)?);
        Ok(())
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
    /// (relative to the engine clock) the event fires immediately at `now`;
    /// the request keeps its own `arrival_time`, so TTFT is still computed
    /// against the intended emission moment.
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

    /// Total preemptions across all pools.
    pub fn aggregate_preemptions(&self) -> u64 {
        self.topology
            .pools
            .iter()
            .flat_map(|p| &p.workers)
            .map(|w| w.scheduler.num_preemptions())
            .sum()
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
            .map(|w| {
                w.scheduler
                    .running()
                    .iter()
                    .filter(|r| r.is_prefill())
                    .count()
            })
            .sum()
    }

    /// Per-second speculative draft-depth series; empty when speculation is
    /// off.
    pub fn spec_depth_series(&self) -> Vec<DepthSample> {
        self.spec
            .as_ref()
            .map(|s| s.depth_series())
            .unwrap_or_default()
    }

    /// Prefix-cache lookup statistics summed over every worker's KV manager.
    pub fn aggregate_prefix_cache(&self) -> PrefixCacheStats {
        let mut total = PrefixCacheStats::default();
        for pool in &self.topology.pools {
            for worker in &pool.workers {
                total += worker.scheduler.kv_cache_manager().prefix_cache_stats();
            }
        }
        total
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
        let ev = self.events.pop().ok_or_else(|| {
            format!(
                "step called with empty event queue (running={}, waiting={}, parked={}, t={})",
                self.aggregate_running(),
                self.aggregate_waiting(),
                self.parked.len(),
                self.current_time
            )
        })?;
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
            EventKind::LinkDrain { link, generation } => {
                self.handle_link_drain(link, generation)?;
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
        let handoff_time = outcome
            .iteration
            .as_ref()
            .map(|i| i.end_time)
            .unwrap_or(now);
        for req in outcome.handed_off {
            self.start_handoff(req, handoff_time);
        }
        if let Some(end) = outcome.iteration.as_ref().map(|i| i.end_time) {
            self.worker_busy[pool][worker] = true;
            self.push(end, EventKind::WorkerReady { pool, worker });
        } else if outcome.preempted {
            // The scheduler preempted but ran nothing (e.g. every runner was
            // preempted, which also blocks admission for the step). State
            // changed, so re-arm immediately; otherwise the worker only
            // wakes on a new arrival and the engine can stall with work
            // queued. This can't loop at one timestamp: a preempt-only pass
            // empties `running`, and the re-run either schedules something
            // or preempts nothing.
            let w = &self.topology.pools[pool].workers[worker];
            if w.scheduler.num_running() > 0 || w.scheduler.num_waiting() > 0 {
                self.maybe_wake_worker(pool, worker, now);
            }
        } else if let Some(ready) = self.topology.pools[pool].workers[worker]
            .scheduler
            .earliest_pending_ready()
        {
            // Nothing ran, but a request is parked on a KV tier promotion.
            // Re-arm at its completion time so the engine doesn't stall
            // waiting for an arrival.
            self.maybe_wake_worker(pool, worker, ready.max(now));
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
        let correction = self.time_correction;
        let w = &mut self.topology.pools[pool].workers[worker];
        let decision = w.scheduler.schedule(now);
        let completed = decision.completed;
        let preempted = decision.num_preempted > 0;

        let batch_indices: Vec<usize> = decision.batch.iter().map(|s| s.idx).collect();
        let tokens_per_request: Vec<u32> = decision.batch.iter().map(|s| s.num_tokens).collect();

        if batch_indices.is_empty() {
            return RunIterationOutcome {
                iteration: None,
                completed,
                handed_off: Vec::new(),
                preempted,
            };
        }

        let batch_size = batch_indices.len();

        // Capture per-request progress (and was_prefill) before mutating.
        let mut progress = Vec::with_capacity(batch_size);
        let mut round_commits: Vec<Option<u32>> = Vec::with_capacity(batch_indices.len());
        {
            let running = w.scheduler.running();
            for (i, &idx) in batch_indices.iter().enumerate() {
                if let Some(req) = running.get(idx) {
                    progress.push(RequestProgress {
                        request_id: req.request_id.clone(),
                        was_prefill: req.is_prefill(),
                        num_tokens: tokens_per_request[i],
                        num_output: 0,
                    });
                    round_commits.push(req.pending_round_commits);
                }
            }
        }

        // Speculative decoding (vLLM-faithful). Each decode request's draft
        // length was decided at the END of the previous iteration and stored
        // as `pending_draft_len`; the scheduler has already reserved a
        // `1 + draft` verify pass in the token budget and KV (trimming `draft`
        // to fit if capacity was tight), which is exactly what the batch's
        // `num_tokens` now carries. So here we only realise the *outcome*:
        // how many of the reserved draft tokens are accepted, and advance by
        // `accepted + 1`. The verify pass itself (`1 + draft` tokens) is the
        // cost. Prefill and chunked-prefill continuations (was_prefill) are
        // never speculated.
        let cost_tokens = tokens_per_request.clone(); // verify width per request
        let mut accepted_extra = vec![0u32; batch_size];
        let mut draft_widths: Vec<u32> = Vec::new(); // decode sequences only
        if self.spec.is_some() {
            for j in 0..batch_size {
                if progress[j].was_prefill {
                    continue;
                }
                let draft = cost_tokens[j].saturating_sub(1);
                draft_widths.push(draft);
                accepted_extra[j] = SpecPlanner::accepted(draft, round_commits[j]);
            }
        }

        let (mut iter_time, measured, bandwidth_util, flops_util) = {
            let running = w.scheduler.running();
            let batch_refs: Vec<&Request> = batch_indices.iter().map(|&i| &running[i]).collect();
            let was_prefill: Vec<bool> = progress.iter().map(|p| p.was_prefill).collect();
            Self::price_step(
                self.spec.as_ref(),
                &w.compute_engine,
                &batch_refs,
                &cost_tokens,
                &was_prefill,
                correction,
            )
        };
        if let Some(spec) = &mut self.spec {
            // Drafter overhead on roofline-priced speculated steps. Table-priced
            // steps skip this — the measured wall gap already embodies the full
            // engine step, drafter included.
            if !measured && draft_widths.iter().any(|&d| d > 0) {
                let peak = w.compute_engine.bf16_peak_flops();
                let bw = w.compute_engine.mem_bandwidth();
                iter_time += spec.drafter_seconds(&draft_widths, peak, bw, iter_time);
            }
            // Constrained-GatedAggregate per-switch stall: a width change
            // decided at the end of the previous round costs the engine a
            // rebuild on the first round executed at the new width — this one.
            iter_time += spec.take_pending_switch_cost((pool, worker));
        }
        let end_time = now + iter_time;

        for (j, &idx) in batch_indices.iter().enumerate() {
            // Decode: advance by the verified tokens (bonus + accepted), NOT
            // the verify width (`num_tokens` = 1 + draft, the cost). Prefill
            // (including chunked continuations and recompute after
            // preemption): advance by the scheduled chunk.
            let adv = if progress[j].was_prefill {
                tokens_per_request[j]
            } else {
                1 + accepted_extra[j]
            };
            progress[j].num_output = w.scheduler.record_progress(idx, adv, end_time);
        }

        let handed_off = if matches!(role, PoolRole::DisaggPrefill) {
            // Anything whose prefill is now complete leaves this worker via
            // the link; the scheduler frees its KV on the way out.
            w.scheduler.take_prefill_complete()
        } else {
            Vec::new()
        };

        // Decide the decode batch's draft depth for its NEXT step. Drafting
        // happens here, at the end of the step -- the one instant when the
        // drafter is about to run AND the carry-over decode set is known.
        // The next scheduler pass reads `pending_draft_len` and reserves
        // `1 + draft` of budget + KV.
        if let Some(spec) = &mut self.spec {
            let dec: Vec<&Request> = w
                .scheduler
                .running()
                .iter()
                .filter(|r| !r.is_prefill() && !r.is_finished())
                .collect();
            let n = dec.len();
            // The measured table's KV-length correction (0 without a table
            // or ref_seq_len): width-independent, but it moves the argmax.
            let kv_delta = match (spec.ref_seq_len(), spec.measured_cost()) {
                (Some(ref_len), Some(_)) => {
                    w.compute_engine.kv_read_seq_delta_seconds(&dec, ref_len)
                }
                _ => 0.0,
            };
            // Analytic verify cost of the decode sub-batch at width 1 + g,
            // through the real roofline (MoE-coupon- and MLA-aware). Prefill
            // is deliberately not priced in: the mandatory prefill is
            // proportional to committed output, so it cancels in the argmax,
            // and routing prefill tokens through the cost model trips the MoE
            // coupon (verify looks free).
            let ce = &w.compute_engine;
            let roofline = |g: u32| ce.calculate_iteration_time(&dec, &vec![g + 1; n]);
            let costs = PlanCosts {
                roofline: &roofline,
                kv_delta,
                peak: ce.bf16_peak_flops(),
                bw: ce.mem_bandwidth(),
            };
            let plans = spec.plan((pool, worker), &dec, &costs, end_time);
            let plans: Vec<(u32, Option<u32>)> = plans
                .iter()
                .map(|p| (p.draft_len, Some(p.commits)))
                .collect();
            w.scheduler.set_draft_plans(&plans);
        }

        // Time-weighted batch accumulator.
        let dt = (end_time - now).max(0.0);
        let acc = &mut self.pool_batch_acc[pool];
        acc.0 += batch_size as f64 * dt;
        acc.1 += dt;

        RunIterationOutcome {
            preempted,
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

    /// Wall-clock time of a step, plus utilisation figures.
    ///
    /// When a measured step-cost table is present, the step's DECODE portion
    /// is priced from the table at (decode batch size, mean decode verify
    /// width) — the same source the policy's C(g) consults — so speculative
    /// and plain-decode steps are priced commensurately (a no-spec step reads
    /// the table's plain-decode rows, not the analytic roofline). Ragged
    /// verify widths interpolate between the nearest *measured* draft-length
    /// cells; a mean width outside the table's measured range falls back to
    /// the roofline for the whole step. Prefill tokens sharing the step are
    /// priced as the roofline time of the prefill sub-batch alone and ADDED
    /// to the table-priced decode portion (the table embodies a full decode
    /// step; the roofline supplies the prefill increment). Pure-prefill steps
    /// always price through the roofline. When the table declares the
    /// sequence length it was benchmarked at (`ref_seq_len`), the decode
    /// portion also gets a bandwidth-roofline KV-read correction for the live
    /// batch's actual KV lengths, floored at 25% of the table value.
    ///
    /// Returns `(iteration_time, priced_from_table, bandwidth_util, flops_util)`.
    fn price_step(
        spec: Option<&SpecPlanner>,
        ce: &ComputeEngine,
        batch_refs: &[&Request],
        cost_tokens: &[u32],
        was_prefill: &[bool],
        correction: Option<(f64, f64)>,
    ) -> (f64, bool, f64, f64) {
        let batch_size = batch_refs.len();
        let measured_time: Option<f64> = spec.and_then(|spec| {
            let table = spec.measured_cost()?;
            let dec_idx: Vec<usize> = (0..batch_size).filter(|&j| !was_prefill[j]).collect();
            if dec_idx.is_empty() {
                return None; // pure prefill: roofline
            }
            let mean_w: f64 =
                dec_idx.iter().map(|&j| cost_tokens[j] as f64).sum::<f64>() / dec_idx.len() as f64;
            let g = (mean_w - 1.0).max(0.0);
            let mut t_dec = table.step_time_frac(dec_idx.len() as u32, g)?;
            if let Some(ref_len) = spec.ref_seq_len() {
                let dec_refs: Vec<&Request> = dec_idx.iter().map(|&j| batch_refs[j]).collect();
                let delta = ce.kv_read_seq_delta_seconds(&dec_refs, ref_len);
                // Recontextualisation, not a license to price below any
                // plausible step: floor at 25% of the table value.
                t_dec = (t_dec + delta).max(0.25 * t_dec);
            }
            let t_pre = if dec_idx.len() < batch_size {
                let pre_refs: Vec<&Request> = (0..batch_size)
                    .filter(|&j| was_prefill[j])
                    .map(|j| batch_refs[j])
                    .collect();
                let pre_tokens: Vec<u32> = (0..batch_size)
                    .filter(|&j| was_prefill[j])
                    .map(|j| cost_tokens[j])
                    .collect();
                ce.calculate_iteration_time(&pre_refs, &pre_tokens)
            } else {
                0.0
            };
            Some(t_dec + t_pre)
        });
        let cost = ce.step_cost(batch_refs, cost_tokens);
        let iter_time = measured_time.unwrap_or(cost.time);
        let iter_time = match correction {
            Some((alpha, beta)) => alpha * iter_time + beta,
            None => iter_time,
        };
        let bw = ce.bandwidth_utilization(&cost, iter_time);
        let flops = ce.flops_utilization(&cost, iter_time);
        (iter_time, measured_time.is_some(), bw, flops)
    }

    fn start_handoff(&mut self, mut req: Request, prefill_done_at: f64) {
        req.prefill_done_time = Some(prefill_done_at);
        let kv_bytes = self
            .topology
            .model
            .kv_storage_bytes(req.num_computed_tokens);
        let id = req.request_id.clone();
        let link = match self.topology.roles {
            Roles::Disagg { handoff, .. } => handoff,
            _ => return,
        };
        // Bring the link up to date under the old contention, then add the
        // new transfer and re-plan the next completion under the new one.
        // (`advance` returns nothing here: any completion due before now was
        // handled by its own drain event.)
        let _ = self.topology.links[link].advance(prefill_done_at);
        self.topology.links[link].submit(id.clone(), kv_bytes, prefill_done_at);
        self.parked.insert(id, req);
        self.schedule_link_drain(link, prefill_done_at);
    }

    /// Schedule the next completion on `link` under its current contention,
    /// invalidating any previously scheduled drain event.
    fn schedule_link_drain(&mut self, link: LinkId, now: f64) {
        self.link_generation[link] += 1;
        let generation = self.link_generation[link];
        if let Some(delay) = self.topology.links[link].next_completion_delay() {
            self.push(now + delay, EventKind::LinkDrain { link, generation });
        }
    }

    fn handle_link_drain(&mut self, link: LinkId, generation: u64) -> Result<(), String> {
        if generation != self.link_generation[link] {
            return Ok(()); // superseded by a later submit / completion
        }
        let now = self.current_time;
        let done = self.topology.links[link].advance(now);
        let decode_pool = match self.topology.roles {
            Roles::Disagg { decode, .. } => decode,
            _ => return Err("link drain on an aggregated topology".to_string()),
        };
        // Route completed hand-offs in a deterministic order.
        let mut done: Vec<String> = done.into_iter().collect();
        done.sort();
        for request_id in done {
            let mut req = self
                .parked
                .remove(&request_id)
                .ok_or_else(|| format!("link complete for unknown request {request_id}"))?;
            req.handoff_done_time = Some(now);
            self.route_into_pool(decode_pool, req);
        }
        self.schedule_link_drain(link, now);
        Ok(())
    }

    fn finalise(&mut self, req: Request, completion_time: f64) -> RequestTiming {
        let first_token = req.first_token_time.unwrap_or(completion_time);
        // On an aggregated topology the prefill pass produces the first
        // token, so prefill ends when the first token is produced.
        let prefill_done = req.prefill_done_time.unwrap_or(first_token);
        let handoff_done = req.handoff_done_time.unwrap_or(prefill_done);
        RequestTiming {
            request_id: req.request_id,
            arrival_time: req.arrival_time,
            prefill_done_time: prefill_done,
            handoff_done_time: handoff_done,
            first_token_time: first_token,
            completion_time,
            num_prompt_tokens: req.num_prompt_tokens,
            num_output_tokens: req.num_output_tokens,
            num_preemptions: req.num_preemptions,
        }
    }
}

struct RunIterationOutcome {
    iteration: Option<IterationInfo>,
    completed: Vec<Request>,
    handed_off: Vec<Request>,
    /// The scheduler preempted at least one request this pass. Relevant when
    /// `iteration` is `None`: state changed even though nothing ran, so the
    /// worker must be re-armed rather than left to wait for a new arrival.
    preempted: bool,
}
