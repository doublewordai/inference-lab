//! Unified discrete-event engine. Pure state machine: knows about pools,
//! workers, the event heap, KV bookkeeping and time. Knows nothing about
//! request generation, metrics, real wall-clock, or how to render progress.
//! Drivers (batch [`super::sim::Simulator`], serve [`crate::serve::engine`])
//! pump it by alternating `next_event_time` / `submit` / `step`.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::compute::ComputeEngine;
use crate::config::{
    ClusterSpec, DisaggTopology, GammaPolicy, ModelConfig, ModelCosts, SchedulerConfig,
    SpeculativeConfig,
};
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
            model.per_sequence_state_bytes(),
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
    /// Speculative decoding config + RNG. `None` = no speculation.
    spec: Option<SpeculativeConfig>,
    spec_rng: Option<StdRng>,
    /// Loaded trace bank when the acceptance model is `TraceRounds`.
    trace_bank: Option<std::sync::Arc<crate::config::TraceBank>>,
    /// Loaded measured step-cost table when `spec.measured_cost` is set.
    /// Both the policy's cost curve C(g) and the wall-clock time of
    /// pure-decode steps read from here (roofline fallback off-grid), so
    /// plain-decode and speculative steps are priced commensurately.
    measured_cost: Option<std::sync::Arc<crate::compute::MeasuredCostTable>>,
    /// Per-second buckets of speculative draft-depth decisions:
    /// time-floor-secs -> (sum of per-seq drafts, decode seqs, steps).
    spec_depth_buckets: std::collections::BTreeMap<u64, (u64, u64, u64)>,
    /// Per-(pool, worker) `GatedAggregate` switching state when
    /// `spec.switch` is constrained (cooldown / bounded candidate walk /
    /// per-switch cost). Empty when unconstrained (the fast path never
    /// touches it, so the raw policy is reproduced bit-for-bit).
    agg_switch: HashMap<(PoolId, usize), AggSwitchState>,
}

/// `GatedAggregate` switching state under engine constraints.
#[derive(Debug, Clone, Copy)]
struct AggSwitchState {
    /// Width currently in force (persists between re-evaluations).
    g: u32,
    /// Decode rounds elapsed since the last re-evaluation.
    rounds_since: u32,
    /// Switch cost (seconds) accrued by a width change, to be paid on the
    /// wall time of the next round (the first executed at the new width).
    pending_cost: f64,
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
            spec: None,
            spec_rng: None,
            trace_bank: None,
            measured_cost: None,
            spec_depth_buckets: std::collections::BTreeMap::new(),
            agg_switch: HashMap::new(),
        }
    }

    /// Enable speculative decoding. Decode steps then verify `gamma + 1` tokens
    /// (cost) and advance by `accepted + 1` (progress) per the acceptance model.
    /// With `TraceRounds` acceptance the bank is loaded here (panics on a bad
    /// path: a missing bank is a configuration error, not a runtime condition).
    pub fn enable_speculative(&mut self, cfg: SpeculativeConfig, seed: u64) {
        if let crate::config::AcceptanceModel::TraceRounds { path } = &cfg.acceptance {
            let bank = crate::config::TraceBank::load(path).expect("loading trace bank");
            self.trace_bank = Some(std::sync::Arc::new(bank));
        }
        if let Some(mc) = &cfg.measured_cost {
            let table = crate::compute::MeasuredCostTable::load(&mc.path)
                .expect("loading measured cost table");
            self.measured_cost = Some(std::sync::Arc::new(table));
        }
        self.spec_rng = Some(StdRng::seed_from_u64(seed));
        self.spec = Some(cfg);
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
            .map(|w| {
                w.scheduler
                    .running()
                    .iter()
                    .filter(|r| r.is_prefill())
                    .count()
            })
            .sum()
    }

    /// Aggregate prefix-cache stats across every worker's KV manager.
    /// Returned as `(hits, misses, hit_size_sum, hit_size_count)`.
    /// Per-second speculative draft-depth series:
    /// (second, mean drafts per decode seq, mean decode batch).
    pub fn spec_depth_series(&self) -> Vec<(u64, f64, f64)> {
        self.spec_depth_buckets
            .iter()
            .map(|(&s, &(drafts, seqs, steps))| {
                (
                    s,
                    drafts as f64 / seqs.max(1) as f64,
                    seqs as f64 / steps.max(1) as f64,
                )
            })
            .collect()
    }

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
        let mut round_commits: Vec<Option<u32>> = Vec::with_capacity(batch_indices.len());
        {
            let running = w.scheduler.running();
            for (i, &idx) in batch_indices.iter().enumerate() {
                if let Some(req) = running.get(idx) {
                    progress.push(RequestProgress {
                        request_id: req.request_id.clone(),
                        was_prefill: req.is_prefill(),
                        num_tokens: tokens_per_request[i],
                    });
                    round_commits.push(req.pending_round_commits);
                }
            }
        }

        // Speculative decoding (vLLM-faithful). Each decode request's draft
        // length was decided at the END of the previous iteration and stored as
        // `pending_draft_len`; the scheduler has already reserved a `1 + draft`
        // verify pass in the token budget and KV (trimming `draft` to fit if
        // capacity was tight), which is exactly what `tokens_for_running` now
        // carries. So here we only realise the *outcome*: sample how many of the
        // reserved draft tokens are accepted and advance by `accepted + 1`. The
        // verify pass itself (`1 + draft` tokens) is the cost. Prefill and
        // chunked-prefill continuations (was_prefill) are never speculated.
        let n_new = decision.scheduled_new.len();
        let cost_tokens = tokens_per_request.clone(); // verify width per request
        let mut accepted_extra = vec![0u32; batch_size];
        let mut step_gamma_max = 0u32; // widest draft this step (for drafter overhead)
        if let Some(spec) = self.spec.clone() {
            for j in 0..batch_size {
                if progress[j].was_prefill {
                    continue;
                }
                let draft = cost_tokens[j].saturating_sub(1);
                step_gamma_max = step_gamma_max.max(draft);
                if draft == 0 {
                    continue;
                }
                // Trace-driven rounds realise the outcome of the round drawn at
                // draft time: `min(commits, draft)` stays correct if the
                // scheduler trimmed the draft to fit budget/KV.
                let accepted = match round_commits.get(j).copied().flatten() {
                    Some(commits) => commits.min(draft),
                    None => match self.spec_rng.as_mut() {
                        Some(rng) => spec.acceptance.sample_accepted(draft, rng),
                        None => spec.acceptance.expected_accepted(draft).round() as u32,
                    },
                };
                accepted_extra[j] = accepted; // bonus (+1) added at progress
                progress[j].num_tokens = accepted + 1; // tokens generated this step
            }
        }

        // Wall-clock pricing. When a measured step-cost table is present, the
        // step's DECODE portion is priced from the table at (decode batch
        // size, mean decode verify width) — the same source the policy's
        // C(g) consults — so speculative and plain-decode steps are priced
        // commensurately (a no-spec step reads the table's plain-decode
        // rows, not the analytic roofline). Ragged verify widths interpolate
        // between the nearest *measured* draft-length cells (interior grid
        // holes are bridged from measured anchors, never the optimistic
        // roofline); a mean width outside the table's measured range falls
        // back to the roofline for the whole step. Prefill tokens sharing
        // the step are priced as the analytic-roofline time of the prefill
        // sub-batch alone and ADDED to the table-priced decode portion (the
        // table embodies a full decode step; the roofline supplies the
        // prefill increment — slightly optimistic, but far closer to the
        // engine than pricing the whole mixed step through the roofline,
        // which under-prices the decode portion by several-fold at large
        // batch). Pure-prefill steps always price through the roofline.
        // When the table declares the sequence length it was benchmarked at
        // (`ref_seq_len`), the decode portion also gets a bandwidth-roofline
        // KV-read correction for the live batch's actual KV lengths.
        let ref_seq_len = self
            .spec
            .as_ref()
            .and_then(|s| s.measured_cost.as_ref())
            .and_then(|m| m.ref_seq_len);
        let (mut iter_time, measured_time, bandwidth_util, flops_util) = {
            let running = w.scheduler.running();
            let batch_refs: Vec<&Request> = batch_indices.iter().map(|&i| &running[i]).collect();
            let measured_time: Option<f64> = self.measured_cost.as_ref().and_then(|table| {
                let dec_idx: Vec<usize> = (0..batch_size)
                    .filter(|&j| !progress[j].was_prefill)
                    .collect();
                if dec_idx.is_empty() {
                    return None; // pure prefill: roofline
                }
                let mean_w: f64 = dec_idx.iter().map(|&j| cost_tokens[j] as f64).sum::<f64>()
                    / dec_idx.len() as f64;
                let g = (mean_w - 1.0).max(0.0);
                let mut t_dec = table.step_time_frac(dec_idx.len() as u32, g)?;
                if let Some(ref_len) = ref_seq_len {
                    let dec_refs: Vec<&Request> = dec_idx.iter().map(|&j| batch_refs[j]).collect();
                    let delta = w
                        .compute_engine
                        .kv_read_seq_delta_seconds(&dec_refs, ref_len);
                    // Recontextualisation, not a license to price below any
                    // plausible step: floor at 25% of the table value.
                    t_dec = (t_dec + delta).max(0.25 * t_dec);
                }
                let t_pre = if dec_idx.len() < batch_size {
                    let pre_refs: Vec<&Request> = (0..batch_size)
                        .filter(|&j| progress[j].was_prefill)
                        .map(|j| batch_refs[j])
                        .collect();
                    let pre_tokens: Vec<u32> = (0..batch_size)
                        .filter(|&j| progress[j].was_prefill)
                        .map(|j| cost_tokens[j])
                        .collect();
                    w.compute_engine
                        .calculate_iteration_time(&pre_refs, &pre_tokens)
                } else {
                    0.0
                };
                Some(t_dec + t_pre)
            });
            let iter_time = measured_time.unwrap_or_else(|| {
                w.compute_engine
                    .calculate_iteration_time(&batch_refs, &cost_tokens)
            });
            let bytes = w
                .compute_engine
                .calculate_bytes_transferred(&batch_refs, &cost_tokens);
            let bw = w
                .compute_engine
                .calculate_bandwidth_utilization(bytes, iter_time);
            let flops =
                w.compute_engine
                    .calculate_flops_utilization(&batch_refs, &cost_tokens, iter_time);
            (iter_time, measured_time, bw, flops)
        };
        // Drafter overhead on roofline-priced speculated steps. The drafter runs
        // only for the decode (speculating) sequences, and the cost follows the
        // ragged per-sequence draft widths: an autoregressive head shrinks its
        // sub-batch with depth, a block head drafts one block sized to the deepest
        // draft. Table-priced steps skip this — the measured wall gap already
        // embodies the full engine step, drafter included.
        if step_gamma_max > 0 && measured_time.is_none() {
            if let Some(spec) = &self.spec {
                let draft_widths: Vec<u32> = (0..batch_size)
                    .filter(|&j| !progress[j].was_prefill)
                    .map(|j| cost_tokens[j].saturating_sub(1))
                    .collect();
                let peak = w.compute_engine.bf16_peak_flops();
                let bw = w.compute_engine.mem_bandwidth();
                iter_time += spec.ragged_drafter_seconds(&draft_widths, peak, bw, iter_time);
            }
        }
        // Constrained-GatedAggregate per-switch stall: a width change decided
        // at the end of the previous round costs the engine a rebuild on the
        // first round executed at the new width — this one.
        if let Some(st) = self.agg_switch.get_mut(&(pool, worker)) {
            if st.pending_cost > 0.0 {
                iter_time += st.pending_cost;
                st.pending_cost = 0.0;
            }
        }
        let end_time = now + iter_time;

        for (i, &idx) in decision.scheduled_new.iter().enumerate() {
            if let Some(req) = w.scheduler.running_mut().get_mut(idx) {
                req.record_generated_tokens(decision.tokens_for_new[i], end_time);
            }
        }
        for (i, &idx) in decision.scheduled_running.iter().enumerate() {
            if let Some(req) = w.scheduler.running_mut().get_mut(idx) {
                let j = n_new + i;
                // Decode: advance by the verified tokens (bonus + accepted), NOT
                // the verify width (`tokens_for_running` = 1 + draft, the cost).
                // Chunked prefill: advance by the scheduled prefill chunk.
                let adv = if progress[j].was_prefill {
                    decision.tokens_for_running[i]
                } else {
                    1 + accepted_extra[j]
                };
                req.record_generated_tokens(adv, end_time);
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

        // Decide the decode batch's draft depth for its NEXT step. Drafting
        // happens here, at the end of the step -- the one instant when the
        // drafter is about to run AND N's survivors (the carry-over decode set)
        // and the waiting prefill queue are both known. The next scheduler pass
        // reads `pending_draft_len` and reserves `1 + draft` of budget + KV.
        //   Fixed         -> constant draft on every decode.
        //   GoodputBudget -> one homogeneous G in 0..=gamma, chosen by pricing
        //                    each candidate against the batch that will actually
        //                    co-occur: the decode sub-batch at width 1+G plus the
        //                    prefill that backfills the leftover token budget. So
        //                    deeper G loses when prefill demands the budget. The
        //                    drafter is charged for G passes; G is the planned
        //                    verify depth, so there is no over-generation.
        if let Some(spec) = self.spec.clone() {
            let (drafts, next_commits): (Vec<u32>, Vec<Option<u32>>) = {
                let dec: Vec<&Request> = w
                    .scheduler
                    .running()
                    .iter()
                    .filter(|r| !r.is_prefill() && !r.is_finished())
                    .collect();
                let n = dec.len();
                // With `TraceRounds` acceptance, draw each decode sequence's
                // next round NOW: the gate's per-depth signal and the realised
                // outcome come from the same real round.
                let round_idx: Option<Vec<usize>> = match (&self.trace_bank, n) {
                    (Some(bank), n) if n > 0 => {
                        let rng = self.spec_rng.as_mut().expect("spec rng");
                        let m = bank.rounds.len();
                        Some((0..n).map(|_| rng.gen_range(0..m)).collect())
                    }
                    _ => None,
                };
                let commits: Vec<Option<u32>> = match (&self.trace_bank, &round_idx) {
                    (Some(bank), Some(idx)) => {
                        idx.iter().map(|&i| Some(bank.rounds[i].commits)).collect()
                    }
                    _ => vec![None; n],
                };
                let drafts: Vec<u32> = if n == 0 {
                    Vec::new()
                } else {
                    match spec.policy {
                        GammaPolicy::Fixed => vec![spec.gamma; n],
                        GammaPolicy::GoodputBudget
                        | GammaPolicy::GatedBudget
                        | GammaPolicy::GatedAggregate => {
                            // Decide one homogeneous draft depth G for the decode
                            // batch by maximising committed decode / step time,
                            // priced through the real roofline cost model (MoE-
                            // coupon- and MLA-aware), drafter charged for G passes.
                            // G is sized from the previous iteration's decode batch
                            // (vLLM-faithful: drafting happens at step end).
                            //
                            // It prices the decode sub-batch only -- prefill is
                            // ignored. That is deliberate, not a TODO: a single-step
                            // prefill term cannot price G correctly (the mandatory
                            // prefill is proportional to committed output, so it
                            // cancels in the argmax; and routing prefill tokens
                            // through the cost model lights up the MoE coupon and
                            // makes verify look free). Measured behaviour (chunked
                            // prefill on, the vLLM-V1 target): within +/-0.5% of the
                            // best fixed gamma when ISL ~ OSL, with a bounded
                            // over-speculation tail (up to ~6%) only at large batch
                            // when ISL >> OSL. That residual is a gamma-dependent
                            // intra-step prefill effect that needs phase-level
                            // modelling; not worth the complexity at this accuracy.
                            let gamma = match &self.trace_bank {
                                Some(bank) => spec.gamma.min(bank.max_depth),
                                None => spec.gamma,
                            };
                            // Homogeneous verify cost curve C(g) on the live
                            // decode batch -- all budget policies start here.
                            // When a measured cost table is present it is the
                            // ONLY price source: a draft depth with no
                            // measured rows is not a real candidate (pricing
                            // it via the optimistic roofline lets a fantasy
                            // width win the argmax), so it carries an
                            // INFINITY sentinel and is excluded. Without a
                            // table, the analytic roofline prices every
                            // depth as before. The table's `ref_seq_len`
                            // KV-read correction is added when set: it is
                            // width-independent, but a constant added to
                            // C(g) still moves the goodput-ratio argmax, so
                            // the policy must see the same prices the wall
                            // clock charges.
                            let kv_delta = match (ref_seq_len, &self.measured_cost) {
                                (Some(ref_len), Some(_)) => {
                                    w.compute_engine.kv_read_seq_delta_seconds(&dec, ref_len)
                                }
                                _ => 0.0,
                            };
                            let c_curve: Vec<f64> = (0..=gamma)
                                .map(|g| {
                                    if let Some(table) = &self.measured_cost {
                                        return match table.step_time(n as u32, g) {
                                            Some(t) => (t + kv_delta).max(0.25 * t),
                                            None => f64::INFINITY,
                                        };
                                    }
                                    let toks = vec![g + 1; n];
                                    w.compute_engine.calculate_iteration_time(&dec, &toks)
                                })
                                .collect();
                            // bf16 roofline rates for pricing the drafter on top
                            // of each verify cost C(g) (all budget policies).
                            let peak = w.compute_engine.bf16_peak_flops();
                            let bw = w.compute_engine.mem_bandwidth();
                            match (spec.policy, &self.trace_bank, &round_idx) {
                                (GammaPolicy::GatedBudget, Some(bank), Some(idx)) => {
                                    Self::gated_drafts_inner(
                                        bank, idx, &c_curve, gamma, &spec, peak, bw, n,
                                    )
                                }
                                (GammaPolicy::GatedAggregate, Some(bank), Some(idx)) => {
                                    let raw = Self::gated_aggregate_inner(
                                        bank, idx, &c_curve, gamma, &spec, peak, bw, n,
                                    )[0];
                                    // Engine switching constraints (no-op by
                                    // default): cooldown between
                                    // re-evaluations, bounded walk through
                                    // the measured candidate widths, and a
                                    // per-switch cost accrued onto the next
                                    // round's wall time.
                                    let g = if spec.switch.is_unconstrained() {
                                        raw
                                    } else {
                                        Self::constrained_aggregate_choice(
                                            &mut self.agg_switch,
                                            (pool, worker),
                                            raw,
                                            &c_curve,
                                            &spec.switch,
                                        )
                                    };
                                    vec![g; n]
                                }
                                _ => {
                                    // Homogeneous argmax (GoodputBudget; also the
                                    // GatedBudget fallback when there is no
                                    // per-sequence signal to gate on).
                                    let mut best_g = 0u32;
                                    let mut best_gp = f64::MIN;
                                    for g in 0..=gamma {
                                        let cv = c_curve[g as usize];
                                        let c = if cv.is_finite() {
                                            cv + spec.drafter_seconds(g, n as u32, peak, bw, cv)
                                        } else {
                                            cv
                                        };
                                        let exp = match &self.trace_bank {
                                            Some(bank) => bank.expected_accepted(g),
                                            None => spec.acceptance.expected_accepted(g),
                                        };
                                        let gp = (exp + 1.0) / c.max(1e-12);
                                        if gp > best_gp {
                                            best_gp = gp;
                                            best_g = g;
                                        }
                                    }
                                    vec![best_g; n]
                                }
                            }
                        }
                    }
                };
                (drafts, commits)
            };
            let mut k = 0usize;
            for req in w.scheduler.running_mut().iter_mut() {
                if !req.is_prefill() && !req.is_finished() {
                    req.pending_draft_len = drafts.get(k).copied().unwrap_or(0);
                    req.pending_round_commits = next_commits.get(k).copied().flatten();
                    k += 1;
                }
            }
            if k > 0 {
                let e = self
                    .spec_depth_buckets
                    .entry(end_time.max(0.0) as u64)
                    .or_insert((0, 0, 0));
                e.0 += drafts.iter().map(|&d| d as u64).sum::<u64>();
                e.1 += k as u64;
                e.2 += 1;
            }
        }

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

    /// Per-sequence confidence-gated draft allocation (`GammaPolicy::GatedBudget`).
    ///
    /// Each decode sequence has drawn a trace round whose per-depth estimated
    /// conditional acceptances are the gate's signal; the cumulative product is
    /// the estimated survival of each candidate draft slot. Slots are granted
    /// one at a time in descending estimated-survival order, while granting the
    /// slot improves expected committed tokens per unit step time. The step
    /// cost is interpolated from the homogeneous curve `c_curve` at the running
    /// mean verify width (the cost model is driven by total batch tokens, which
    /// a mean-width interpolation captures; the per-sequence attention split is
    /// second-order at these widths), and the drafter is charged on the widest
    /// draft present, as in the engine's timing. A slot rejected once is never
    /// reconsidered: the acceptance bar (current goodput) only rises and the
    /// marginal cost only grows as slots are added, so the greedy order is
    /// decision-complete.
    /// Per-sequence confidence-gated allocation. The drafter architecture decides
    /// what "ragged" means: an autoregressive head can ragged the *draft* (each
    /// sequence drafts to its own depth, the drafter sub-batch shrinking with
    /// depth), while a block head must draft one uniform block and can only ragged
    /// the *verify*. So we dispatch.
    fn gated_drafts_inner(
        bank: &crate::config::TraceBank,
        round_idx: &[usize],
        c_curve: &[f64],
        gamma: u32,
        spec: &crate::config::SpeculativeConfig,
        peak: f64,
        bw: f64,
        n: usize,
    ) -> Vec<u32> {
        if spec.drafter_is_block() {
            Self::gated_block_verify(bank, round_idx, c_curve, gamma, spec, peak, bw, n)
        } else {
            Self::gated_ragged_draft(bank, round_idx, c_curve, gamma, spec, peak, bw, n)
        }
    }

    /// Estimated survival chains: `surv[k][d]` = P(slots 0..=d all accepted) for
    /// sequence `k`, from its drawn round's per-depth confidences.
    fn survival_chains(
        bank: &crate::config::TraceBank,
        round_idx: &[usize],
        gamma: u32,
    ) -> Vec<Vec<f64>> {
        round_idx
            .iter()
            .map(|&i| {
                let mut s = 1.0;
                bank.rounds[i]
                    .a_hat
                    .iter()
                    .take(gamma as usize)
                    .map(|a| {
                        s *= a.clamp(0.0, 1.0);
                        s
                    })
                    .collect()
            })
            .collect()
    }

    /// Interpolate the homogeneous verify-cost curve at the mean verify width that
    /// `t` total verify tokens over `n` sequences implies.
    fn verify_cost(c_curve: &[f64], gamma: u32, n: usize, t: f64) -> f64 {
        let w_mean = (t / n as f64 - 1.0).clamp(0.0, gamma as f64);
        let seg = (w_mean.floor() as usize).min(gamma as usize - 1);
        let frac = w_mean - seg as f64;
        if frac > 0.0 {
            c_curve[seg] + frac * (c_curve[seg + 1] - c_curve[seg])
        } else {
            c_curve[seg]
        }
    }

    /// Batch-adapted draft depth `G`: the width the realizable (aggregate) gate
    /// would pick, maximising `(Σ_i E[accepted_i | g] + n) / cost(g)`. Choosing
    /// the depth globally amortises each pass's shared weight read; ragged
    /// allocation is then a refinement within `G`, not something a per-sequence
    /// greedy has to bootstrap a pass open for.
    fn aggregate_depth(
        surv: &[Vec<f64>],
        c_curve: &[f64],
        gamma: u32,
        spec: &crate::config::SpeculativeConfig,
        peak: f64,
        bw: f64,
        n: usize,
    ) -> u32 {
        let mut e_sum = vec![0.0f64; gamma as usize + 1];
        for chain in surv {
            let mut acc = 0.0;
            for d in 0..(gamma as usize).min(chain.len()) {
                acc += chain[d];
                e_sum[d + 1] += acc;
            }
        }
        let mut best_g = 0u32;
        let mut best_gp = f64::MIN;
        for g in 0..=gamma {
            let cv = c_curve[g as usize];
            if !cv.is_finite() {
                continue;
            }
            let cost = cv + spec.drafter_seconds(g, n as u32, peak, bw, cv);
            let gp = (e_sum[g as usize] + n as f64) / cost.max(1e-12);
            if gp > best_gp {
                best_gp = gp;
                best_g = g;
            }
        }
        best_g
    }

    /// Autoregressive (MTP-style) gating: ragged *draft* depth. Start every
    /// sequence at the global depth `G` (so the per-pass weight reads are already
    /// amortised), then greedily *remove* the least-confident deepest slots while
    /// that improves goodput. Removing the last sequence from a pass closes it and
    /// reclaims its read, which the removal correctly credits — the dual of the
    /// add-from-empty greedy that could never justify opening a pass.
    fn gated_ragged_draft(
        bank: &crate::config::TraceBank,
        round_idx: &[usize],
        c_curve: &[f64],
        gamma: u32,
        spec: &crate::config::SpeculativeConfig,
        peak: f64,
        bw: f64,
        n: usize,
    ) -> Vec<u32> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        let surv = Self::survival_chains(bank, round_idx, gamma);
        let gd = Self::aggregate_depth(&surv, c_curve, gamma, spec, peak, bw, n);
        if gd == 0 {
            return vec![0u32; n];
        }
        let pass = |m: u32| spec.drafter_pass_seconds(m, peak, bw);
        // Start every sequence at G (capped by its round's available signal).
        let mut g: Vec<u32> = (0..n)
            .map(|k| (gd as usize).min(surv[k].len()) as u32)
            .collect();
        let mut nk = vec![0u32; gd as usize]; // nk[d] = #sequences drafting pass d+1
        for &gi in &g {
            for d in 0..gi as usize {
                nk[d] += 1;
            }
        }
        let mut drafter: f64 = (0..gd as usize).map(|d| pass(nk[d])).sum();
        let mut expected: f64 = n as f64
            + (0..n)
                .map(|k| (0..g[k] as usize).map(|d| surv[k][d]).sum::<f64>())
                .sum::<f64>();
        let mut t: f64 = g.iter().map(|&gi| (gi + 1) as f64).sum();
        // min-heap on the deepest slot's survival: least confident removed first.
        let mut heap: BinaryHeap<(Reverse<u64>, usize)> = (0..n)
            .filter(|&k| g[k] > 0)
            .map(|k| (Reverse(surv[k][g[k] as usize - 1].to_bits()), k))
            .collect();
        while let Some((Reverse(sb), k)) = heap.pop() {
            let d = g[k] as usize;
            if d == 0 {
                continue;
            }
            let s = f64::from_bits(sb);
            let dmarg = pass(nk[d - 1]) - pass(nk[d - 1] - 1);
            let vc = Self::verify_cost(c_curve, gamma, n, t);
            let vp = Self::verify_cost(c_curve, gamma, n, t - 1.0);
            let cur = expected / (vc + drafter).max(1e-12);
            let rem = (expected - s) / (vp + drafter - dmarg).max(1e-12);
            if rem > cur {
                g[k] -= 1;
                nk[d - 1] -= 1;
                expected -= s;
                t -= 1.0;
                drafter -= dmarg;
                if g[k] > 0 {
                    heap.push((Reverse(surv[k][g[k] as usize - 1].to_bits()), k));
                }
            }
        }
        g
    }

    /// Block (DFlash-style) gating: the head drafts one uniform block, so the
    /// block depth `B` is a single global choice (the homogeneous goodput-optimal
    /// block), and only the *verify* is ragged within it. The drafter cost is the
    /// fixed `block(B, n)`, so the per-sequence verify greedy never pays a
    /// whole-batch deepening for an individual sequence.
    fn gated_block_verify(
        bank: &crate::config::TraceBank,
        round_idx: &[usize],
        c_curve: &[f64],
        gamma: u32,
        spec: &crate::config::SpeculativeConfig,
        peak: f64,
        bw: f64,
        n: usize,
    ) -> Vec<u32> {
        use std::collections::BinaryHeap;
        // Block depth B: the batch-adapted (aggregate) depth, so this rung sits at
        // or above the realizable gate and the only added freedom is ragged verify.
        let surv_full = Self::survival_chains(bank, round_idx, gamma);
        let b = Self::aggregate_depth(&surv_full, c_curve, gamma, spec, peak, bw, n);
        if b == 0 {
            return vec![0u32; n];
        }
        // Drafter is fixed: one block of depth B over the whole batch.
        let draft_const = spec.drafter_seconds(b, n as u32, peak, bw, c_curve[b as usize]);
        let surv: Vec<Vec<f64>> = surv_full
            .into_iter()
            .map(|mut c| {
                c.truncate(b as usize);
                c
            })
            .collect();
        let mut heap: BinaryHeap<(u64, usize)> = (0..n)
            .filter(|&k| !surv[k].is_empty())
            .map(|k| (surv[k][0].to_bits(), k))
            .collect();
        let mut v = vec![0u32; n]; // ragged verify widths, capped at B
        let mut expected = n as f64;
        let mut t = n as f64;
        while let Some((sb, k)) = heap.pop() {
            let s = f64::from_bits(sb);
            let vc = Self::verify_cost(c_curve, gamma, n, t) + draft_const;
            let vn = Self::verify_cost(c_curve, gamma, n, t + 1.0) + draft_const;
            if !vn.is_finite() {
                continue;
            }
            let cur = expected / vc.max(1e-12);
            let nxt = (expected + s) / vn.max(1e-12);
            if nxt > cur {
                v[k] += 1;
                expected += s;
                t += 1.0;
                if v[k] < b {
                    heap.push((surv[k][v[k] as usize].to_bits(), k));
                }
            } else {
                break; // drafter fixed: the bar only rises, so we're done
            }
        }
        v
    }

    /// Engine-realizable aggregated gate (`GammaPolicy::GatedAggregate`).
    ///
    /// Same per-sequence signal as `gated_drafts_inner` (each sequence's drawn
    /// round's estimated survival chain), but the verify width must be
    /// batch-uniform: one global `g` maximising
    /// `(Σ_i E[accepted_i | g] + n) / (C(g) · (1 + c_draft · g))`,
    /// where `E[accepted_i | g] = Σ_{d<g} surv_i[d]` and `C` is the same
    /// homogeneous cost curve the other budget policies price against. This
    /// models an engine that can read per-sequence acceptance signals but can
    /// only issue rectangular verify batches.
    fn gated_aggregate_inner(
        bank: &crate::config::TraceBank,
        round_idx: &[usize],
        c_curve: &[f64],
        gamma: u32,
        spec: &crate::config::SpeculativeConfig,
        peak: f64,
        bw: f64,
        n: usize,
    ) -> Vec<u32> {
        // e_sum[g] = Σ_i E[accepted_i | g], built from each sequence's
        // cumulative survival; depths beyond a round's signal are never
        // accepted (the curve goes flat), matching `AcceptanceModel`.
        let mut e_sum = vec![0.0f64; gamma as usize + 1];
        for &i in round_idx {
            let a_hat = &bank.rounds[i].a_hat;
            let mut surv = 1.0;
            let mut acc = 0.0;
            for d in 0..gamma as usize {
                surv *= a_hat.get(d).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                acc += surv;
                e_sum[d + 1] += acc;
            }
        }
        let mut best_g = 0u32;
        let mut best_gp = f64::MIN;
        for g in 0..=gamma {
            let cv = c_curve[g as usize];
            let c = if cv.is_finite() {
                cv + spec.drafter_seconds(g, n as u32, peak, bw, cv)
            } else {
                cv
            };
            let gp = (e_sum[g as usize] + n as f64) / c.max(1e-12);
            if gp > best_gp {
                best_gp = gp;
                best_g = g;
            }
        }
        vec![best_g; n]
    }

    /// Apply the engine's switching constraints to the aggregated gate's raw
    /// argmax `raw`. The candidate set is the measured widths (finite entries
    /// of `c_curve` — the same exclusion the argmax itself applies), sorted
    /// ascending. The first decision a worker ever makes is free (there is no
    /// previous width to persist); thereafter the width re-evaluates only
    /// every `switch.cooldown_rounds` decode rounds, each re-evaluation walks
    /// at most `switch.max_step` candidate indices toward `raw`, and a width
    /// change accrues `switch.cost_ms` onto the next round's wall time.
    fn constrained_aggregate_choice(
        state: &mut HashMap<(PoolId, usize), AggSwitchState>,
        key: (PoolId, usize),
        raw: u32,
        c_curve: &[f64],
        switch: &crate::config::SwitchConstraints,
    ) -> u32 {
        let st = match state.get_mut(&key) {
            Some(st) => st,
            None => {
                state.insert(
                    key,
                    AggSwitchState {
                        g: raw,
                        rounds_since: 0,
                        pending_cost: 0.0,
                    },
                );
                return raw;
            }
        };
        st.rounds_since += 1;
        if st.rounds_since >= switch.cooldown_rounds.max(1) {
            st.rounds_since = 0;
            let cands: Vec<u32> = (0..c_curve.len() as u32)
                .filter(|&g| c_curve[g as usize].is_finite())
                .collect();
            let new_g = Self::walk_candidates(&cands, st.g, raw, switch.max_step);
            if new_g != st.g {
                st.g = new_g;
                st.pending_cost += 1e-3 * switch.cost_ms;
            }
        }
        st.g
    }

    /// Move from `cur` toward `target` through the sorted candidate list,
    /// at most `max_step` indices (`None` = land on `target`). A `cur` that
    /// is no longer a candidate snaps to the nearest candidate index first.
    fn walk_candidates(cands: &[u32], cur: u32, target: u32, max_step: Option<u32>) -> u32 {
        if cands.is_empty() {
            return cur;
        }
        let cur_i = match cands.binary_search(&cur) {
            Ok(i) => i,
            Err(i) => i.min(cands.len() - 1),
        };
        let tgt_i = cands.binary_search(&target).unwrap_or(cur_i);
        let new_i = match max_step {
            None => tgt_i,
            Some(d) => {
                let d = d as usize;
                if tgt_i > cur_i {
                    (cur_i + d).min(tgt_i)
                } else {
                    cur_i.saturating_sub(d).max(tgt_i)
                }
            }
        };
        cands[new_i]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SwitchConstraints;

    #[test]
    fn walk_candidates_bounded_and_unbounded() {
        let cands = [0u32, 1, 2, 3, 4, 6, 8];
        // Unbounded: jump straight to the target.
        assert_eq!(Engine::walk_candidates(&cands, 0, 8, None), 8);
        // Bounded: at most two INDICES through the list, both directions.
        assert_eq!(Engine::walk_candidates(&cands, 0, 8, Some(2)), 2);
        assert_eq!(Engine::walk_candidates(&cands, 4, 8, Some(2)), 8); // 4 -> 6 -> 8
        assert_eq!(Engine::walk_candidates(&cands, 8, 0, Some(2)), 4);
        // At / near the target: clamp, never overshoot.
        assert_eq!(Engine::walk_candidates(&cands, 3, 3, Some(2)), 3);
        assert_eq!(Engine::walk_candidates(&cands, 3, 4, Some(2)), 4);
    }

    #[test]
    fn constrained_choice_cooldown_walk_and_cost() {
        // Finite cost at g in {0,1,2,3,4,6,8}; INF (unmeasured) at 5 and 7,
        // so the candidate walk skips them as single index moves.
        let mut c = vec![1.0f64; 9];
        c[5] = f64::INFINITY;
        c[7] = f64::INFINITY;
        let sw = SwitchConstraints {
            cooldown_rounds: 4,
            max_step: Some(2),
            cost_ms: 0.5,
        };
        let mut st: HashMap<(PoolId, usize), AggSwitchState> = HashMap::new();
        let key = (0usize, 0usize);
        // First decision is free (no previous width to persist).
        assert_eq!(
            Engine::constrained_aggregate_choice(&mut st, key, 0, &c, &sw),
            0
        );
        // Rounds 1..3 of the cooldown hold the width even as the argmax moves.
        for _ in 0..3 {
            assert_eq!(
                Engine::constrained_aggregate_choice(&mut st, key, 8, &c, &sw),
                0
            );
        }
        // Round 4 re-evaluates: walk two indices toward 8 -> g = 2; per-switch
        // cost accrued for the next round to pay.
        assert_eq!(
            Engine::constrained_aggregate_choice(&mut st, key, 8, &c, &sw),
            2
        );
        assert!((st[&key].pending_cost - 0.5e-3).abs() < 1e-12);
        // Hold, then 2 -> 4; hold, then 4 -> 8 (6 and 8 are one index each).
        for _ in 0..3 {
            assert_eq!(
                Engine::constrained_aggregate_choice(&mut st, key, 8, &c, &sw),
                2
            );
        }
        assert_eq!(
            Engine::constrained_aggregate_choice(&mut st, key, 8, &c, &sw),
            4
        );
        for _ in 0..3 {
            assert_eq!(
                Engine::constrained_aggregate_choice(&mut st, key, 8, &c, &sw),
                4
            );
        }
        assert_eq!(
            Engine::constrained_aggregate_choice(&mut st, key, 8, &c, &sw),
            8
        );
        // A re-evaluation that does not change the width accrues no cost.
        let before = st[&key].pending_cost;
        for _ in 0..4 {
            Engine::constrained_aggregate_choice(&mut st, key, 8, &c, &sw);
        }
        assert_eq!(st[&key].pending_cost, before);
    }
}
