//! Unified discrete-event engine. Pure state machine: knows about pools,
//! workers, the event heap, KV bookkeeping and time. Knows nothing about
//! request generation, metrics, real wall-clock, or how to render progress.
//! Drivers (batch [`super::simulator::Simulator`], serve `crate::serve::engine`)
//! pump it by alternating `next_event_time` / `submit` / `step`.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use super::spec::{DepthSample, PlanCosts, SpecPlanner};
use crate::compute::ComputeEngine;
use crate::config::{
    ClusterSpec, DisaggTopology, ModelSpec, RouterConfig, SchedulerConfig, SourcePolicy,
    SpeculativeConfig,
};
use crate::kv_cache::{
    KVCacheManager, MemoryGraph, Owner, PrefixCacheStats, SharedMemoryGraph, SharedRadix,
};
use crate::request::{Request, SessionStep};
use crate::router::{build_router, PrefixSignal, Router, RouterStats, WorkerSignal};
use crate::scheduler::{RecomputeFn, Scheduler};

pub type PoolId = usize;

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
    /// Time the first output token reached the client (= TTFT relative to
    /// arrival): the end of the pass that completed the prompt, or on a
    /// disaggregated topology the end of the KV hand-off, which carries it.
    pub first_token_time: f64,
    pub completion_time: f64,
    pub num_prompt_tokens: u32,
    pub num_output_tokens: u32,
    /// Prompt tokens served from the prefix cache at admission.
    pub num_cached_tokens: u32,
    /// Prompt-prefix tokens already resident on the selected decoder when
    /// the hand-off began (`None` in aggregated mode).
    pub decode_cached_tokens: Option<u32>,
    /// Session workloads: which step of which session this was.
    pub session: Option<Box<SessionStep>>,
    /// Memory-graph id of the worker that served the request.
    pub worker: Option<u32>,
    /// Times the request was preempted (and recomputed) before completing.
    pub num_preemptions: u32,
    /// Refused at submission (context larger than the worker's KV cache).
    pub rejected: bool,
    /// The prefix lookup that fixed `num_cached_tokens` (see
    /// [`crate::request::LookupRecord`]).
    pub lookup: Option<crate::request::LookupRecord>,
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
    /// This worker's id in the topology's memory graph (pools numbered in
    /// order: prefill workers, then decode workers; a DP-attention replica
    /// is `ranks` consecutive ids).
    pub global_id: usize,
    /// DP-attention: this worker's rank within its replica (0 for a whole
    /// replica). The ranks of a replica step in lockstep as one group (see
    /// `WorkerPool::groups`).
    pub rank: usize,
}

impl Worker {
    /// `graph` is the topology's KV memory graph and `global_id` this
    /// worker's id in it; the manager is attached to the graph only when
    /// the worker has tiers there. A DP-attention rank gets `1 / ranks` of
    /// the replica's KV capacity and the replica-wide compute engine.
    pub fn new(
        cluster: &ClusterSpec,
        model: ModelSpec,
        scheduler_config: SchedulerConfig,
        graph: &SharedMemoryGraph,
        global_id: usize,
        rank: usize,
        ranks: usize,
    ) -> Result<Self, String> {
        let kv_capacity = cluster
            .kv_cache_capacity(&scheduler_config, cluster.resident_weight_bytes(&model))
            / ranks.max(1) as u64;

        // Blocks are the model's content KV per block of tokens (the part
        // that grows for life: full-context layers, compressed history);
        // sliding windows and per-sequence state ride in auxiliary blocks.
        let policies = cluster.memory.policies();
        if let Some(bs) = scheduler_config.balance_set {
            bs.validate()?;
        }
        let mut kv_cache_manager =
            KVCacheManager::for_model(kv_capacity, scheduler_config.block_size, &model, true)
                .with_hbm_eviction(policies.hbm_eviction);
        if graph.lock().unwrap().num_tiers(global_id) > 0 {
            kv_cache_manager = kv_cache_manager.with_memory(graph.clone(), global_id);
        }
        if kv_cache_manager.total_blocks() == 0 {
            return Err(format!(
                "KV cache capacity ({} bytes) holds less than one {}-token block ({} bytes) of {}",
                kv_capacity,
                scheduler_config.block_size,
                kv_cache_manager.bytes_per_block(),
                model.name
            ));
        }

        let compute_engine = cluster
            .compute_engine(model.clone())
            .with_cascade_attention(
                scheduler_config.enable_cascade_attention,
                scheduler_config.block_size,
            );
        // `source = min_time` prices recomputing a tier-held prefix at the
        // worker's own roofline: a second engine over the same hardware.
        let recompute: Option<RecomputeFn> = match policies.source {
            SourcePolicy::Promote {} => None,
            SourcePolicy::MinTime {} => {
                let pricer = cluster.compute_engine(model);
                Some(Box::new(move |req: &Request, from: u32, tokens: u32| {
                    let mut probe = req.clone();
                    probe.num_computed_tokens = from;
                    probe.kv_blocks.clear();
                    pricer.calculate_iteration_time(&[&probe], &[tokens])
                }))
            }
        };
        let scheduler = Scheduler::new(scheduler_config.clone(), kv_cache_manager)
            .with_source(policies.source, recompute)
            .with_prefetch(policies.prefetch)
            .with_hicache(
                policies.promote_fill,
                policies.storage_prefetch,
                policies.load_overlap,
            );
        Ok(Self {
            scheduler,
            compute_engine,
            global_id,
            rank,
        })
    }

    /// An empty rank need not take the shared graph lock just to discover
    /// that [`Scheduler::schedule`] has nothing to do. A queued transfer
    /// completion is graph-owned state and therefore keeps the rank live.
    fn can_skip_schedule(&self, now: f64, completed_owners: &[Owner]) -> bool {
        !self.scheduler.has_local_work_for_schedule(now)
            && !completed_owners.contains(&Owner::Worker(self.global_id))
    }
}

pub(crate) struct WorkerPool {
    pub workers: Vec<Worker>,
    /// Picks the worker each request enters. Round-robin unless
    /// [`Topology::with_router`] set another. Routes flat over every
    /// worker, so on a DP-attention pool it places requests on ranks.
    router: Box<dyn Router>,
    router_stats: RouterStats,
    /// Tier-attached managers in a pool share one radix. Keep that handle
    /// and its radix-local worker ids so a KV-aware routing decision resolves
    /// the prompt path once, then reads each worker's view. Multi-worker
    /// HBM-only pools retain private trees and use the per-manager fallback.
    routing_radix: Option<SharedRadix>,
    worker_ids: Vec<usize>,
    /// Lockstep groups: the ranks of one replica, stepped together as one
    /// iteration (they meet at every layer's FFN collective). Each group
    /// lists its members' worker indices; the first is the leader that
    /// carries the group's events.
    groups: Vec<Vec<usize>>,
    group_of: Vec<usize>,
}

impl WorkerPool {
    pub fn new(workers: Vec<Worker>) -> Self {
        let n = workers.len();
        let worker_ids: Vec<usize> = workers
            .iter()
            .map(|w| w.scheduler.kv_cache_manager().radix_worker())
            .collect();
        let routing_radix = workers
            .first()
            .map(|w| w.scheduler.kv_cache_manager().radix().clone())
            .filter(|radix| {
                workers
                    .iter()
                    .all(|w| std::sync::Arc::ptr_eq(radix, w.scheduler.kv_cache_manager().radix()))
            });
        let mut groups: Vec<Vec<usize>> = Vec::new();
        let mut group_of = Vec::with_capacity(n);
        for (i, w) in workers.iter().enumerate() {
            if w.rank == 0 || groups.is_empty() {
                groups.push(Vec::new());
            }
            group_of.push(groups.len() - 1);
            groups.last_mut().unwrap().push(i);
        }
        Self {
            workers,
            router: build_router(&RouterConfig::RoundRobin {}),
            router_stats: RouterStats::new(n),
            routing_radix,
            worker_ids,
            groups,
            group_of,
        }
    }

    /// The worker that carries `worker`'s group's events.
    fn leader_of(&self, worker: usize) -> usize {
        self.groups[self.group_of[worker]][0]
    }

    /// The workers stepped together with `worker` (itself included).
    fn members_of(&self, worker: usize) -> &[usize] {
        &self.groups[self.group_of[worker]]
    }

    /// Whether any group has more than one rank.
    pub fn has_rank_groups(&self) -> bool {
        self.groups.iter().any(|g| g.len() > 1)
    }

    fn set_router(&mut self, cfg: &RouterConfig) {
        self.router = build_router(cfg);
        self.router_stats = RouterStats::new(self.workers.len());
    }

    /// Ask the router where `req` goes. Builds one signal per worker; the
    /// per-worker prefix estimate is only computed for routers that ask.
    fn pick(&mut self, req: &Request) -> usize {
        let prefix = self.router.prefix_signal();
        let hashes = &req.prompt_block_hashes;
        let shared_prefix_tokens: Option<Vec<u32>> = match prefix {
            PrefixSignal::None => None,
            _ if hashes.is_empty() => Some(vec![0; self.workers.len()]),
            PrefixSignal::Cached => self.routing_radix.as_ref().map(|radix| {
                radix
                    .lock()
                    .unwrap()
                    .lookup_workers(&self.worker_ids, hashes)
                    .into_iter()
                    .zip(&self.workers)
                    .map(|(lk, w)| lk.cached() * w.scheduler.kv_cache_manager().block_size())
                    .collect()
            }),
            PrefixSignal::Resident => self.routing_radix.as_ref().map(|radix| {
                radix
                    .lock()
                    .unwrap()
                    .resident_prefix_workers(&self.worker_ids, hashes)
                    .into_iter()
                    .zip(&self.workers)
                    .map(|(blocks, w)| blocks * w.scheduler.kv_cache_manager().block_size())
                    .collect()
            }),
        };
        let signals: Vec<WorkerSignal> = self
            .workers
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let sched = &w.scheduler;
                let mgr = sched.kv_cache_manager();
                WorkerSignal {
                    running: sched.num_running(),
                    waiting: sched.num_waiting(),
                    queued_prefill_tokens: sched.queued_prefill_tokens(),
                    kv_util: mgr.utilization(),
                    free_kv_tokens: mgr.num_free_blocks() as u64 * mgr.block_size() as u64,
                    cached_prefix_tokens: shared_prefix_tokens.as_ref().map_or_else(
                        || match prefix {
                            PrefixSignal::None => None,
                            PrefixSignal::Cached => Some(mgr.cached_prefix_tokens_estimate(hashes)),
                            PrefixSignal::Resident => Some(mgr.hbm_prefix_tokens(hashes)),
                        },
                        |tokens| Some(tokens[i]),
                    ),
                }
            })
            .collect();
        let idx = self.router.route(req, &signals).min(self.workers.len() - 1);
        self.router_stats.record(&signals, idx);
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
    Aggregated { pool: PoolId },
    Disagg { prefill: PoolId, decode: PoolId },
}

pub struct Topology {
    pub(crate) pools: Vec<WorkerPool>,
    pub(crate) roles: Roles,
    /// The KV memory graph every worker's tiers and every hand-off move
    /// over.
    pub(crate) memory: SharedMemoryGraph,
    model: ModelSpec,
}

impl Topology {
    pub fn aggregated(
        cluster: ClusterSpec,
        model: ModelSpec,
        scheduler_config: SchedulerConfig,
    ) -> Result<Self, String> {
        let memory = MemoryGraph::build(
            &[&cluster],
            scheduler_config.block_size,
            KVCacheManager::content_curve(&model, scheduler_config.block_size),
            None,
        )?
        .shared_handle();
        let workers = Self::build_pool(&cluster, &model, &scheduler_config, &memory, 0)?;
        Ok(Self {
            pools: vec![WorkerPool::new(workers)],
            roles: Roles::Aggregated { pool: 0 },
            memory,
            model,
        })
    }

    /// The workers of one pool, numbered `first_id..` in the memory graph.
    fn build_pool(
        cluster: &ClusterSpec,
        model: &ModelSpec,
        scheduler_config: &SchedulerConfig,
        memory: &SharedMemoryGraph,
        first_id: usize,
    ) -> Result<Vec<Worker>, String> {
        let n = cluster.num_workers.max(1) as usize;
        let ranks = cluster.kv_ranks() as usize;
        let mut workers = Vec::with_capacity(n * ranks);
        for w in 0..n {
            for r in 0..ranks {
                workers.push(Worker::new(
                    cluster,
                    model.clone(),
                    scheduler_config.clone(),
                    memory,
                    first_id + w * ranks + r,
                    r,
                    ranks,
                )?);
            }
        }
        Ok(workers)
    }

    pub fn from_disagg(
        topology: &DisaggTopology,
        model: ModelSpec,
        scheduler_config: SchedulerConfig,
    ) -> Result<Self, String> {
        let memory = MemoryGraph::build(
            &[&topology.prefill, &topology.decode],
            scheduler_config.block_size,
            KVCacheManager::content_curve(&model, scheduler_config.block_size),
            topology.kv_link_bw,
        )?
        .shared_handle();
        let p_count = topology.prefill.graph_workers().0;
        let p_workers = Self::build_pool(&topology.prefill, &model, &scheduler_config, &memory, 0)?;
        let d_workers = Self::build_pool(
            &topology.decode,
            &model,
            &scheduler_config,
            &memory,
            p_count,
        )?;
        // Every hand-off route must exist up front (a missing NIC template
        // and no `kv_link_bw` is a config error, not a run-time one).
        {
            let mut g = memory.lock().unwrap();
            let d_count = topology.decode.graph_workers().0;
            for p in 0..p_count {
                for d in 0..d_count {
                    g.handoff_path(p, p_count + d)?;
                }
            }
        }
        Ok(Self {
            pools: vec![WorkerPool::new(p_workers), WorkerPool::new(d_workers)],
            roles: Roles::Disagg {
                prefill: 0,
                decode: 1,
            },
            memory,
            model,
        })
    }

    /// The topology's memory graph.
    pub fn memory(&self) -> &SharedMemoryGraph {
        &self.memory
    }

    /// Front every pool with the router `cfg` names. On a disaggregated
    /// topology the same policy routes arrivals into the prefill pool and
    /// hand-offs into the decode pool; see `with_routers` to split them.
    pub fn with_router(mut self, cfg: &RouterConfig) -> Self {
        for pool in &mut self.pools {
            pool.set_router(cfg);
        }
        self
    }

    /// Front the pool arrivals enter with `entry` and, on a disaggregated
    /// topology, the decode pool with `decode`.
    pub fn with_routers(mut self, entry: &RouterConfig, decode: &RouterConfig) -> Self {
        let entry_pool = self.entry_pool();
        for (i, pool) in self.pools.iter_mut().enumerate() {
            pool.set_router(if i == entry_pool { entry } else { decode });
        }
        self
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
    /// The next transfer on the memory graph (a hand-off or a tier
    /// promotion) is due to complete under the rates in force when it was
    /// scheduled. `generation` invalidates events made stale by a later
    /// change to the in-flight set.
    FlowDrain {
        generation: u64,
    },
    /// A request's prefill finishes at this instant on the P pool: pick its
    /// decode worker and put its KV on the hand-off link. Its own event (not
    /// done inside the iteration that produced it) so the link is advanced
    /// to a time no later than any transfer's pending drain event.
    HandoffStart(Request),
    /// A worker's next prefetch plan is due: wake it if idle. Its own
    /// event (not a `WorkerReady` armed ahead of time) so an idle worker
    /// still wakes for arrivals in between.
    PrefetchDue {
        pool: PoolId,
        worker: usize,
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
    HandoffStart,
    PrefetchDue,
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
    /// the link, with the decode worker each was routed to when the transfer
    /// began.
    parked: HashMap<String, (Request, usize)>,
    /// Memory-graph id of the prefill worker each in-flight hand-off left.
    handoff_source: HashMap<String, usize>,
    /// Generation of the scheduled `FlowDrain` event and the time it is
    /// due; only the event carrying the current generation is acted on.
    flow_generation: u64,
    flow_due: Option<f64>,
    /// `worker_busy[pool][worker]` is true iff a `WorkerReady` for that
    /// worker is currently scheduled in the queue.
    worker_busy: Vec<Vec<bool>>,
    /// Per worker: the time of its armed `PrefetchDue` event, if any.
    prefetch_armed: Vec<Vec<Option<f64>>>,
    /// Hand-off transfers started, bytes moved, and bytes skipped because
    /// the chosen decoder already held the prompt prefix.
    handoff_stats: HandoffStats,
    /// Joint prefill/decode residency of reusable session prefixes, one
    /// counter set per decode worker.
    reusable_kv_by_decode_worker: Vec<ReusableKvStats>,
    /// Unique prompt positions computed by the prefill pool, partitioned by
    /// their role in the session turn.
    session_prefill_work: SessionPrefillWorkStats,
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
        let mut prefetch_armed = Vec::with_capacity(topology.pools.len());
        for pool in &topology.pools {
            worker_busy.push(vec![false; pool.workers.len()]);
            prefetch_armed.push(vec![None; pool.workers.len()]);
        }
        let pool_batch_acc = vec![(0.0_f64, 0.0_f64); topology.pools.len()];
        let decode_workers = match topology.roles {
            Roles::Disagg { decode, .. } => topology.pools[decode].workers.len(),
            Roles::Aggregated { .. } => 0,
        };
        Self {
            topology,
            events: BinaryHeap::new(),
            parked: HashMap::new(),
            handoff_source: HashMap::new(),
            flow_generation: 0,
            flow_due: None,
            worker_busy,
            prefetch_armed,
            handoff_stats: HandoffStats::default(),
            reusable_kv_by_decode_worker: vec![ReusableKvStats::default(); decode_workers],
            session_prefill_work: SessionPrefillWorkStats::default(),
            pool_batch_acc,
            current_time: 0.0,
            seq_counter: 0,
            spec: None,
            time_correction: None,
        }
    }

    /// Apply an affine empirical correction to every roofline-priced
    /// iteration time: `t = alpha * t_model + beta`. `alpha` captures the
    /// kernel-efficiency gap to the roofline (dominant at large batch);
    /// `beta` captures fixed per-iteration overhead — scheduler, CPU, launch
    /// latency (dominant at small batch). Applied to the actual step time
    /// only, never to policy candidate pricing (so speculative width choices
    /// stay model-relative) and never on top of a measured step-cost table.
    /// Config: `[hardware.<name>] time_correction = { alpha, beta }`.
    pub fn set_time_correction(&mut self, alpha: f64, beta: f64) {
        self.time_correction = Some((alpha, beta));
    }

    /// Enable speculative decoding. Decode steps then verify `gamma + 1` tokens
    /// (cost) and advance by `accepted + 1` (progress) per the acceptance model.
    /// Loads the trace bank / measured cost table the config names; a bad
    /// path is returned as an error.
    pub fn enable_speculative(&mut self, cfg: SpeculativeConfig, seed: u64) -> Result<(), String> {
        if self.topology.pools.iter().any(|p| p.has_rank_groups()) {
            return Err(
                "speculative decoding is not modelled with dp_attention rank groups (tp > 1 with dp_attention = true)"
                    .into(),
            );
        }
        self.spec = Some(SpecPlanner::new(cfg, seed)?);
        Ok(())
    }

    fn record_session_prefill_work(&mut self, req: &Request) {
        let Some(step) = req.session.as_ref() else {
            return;
        };
        let prompt = req.num_prompt_tokens;
        let shared = step.shared_tokens.min(prompt);
        let parent_prefill = step.shared_prefill_tokens.min(shared);
        let parent_decode_end = parent_prefill
            .saturating_add(step.shared_decode_tokens)
            .min(shared);
        let cached = req.num_cached_tokens.min(prompt);
        let cached_inherited = cached.min(shared);
        let amount = |start: u32, end: u32| KvAmountStats {
            tokens: end.saturating_sub(start) as u64,
            bytes: self
                .topology
                .model
                .kv_storage_bytes(end)
                .saturating_sub(self.topology.model.kv_storage_bytes(start)),
        };
        self.session_prefill_work += SessionPrefillWorkStats {
            new_prompt: amount(cached.max(shared), prompt),
            parent_prefill_recompute: amount(cached_inherited, parent_prefill),
            parent_decode_recompute: amount(
                cached_inherited.max(parent_prefill),
                parent_decode_end,
            ),
            unattributed_recompute: amount(cached_inherited.max(parent_decode_end), shared),
        };
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

    /// One line per worker holding work, for a stall report: what is
    /// waiting, parked, and free.
    pub fn describe_stuck_workers(&self) -> String {
        let mut out = String::new();
        for (p, pool) in self.topology.pools.iter().enumerate() {
            for (i, w) in pool.workers.iter().enumerate() {
                let s = &w.scheduler;
                if s.num_running() == 0 && s.num_waiting() == 0 {
                    continue;
                }
                let mgr = s.kv_cache_manager();
                let parked = s.pending_transfers();
                let first_parked = parked.first().map(|r| {
                    format!(
                        "{}(cached {}, ready_at {:?}, blocks {})",
                        r.request_id,
                        r.num_cached_tokens,
                        r.ready_at,
                        r.kv_blocks.len()
                    )
                });
                let first_waiting = s.waiting().front().map(|r| {
                    let l = mgr.peek_prefix_cache(r);
                    format!(
                        "{}(prompt {}, computed {}, blocks {}, needs {} blocks; lookup: cached {}, hbm {}, in_flight {}, promote {:?}, needs_staging {}, abandoned {}, ready_at {:?})",
                        r.request_id,
                        r.num_prompt_tokens,
                        r.num_computed_tokens,
                        r.kv_blocks.len(),
                        mgr.blocks_for_context(r.planned_positions()),
                        l.total_cached_tokens,
                        l.hbm_tokens,
                        l.in_flight_tokens,
                        l.promote_tokens_per_tier,
                        mgr.needs_staging(&l),
                        r.storage_prefetch_abandoned,
                        r.ready_at,
                    )
                });
                let staging = s.pending_storage();
                let first_staging = staging.first().map(|r| {
                    format!(
                        "{}(cached {}, ready_at {:?}, abandoned {}, lookups {:?})",
                        r.request_id,
                        r.num_cached_tokens,
                        r.ready_at,
                        r.storage_prefetch_abandoned,
                        r.lookup.as_ref().map(|l| l.lookups)
                    )
                });
                let (held, refs, free) = mgr.ref_summary();
                let by_queue = s.held_blocks_by_queue();
                out.push_str(&format!(
                    "\n  staging {}: [0] {:?}",
                    staging.len(),
                    first_staging
                ));
                out.push_str(&format!(
                    "\n  refs: {held} blocks referenced ({refs} refs), {free} free; held by (running, waiting, parked, prefetches) = {by_queue:?}"
                ));
                out.push_str(&format!(
                    "\n  pool {p} worker {i} (gid {}): running {}, waiting {}, parked {}, free {}/{} blocks, busy {}; parked[0] {:?}; waiting[0] {:?}",
                    w.global_id,
                    s.num_running(),
                    s.num_waiting(),
                    parked.len(),
                    mgr.num_free_blocks(),
                    mgr.total_blocks(),
                    self.worker_busy[p][pool.leader_of(i)],
                    first_parked,
                    first_waiting
                ));
            }
        }
        out
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

    /// Routing statistics for the pool arrivals enter (the prefill pool on
    /// a disaggregated topology).
    pub fn router_stats(&self) -> &RouterStats {
        &self.topology.pools[self.topology.entry_pool()].router_stats
    }

    /// Routing statistics per pool, in pool order (prefill then decode on a
    /// disaggregated topology).
    pub fn pool_router_stats(&self) -> Vec<&RouterStats> {
        self.topology
            .pools
            .iter()
            .map(|p| &p.router_stats)
            .collect()
    }

    /// Routing statistics for the decode pool of a disaggregated topology.
    pub fn decode_router_stats(&self) -> Option<&RouterStats> {
        match self.topology.roles {
            Roles::Disagg { decode, .. } => Some(&self.topology.pools[decode].router_stats),
            Roles::Aggregated { .. } => None,
        }
    }

    /// Hand-off transfer totals (zero on an aggregated topology).
    pub fn handoff_stats(&self) -> HandoffStats {
        self.handoff_stats
    }

    pub fn handoffs_in_flight(&self) -> usize {
        self.parked.len()
    }

    /// Joint reusable-prefix residency totals and per-decode-worker detail.
    pub fn reusable_kv_stats(
        &self,
    ) -> Option<(
        ReusableKvStats,
        Vec<RankReusableKvStats>,
        SessionPrefillWorkStats,
    )> {
        let decode = match self.topology.roles {
            Roles::Disagg { decode, .. } => decode,
            Roles::Aggregated { .. } => return None,
        };
        let mut total = ReusableKvStats::default();
        let ranks = self
            .topology
            .pools[decode]
            .workers
            .iter()
            .zip(&self.reusable_kv_by_decode_worker)
            .map(|(worker, stats)| {
                total += *stats;
                RankReusableKvStats {
                    worker: worker.global_id as u32,
                    rank: worker.rank as u32,
                    stats: *stats,
                }
            })
            .collect();
        Some((total, ranks, self.session_prefill_work))
    }

    /// HBM state and eviction accounting by topology pool and worker.
    pub fn hbm_pool_stats(&self) -> Vec<HbmPoolStats> {
        self.topology
            .pools
            .iter()
            .enumerate()
            .map(|(pool_id, pool)| {
                let role = match self.topology.role_for_pool(pool_id) {
                    PoolRole::Aggregated => "aggregated",
                    PoolRole::DisaggPrefill => "prefill",
                    PoolRole::DisaggDecode => "decode",
                }
                .to_string();
                let workers = pool
                    .workers
                    .iter()
                    .map(|worker| {
                        let manager = worker.scheduler.kv_cache_manager();
                        HbmWorkerStats {
                            worker: worker.global_id as u32,
                            rank: worker.rank as u32,
                            running: worker.scheduler.num_running() as u64,
                            waiting: worker.scheduler.num_waiting() as u64,
                            capacity_bytes: manager.capacity_bytes(),
                            resident_prefix_bytes: manager.resident_prefix_bytes(),
                            active_or_reserved_bytes: manager.active_or_reserved_bytes(),
                            eviction_events: manager.hbm_eviction_events(),
                            evicted_bytes: manager.hbm_evicted_bytes(),
                        }
                    })
                    .collect();
                HbmPoolStats { role, workers }
            })
            .collect()
    }

    /// The memory graph's stores and links over the run so far, `None`
    /// when no worker has tiers.
    pub fn memory_metrics(&self) -> Option<crate::metrics::MemoryMetrics> {
        let g = self.topology.memory.lock().unwrap();
        let tiered = (0..g.num_workers()).any(|w| g.num_tiers(w) > 0);
        if !tiered {
            return None;
        }
        let elapsed = self.current_time.max(f64::MIN_POSITIVE);
        let stores = g
            .store_totals()
            .into_iter()
            .map(|(name, t)| crate::metrics::StoreMetrics {
                name,
                instances: t.instances,
                capacity_blocks: t.capacity_blocks,
                held_blocks: t.held_blocks,
                bytes_written: t.bytes_written,
                bytes_read: t.bytes_read,
                dead_bytes: t.dead_bytes,
                evictions: t.evictions,
                expired: t.expired,
            })
            .collect();
        let links = g
            .edge_totals()
            .into_iter()
            .map(|(name, t)| crate::metrics::EdgeMetrics {
                name,
                instances: t.instances,
                capacity: t.capacity,
                bytes_moved: t.bytes_moved,
                utilisation: if t.capacity > 0.0 && t.instances > 0 {
                    // `instances` counts both directions of each link.
                    t.bytes_moved / (t.instances as f64 * t.capacity * elapsed)
                } else {
                    0.0
                },
            })
            .collect();
        let write_policy = g.write_policy(0).name().to_string();
        let eviction_policy = g
            .stores()
            .first()
            .map(|s| s.eviction.name().to_string())
            .unwrap_or_default();
        Some(crate::metrics::MemoryMetrics {
            write_policy,
            eviction_policy,
            stores,
            links,
            bytes_written: g.flows().bytes_submitted_write,
            bytes_promoted: g.flows().bytes_submitted_worker,
            write_race_waits: g.write_race_waits,
            peak_transfers_in_flight: g.flows().peak_in_flight() as u64,
            peer_hbm_bytes_promoted: g.peer_hbm_bytes_promoted,
            pin_stalls: g.pin_stalls(),
            partial_landings: g.partial_landings,
        })
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
    /// KV bytes written into HBM so far, summed over workers: every fresh
    /// block allocation, whether for computed or promoted content. Monotone;
    /// the difference between two readings is the distinct content that
    /// entered the caches in between (a reuse distance, in bytes).
    pub fn kv_bytes_written(&self) -> u64 {
        self.topology
            .pools
            .iter()
            .flat_map(|p| p.workers.iter())
            .map(|w| w.scheduler.kv_cache_manager().bytes_written())
            .sum()
    }

    /// `kv_bytes_written` plus the bytes of free (evictable) blocks that
    /// prefix hits pulled back into use, summed over workers. Fresh writes
    /// alone undercount the LRU stack distance: a hit moves its content to
    /// the recently-used end too. The two readings bracket it.
    pub fn kv_bytes_touched(&self) -> u64 {
        self.topology
            .pools
            .iter()
            .flat_map(|p| p.workers.iter())
            .map(|w| w.scheduler.kv_cache_manager().bytes_touched())
            .sum()
    }

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

        let outcome = match ev.kind {
            EventKind::Arrival(mut req) => {
                if let Some(step) = &mut req.session {
                    if let Some(at_parent) = step.parent_bytes_written {
                        step.reuse_distance_bytes =
                            Some(self.kv_bytes_written().saturating_sub(at_parent));
                    }
                    if let Some(at_parent) = step.parent_bytes_touched {
                        step.reuse_touched_bytes =
                            Some(self.kv_bytes_touched().saturating_sub(at_parent));
                    }
                }
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
            EventKind::FlowDrain { generation } => {
                if generation == self.flow_generation {
                    self.flow_due = None;
                }
                Ok(StepOutcome {
                    time: self.current_time,
                    kind: StepKind::LinkComplete,
                    iteration: None,
                    completions: Vec::new(),
                })
            }
            EventKind::PrefetchDue { pool, worker } => {
                self.prefetch_armed[pool][worker] = None;
                let now = self.current_time;
                self.maybe_wake_worker(pool, worker, now);
                Ok(StepOutcome {
                    time: self.current_time,
                    kind: StepKind::PrefetchDue,
                    iteration: None,
                    completions: Vec::new(),
                })
            }
            EventKind::HandoffStart(req) => {
                self.start_handoff(req)?;
                Ok(StepOutcome {
                    time: self.current_time,
                    kind: StepKind::HandoffStart,
                    iteration: None,
                    completions: Vec::new(),
                })
            }
        };
        // Whatever the event was, move the memory graph's transfers to now,
        // hand finished hand-offs to their decode workers, wake workers with
        // finished promotions, and re-arm the drain event.
        self.pump_flows()?;
        outcome
    }

    /// Bring the memory graph to `now`; deliver completed hand-offs; wake
    /// workers whose promotions completed (their next `schedule()` collects
    /// them); (re)schedule the `FlowDrain` for the next completion.
    fn pump_flows(&mut self) -> Result<(), String> {
        let now = self.current_time;
        let (owners, next) = {
            let mut g = self.topology.memory.lock().unwrap();
            g.advance(now);
            (g.owners_with_completions(), g.next_completion_delay())
        };
        for owner in owners {
            match owner {
                Owner::Handoff => {
                    let done = self
                        .topology
                        .memory
                        .lock()
                        .unwrap()
                        .take_completed(Owner::Handoff);
                    if let Some(done) = done {
                        self.deliver_drained(done, now)?;
                    }
                }
                Owner::Worker(w) => {
                    if let Some((pool, idx)) = self.locate_worker(w) {
                        self.maybe_wake_worker(pool, idx, now);
                    }
                }
                // Writes land inside the graph's own `advance`.
                Owner::Write => {}
            }
        }
        match next {
            Some(delay) => {
                let due = now + delay;
                if self.flow_due.is_none_or(|t| (t - due).abs() > 1e-12) {
                    self.flow_generation += 1;
                    self.flow_due = Some(due);
                    self.push(
                        due,
                        EventKind::FlowDrain {
                            generation: self.flow_generation,
                        },
                    );
                }
            }
            None => {
                if self.flow_due.is_some() {
                    self.flow_generation += 1;
                    self.flow_due = None;
                }
            }
        }
        Ok(())
    }

    /// `(pool, index)` of the worker with memory-graph id `global`.
    fn locate_worker(&self, global: usize) -> Option<(PoolId, usize)> {
        for (p, pool) in self.topology.pools.iter().enumerate() {
            if let Some(i) = pool.workers.iter().position(|w| w.global_id == global) {
                return Some((p, i));
            }
        }
        None
    }

    fn handle_arrival(&mut self, req: Request) {
        let entry = self.topology.entry_pool();
        self.route_into_pool(entry, req);
    }

    fn route_into_pool(&mut self, pool_id: PoolId, req: Request) {
        let worker_idx = self.topology.pools[pool_id].pick(&req);
        self.deliver_to_worker(pool_id, worker_idx, req);
    }

    fn deliver_to_worker(&mut self, pool_id: PoolId, worker_idx: usize, mut req: Request) {
        req.worker = Some(self.topology.pools[pool_id].workers[worker_idx].global_id as u32);
        self.topology.pools[pool_id].workers[worker_idx]
            .scheduler
            .add_request_at(req, self.current_time);
        self.maybe_wake_worker(pool_id, worker_idx, self.current_time);
    }

    /// Wake `worker`'s lockstep group (through its leader) at `when`,
    /// unless it is already busy or armed.
    fn maybe_wake_worker(&mut self, pool: PoolId, worker: usize, when: f64) {
        let leader = self.topology.pools[pool].leader_of(worker);
        if !self.worker_busy[pool][leader] {
            self.worker_busy[pool][leader] = true;
            self.push(
                when,
                EventKind::WorkerReady {
                    pool,
                    worker: leader,
                },
            );
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
        // ran *during* this step, so they start at `end_time`: each gets its
        // own event there (queued before this worker's next `WorkerReady`,
        // so it fires first at that instant).
        let mut timings = Vec::with_capacity(outcome.completed.len());
        for req in outcome.completed {
            if matches!(role, PoolRole::DisaggPrefill) {
                self.record_session_prefill_work(&req);
            }
            timings.push(self.finalise(req, now));
        }
        let handoff_time = outcome
            .iteration
            .as_ref()
            .map(|i| i.end_time)
            .unwrap_or(now);
        for (source, req) in outcome.handed_off {
            self.handoff_source.insert(req.request_id.clone(), source);
            self.push(handoff_time, EventKind::HandoffStart(req));
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
            let p = &self.topology.pools[pool];
            let has_work = p.members_of(worker).iter().any(|&m| {
                let sch = &p.workers[m].scheduler;
                sch.num_running() > 0 || sch.num_waiting() > 0
            });
            if has_work {
                self.maybe_wake_worker(pool, worker, now);
            }
        }
        // Requests parked on a KV promotion need no timer: the memory
        // graph's `FlowDrain` wakes this group when the transfer completes.
        // Arm each member's next prefetch plan as its own event, so it
        // fires whether or not the group is busy then.
        let members: Vec<usize> = self.topology.pools[pool].members_of(worker).to_vec();
        for m in members {
            if let Some(t) = self.topology.pools[pool].workers[m]
                .scheduler
                .next_prefetch_at()
            {
                if t > now && self.prefetch_armed[pool][m] != Some(t) {
                    self.prefetch_armed[pool][m] = Some(t);
                    self.push(t, EventKind::PrefetchDue { pool, worker: m });
                }
            }
        }
        (outcome.iteration, timings)
    }

    /// One iteration of `worker`'s lockstep group (a whole replica, or the
    /// `tp` ranks of a DP-attention replica stepped together). Members with
    /// no local work or queued transfer completion are skipped;
    /// the remaining schedules form a union batch priced on the replica-wide
    /// roofline, plus — with more than one rank — the attention skew: the
    /// slowest rank's own-GPU attention time over the mean rank's, since
    /// the ranks meet at every layer's FFN collective.
    fn run_iteration(
        &mut self,
        pool: PoolId,
        worker: usize,
        role: PoolRole,
        now: f64,
    ) -> RunIterationOutcome {
        let correction = self.time_correction;
        let members: Vec<usize> = self.topology.pools[pool].members_of(worker).to_vec();
        let grouped = members.len() > 1;

        // Every member's schedule used to advance the shared graph before
        // collecting its own completions. Advance once up front so a
        // completion landing exactly at `now` is visible before deciding an
        // otherwise-empty rank can be skipped; inspect without draining so
        // the live rank's schedule remains the sole collector.
        let completed_owners = if grouped {
            let mut graph = self.topology.memory.lock().unwrap();
            graph.advance(now);
            graph.owners_with_completions()
        } else {
            Vec::new()
        };

        // Schedule each live member; the batch is the union, each entry
        // tagged with its member and index into that member's running set.
        let mut completed = Vec::new();
        let mut preempted = false;
        let mut entries: Vec<(usize, usize, u32)> = Vec::new();
        let mut active_members = Vec::new();
        for &m in &members {
            let worker = &mut self.topology.pools[pool].workers[m];
            if grouped && worker.can_skip_schedule(now, &completed_owners) {
                continue;
            }
            let decision = worker.scheduler.schedule(now);
            if !decision.batch.is_empty() {
                active_members.push(m);
            }
            completed.extend(decision.completed);
            preempted |= decision.num_preempted > 0;
            entries.extend(decision.batch.iter().map(|s| (m, s.idx, s.num_tokens)));
        }
        if entries.is_empty() {
            return RunIterationOutcome {
                iteration: None,
                completed,
                handed_off: Vec::new(),
                preempted,
            };
        }
        let batch_size = entries.len();
        let tokens_per_request: Vec<u32> = entries.iter().map(|e| e.2).collect();

        // Capture per-request progress (and was_prefill) before mutating.
        let mut progress = Vec::with_capacity(batch_size);
        let mut round_commits: Vec<Option<u32>> = Vec::with_capacity(batch_size);
        {
            let ws = &self.topology.pools[pool].workers;
            for &(m, idx, tokens) in &entries {
                let req = &ws[m].scheduler.running()[idx];
                progress.push(RequestProgress {
                    request_id: req.request_id.clone(),
                    was_prefill: req.is_prefill(),
                    num_tokens: tokens,
                    num_output: 0,
                });
                round_commits.push(req.pending_round_commits);
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
        // never speculated. Not modelled across rank groups.
        let cost_tokens = tokens_per_request.clone(); // verify width per request
        let mut accepted_extra = vec![0u32; batch_size];
        let mut draft_widths: Vec<u32> = Vec::new(); // decode sequences only
        if self.spec.is_some() && !grouped {
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
            let ws = &self.topology.pools[pool].workers;
            let batch_refs: Vec<&Request> = entries
                .iter()
                .map(|&(m, idx, _)| &ws[m].scheduler.running()[idx])
                .collect();
            let was_prefill: Vec<bool> = progress.iter().map(|p| p.was_prefill).collect();
            let ce = &ws[worker].compute_engine;
            let (mut t, measured, bw_util, fl_util) = Self::price_step(
                self.spec.as_ref(),
                ce,
                &batch_refs,
                &cost_tokens,
                &was_prefill,
                correction,
            );
            if grouped {
                // Attention skew across the ranks: max own-GPU attention
                // time minus the mean the union pricing already charged.
                // Empty ranks contribute exactly zero. The common one-live-
                // rank case can reuse the union vectors and avoid building
                // and filtering one pair of vectors per replica rank.
                let (max, sum) = if active_members.len() == 1 {
                    let attention = ce.attention_seconds_on_one_gpu(&batch_refs, &cost_tokens);
                    (attention, attention)
                } else {
                    let mut max = 0.0_f64;
                    let mut sum = 0.0_f64;
                    for &m in &active_members {
                        let refs: Vec<&Request> = entries
                            .iter()
                            .filter(|e| e.0 == m)
                            .map(|&(_, idx, _)| &ws[m].scheduler.running()[idx])
                            .collect();
                        let toks: Vec<u32> =
                            entries.iter().filter(|e| e.0 == m).map(|e| e.2).collect();
                        let attention = ce.attention_seconds_on_one_gpu(&refs, &toks);
                        max = max.max(attention);
                        sum += attention;
                    }
                    (max, sum)
                };
                let mean = sum / members.len() as f64;
                t += (max - mean).max(0.0);
            }
            (t, measured, bw_util, fl_util)
        };
        if let Some(spec) = &mut self.spec {
            if !grouped {
                let ce = &self.topology.pools[pool].workers[worker].compute_engine;
                // Drafter overhead on roofline-priced speculated steps.
                // Table-priced steps skip this — the measured wall gap
                // already embodies the full engine step, drafter included.
                if !measured && draft_widths.iter().any(|&d| d > 0) {
                    let peak = ce.bf16_peak_flops();
                    let bw = ce.mem_bandwidth();
                    iter_time += spec.drafter_seconds(&draft_widths, peak, bw, iter_time);
                }
                // Constrained-GatedAggregate per-switch stall: a width change
                // decided at the end of the previous round costs the engine a
                // rebuild on the first round executed at the new width.
                iter_time += spec.take_pending_switch_cost((pool, worker));
            }
        }
        let end_time = now + iter_time;

        for (j, &(m, idx, tokens)) in entries.iter().enumerate() {
            // Decode: advance by the verified tokens (bonus + accepted), NOT
            // the verify width (`num_tokens` = 1 + draft, the cost). Prefill
            // (including chunked continuations and recompute after
            // preemption): advance by the scheduled chunk.
            let adv = if progress[j].was_prefill {
                tokens
            } else {
                1 + accepted_extra[j]
            };
            progress[j].num_output = self.topology.pools[pool].workers[m]
                .scheduler
                .record_progress(idx, adv, end_time);
        }

        let mut handed_off = Vec::new();
        if matches!(role, PoolRole::DisaggPrefill) {
            // Anything whose prefill is now complete leaves its worker via
            // the link; its KV stays allocated here until the transfer
            // drains (`deliver_drained`).
            for &m in &members {
                let w = &mut self.topology.pools[pool].workers[m];
                let gid = w.global_id;
                handed_off.extend(
                    w.scheduler
                        .take_prefill_complete()
                        .into_iter()
                        .map(|r| (gid, r)),
                );
            }
        }

        // Decide the decode batch's draft depth for its NEXT step. Drafting
        // happens here, at the end of the step -- the one instant when the
        // drafter is about to run AND the carry-over decode set is known.
        // The next scheduler pass reads `pending_draft_len` and reserves
        // `1 + draft` of budget + KV.
        if let (Some(spec), false) = (&mut self.spec, grouped) {
            let w = &mut self.topology.pools[pool].workers[worker];
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
        // A measured table already embodies the engine's overheads; the
        // correction calibrates roofline-priced steps only.
        let mut iter_time = match (measured_time, correction) {
            (Some(t), _) => t,
            (None, Some((alpha, beta))) => alpha * cost.time + beta,
            (None, None) => cost.time,
        };
        let overlap_credit = batch_refs
            .iter()
            .zip(was_prefill)
            .filter(|(_, prefill)| **prefill)
            .map(|(request, _)| request.load_overlap_credit)
            .fold(0.0_f64, f64::max);
        if overlap_credit > 0.0 {
            let pre_idx: Vec<usize> = (0..batch_size).filter(|&i| was_prefill[i]).collect();
            let pre_refs: Vec<&Request> = pre_idx.iter().map(|&i| batch_refs[i]).collect();
            let pre_tokens: Vec<u32> = pre_idx.iter().map(|&i| cost_tokens[i]).collect();
            let pre_time = ce.calculate_iteration_time(&pre_refs, &pre_tokens);
            let dec_idx: Vec<usize> = (0..batch_size).filter(|&i| !was_prefill[i]).collect();
            let decode_floor = if dec_idx.is_empty() {
                0.0
            } else {
                let refs: Vec<&Request> = dec_idx.iter().map(|&i| batch_refs[i]).collect();
                let tokens: Vec<u32> = dec_idx.iter().map(|&i| cost_tokens[i]).collect();
                ce.calculate_iteration_time(&refs, &tokens)
            };
            iter_time = (iter_time - overlap_credit.min(pre_time)).max(decode_floor);
        }
        let bw = ce.bandwidth_utilization(&cost, iter_time);
        let flops = ce.flops_utilization(&cost, iter_time);
        (iter_time, measured_time.is_some(), bw, flops)
    }

    /// A request's prefill on the P pool finishes now. Pick its decode
    /// worker (the router sees the decode pool's load and, for the KV-aware
    /// policies, which decoders already hold part of this context), size the
    /// transfer at the context minus the prompt prefix that decoder already
    /// has resident, and start it on the hand-off link. The request is
    /// delivered to that worker when the transfer drains.
    fn start_handoff(&mut self, mut req: Request) -> Result<(), String> {
        self.record_session_prefill_work(&req);
        let prefill_done_at = self.current_time;
        req.prefill_done_time = Some(prefill_done_at);
        let decode = match self.topology.roles {
            Roles::Disagg { decode, .. } => decode,
            _ => return Err("hand-off on an aggregated topology".to_string()),
        };
        let worker_idx = self.topology.pools[decode].pick(&req);
        let resident = self.topology.pools[decode].workers[worker_idx]
            .scheduler
            .kv_cache_manager()
            .hbm_prefix_tokens(&req.prompt_block_hashes)
            .min(req.num_computed_tokens);
        req.decode_cached_tokens = Some(resident);
        if let Some(step) = req.session.as_ref() {
            let shared = step.shared_tokens.min(req.num_computed_tokens);
            let parent_prefill = step.shared_prefill_tokens.min(shared);
            let parent_decode = step.shared_decode_tokens.min(shared);
            let prefill = req
                .lookup
                .map(|lookup| lookup.hbm_tokens + lookup.tier_tokens)
                .unwrap_or(req.num_cached_tokens)
                .min(req.num_computed_tokens);
            let decoder = resident.min(shared);
            self.reusable_kv_by_decode_worker[worker_idx] += classify_reusable_kv(
                &self.topology.model,
                ReusableKvObservation {
                    prompt: req.num_computed_tokens,
                    shared,
                    prefill,
                    decoder,
                    parent_prefill,
                    parent_decode,
                },
            );
        }
        let full_bytes = self
            .topology
            .model
            .kv_storage_bytes(req.num_computed_tokens);
        let kv_bytes = full_bytes.saturating_sub(self.topology.model.kv_storage_bytes(resident));
        self.handoff_stats.transfers += 1;
        self.handoff_stats.bytes += kv_bytes;
        self.handoff_stats.bytes_skipped += full_bytes - kv_bytes;
        let id = req.request_id.clone();
        let from = self
            .handoff_source
            .get(&id)
            .copied()
            .ok_or_else(|| format!("hand-off of {id} has no source worker"))?;
        let to = self.topology.pools[decode].workers[worker_idx].global_id;
        self.topology.memory.lock().unwrap().submit_handoff(
            &id,
            from,
            to,
            kv_bytes,
            prefill_done_at,
        )?;
        self.parked.insert(id, (req, worker_idx));
        Ok(())
    }

    /// Deliver drained hand-offs, in a deterministic order, each to the
    /// decode worker chosen when its transfer began.
    fn deliver_drained(
        &mut self,
        done: std::collections::HashSet<String>,
        now: f64,
    ) -> Result<(), String> {
        let decode_pool = match self.topology.roles {
            Roles::Disagg { decode, .. } => decode,
            _ => return Err("hand-off drain on an aggregated topology".to_string()),
        };
        let mut done: Vec<String> = done.into_iter().collect();
        done.sort();
        for request_id in done {
            let (mut req, worker_idx) = self
                .parked
                .remove(&request_id)
                .ok_or_else(|| format!("hand-off complete for unknown request {request_id}"))?;
            // The prefill worker held the request's KV for the transfer;
            // release it there now (hashes stay hittable).
            let source = self
                .handoff_source
                .remove(&request_id)
                .ok_or_else(|| format!("hand-off of {request_id} has no source worker"))?;
            let (sp, si) = self
                .locate_worker(source)
                .ok_or_else(|| format!("hand-off source worker {source} not found"))?;
            self.topology.pools[sp].workers[si]
                .scheduler
                .release_handed_off(&mut req);
            // Those blocks may be what its queue was waiting for; nothing
            // else wakes it (the next arrival would, some time later).
            if self.topology.pools[sp].workers[si].scheduler.num_waiting() > 0 {
                self.maybe_wake_worker(sp, si, now);
            }
            req.handoff_done_time = Some(now);
            // The first token travels with the KV: the decode side emits it
            // on receipt (Dynamo / sglang PD), so that is when the client
            // sees it. Decode-side TPOT then runs from here.
            req.first_token_time = Some(now);
            self.deliver_to_worker(decode_pool, worker_idx, req);
        }
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
            num_cached_tokens: req.num_cached_tokens,
            decode_cached_tokens: req.decode_cached_tokens,
            session: req.session,
            worker: req.worker,
            num_preemptions: req.num_preemptions,
            rejected: req.rejected,
            lookup: req.lookup,
        }
    }
}

/// Hand-off transfer totals over the run.
#[derive(Debug, Clone, Copy, Default)]
pub struct HandoffStats {
    pub transfers: u64,
    /// Bytes put on the hand-off link.
    pub bytes: u64,
    /// Bytes not transferred because the chosen decoder already held that
    /// prompt prefix in HBM.
    pub bytes_skipped: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct KvAmountStats {
    pub tokens: u64,
    pub bytes: u64,
}

impl std::ops::AddAssign for KvAmountStats {
    fn add_assign(&mut self, other: Self) {
        self.tokens += other.tokens;
        self.bytes += other.bytes;
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ReusableKvStats {
    pub requests: u64,
    pub reusable: KvAmountStats,
    pub both: KvAmountStats,
    pub decoder_only: KvAmountStats,
    pub prefill_only: KvAmountStats,
    pub neither: KvAmountStats,
    pub prefiller_miss_parent_prefill: KvAmountStats,
    pub prefiller_miss_parent_decode: KvAmountStats,
    pub prefiller_miss_unattributed: KvAmountStats,
}

impl std::ops::AddAssign for ReusableKvStats {
    fn add_assign(&mut self, other: Self) {
        self.requests += other.requests;
        self.reusable += other.reusable;
        self.both += other.both;
        self.decoder_only += other.decoder_only;
        self.prefill_only += other.prefill_only;
        self.neither += other.neither;
        self.prefiller_miss_parent_prefill += other.prefiller_miss_parent_prefill;
        self.prefiller_miss_parent_decode += other.prefiller_miss_parent_decode;
        self.prefiller_miss_unattributed += other.prefiller_miss_unattributed;
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RankReusableKvStats {
    pub worker: u32,
    pub rank: u32,
    pub stats: ReusableKvStats,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SessionPrefillWorkStats {
    pub new_prompt: KvAmountStats,
    pub parent_prefill_recompute: KvAmountStats,
    pub parent_decode_recompute: KvAmountStats,
    pub unattributed_recompute: KvAmountStats,
}

impl std::ops::AddAssign for SessionPrefillWorkStats {
    fn add_assign(&mut self, other: Self) {
        self.new_prompt += other.new_prompt;
        self.parent_prefill_recompute += other.parent_prefill_recompute;
        self.parent_decode_recompute += other.parent_decode_recompute;
        self.unattributed_recompute += other.unattributed_recompute;
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HbmWorkerStats {
    pub worker: u32,
    pub rank: u32,
    pub running: u64,
    pub waiting: u64,
    pub capacity_bytes: u64,
    pub resident_prefix_bytes: u64,
    pub active_or_reserved_bytes: u64,
    pub eviction_events: u64,
    pub evicted_bytes: u64,
}

#[derive(Debug, Clone, Default)]
pub struct HbmPoolStats {
    pub role: String,
    pub workers: Vec<HbmWorkerStats>,
}

#[derive(Debug, Clone, Copy)]
struct ReusableKvObservation {
    prompt: u32,
    shared: u32,
    prefill: u32,
    decoder: u32,
    parent_prefill: u32,
    parent_decode: u32,
}

fn classify_reusable_kv(model: &ModelSpec, observation: ReusableKvObservation) -> ReusableKvStats {
    let ReusableKvObservation {
        prompt,
        shared,
        prefill,
        decoder,
        parent_prefill,
        parent_decode,
    } = observation;
    let prompt = prompt.max(shared);
    let prefill_total = prefill.min(prompt);
    let prefill = prefill_total.min(shared);
    let decoder = decoder.min(shared);
    let parent_prefill = parent_prefill.min(shared);
    let parent_decode_end = parent_prefill.saturating_add(parent_decode).min(shared);
    let lo = prefill.min(decoder);
    let hi = prefill.max(decoder);
    let amount = |start: u32, end: u32| KvAmountStats {
        tokens: end.saturating_sub(start) as u64,
        bytes: model
            .kv_storage_bytes(end)
            .saturating_sub(model.kv_storage_bytes(start)),
    };
    let mut stats = ReusableKvStats {
        requests: u64::from(shared > 0),
        reusable: amount(0, shared),
        both: amount(0, lo),
        neither: amount(hi, shared),
        prefiller_miss_parent_prefill: amount(prefill, parent_prefill),
        prefiller_miss_parent_decode: amount(prefill.max(parent_prefill), parent_decode_end),
        prefiller_miss_unattributed: amount(prefill.max(parent_decode_end), shared),
        ..Default::default()
    };
    if decoder > prefill {
        stats.decoder_only = amount(lo, hi);
    } else {
        stats.prefill_only = amount(lo, hi);
    }
    stats
}

struct RunIterationOutcome {
    iteration: Option<IterationInfo>,
    completed: Vec<Request>,
    /// Requests whose prefill finished, with the memory-graph id of the
    /// worker they leave.
    handed_off: Vec<(usize, Request)>,
    /// The scheduler preempted at least one request this pass. Relevant when
    /// `iteration` is `None`: state changed even though nothing ran, so the
    /// worker must be re-armed rather than left to wait for a new arrival.
    preempted: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, FabricConfig, FabricLink};

    #[test]
    fn reusable_kv_classification_is_a_complete_joint_partition() {
        let model = Config::test_default().model;
        let observe = |prompt, shared, prefill, decoder, parent_prefill, parent_decode| {
            ReusableKvObservation {
                prompt,
                shared,
                prefill,
                decoder,
                parent_prefill,
                parent_decode,
            }
        };
        let p_longer = classify_reusable_kv(&model, observe(125, 100, 70, 40, 80, 20));
        assert_eq!(p_longer.requests, 1);
        assert_eq!(p_longer.reusable.tokens, 100);
        assert_eq!(p_longer.both.tokens, 40);
        assert_eq!(p_longer.prefill_only.tokens, 30);
        assert_eq!(p_longer.decoder_only.tokens, 0);
        assert_eq!(p_longer.neither.tokens, 30);
        assert_eq!(p_longer.prefiller_miss_parent_prefill.tokens, 10);
        assert_eq!(p_longer.prefiller_miss_parent_decode.tokens, 20);
        assert_eq!(p_longer.prefiller_miss_unattributed.tokens, 0);
        assert_eq!(
            p_longer.reusable.bytes,
            p_longer.both.bytes
                + p_longer.prefill_only.bytes
                + p_longer.decoder_only.bytes
                + p_longer.neither.bytes
        );
        assert_eq!(
            p_longer.decoder_only.bytes + p_longer.neither.bytes,
            p_longer.prefiller_miss_parent_prefill.bytes
                + p_longer.prefiller_miss_parent_decode.bytes
                + p_longer.prefiller_miss_unattributed.bytes
        );

        let d_longer = classify_reusable_kv(&model, observe(125, 100, 30, 80, 60, 40));
        assert_eq!(d_longer.both.tokens, 30);
        assert_eq!(d_longer.decoder_only.tokens, 50);
        assert_eq!(d_longer.prefill_only.tokens, 0);
        assert_eq!(d_longer.neither.tokens, 20);
        assert_eq!(d_longer.prefiller_miss_parent_prefill.tokens, 30);
        assert_eq!(d_longer.prefiller_miss_parent_decode.tokens, 40);
        assert_eq!(d_longer.prefiller_miss_unattributed.tokens, 0);

        let inherited = classify_reusable_kv(&model, observe(125, 100, 0, 0, 0, 0));
        assert_eq!(inherited.prefiller_miss_unattributed.tokens, 100);

    }

    /// One replica of `tp` GPUs with a fabric, DP-attention on or off.
    fn cluster(tp: u32, dp_attention: bool) -> (ClusterSpec, ModelSpec, SchedulerConfig) {
        let base = Config::test_default();
        let mut cluster = base.cluster();
        cluster.hardware.fabric = Some(FabricConfig {
            gpus_per_node: 8,
            scale_up: FabricLink {
                bandwidth: 1e12,
                latency: 1e-6,
                in_network_reduction: false,
            },
            scale_out: None,
        });
        cluster.parallel.tp = tp;
        cluster.parallel.dp_attention = dp_attention;
        cluster.num_workers = 1;
        (cluster, base.model.clone(), base.scheduler.clone())
    }

    fn engine(tp: u32, dp_attention: bool) -> Engine {
        let (c, m, s) = cluster(tp, dp_attention);
        Engine::new(Topology::aggregated(c, m, s).unwrap())
    }

    /// Run until idle; return every iteration in order.
    fn run(engine: &mut Engine) -> Vec<IterationInfo> {
        let mut iters = Vec::new();
        while !engine.is_idle() {
            if let Some(it) = engine.step().unwrap().iteration {
                iters.push(it);
            }
        }
        iters
    }

    #[test]
    fn dp_attention_replica_is_tp_rank_workers_stepped_as_one_group() {
        let e = engine(2, true);
        let pool = &e.topology.pools[0];
        assert_eq!(pool.workers.len(), 2, "one worker per rank");
        assert_eq!(pool.groups, vec![vec![0, 1]]);
        assert!(pool.has_rank_groups());
        assert_eq!(pool.workers[0].global_id, 0);
        assert_eq!(pool.workers[1].global_id, 1);
        // Each rank holds half the replica's KV.
        let (c, m, s) = cluster(2, true);
        let whole = c.kv_cache_capacity(&s, c.resident_weight_bytes(&m));
        let per_block = m.kv_storage_bytes(s.block_size);
        let rank_blocks = pool.workers[0].scheduler.kv_cache_manager().total_blocks() as u64;
        assert_eq!(rank_blocks, whole / 2 / per_block);
        // Without DP-attention the same tp is one worker with all of it.
        let e1 = engine(2, false);
        assert_eq!(e1.topology.pools[0].workers.len(), 1);
        assert!(!e1.topology.pools[0].has_rank_groups());
        assert_eq!(
            e1.topology.pools[0].workers[0]
                .scheduler
                .kv_cache_manager()
                .total_blocks() as u64,
            whole / per_block
        );
    }

    #[test]
    fn idle_rank_skip_guard_includes_graph_completions() {
        let e = engine(2, true);
        let rank = &e.topology.pools[0].workers[1];
        assert!(rank.can_skip_schedule(0.0, &[]));
        assert!(!rank.can_skip_schedule(0.0, &[Owner::Worker(rank.global_id)]));
    }

    #[test]
    fn ranks_route_flat_and_an_iteration_covers_every_rank() {
        let mut e = engine(2, true);
        for i in 0..4 {
            e.submit(Request::new(format!("r{i}"), 0, 0.0, 64, 4));
        }
        let iters = run(&mut e);
        // Round-robin over ranks: 2 each.
        assert_eq!(e.router_stats().per_worker, vec![2, 2]);
        // The first iteration is one group step carrying all four prefills,
        // reported under the leader.
        let first = &iters[0];
        assert_eq!(first.worker, 0);
        assert_eq!(first.batch_size, 4);
        assert!(
            iters.iter().all(|i| i.worker == 0),
            "events ride the leader"
        );
    }

    #[test]
    fn a_lopsided_rank_pays_the_attention_skew() {
        // One long prefill on a 2-rank replica lands on one rank: the step
        // costs the union price plus (max − mean) of the ranks' own-GPU
        // attention time = a/2 for one loaded rank and one idle one.
        let mut e = engine(2, true);
        e.submit(Request::new("long".into(), 0, 0.0, 4096, 1));
        let iters = run(&mut e);
        let step = &iters[0];
        assert_eq!(step.batch_size, 1);
        // The first step is the first prefill chunk (chunked prefill).
        let chunk = step.progress[0].num_tokens;
        let (c, m, _) = cluster(2, true);
        let ce = c.compute_engine(m);
        let req = Request::new("probe".into(), 0, 0.0, 4096, 1);
        let union = ce.calculate_iteration_time(&[&req], &[chunk]);
        let a = ce.attention_seconds_on_one_gpu(&[&req], &[chunk]);
        assert!(a > 0.0);
        assert!(
            (step.iteration_time - (union + a / 2.0)).abs() < 1e-12,
            "{} vs {} + {}",
            step.iteration_time,
            union,
            a
        );
        // Two equal prefills, one per rank: no skew — the union price alone.
        let mut e = engine(2, true);
        e.submit(Request::new("a".into(), 0, 0.0, 4096, 1));
        e.submit(Request::new("b".into(), 0, 0.0, 4096, 1));
        let iters = run(&mut e);
        let step = &iters[0];
        assert_eq!(step.batch_size, 2);
        let (ca, cb) = (step.progress[0].num_tokens, step.progress[1].num_tokens);
        assert_eq!(ca, cb);
        let r2 = Request::new("p".into(), 0, 0.0, 4096, 1);
        let union2 = ce.calculate_iteration_time(&[&r2, &r2], &[ca, cb]);
        assert!((step.iteration_time - union2).abs() < 1e-12);
    }

    #[test]
    fn layerwise_load_credit_prices_load_and_prefill_as_a_pipeline() {
        let (cluster, model, _) = cluster(1, false);
        let ce = cluster.compute_engine(model);
        let mut request = Request::new("prefill".into(), 0, 0.0, 1024, 1);
        let tokens = 512;
        let compute = ce.calculate_iteration_time(&[&request], &[tokens]);

        let load = compute * 0.4;
        request.load_overlap_credit = load;
        let (after_load, ..) = Engine::price_step(None, &ce, &[&request], &[tokens], &[true], None);
        assert!((load + after_load - compute).abs() < 1e-12);

        let load = compute * 2.0;
        request.load_overlap_credit = load;
        let (after_load, ..) = Engine::price_step(None, &ce, &[&request], &[tokens], &[true], None);
        assert!(after_load.abs() < 1e-12);
        assert!((load + after_load - load.max(compute)).abs() < 1e-12);
    }
}
