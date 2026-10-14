//! The engine seams a fleet simulator drives: workers added, drained and
//! killed at run time; a caller's router fed by engine notifications and KV
//! events; and a linear step-cost model.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use inference_lab::config::{
    ClusterSpec, DisaggTopology, FabricConfig, FabricLink, HardwareConfig, LayerClass,
    LinearStepCost, MemoryConfig, ModelSpec, ParallelConfig, Precision, RouterConfig,
    SchedulerConfig, StepCostModel, WeightStream,
};
use inference_lab::request::Request;
use inference_lab::router::{Router, WorkerSignal};
use inference_lab::scheduler::SchedulingPolicy;
use inference_lab::simulation::{
    Engine, FailureReason, RequestTiming, StepKind, StepOutcome, Topology, WorkerEvent, WorkerId,
    WorkerState,
};

fn model() -> ModelSpec {
    ModelSpec {
        name: "test-dense".into(),
        hidden_dim: 1024,
        max_seq_len: 4096,
        attention_precision: Precision::Bf16,
        activation_bytes: 2,
        weights: vec![WeightStream {
            scale: None,
            precision: Precision::Bf16,
            active_params: 1_000_000_000,
            resident_params: 1_000_000_000,
            routing: None,
        }],
        layers: vec![LayerClass::Attention {
            count: 8,
            heads: 8,
            head_dim: 128,
            kv_heads: 8,
            kv_shared: false,
            window: 0,
            kv_precision: Precision::Bf16,
        }],
    }
}

fn scheduler(kv_blocks: u64, chunked: bool) -> SchedulerConfig {
    SchedulerConfig {
        max_num_batched_tokens: if chunked { 256 } else { 8192 },
        max_num_seqs: 16,
        max_waiting: 0,
        enable_chunked_prefill: chunked,
        mix_prefill_with_decode: true,
        overlap_schedule: false,
        prefill_delayer_max_passes: 0,
        idle_pass_s: 0.0,
        long_prefill_decode_interval: 0,
        long_prefill_min_tokens: 262_144,
        long_prefill_token_threshold: 0,
        max_num_partial_prefills: 1,
        block_size: 16,
        gpu_memory_utilization: 0.9,
        kv_cache_capacity: kv_blocks * model().kv_storage_bytes(16),
        max_model_len: None,
        policy: SchedulingPolicy::FCFS,
        enable_preemption_free: false,
        enable_cascade_attention: false,
        balance_set: None,
    }
}

fn cluster(num_workers: u32) -> ClusterSpec {
    let mut c: ClusterSpec = toml::from_str("hardware = \"b200\"").unwrap();
    c.hardware = HardwareConfig {
        name: "test".into(),
        flops_fp4: None,
        flops_fp8: None,
        flops_bf16: Some(1e15),
        flops_fp16: Some(1e15),
        memory_bandwidth: 1e12,
        memory_capacity: 80_000_000_000,
        memory: None,
        fabric: None,
    };
    c.parallel = ParallelConfig::default();
    c.num_workers = num_workers;
    c.memory = MemoryConfig::default();
    c
}

/// One GPU per node and a topology-wide store behind the network, whose
/// shared aggregate is its per-node bandwidth times the nodes.
fn cluster_tiered(mut c: ClusterSpec) -> ClusterSpec {
    const MEM: &str = r#"
gpus_per_node = 1
[[stores]]
name = "cluster_nvme"
per = "cluster"
capacity = 1e12
bandwidth = 2e9
stripe = 1
[[links]]
name = "nic"
from = "gpu"
to = "network"
bandwidth = 1e12
[[links]]
name = "core"
from = "network"
to = "cluster_nvme"
bandwidth = 1e12
"#;
    c.hardware.memory = Some(toml::from_str(MEM).unwrap());
    c.memory = toml::from_str("tiers = [\"cluster_nvme\"]").unwrap();
    c
}

/// `c` as DP-attention replicas of `tp` ranks, on 8-GPU nodes unless its
/// memory template says otherwise.
fn dp(mut c: ClusterSpec, tp: u32) -> ClusterSpec {
    c.parallel = ParallelConfig {
        tp,
        ep: 1,
        dp_attention: true,
    };
    c.hardware.fabric = Some(FabricConfig {
        gpus_per_node: 8,
        scale_up: FabricLink {
            bandwidth: 4e11,
            latency: 2e-6,
            in_network_reduction: false,
        },
        scale_out: None,
    });
    c
}

/// Two GPUs per node, a node-shared host store and NVLink to the node's
/// switch; `tiers` picks which of `host` and `peer_hbm` the workers use.
fn tiered_with(mut c: ClusterSpec, tiers: &str) -> ClusterSpec {
    const MEM: &str = r#"
gpus_per_node = 2
[[stores]]
name = "host"
per = "node"
capacity = 1e9
[[stores]]
name = "peer_hbm"
per = "node"
kind = "peer_hbm"
[[links]]
name = "pcie"
from = "gpu"
to = "host"
bandwidth = 5e10
[[links]]
name = "nvlink"
from = "gpu"
to = "switch"
bandwidth = 4e11
"#;
    c.hardware.memory = Some(toml::from_str(MEM).unwrap());
    c.memory = toml::from_str(&format!("tiers = {tiers}")).unwrap();
    c
}

/// A request with block hashes: `shared` blocks of `session`'s prefix,
/// then `own` blocks of its own.
fn hashed(id: &str, t: f64, session: u64, shared: u64, own: u64, output: u32) -> Request {
    let mut seq: Vec<u64> = (1..=shared).map(|b| session * 1_000 + b).collect();
    let tag = id.bytes().fold(0u64, |h, b| h * 131 + b as u64);
    seq.extend((0..own).map(|b| (1 << 40) + tag * 64 + b));
    let mut r = Request::new(id.into(), 0, t, seq.len() as u32 * 16 - 3, output);
    // Local hashes: content alone, distinct from the cumulative ones.
    r.prompt_local_block_hashes = seq.iter().map(|h| h.rotate_left(17) ^ 0x5bd1).collect();
    r.prompt_block_hashes = seq;
    r
}

/// Six sessions, growing prefixes, bursty arrivals.
fn sessions(n: u64) -> Vec<Request> {
    sessions_of(6, n)
}

/// `k` sessions taking turns, `n` requests in all.
fn sessions_of(k: u64, n: u64) -> Vec<Request> {
    (0..n)
        .map(|i| {
            let (s, turn) = (i % k, i / k);
            hashed(
                &format!("r{i:03}"),
                i as f64 * 0.004,
                s,
                4 + turn,
                2,
                4 + (i as u32 * 7) % 23,
            )
        })
        .collect()
}

/// Step until idle, collecting every outcome.
fn run(engine: &mut Engine) -> Vec<StepOutcome> {
    let mut out = Vec::new();
    while !engine.is_idle() {
        out.push(engine.step().unwrap());
    }
    out
}

fn completions(outcomes: &[StepOutcome]) -> Vec<RequestTiming> {
    let mut done: Vec<RequestTiming> = outcomes
        .iter()
        .flat_map(|o| o.completions.iter().cloned())
        .collect();
    done.sort_by(|a, b| a.request_id.cmp(&b.request_id));
    done
}

fn worker_events(outcomes: &[StepOutcome]) -> Vec<(f64, WorkerEvent)> {
    outcomes
        .iter()
        .filter_map(|o| o.worker_event.map(|e| (o.time, e)))
        .collect()
}

/// Everything a router can be told, as it was told.
#[derive(Debug, Clone, PartialEq)]
enum Call {
    /// A replica and its number of ranks.
    Added(WorkerId, u32, f64),
    Removed(WorkerId, f64),
    Routed(String, WorkerId, f64),
    Prefilled(String, WorkerId, f64),
    Completed(String, WorkerId, f64),
}

/// A shadow of each worker's prefix cache, rebuilt from KV events alone the
/// way an external router's index would be: sequence hash → (parent, local).
#[derive(Debug, Default)]
struct Shadow {
    blocks: HashMap<WorkerId, HashMap<u64, (Option<u64>, u64)>>,
    stored: u64,
    removed: u64,
    /// Every local hash seen for a sequence hash.
    locals: HashMap<u64, u64>,
}

#[derive(Debug, Default)]
struct Log {
    calls: Vec<Call>,
    shadow: Shadow,
    /// Each joined replica's ranks, as the router heard them.
    replicas: HashMap<WorkerId, u32>,
    /// `now_s` of every call, in call order.
    times: Vec<f64>,
}

/// Least-loaded routing that records every notification and replays KV
/// events into its shadow index.
struct Recorder {
    log: Arc<Mutex<Log>>,
    kv_events: bool,
    pick: Pick,
}

/// Where a recorder sends a request.
type Pick = fn(&Request, &[WorkerSignal]) -> WorkerId;

fn least_loaded(_req: &Request, workers: &[WorkerSignal]) -> WorkerId {
    workers
        .iter()
        .min_by_key(|w| (w.in_system(), w.queued_prefill_tokens, w.worker))
        .unwrap()
        .worker
}

impl Recorder {
    fn boxed(kv_events: bool) -> (Box<dyn Router>, Arc<Mutex<Log>>) {
        Self::picking(kv_events, least_loaded)
    }

    fn picking(kv_events: bool, pick: Pick) -> (Box<dyn Router>, Arc<Mutex<Log>>) {
        let log = Arc::new(Mutex::new(Log::default()));
        let router = Recorder {
            log: log.clone(),
            kv_events,
            pick,
        };
        (Box::new(router), log)
    }

    fn note(&self, call: Option<Call>, now: f64) {
        let mut log = self.log.lock().unwrap();
        log.times.push(now);
        if let Some(c) = call {
            log.calls.push(c);
        }
    }
}

impl Router for Recorder {
    fn wants_kv_events(&self) -> bool {
        self.kv_events
    }

    fn route(&mut self, req: &Request, workers: &[WorkerSignal], now_s: f64) -> WorkerId {
        self.note(None, now_s);
        (self.pick)(req, workers)
    }

    fn on_worker_added(&mut self, worker: WorkerId, ranks: u32, now_s: f64) {
        self.note(Some(Call::Added(worker, ranks, now_s)), now_s);
        let mut log = self.log.lock().unwrap();
        log.replicas.insert(worker, ranks);
        for rank in 0..ranks {
            log.shadow
                .blocks
                .entry(WorkerId(worker.0 + rank))
                .or_default();
        }
    }

    fn on_worker_removed(&mut self, worker: WorkerId, now_s: f64) {
        self.note(Some(Call::Removed(worker, now_s)), now_s);
        let mut log = self.log.lock().unwrap();
        let ranks = log
            .replicas
            .remove(&worker)
            .expect("a replica it heard join");
        for rank in 0..ranks {
            log.shadow.blocks.remove(&WorkerId(worker.0 + rank));
        }
    }

    fn on_routed(&mut self, req: &Request, worker: WorkerId, now_s: f64) {
        self.note(
            Some(Call::Routed(req.request_id.clone(), worker, now_s)),
            now_s,
        );
    }

    fn on_prefill_complete(&mut self, request_id: &str, worker: WorkerId, now_s: f64) {
        self.note(
            Some(Call::Prefilled(request_id.into(), worker, now_s)),
            now_s,
        );
    }

    fn on_request_complete(&mut self, request_id: &str, worker: WorkerId, now_s: f64) {
        self.note(
            Some(Call::Completed(request_id.into(), worker, now_s)),
            now_s,
        );
    }

    fn on_kv_stored(
        &mut self,
        worker: WorkerId,
        parent: Option<u64>,
        blocks: &[(u64, u64)],
        now_s: f64,
    ) {
        self.note(None, now_s);
        let mut log = self.log.lock().unwrap();
        let shadow = &mut log.shadow;
        let index = shadow.blocks.get_mut(&worker).expect("a joined worker");
        if let Some(p) = parent {
            assert!(index.contains_key(&p), "stored under an unknown parent");
        }
        let mut prev = parent;
        for &(local, seq) in blocks {
            assert!(
                index.insert(seq, (prev, local)).is_none(),
                "{worker} stored block {seq:x} twice"
            );
            let seen = *shadow.locals.entry(seq).or_insert(local);
            assert_eq!(seen, local, "one sequence hash, two local hashes");
            prev = Some(seq);
        }
        shadow.stored += blocks.len() as u64;
    }

    fn on_kv_removed(&mut self, worker: WorkerId, hashes: &[u64], now_s: f64) {
        self.note(None, now_s);
        let mut log = self.log.lock().unwrap();
        let index = log.shadow.blocks.get_mut(&worker).expect("a joined worker");
        for h in hashes {
            assert!(
                index.remove(h).is_some(),
                "{worker} removed block {h:x} it never stored"
            );
        }
        log.shadow.removed += hashes.len() as u64;
    }
}

/// The shadow's view of every rank of every joined replica equals that
/// rank's cache.
fn assert_shadow_matches(engine: &Engine, log: &Arc<Mutex<Log>>, pool: usize) {
    let log = log.lock().unwrap();
    for (replica, state) in engine.pool_members(pool) {
        if state == WorkerState::Starting {
            continue;
        }
        for id in engine.replica_ranks(replica).unwrap() {
            let mut shadow: Vec<u64> = log.shadow.blocks[&id].keys().copied().collect();
            shadow.sort_unstable();
            assert_eq!(
                shadow,
                engine.worker_cached_blocks(id).unwrap(),
                "{id}: shadow index diverged from the cache at t={}",
                engine.current_time()
            );
        }
    }
}

// ----------------------------------------------------------------------
// Runtime lifecycle

#[test]
fn an_added_worker_joins_routing_only_once_ready() {
    let mut e = Engine::new(
        Topology::aggregated(cluster(1), model(), scheduler(512, true))
            .unwrap()
            .with_router(&RouterConfig::RoundRobin {}),
    );
    let pool = e.entry_pool();
    let added = e.add_worker(pool, 1.0).unwrap();
    assert_eq!(added, WorkerId(1), "ids continue after the fixed workers");
    assert_eq!(
        e.pool_members(pool),
        vec![
            (WorkerId(0), WorkerState::Active),
            (added, WorkerState::Starting)
        ]
    );
    for i in 0..8 {
        e.submit(Request::new(format!("r{i}"), 0, i as f64 * 0.25, 64, 4));
    }
    let outcomes = run(&mut e);
    assert_eq!(
        worker_events(&outcomes),
        vec![(
            1.0,
            WorkerEvent::Joined {
                pool,
                worker: added
            }
        )]
    );
    let joined = outcomes
        .iter()
        .find(|o| o.kind == StepKind::WorkerJoined)
        .unwrap();
    assert_eq!(joined.time, 1.0);
    for t in completions(&outcomes) {
        let expect = if t.arrival_time < 1.0 {
            vec![0]
        } else {
            vec![0, 1]
        };
        assert!(expect.contains(&t.worker.unwrap()), "{t:?}");
    }
    // Round-robin over both once the new worker is in.
    assert_eq!(e.router_stats().per_worker, vec![6, 2]);
    assert_eq!(e.worker_state(added), Some(WorkerState::Active));
}

/// A worker added at t = 0 is the same worker a config would have built:
/// one fixed worker plus one added gives the run two fixed ones give, KV
/// tiers and node placement included.
#[test]
fn a_runtime_worker_runs_exactly_like_a_configured_one() {
    let fingerprint = |e: &mut Engine, k: u64| {
        for r in sessions_of(k, 90) {
            e.submit(r);
        }
        completions(&run(e))
            .iter()
            .map(|t| {
                (
                    t.request_id.clone(),
                    t.first_token_time.to_bits(),
                    t.completion_time.to_bits(),
                    t.num_cached_tokens,
                    t.worker,
                )
            })
            .collect::<Vec<_>>()
    };
    let kv_aware = RouterConfig::KvAware { load_weight: 1.0 };
    for (tiers, router, k) in [
        (None, &kv_aware, 6),
        (Some("[\"host\"]"), &kv_aware, 6),
        // Round-robin alternates each of five sessions between the
        // workers, so prefixes move GPU to GPU: the added worker must be
        // its sibling's peer.
        (
            Some("[\"peer_hbm\", \"host\"]"),
            &RouterConfig::RoundRobin {},
            5,
        ),
        // One GPU per node: the added worker opens a node, and the cluster
        // store's shared aggregate grows with it.
        (Some("cluster"), &RouterConfig::RoundRobin {}, 5),
    ] {
        let build = |n| {
            let c = match tiers {
                Some("cluster") => cluster_tiered(cluster(n)),
                Some(t) => tiered_with(cluster(n), t),
                None => cluster(n),
            };
            Topology::aggregated(c, model(), scheduler(64, true))
                .unwrap()
                .with_router(router)
        };
        let mut fixed = Engine::new(build(2));
        let mut grown = Engine::new(build(1));
        assert_eq!(grown.add_worker(0, 0.0).unwrap(), WorkerId(1));
        let a = fingerprint(&mut fixed, k);
        let b = fingerprint(&mut grown, k);
        assert_eq!(a.len(), 90);
        assert_eq!(a, b, "tiers = {tiers:?}");
        if let Some(t) = tiers {
            let m = fixed.memory_metrics().unwrap();
            assert!(m.bytes_promoted > 0.0);
            if t.contains("peer") {
                assert!(m.peer_hbm_bytes_promoted > 0, "{t}");
            }
        }
    }
}

#[test]
fn a_drained_worker_finishes_its_work_then_leaves() {
    let (router, log) = Recorder::boxed(false);
    let mut e = Engine::new(
        Topology::aggregated(cluster(2), model(), scheduler(512, true))
            .unwrap()
            .with_pool_router(0, router),
    );
    for i in 0..6 {
        e.submit(Request::new(format!("early{i}"), 0, 0.0, 200, 40));
    }
    for i in 0..6 {
        e.submit(Request::new(format!("late{i}"), 0, 0.05, 200, 40));
    }
    // Run until both workers are busy, then drain worker 1.
    while e.current_time() < 0.01 {
        e.step().unwrap();
    }
    e.drain_worker(0, WorkerId(1)).unwrap();
    assert_eq!(e.worker_state(WorkerId(1)), Some(WorkerState::Draining));
    let drained_at = e.current_time();
    let outcomes = run(&mut e);
    let done = completions(&outcomes);
    assert_eq!(done.len(), 12, "everything finished");
    for t in &done {
        if t.request_id.starts_with("late") {
            assert_eq!(t.worker, Some(0), "routed after the drain: {t:?}");
        }
    }
    let on_one: Vec<_> = done.iter().filter(|t| t.worker == Some(1)).collect();
    assert_eq!(on_one.len(), 3, "its in-flight work finished there");
    let last = on_one.iter().map(|t| t.completion_time).fold(0.0, f64::max);
    let events = worker_events(&outcomes);
    assert_eq!(
        events,
        vec![(
            last,
            WorkerEvent::Removed {
                pool: 0,
                worker: WorkerId(1),
                killed: false
            }
        )]
    );
    assert!(last > drained_at);
    assert_eq!(e.pool_members(0), vec![(WorkerId(0), WorkerState::Active)]);
    assert_eq!(e.worker_state(WorkerId(1)), None);
    // The router hears it leave, and nothing of it after.
    let calls = log.lock().unwrap().calls.clone();
    let gone = calls
        .iter()
        .position(|c| *c == Call::Removed(WorkerId(1), last))
        .expect("told of the removal");
    assert!(calls[gone + 1..].iter().all(|c| match c {
        Call::Routed(_, w, _) | Call::Prefilled(_, w, _) | Call::Completed(_, w, _) =>
            *w != WorkerId(1),
        _ => true,
    }));
    // An idle worker drains at once.
    e.drain_worker(0, WorkerId(0)).unwrap();
    let removed = e.step().unwrap();
    assert_eq!(
        removed.worker_event,
        Some(WorkerEvent::Removed {
            pool: 0,
            worker: WorkerId(0),
            killed: false
        })
    );
    // With nobody left, arrivals are handed back.
    e.submit(Request::new("orphan".into(), 0, e.current_time(), 64, 4));
    let out = e.step().unwrap();
    assert_eq!(out.failed.len(), 1);
    assert_eq!(out.failed[0].reason, FailureReason::NoWorker);
    assert_eq!(out.failed[0].worker, None);
    assert!(e.is_idle());
}

#[test]
fn killed_requests_come_back_for_resubmission_and_finish_elsewhere() {
    let mut e = Engine::new(
        Topology::aggregated(cluster(2), model(), scheduler(512, true))
            .unwrap()
            .with_router(&RouterConfig::RoundRobin {}),
    );
    for i in 0..10 {
        e.submit(Request::new(format!("r{i}"), 0, 0.0, 300, 60));
    }
    // Let decoding get under way, then kill worker 1.
    let mut first = Vec::new();
    while e.current_time() < 0.02 {
        first.extend(e.step().unwrap().completions);
    }
    assert!(first.is_empty());
    e.kill_worker(0, WorkerId(1)).unwrap();
    assert_eq!(e.worker_state(WorkerId(1)), None);
    let out = e.step().unwrap();
    assert_eq!(out.kind, StepKind::WorkerRemoved);
    assert_eq!(
        out.worker_event,
        Some(WorkerEvent::Removed {
            pool: 0,
            worker: WorkerId(1),
            killed: true
        })
    );
    assert_eq!(out.failed.len(), 5, "everything round-robin put on it");
    for f in &out.failed {
        assert_eq!(f.reason, FailureReason::WorkerKilled);
        assert_eq!(f.worker, Some(WorkerId(1)));
        assert_eq!(f.request.num_computed_tokens, 0, "its KV is gone");
        assert!(f.request.kv_blocks.is_empty());
        assert!(f.request.num_output_tokens > 0, "it had started decoding");
    }
    let resumed: HashMap<String, u32> = out
        .failed
        .iter()
        .map(|f| (f.request.request_id.clone(), f.request.num_output_tokens))
        .collect();
    for f in out.failed {
        e.submit(f.request);
    }
    let done = completions(&run(&mut e));
    assert_eq!(done.len(), 10);
    for t in &done {
        assert_eq!(t.worker, Some(0));
        assert_eq!(t.num_output_tokens, 60, "{t:?}");
        if resumed.contains_key(&t.request_id) {
            // Migrated: re-prefilled prompt and generated tokens, then went
            // on decoding; its first token is the one produced before.
            assert!(t.first_token_time < 0.02);
        }
    }
}

#[test]
fn killing_either_end_of_a_hand_off_fails_it_and_frees_the_other() {
    for kill_decoder in [true, false] {
        let topo = DisaggTopology {
            prefill: cluster(2),
            decode: cluster(2),
            // Slow enough that hand-offs are in flight at the kill.
            kv_link_bw: Some(5e8),
        };
        let mut e = Engine::new(
            Topology::from_disagg(&topo, model(), scheduler(512, false))
                .unwrap()
                .with_router(&RouterConfig::RoundRobin {}),
        );
        let decode = e.decode_pool().unwrap();
        for i in 0..8 {
            e.submit(Request::new(format!("r{i}"), 0, i as f64 * 1e-3, 512, 8));
        }
        while e.handoffs_in_flight() < 2 {
            e.step().unwrap();
        }
        let (pool, victim) = if kill_decoder {
            (decode, WorkerId(2))
        } else {
            (e.entry_pool(), WorkerId(0))
        };
        e.kill_worker(pool, victim).unwrap();
        let mut failed = Vec::new();
        let mut done = Vec::new();
        while !e.is_idle() {
            let out = e.step().unwrap();
            failed.extend(out.failed);
            done.extend(out.completions);
        }
        assert!(!failed.is_empty(), "kill_decoder = {kill_decoder}");
        assert_eq!(done.len() + failed.len(), 8);
        assert!(failed
            .iter()
            .all(|f| f.reason == FailureReason::WorkerKilled && f.worker == Some(victim)));
        assert!(done.iter().all(|t| t.worker != Some(victim.0)));
        // Nothing leaked: every surviving worker's KV is free again.
        assert_eq!(e.kv_cache_util(), 0.0, "kill_decoder = {kill_decoder}");
        // The failed ones finish once resubmitted.
        for f in failed {
            e.submit(f.request);
        }
        let n = completions(&run(&mut e)).len();
        assert_eq!(done.len() + n, 8);
    }
}

/// A worker killed mid-promotion takes its transfers with it: the sibling
/// HBM and host-store blocks they were reading are released, so once the
/// failed requests are resubmitted and done, nothing is left pinned. Slow
/// links keep each promotion in flight for a while; the kill lands just
/// after the `k`-th one starts.
#[test]
fn a_killed_worker_releases_what_its_promotions_pinned() {
    for k in 1..=6 {
        let mut c = tiered_with(cluster(2), "[\"peer_hbm\", \"host\"]");
        for link in &mut c.hardware.memory.as_mut().unwrap().links {
            link.bandwidth = 2e9;
        }
        let mut e = Engine::new(
            Topology::aggregated(c, model(), scheduler(48, true))
                .unwrap()
                .with_router(&RouterConfig::RoundRobin {}),
        );
        for r in sessions_of(5, 120) {
            e.submit(r);
        }
        let promoted = |e: &Engine| e.memory_metrics().unwrap().bytes_promoted;
        let (mut done, mut starts, mut last) = (0, 0, 0.0);
        while starts < k {
            done += e.step().unwrap().completions.len();
            if promoted(&e) > last {
                last = promoted(&e);
                starts += 1;
            }
        }
        e.kill_worker(0, WorkerId(1)).unwrap();
        let mut failed = 0;
        while !e.is_idle() {
            let out = e.step().unwrap();
            done += out.completions.len();
            for f in out.failed {
                failed += 1;
                e.submit(f.request);
            }
        }
        assert_eq!(done, 120, "kill after promotion {k}");
        assert!(failed > 0);
        assert_eq!(
            e.kv_cache_util(),
            0.0,
            "kill after promotion {k}: HBM left pinned"
        );
    }
}

#[test]
fn lifecycle_calls_check_their_arguments() {
    let mut e =
        Engine::new(Topology::aggregated(cluster(1), model(), scheduler(64, true)).unwrap());
    assert!(e.drain_worker(0, WorkerId(7)).is_err());
    assert!(e.add_worker(3, 0.0).is_err());
    let topo = DisaggTopology {
        prefill: cluster(1),
        decode: cluster(1),
        kv_link_bw: Some(1e10),
    };
    let mut d = Engine::new(Topology::from_disagg(&topo, model(), scheduler(64, true)).unwrap());
    assert!(d.kill_worker(0, WorkerId(1)).is_err(), "wrong pool");
    let added = d.add_worker(1, 100.0).unwrap();
    assert_eq!(added, WorkerId(2));
    // A starting worker can be killed before it ever joins; its join goes
    // with it, so the clock does not run on to t = 100.
    d.kill_worker(1, added).unwrap();
    let outcomes = run(&mut d);
    assert_eq!(
        worker_events(&outcomes),
        vec![(
            0.0,
            WorkerEvent::Removed {
                pool: 1,
                worker: added,
                killed: true
            }
        )]
    );
    assert_eq!(d.current_time(), 0.0);
}

/// A prefill worker killed mid-pass: the requests whose hand-off that pass
/// was to start fail at the kill, without the first token the pass never
/// produced, and the engine stops at the kill rather than at the pass's
/// would-be end.
#[test]
fn a_prefill_pass_cut_short_by_a_kill_fails_its_hand_offs_at_once() {
    let topo = DisaggTopology {
        prefill: cluster(1),
        decode: cluster(1),
        kv_link_bw: Some(1e10),
    };
    let mut e = Engine::new(Topology::from_disagg(&topo, model(), scheduler(512, false)).unwrap());
    e.submit(Request::new("a".into(), 0, 0.0, 2048, 4));
    e.submit(Request::new("b".into(), 0, 0.0, 2048, 4));
    // Step until the prefill pass that completes both prompts is running.
    let pass_end = loop {
        let out = e.step().unwrap();
        if let Some(it) = out.iteration {
            assert_eq!(it.pool, 0);
            break it.end_time;
        }
    };
    let killed_at = e.current_time();
    assert!(killed_at < pass_end);
    e.kill_worker(0, WorkerId(0)).unwrap();
    let out = e.step().unwrap();
    assert_eq!(out.kind, StepKind::WorkerRemoved);
    assert_eq!(out.time, killed_at);
    let mut failed: Vec<_> = out.failed.iter().map(|f| &f.request).collect();
    failed.sort_by(|x, y| x.request_id.cmp(&y.request_id));
    assert_eq!(failed.len(), 2);
    for r in failed {
        assert_eq!(r.num_output_tokens, 0, "{}", r.request_id);
        assert_eq!(r.first_token_time, None);
    }
    assert!(e.is_idle());
    assert_eq!(e.current_time(), killed_at);
}

// ----------------------------------------------------------------------
// Router seam

#[test]
fn a_callers_router_hears_the_whole_request_lifecycle_in_time_order() {
    let (router, log) = Recorder::boxed(false);
    let mut e = Engine::new(
        Topology::aggregated(cluster(2), model(), scheduler(512, true))
            .unwrap()
            .with_pool_router(0, router),
    );
    let reqs = sessions(24);
    for r in reqs.clone() {
        e.submit(r);
    }
    let added = e.add_worker(0, 0.05).unwrap();
    let done = completions(&run(&mut e));
    assert_eq!(done.len(), 24);
    let log = log.lock().unwrap();
    assert!(
        log.times.windows(2).all(|w| w[0] <= w[1]),
        "calls arrive in time order"
    );
    let calls = &log.calls;
    assert_eq!(calls[0], Call::Added(WorkerId(0), 1, 0.0));
    assert_eq!(calls[1], Call::Added(WorkerId(1), 1, 0.0));
    assert!(calls.contains(&Call::Added(added, 1, 0.05)));
    for t in &done {
        let w = WorkerId(t.worker.unwrap());
        let mine: Vec<&Call> = calls
            .iter()
            .filter(|c| match c {
                Call::Routed(id, ..) | Call::Prefilled(id, ..) | Call::Completed(id, ..) => {
                    *id == t.request_id
                }
                _ => false,
            })
            .collect();
        assert_eq!(
            mine,
            vec![
                &Call::Routed(t.request_id.clone(), w, t.arrival_time),
                &Call::Prefilled(t.request_id.clone(), w, t.first_token_time),
                &Call::Completed(t.request_id.clone(), w, t.completion_time),
            ]
        );
    }
}

#[test]
fn a_router_answering_with_a_stranger_is_an_error() {
    struct Wrong;
    impl Router for Wrong {
        fn route(&mut self, _: &Request, _: &[WorkerSignal], _: f64) -> WorkerId {
            WorkerId(99)
        }
    }
    let mut e = Engine::new(
        Topology::aggregated(cluster(2), model(), scheduler(64, true))
            .unwrap()
            .with_pool_router(0, Box::new(Wrong)),
    );
    e.submit(Request::new("r".into(), 0, 0.0, 64, 4));
    assert!(e.step().is_err());
}

#[test]
fn disaggregated_routers_hear_their_own_side_of_each_request() {
    let (prefill_router, prefill_log) = Recorder::boxed(false);
    let (decode_router, decode_log) = Recorder::boxed(false);
    let topo = DisaggTopology {
        prefill: cluster(2),
        decode: cluster(2),
        kv_link_bw: Some(2e10),
    };
    let mut e = Engine::new(
        Topology::from_disagg(&topo, model(), scheduler(512, true))
            .unwrap()
            .with_pool_router(0, prefill_router)
            .with_pool_router(1, decode_router),
    );
    for r in sessions(12) {
        e.submit(r);
    }
    let done = completions(&run(&mut e));
    assert_eq!(done.len(), 12);
    let (p, d) = (prefill_log.lock().unwrap(), decode_log.lock().unwrap());
    for t in &done {
        let id = &t.request_id;
        let side = |calls: &[Call]| -> Vec<Call> {
            calls
                .iter()
                .filter(|c| match c {
                    Call::Routed(r, ..) | Call::Prefilled(r, ..) | Call::Completed(r, ..) => {
                        r == id
                    }
                    _ => false,
                })
                .cloned()
                .collect()
        };
        let pre = side(&p.calls);
        // Routed in, prefilled (the hand-off starts), left when it drained.
        assert!(
            matches!(&pre[..], [
            Call::Routed(_, a, t0),
            Call::Prefilled(_, b, t1),
            Call::Completed(_, c, t2),
        ] if a == b && b == c && *t0 == t.arrival_time && *t1 == t.prefill_done_time
            && *t2 == t.handoff_done_time),
            "{pre:?}"
        );
        let dec = side(&d.calls);
        let w = WorkerId(t.worker.unwrap());
        assert_eq!(
            dec,
            vec![
                Call::Routed(id.clone(), w, t.prefill_done_time),
                Call::Completed(id.clone(), w, t.completion_time),
            ]
        );
    }
}

/// Every KV event replayed into a shadow index keeps it equal to each
/// worker's real cache after every step: stores (fresh blocks and landed
/// promotions) and evictions (whole spans split into blocks), across
/// churn, preemption, a worker joining and one draining away.
#[test]
fn kv_events_keep_an_external_index_equal_to_the_cache() {
    let host = Some("[\"host\"]");
    let peer = Some("[\"peer_hbm\", \"host\"]");
    // DP-attention: one replica of two ranks, each with its own cache.
    for (tiers, dp_attention) in [
        (None, false),
        (host, false),
        (peer, false),
        (None, true),
        (peer, true),
    ] {
        let (router, log) = Recorder::boxed(true);
        let c = match (tiers, dp_attention) {
            (Some(t), false) => tiered_with(cluster(2), t),
            (None, false) => cluster(2),
            (Some(t), true) => dp(tiered_with(cluster(1), t), 2),
            (None, true) => dp(cluster(1), 2),
        };
        let mut e = Engine::new(
            // Each rank holds its share of the replica's blocks.
            Topology::aggregated(
                c,
                model(),
                scheduler(if dp_attention { 96 } else { 48 }, true),
            )
            .unwrap()
            .with_pool_router(0, router),
        );
        for r in sessions(150) {
            e.submit(r);
        }
        let mut grown = false;
        while !e.is_idle() {
            e.step().unwrap();
            assert_shadow_matches(&e, &log, 0);
            if !grown && e.current_time() > 0.1 {
                e.add_worker(0, e.current_time() + 0.05).unwrap();
                e.drain_worker(0, WorkerId(0)).unwrap();
                grown = true;
            }
        }
        let log = log.lock().unwrap();
        assert!(log.shadow.stored > 500, "{}", log.shadow.stored);
        assert!(log.shadow.removed > 100, "{}", log.shadow.removed);
        assert!(!log.shadow.blocks.contains_key(&WorkerId(0)), "drained");
        if dp_attention {
            // The drained replica's two ranks went; the added one's came.
            assert!(!log.shadow.blocks.contains_key(&WorkerId(1)));
            let joined: Vec<_> = log.replicas.iter().map(|(&w, &r)| (w, r)).collect();
            assert_eq!(joined, vec![(WorkerId(2), 2)]);
        }
        if let Some(t) = tiers {
            let m = e.memory_metrics().unwrap();
            assert!(m.bytes_promoted > 0.0, "promotions landed");
            assert_eq!(m.peer_hbm_bytes_promoted > 0, t.contains("peer"));
        }
    }
}

#[test]
fn kv_events_carry_the_requests_local_hashes() {
    let (router, log) = Recorder::boxed(true);
    let mut e = Engine::new(
        Topology::aggregated(cluster(1), model(), scheduler(512, false))
            .unwrap()
            .with_pool_router(0, router),
    );
    let a = hashed("a", 0.0, 7, 3, 1, 2);
    let b = hashed("b", 1.0, 7, 3, 2, 2);
    let expected: HashSet<(u64, u64)> = [&a, &b]
        .iter()
        .flat_map(|r| {
            r.prompt_local_block_hashes
                .iter()
                .copied()
                .zip(r.prompt_block_hashes.iter().copied())
        })
        .collect();
    e.submit(a);
    e.submit(b);
    run(&mut e);
    let log = log.lock().unwrap();
    let stored: HashSet<(u64, u64)> = log
        .shadow
        .locals
        .iter()
        .map(|(&seq, &local)| (local, seq))
        .collect();
    assert_eq!(stored, expected, "a's 4 blocks, then b's 2 new ones");
}

// ----------------------------------------------------------------------
// Linear step cost

#[test]
fn a_linear_pool_prices_each_step_from_its_batch() {
    let linear = LinearStepCost {
        t0: 0.003,
        tp: 2e-5,
        td: 1e-4,
        tk: 1e-7,
        t0p: 0.0,
        graph_prefill_max_tokens: 0,
        t0p_graph: 0.0,
    };
    let mut c = cluster(1);
    // A topology checks its pools' models, whichever way they were built.
    c.step_cost = StepCostModel::Linear(LinearStepCost {
        tk: -1e-7,
        t0p: 0.0,
        graph_prefill_max_tokens: 0,
        t0p_graph: 0.0,
        ..linear
    });
    assert!(Topology::aggregated(c.clone(), model(), scheduler(512, false)).is_err());
    c.step_cost = StepCostModel::Linear(linear);
    let mut e = Engine::new(Topology::aggregated(c, model(), scheduler(512, false)).unwrap());
    // A roofline correction is for roofline-priced steps only.
    e.set_time_correction(3.0, 0.5);
    e.submit(Request::new("a".into(), 0, 0.0, 100, 3));
    e.submit(Request::new("b".into(), 0, 0.0, 40, 2));
    let iters: Vec<_> = run(&mut e)
        .into_iter()
        .filter_map(|o| o.iteration)
        .map(|i| i.iteration_time)
        .collect();
    let expected = [
        // Both prompts in one step: 140 prefill tokens.
        0.003 + 2e-5 * 140.0,
        // Both decode: a attends 100 + 1, b 40 + 1.
        0.003 + 1e-4 * 2.0 + 1e-7 * (101.0 + 41.0),
        // b is done; a decodes its last token at 101 + 1.
        0.003 + 1e-4 + 1e-7 * 102.0,
    ];
    assert_eq!(iters.len(), expected.len(), "{iters:?}");
    for (got, want) in iters.iter().zip(expected) {
        assert!((got - want).abs() < 1e-15, "{got} vs {want}");
    }
}

// ----------------------------------------------------------------------
// DP-attention replicas

/// A replica of DP-attention ranks joins, drains and leaves whole: one id
/// names it, its ranks follow that id, and every lifecycle call takes any
/// rank's id.
#[test]
fn a_dp_replica_joins_drains_and_leaves_with_all_its_ranks() {
    let (router, log) = Recorder::boxed(true);
    let mut e = Engine::new(
        Topology::aggregated(dp(cluster(1), 2), model(), scheduler(256, true))
            .unwrap()
            .with_pool_router(0, router),
    );
    assert_eq!(e.pool_members(0), vec![(WorkerId(0), WorkerState::Active)]);
    assert_eq!(
        e.replica_ranks(WorkerId(1)),
        Some(vec![WorkerId(0), WorkerId(1)])
    );
    let added = e.add_worker(0, 0.05).unwrap();
    assert_eq!(added, WorkerId(2));
    assert_eq!(e.replica_ranks(added), Some(vec![WorkerId(2), WorkerId(3)]));
    assert_eq!(e.worker_state(WorkerId(3)), Some(WorkerState::Starting));
    for r in sessions(120) {
        e.submit(r);
    }
    let mut outcomes = Vec::new();
    while e.current_time() < 0.1 {
        outcomes.push(e.step().unwrap());
        assert_shadow_matches(&e, &log, 0);
    }
    // Name the replica by its second rank.
    e.drain_worker(0, WorkerId(1)).unwrap();
    let drained_at = e.current_time();
    assert_eq!(e.worker_state(WorkerId(0)), Some(WorkerState::Draining));
    while !e.is_idle() {
        outcomes.push(e.step().unwrap());
        assert_shadow_matches(&e, &log, 0);
    }
    let done = completions(&outcomes);
    assert_eq!(done.len(), 120);
    let served: HashSet<u32> = done.iter().map(|t| t.worker.unwrap()).collect();
    assert_eq!(served, HashSet::from([0, 1, 2, 3]), "every rank served");
    for t in &done {
        if t.arrival_time > drained_at {
            assert!(t.worker.unwrap() >= 2, "routed to a draining rank: {t:?}");
        }
    }
    let events: Vec<WorkerEvent> = worker_events(&outcomes)
        .into_iter()
        .map(|(_, e)| e)
        .collect();
    assert_eq!(
        events,
        vec![
            WorkerEvent::Joined {
                pool: 0,
                worker: added
            },
            WorkerEvent::Removed {
                pool: 0,
                worker: WorkerId(0),
                killed: false
            },
        ]
    );
    assert_eq!(e.pool_members(0), vec![(added, WorkerState::Active)]);
    assert_eq!(e.worker_state(WorkerId(1)), None);
    let log = log.lock().unwrap();
    assert_eq!(log.calls[0], Call::Added(WorkerId(0), 2, 0.0));
    assert!(log.calls.contains(&Call::Added(added, 2, 0.05)));
    assert_eq!(
        log.calls
            .iter()
            .filter(|c| matches!(c, Call::Removed(..)))
            .count(),
        1
    );
}

/// A router places a request on a rank of its choosing — a replica, then
/// a rank of it — and every notification about that request names the
/// rank.
#[test]
fn a_router_pins_each_request_to_a_rank_and_hears_back_from_it() {
    /// Request `i` goes to replica `i % 2` (in id order), rank `i / 2 % 4`.
    fn pin(req: &Request, workers: &[WorkerSignal]) -> WorkerId {
        let i: u32 = req.request_id[1..].parse().unwrap();
        let mut replicas: Vec<WorkerId> = workers.iter().map(|w| w.replica).collect();
        replicas.sort();
        replicas.dedup();
        let replica = replicas[i as usize % replicas.len()];
        workers
            .iter()
            .find(|w| w.replica == replica && w.rank == i / 2 % 4)
            .unwrap()
            .worker
    }
    let (router, log) = Recorder::picking(true, pin);
    let mut e = Engine::new(
        Topology::aggregated(dp(cluster(2), 4), model(), scheduler(1024, true))
            .unwrap()
            .with_pool_router(0, router),
    );
    assert_eq!(
        e.pool_members(0),
        vec![
            (WorkerId(0), WorkerState::Active),
            (WorkerId(4), WorkerState::Active)
        ]
    );
    for i in 0..32u32 {
        let mut r = hashed(&format!("r{i}"), i as f64 * 1e-3, i as u64, 3, 2, 6);
        r.request_id = format!("r{i}");
        e.submit(r);
    }
    let done = completions(&run(&mut e));
    assert_eq!(done.len(), 32);
    let log = log.lock().unwrap();
    for t in &done {
        let i: u32 = t.request_id[1..].parse().unwrap();
        let rank = WorkerId(4 * (i % 2) + i / 2 % 4);
        assert_eq!(t.worker, Some(rank.0), "{}", t.request_id);
        let heard: Vec<&Call> = log
            .calls
            .iter()
            .filter(|c| match c {
                Call::Routed(id, ..) | Call::Prefilled(id, ..) | Call::Completed(id, ..) => {
                    *id == t.request_id
                }
                _ => false,
            })
            .collect();
        assert_eq!(
            heard,
            vec![
                &Call::Routed(t.request_id.clone(), rank, t.arrival_time),
                &Call::Prefilled(t.request_id.clone(), rank, t.first_token_time),
                &Call::Completed(t.request_id.clone(), rank, t.completion_time),
            ]
        );
    }
    // Each rank's KV events built its own index, equal to its cache.
    for rank in 0..8 {
        let id = WorkerId(rank);
        let mut shadow: Vec<u64> = log.shadow.blocks[&id].keys().copied().collect();
        shadow.sort_unstable();
        assert!(!shadow.is_empty(), "{id} cached nothing");
        assert_eq!(shadow, e.worker_cached_blocks(id).unwrap(), "{id}");
    }
}

/// A DP replica added at t = 0 is the replica a config would have built.
#[test]
fn a_runtime_dp_replica_runs_exactly_like_a_configured_one() {
    let fingerprint = |e: &mut Engine| {
        for r in sessions_of(5, 90) {
            e.submit(r);
        }
        completions(&run(e))
            .iter()
            .map(|t| {
                (
                    t.request_id.clone(),
                    t.first_token_time.to_bits(),
                    t.completion_time.to_bits(),
                    t.num_cached_tokens,
                    t.worker,
                )
            })
            .collect::<Vec<_>>()
    };
    for tiers in [None, Some("[\"peer_hbm\", \"host\"]")] {
        let build = |n| {
            let c = match tiers {
                Some(t) => dp(tiered_with(cluster(n), t), 2),
                None => dp(cluster(n), 2),
            };
            Topology::aggregated(c, model(), scheduler(128, true))
                .unwrap()
                .with_router(&RouterConfig::KvAware { load_weight: 1.0 })
        };
        let mut fixed = Engine::new(build(2));
        let mut grown = Engine::new(build(1));
        assert_eq!(grown.add_worker(0, 0.0).unwrap(), WorkerId(2));
        let a = fingerprint(&mut fixed);
        assert_eq!(a.len(), 90);
        assert_eq!(a, fingerprint(&mut grown), "tiers = {tiers:?}");
    }
}

/// Killing a DP replica fails what every one of its ranks held; resubmitted,
/// those requests finish on the other replica, and nothing stays held.
#[test]
fn killing_a_dp_replica_fails_every_rank_and_frees_everything() {
    let mut e = Engine::new(
        Topology::aggregated(dp(cluster(2), 2), model(), scheduler(512, true))
            .unwrap()
            .with_router(&RouterConfig::RoundRobin {}),
    );
    for i in 0..12 {
        e.submit(Request::new(format!("r{i}"), 0, 0.0, 300, 60));
    }
    let mut done = 0;
    while e.current_time() < 0.02 {
        done += e.step().unwrap().completions.len();
    }
    // Name the second replica by its second rank.
    e.kill_worker(0, WorkerId(3)).unwrap();
    let out = e.step().unwrap();
    assert_eq!(
        out.worker_event,
        Some(WorkerEvent::Removed {
            pool: 0,
            worker: WorkerId(2),
            killed: true
        })
    );
    // Round-robin put three requests on each rank.
    assert_eq!(out.failed.len(), 6);
    let on: HashSet<Option<WorkerId>> = out.failed.iter().map(|f| f.worker).collect();
    assert_eq!(on, HashSet::from([Some(WorkerId(2)), Some(WorkerId(3))]));
    assert_eq!(e.pool_members(0), vec![(WorkerId(0), WorkerState::Active)]);
    for f in out.failed {
        e.submit(f.request);
    }
    let rest = completions(&run(&mut e));
    assert_eq!(done + rest.len(), 12);
    assert!(rest.iter().all(|t| t.worker.unwrap() < 2));
    assert_eq!(e.kv_cache_util(), 0.0);
}

/// A decode pool of DP replicas: killing one mid-hand-off fails the
/// hand-offs bound for any of its ranks and frees the prefill side.
#[test]
fn killing_a_dp_decode_replica_fails_the_hand_offs_bound_for_it() {
    let topo = DisaggTopology {
        prefill: cluster(2),
        decode: dp(cluster(2), 2),
        kv_link_bw: Some(5e8),
    };
    let mut e = Engine::new(
        Topology::from_disagg(&topo, model(), scheduler(512, false))
            .unwrap()
            .with_router(&RouterConfig::RoundRobin {}),
    );
    let decode = e.decode_pool().unwrap();
    // Prefill workers 0 and 1; decode replicas 2 (ranks 2, 3) and 4 (4, 5).
    assert_eq!(
        e.pool_members(decode),
        vec![
            (WorkerId(2), WorkerState::Active),
            (WorkerId(4), WorkerState::Active)
        ]
    );
    for i in 0..8 {
        e.submit(Request::new(format!("r{i}"), 0, i as f64 * 1e-3, 512, 8));
    }
    while e.handoffs_in_flight() < 4 {
        e.step().unwrap();
    }
    e.kill_worker(decode, WorkerId(2)).unwrap();
    let mut failed = Vec::new();
    let mut done = Vec::new();
    while !e.is_idle() {
        let out = e.step().unwrap();
        failed.extend(out.failed);
        done.extend(out.completions);
    }
    assert!(!failed.is_empty());
    assert!(failed
        .iter()
        .all(|f| matches!(f.worker, Some(WorkerId(2) | WorkerId(3)))));
    assert_eq!(done.len() + failed.len(), 8);
    assert!(done.iter().all(|t| matches!(t.worker, Some(4 | 5))));
    assert_eq!(e.kv_cache_util(), 0.0);
}

/// A DP pool built with one replica and drained at once starts empty:
/// arrivals are handed back until a replica joins, and replicas added
/// later take the ids after it.
#[test]
fn a_dp_pool_drained_at_start_is_empty_until_a_replica_joins() {
    let mut e = Engine::new(
        Topology::aggregated(dp(cluster(1), 4), model(), scheduler(512, true)).unwrap(),
    );
    e.advance_to(10.0);
    e.drain_worker(0, WorkerId(0)).unwrap();
    let out = e.step().unwrap();
    assert_eq!(
        out.worker_event,
        Some(WorkerEvent::Removed {
            pool: 0,
            worker: WorkerId(0),
            killed: false
        })
    );
    assert_eq!(out.time, 10.0);
    assert!(e.is_idle());
    assert!(e.pool_members(0).is_empty());
    e.submit(Request::new("early".into(), 0, 10.0, 64, 4));
    let out = e.step().unwrap();
    assert_eq!(out.failed[0].reason, FailureReason::NoWorker);
    let replica = e.add_worker(0, 10.5).unwrap();
    assert_eq!(replica, WorkerId(4));
    assert_eq!(e.replica_ranks(replica).unwrap().len(), 4);
    for i in 0..16 {
        e.submit(Request::new(format!("r{i}"), 0, 11.0, 128, 8));
    }
    let done = completions(&run(&mut e));
    assert_eq!(done.len(), 16);
    let served: HashSet<u32> = done.iter().map(|t| t.worker.unwrap()).collect();
    assert_eq!(served, HashSet::from([4, 5, 6, 7]));
}

// ----------------------------------------------------------------------
// Speculative decoding on DP-attention replicas

/// A fixed-depth draft with i.i.d. acceptance `alpha`, as SGLang's EAGLE
/// with `--speculative-num-steps gamma --speculative-eagle-topk 1`.
fn eagle(gamma: u32, alpha: f64) -> inference_lab::config::SpeculativeConfig {
    toml::from_str(&format!(
        "gamma = {gamma}\n[acceptance]\nkind = \"constant\"\nalpha = {alpha}\n"
    ))
    .unwrap()
}

/// Every rank of a DP replica verifies its own requests' drafts inside the
/// replica's lockstep step: each decode entry is `1 + gamma` wide, its
/// accepted tokens drawn per request, and the mean advance is the
/// acceptance model's.
#[test]
fn dp_ranks_verify_their_own_drafts_in_the_lockstep_step() {
    let mut e = Engine::new(
        Topology::aggregated(dp(cluster(1), 4), model(), scheduler(1024, true))
            .unwrap()
            .with_router(&RouterConfig::RoundRobin {}),
    );
    e.enable_speculative(eagle(3, 0.7), 11).unwrap();
    for i in 0..32 {
        e.submit(Request::new(format!("r{i}"), 0, 0.0, 200, 64));
    }
    let outcomes = run(&mut e);
    let done = completions(&outcomes);
    assert_eq!(done.len(), 32);
    assert!(done.iter().all(|t| t.num_output_tokens == 64));
    let served: HashSet<u32> = done.iter().map(|t| t.worker.unwrap()).collect();
    assert_eq!(served, HashSet::from([0, 1, 2, 3]));
    let (mut verifies, mut advanced, mut mixed_steps) = (0u32, 0u32, 0u32);
    let mut drafting_ranks = HashSet::new();
    for it in outcomes.iter().filter_map(|o| o.iteration.as_ref()) {
        let decodes: Vec<_> = it.progress.iter().filter(|p| !p.was_prefill).collect();
        // Ranks share the step: decode entries from several ranks at once.
        let ranks = decodes
            .iter()
            .map(|p| {
                done.iter()
                    .find(|t| t.request_id == p.request_id)
                    .unwrap()
                    .worker
            })
            .collect::<HashSet<_>>();
        if ranks.len() > 1 {
            mixed_steps += 1;
        }
        for p in decodes {
            // Draft depth 3: a verify pass of 4 positions, unless the
            // request's last tokens need fewer.
            assert!(p.num_tokens <= 4 && p.num_output <= p.num_tokens, "{p:?}");
            if p.num_tokens == 4 {
                verifies += 1;
                advanced += p.num_output;
                let on = done.iter().find(|t| t.request_id == p.request_id).unwrap();
                drafting_ranks.insert(on.worker.unwrap());
            }
        }
    }
    assert!(mixed_steps > 10, "{mixed_steps}");
    assert_eq!(
        drafting_ranks,
        HashSet::from([0, 1, 2, 3]),
        "every rank drafts"
    );
    // E[1 + accepted] at depth 3 and alpha 0.7: 1 + 0.7 + 0.49 + 0.343.
    let mean = advanced as f64 / verifies as f64;
    assert!(
        (mean - 2.533).abs() < 0.15,
        "{mean} over {verifies} verifies"
    );
}

/// A linear step cost prices a DP replica's lockstep step on the replica's
/// whole batch — every rank's prefill tokens, decoding requests and
/// attended KV, verify widths included — with or without speculation.
#[test]
fn a_linear_dp_step_prices_the_replicas_whole_batch() {
    let linear = LinearStepCost {
        t0: 0.004,
        tp: 2e-5,
        td: 1e-4,
        tk: 1e-7,
        t0p: 0.0,
        graph_prefill_max_tokens: 0,
        t0p_graph: 0.0,
    };
    for speculate in [false, true] {
        let mut c = dp(cluster(1), 4);
        c.step_cost = StepCostModel::Linear(linear);
        let mut e = Engine::new(
            Topology::aggregated(c, model(), scheduler(1024, false))
                .unwrap()
                .with_router(&RouterConfig::RoundRobin {}),
        );
        if speculate {
            e.enable_speculative(eagle(3, 0.7), 5).unwrap();
        }
        for i in 0..12 {
            e.submit(Request::new(
                format!("r{i}"),
                0,
                i as f64 * 1e-3,
                100 + 10 * i,
                24,
            ));
        }
        let mut wide = 0;
        for it in run(&mut e).iter().filter_map(|o| o.iteration.as_ref()) {
            let (mut prefill, mut decode, mut kv) = (0u64, 0u64, 0u64);
            for p in &it.progress {
                if p.was_prefill {
                    prefill += p.num_tokens as u64;
                } else {
                    decode += 1;
                    kv += (p.context_tokens + p.num_tokens) as u64;
                    wide += u32::from(p.num_tokens > 1);
                }
            }
            let expected = linear.seconds(prefill, decode, kv);
            assert!(
                (it.iteration_time - expected).abs() < 1e-15,
                "speculate = {speculate}: {} vs {expected}",
                it.iteration_time
            );
        }
        assert_eq!(wide > 0, speculate);
    }
}

/// Killing a speculating DP replica mid-decode unwinds every rank's
/// unfinished multi-token verify: resubmitted, each request still produces
/// exactly its target.
#[test]
fn killing_a_speculating_dp_replica_unwinds_its_verify_passes() {
    let mut e = Engine::new(
        Topology::aggregated(dp(cluster(2), 2), model(), scheduler(512, true))
            .unwrap()
            .with_router(&RouterConfig::RoundRobin {}),
    );
    e.enable_speculative(eagle(3, 0.8), 3).unwrap();
    for i in 0..16 {
        e.submit(Request::new(format!("r{i}"), 0, 0.0, 300, 80));
    }
    let mut done = Vec::new();
    let mut iterations = Vec::new();
    while e.current_time() < 0.03 {
        let out = e.step().unwrap();
        done.extend(out.completions);
        iterations.extend(out.iteration);
    }
    let killed_at = e.current_time();
    e.kill_worker(0, WorkerId(2)).unwrap();
    let out = e.step().unwrap();
    assert_eq!(out.failed.len(), 8);
    // Each failed request keeps the tokens of the passes that ended by the
    // kill, and none of the verify still running.
    let mut unwound = 0;
    for f in &out.failed {
        let produced: u32 = iterations
            .iter()
            .filter(|it| it.end_time <= killed_at)
            .flat_map(|it| &it.progress)
            .filter(|p| p.request_id == f.request.request_id)
            .map(|p| p.num_output)
            .sum();
        let in_flight: u32 = iterations
            .iter()
            .filter(|it| it.end_time > killed_at)
            .flat_map(|it| &it.progress)
            .filter(|p| p.request_id == f.request.request_id)
            .map(|p| p.num_output)
            .sum();
        assert_eq!(
            f.request.num_output_tokens, produced,
            "{}",
            f.request.request_id
        );
        assert!(produced > 0 && produced < 80);
        unwound += u32::from(in_flight > 1);
    }
    assert!(unwound > 0, "some killed verify had accepted drafts");
    for f in out.failed {
        e.submit(f.request);
    }
    done.extend(completions(&run(&mut e)));
    assert_eq!(done.len(), 16);
    assert!(done.iter().all(|t| t.num_output_tokens == 80));
    assert_eq!(e.kv_cache_util(), 0.0);
}

/// SGLang's batching (no mixed chunk): a step prefills or decodes, never both, and under its
/// overlap scheduler a prefill's first token reaches the client once the next step is done.
#[test]
fn an_sglang_engine_alternates_steps_and_emits_first_tokens_a_step_late() {
    let linear = LinearStepCost {
        t0: 0.01,
        tp: 0.0,
        td: 0.0,
        tk: 0.0,
        t0p: 0.1,
        graph_prefill_max_tokens: 0,
        t0p_graph: 0.0,
    };
    let engine = |mixed: bool, overlap: bool| {
        let mut c = cluster(1);
        c.step_cost = StepCostModel::Linear(linear);
        let mut s = scheduler(512, false);
        s.mix_prefill_with_decode = mixed;
        s.overlap_schedule = overlap;
        let mut e = Engine::new(Topology::aggregated(c, model(), s).unwrap());
        e.submit(Request::new("a".into(), 0, 0.0, 100, 3));
        e.submit(Request::new("b".into(), 0, 0.05, 40, 2));
        let outcomes = run(&mut e);
        let iters: Vec<f64> = outcomes
            .iter()
            .filter_map(|o| o.iteration.as_ref())
            .map(|i| i.iteration_time)
            .collect();
        let ttft: Vec<f64> = completions(&outcomes).iter().map(|t| t.ttft()).collect();
        (iters, ttft)
    };
    let close = |got: &[f64], want: &[f64]| {
        got.len() == want.len() && got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-12)
    };
    // vLLM's batching: b's prefill carries a's first decode.
    let (iters, ttft) = engine(true, false);
    assert!(close(&iters, &[0.11, 0.11, 0.01]), "{iters:?}");
    assert!(close(&ttft, &[0.11, 0.17]), "{ttft:?}");
    // SGLang's: b's prefill runs alone, then both decode, then a alone.
    let (iters, ttft) = engine(false, false);
    assert!(close(&iters, &[0.11, 0.11, 0.01, 0.01]), "{iters:?}");
    assert!(close(&ttft, &[0.11, 0.17]), "{ttft:?}");
    // Overlapped: a's token waits for b's prefill, b's for the decode after it.
    let (iters, ttft) = engine(false, true);
    assert!(close(&iters, &[0.11, 0.11, 0.01, 0.01]), "{iters:?}");
    assert!(close(&ttft, &[0.22, 0.18]), "{ttft:?}");
}

/// An SGLang DP-attention replica: a rank's prefill idles the other ranks for that step, and
/// the prefill delayer holds a lone rank's new prompt while the others decode, for up to its
/// pass limit.
#[test]
fn a_dp_replica_prefills_in_lockstep_and_delays_lone_prefills() {
    let linear = LinearStepCost {
        t0: 0.01,
        tp: 0.0,
        td: 0.001,
        tk: 0.0,
        t0p: 0.1,
        graph_prefill_max_tokens: 0,
        t0p_graph: 0.0,
    };
    let engine = |delay_passes: u32| {
        let mut c = dp(cluster(1), 2);
        c.step_cost = StepCostModel::Linear(linear);
        let mut s = scheduler(4096, false);
        s.mix_prefill_with_decode = false;
        s.prefill_delayer_max_passes = delay_passes;
        let mut e = Engine::new(Topology::aggregated(c, model(), s).unwrap());
        // `a` lands on one rank and decodes; `b` arrives on the other mid-decode.
        e.submit(Request::new("a".into(), 0, 0.0, 64, 100));
        e.submit(Request::new("b".into(), 0, 0.505, 64, 2));
        let outcomes = run(&mut e);
        let iters: Vec<f64> = outcomes
            .iter()
            .filter_map(|o| o.iteration.as_ref())
            .map(|i| i.iteration_time)
            .collect();
        let ttft: Vec<f64> = completions(&outcomes).iter().map(|t| t.ttft()).collect();
        (iters, ttft)
    };
    // a's decode steps cost 0.011 s and end at 0.11 + 0.011·k; b arrives during the one that
    // ends at 0.506.
    let boundary = 0.11 + 0.011 * 36.0;
    // No delayer: b's prefill runs at that boundary, alone (0.11 s: a's decode sits it out),
    // and `a`, alone at t = 0, never waits.
    let (iters, ttft) = engine(0);
    assert!((ttft[0] - 0.11).abs() < 1e-9, "{ttft:?}");
    assert!(
        (ttft[1] - (boundary + 0.11 - 0.505)).abs() < 1e-9,
        "{ttft:?}"
    );
    let prefills = iters.iter().filter(|&&t| (t - 0.11).abs() < 1e-9).count();
    assert_eq!(prefills, 2, "two pure prefill steps: {iters:?}");
    assert!(iters
        .iter()
        .all(|&t| (t - 0.11).abs() < 1e-9 || (t - 0.011).abs() < 1e-9 || (t - 0.012).abs() < 1e-9));
    // A delayer of 3 passes holds b for two decode steps of a, then lets it run.
    let (_, ttft) = engine(3);
    assert!((ttft[0] - 0.11).abs() < 1e-9, "{ttft:?}");
    assert!(
        (ttft[1] - (boundary + 2.0 * 0.011 + 0.11 - 0.505)).abs() < 1e-9,
        "{ttft:?}"
    );
}

/// With a cost for passes that run nothing, the prefill delayer holds a lone rank's prompt on an
/// otherwise idle replica too: the replica spins its passes, each `idle_pass_s`.
#[test]
fn a_dp_replica_spins_while_its_delayer_holds_a_prompt() {
    let linear = LinearStepCost {
        t0: 0.01,
        tp: 0.0,
        td: 0.001,
        tk: 0.0,
        t0p: 0.1,
        graph_prefill_max_tokens: 0,
        t0p_graph: 0.0,
    };
    let ttft = |idle_pass_s: f64| {
        let mut c = dp(cluster(1), 2);
        c.step_cost = StepCostModel::Linear(linear);
        let mut s = scheduler(4096, false);
        s.mix_prefill_with_decode = false;
        s.prefill_delayer_max_passes = 30;
        s.idle_pass_s = idle_pass_s;
        let mut e = Engine::new(Topology::aggregated(c, model(), s).unwrap());
        e.submit(Request::new("a".into(), 0, 0.0, 64, 2));
        completions(&run(&mut e))[0].ttft()
    };
    // Spinning in no time, the replica does not hold it.
    assert!((ttft(0.0) - 0.11).abs() < 1e-9);
    // 29 held passes of 2 ms, then the prefill.
    assert!(
        (ttft(0.002) - (29.0 * 0.002 + 0.11)).abs() < 1e-9,
        "{}",
        ttft(0.002)
    );
}

/// Doubleword's SGLang: after each chunk of a long prompt, every rank decodes for the decode
/// interval before the next chunk, and while the long prompt continues, the delayer lets another
/// rank's new prompt prefill with its next chunk.
#[test]
fn a_dp_replica_decodes_between_a_long_prompts_chunks() {
    let linear = LinearStepCost {
        t0: 0.01,
        tp: 0.0,
        td: 0.001,
        tk: 0.0,
        t0p: 0.1,
        graph_prefill_max_tokens: 0,
        t0p_graph: 0.0,
    };
    let engine = |interval: u32, delayer: u32| {
        let mut c = dp(cluster(1), 2);
        c.step_cost = StepCostModel::Linear(linear);
        let mut s = scheduler(4096, true);
        s.mix_prefill_with_decode = false;
        s.prefill_delayer_max_passes = delayer;
        s.long_prefill_decode_interval = interval;
        s.long_prefill_min_tokens = 600;
        s.idle_pass_s = 0.002;
        Engine::new(Topology::aggregated(c, model(), s).unwrap())
    };
    let kinds = |e: &mut Engine| -> String {
        run(e)
            .iter()
            .filter_map(|o| o.iteration.as_ref())
            .map(|i| if i.iteration_time > 0.05 { 'P' } else { 'd' })
            .collect()
    };
    // `a` decodes on one rank; a 700-token prompt on the other prefills in chunks of 256.
    let long = |e: &mut Engine| {
        e.submit(Request::new("a".into(), 0, 0.0, 64, 20));
        e.submit(Request::new("long".into(), 0, 0.2, 700, 2));
    };
    let mut e = engine(0, 0);
    long(&mut e);
    let steps = kinds(&mut e);
    assert!(steps.contains("dPPPd"), "chunks back to back: {steps}");
    let mut e = engine(2, 0);
    long(&mut e);
    let steps = kinds(&mut e);
    assert!(
        steps.contains("dPddPddPd"),
        "two decode passes after each chunk with more to come: {steps}"
    );

    // On an idle replica, a prompt arriving on the other rank during the interval prefills with
    // the long prompt's next chunk: 2 passes of 2 ms after the first chunk, then that chunk.
    let mut e = engine(2, 30);
    e.submit(Request::new("long".into(), 0, 0.0, 1000, 2));
    e.submit(Request::new("short".into(), 0, 0.111, 64, 2));
    let done = completions(&run(&mut e));
    let short = done.iter().find(|t| t.request_id == "short").unwrap();
    // The long prompt's first chunk waits out the delayer's 29 passes (the other rank has none).
    let chunk2_start = 29.0 * 0.002 + 0.11 + 2.0 * 0.002;
    assert!(
        (short.ttft() - (chunk2_start + 0.11 - 0.111)).abs() < 1e-9,
        "{}",
        short.ttft()
    );
}

/// A DP replica's prefill step in which no rank overflows the captured graph's size costs
/// `t0p_graph`; one that overflows it on any rank costs `t0p`.
#[test]
fn a_dp_replica_runs_small_prefill_steps_as_a_graph() {
    let linear = LinearStepCost {
        t0: 0.01,
        tp: 0.0,
        td: 0.0,
        tk: 0.0,
        t0p: 0.5,
        graph_prefill_max_tokens: 256,
        t0p_graph: 0.1,
    };
    let first_step = |prompts: &[u32]| {
        let mut c = dp(cluster(1), 2);
        c.step_cost = StepCostModel::Linear(linear);
        let mut s = scheduler(4096, false);
        s.mix_prefill_with_decode = false;
        let mut e = Engine::new(Topology::aggregated(c, model(), s).unwrap());
        for (i, &p) in prompts.iter().enumerate() {
            e.submit(Request::new(format!("r{i}"), 0, 0.0, p, 2));
        }
        run(&mut e)
            .iter()
            .find_map(|o| o.iteration.as_ref().map(|i| i.iteration_time))
            .unwrap()
    };
    // Both ranks prefill 128 tokens, or one does: the graph.
    assert!((first_step(&[128, 128]) - 0.11).abs() < 1e-12);
    assert!((first_step(&[128]) - 0.11).abs() < 1e-12);
    // A rank over the graph's size: eager.
    assert!((first_step(&[128, 512]) - 0.51).abs() < 1e-12);
}
