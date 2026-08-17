//! The KV memory of a topology beyond HBM, as a graph: stores and links
//! instantiated from each pool's hardware `[memory]` template per GPU, per
//! node, or once for the cluster, joined across nodes by the scale-out
//! network.
//!
//! *Vertices*: each worker's GPU (a `tp`-GPU replica, its ports pooled),
//! each node's switch, the network core, every store instance and every
//! junction instance. *Edges*: directed, one each way per template link
//! instance, with a capacity in bytes/s; a store with its own throughput
//! adds an edge every transfer in or out of it crosses. Rates are max-min
//! fair over all in-flight transfers (`Flows`).
//!
//! A worker sees an ordered list of tiers, closest first: the deployment's
//! `[memory] tiers`. A `per = "gpu"` store is private to the worker (its
//! `tp` GPUs' instances pooled); a `per = "node"` store is one instance
//! shared by every worker on that node — what one worker demotes, its
//! neighbours can promote. A `per = "cluster"` store is one instance behind
//! the scale-out network and visible to every worker. Its striped bandwidth
//! caps one transfer while an aggregate edge is shared topology-wide. A
//! promotion moves bytes along the shortest path from the store to the
//! worker's GPU; a prefill → decode hand-off along the shortest path from one
//! GPU to the other (through the network when they are on different nodes,
//! with an optional core capacity).
//!
//! The graph is shared by the topology's workers behind a mutex; the
//! engine is single-threaded, so the lock only serialises.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use super::flows::{EdgeId, Flows, Owner};
use super::radix::{HbmEvicted, KvBytesFn, NodeId, Radix, SharedRadix, Span, TierEvicted};
use crate::config::{ClusterSpec, EvictionPolicy, MemoryTemplate, Scope, WritePolicy};

pub type StoreId = usize;
pub type WorkerId = usize;
pub type VertexId = usize;

/// One instance of a store: a capacity, an eviction policy, and byte
/// accounting. What it holds lives in the topology's [`Radix`] tree as
/// ranges per node, under this store's index.
#[derive(Debug)]
pub struct Store {
    pub name: String,
    pub scope: Scope,
    /// Node (for `Node` scope), worker (for `Gpu` scope), or zero (for the
    /// topology-wide `Cluster` scope) this instance belongs to.
    pub owner: usize,
    pub capacity_blocks: u64,
    pub vertex: VertexId,
    /// Shared throughput: the instance rate for GPU/node stores or the
    /// topology-wide aggregate rate for a cluster store.
    pub throughput_edge: Option<EdgeId>,
    /// Per-transfer cap from striping, bytes/s. This is a private edge in
    /// the flow solver rather than a shared graph edge.
    pub transfer_bandwidth: Option<f64>,
    /// Fixed cost paid by every fetch or write touching the store.
    pub latency: f64,
    pub eviction: EvictionPolicy,
    /// The store a full instance evicts into (the next tier), if any.
    pub next: Option<StoreId>,
    /// Blocks evicted for capacity.
    pub num_evictions: u64,
    /// Blocks dropped by TTL.
    pub num_expired: u64,
    /// Bytes whose write landed here.
    pub bytes_written: u64,
    /// Bytes promoted from here.
    pub bytes_read: u64,
    /// Bytes evicted or expired without ever being promoted.
    pub dead_bytes: u64,
}

/// A route through the graph: edges in order and the latency paid before
/// bytes flow.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Path {
    pub edges: Vec<EdgeId>,
    pub latency: f64,
}

/// One worker's access to one store: the store, the path a promotion
/// from it takes into the worker's HBM, and the path a write takes out.
#[derive(Debug, Clone)]
pub struct Tier {
    pub store: StoreId,
    pub fetch_path: Path,
    pub write_path: Path,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum VertexKey {
    Gpu(WorkerId),
    Switch(usize),
    Network,
    /// Store or junction: name and scope key (worker for `Gpu` scope, node
    /// for `Node` scope, zero for `Cluster` scope).
    Named(String, usize),
}

#[derive(Debug, Clone)]
struct Hop {
    to: VertexId,
    edge: EdgeId,
    latency: f64,
}

/// The topology's KV memory beyond HBM.
#[derive(Debug)]
pub struct MemoryGraph {
    /// The KV tree every store's ranges live in (shared with the workers'
    /// managers).
    radix: SharedRadix,
    flows: Flows,
    vertex_index: HashMap<VertexKey, VertexId>,
    adj: Vec<Vec<Hop>>,
    stores: Vec<Store>,
    /// `tiers[worker]`: that worker's tiers, closest first.
    tiers: Vec<Vec<Tier>>,
    gpu_vertex: Vec<VertexId>,
    node_of: Vec<usize>,
    /// Hand-off core capacity between the network's ingress and egress
    /// (`DisaggTopology.kv_link_bw`).
    core_edge: Option<EdgeId>,
    handoff_paths: HashMap<(WorkerId, WorkerId), Path>,
    /// Instantiated `(link name, from vertex)` pairs, to create node-level
    /// links once.
    made_links: HashSet<(String, VertexId)>,
    /// Per worker: when its blocks are written to its first tier, and
    /// whether HBM prefers recycling blocks a tier already holds.
    write_of: Vec<WritePolicy>,
    evict_backed_first: Vec<bool>,
    /// In-flight writes: transfer id → the (destination store, span)
    /// ranges it carries that have not landed or been dropped. A batch of
    /// blocks written together moves as one transfer.
    pending_writes: HashMap<String, Vec<(StoreId, Span)>>,
    next_write_seq: u64,
    /// Promotions that had to wait for a write still arriving.
    pub write_race_waits: u64,
}

/// A graph shared by the topology's workers.
pub type SharedMemoryGraph = Arc<Mutex<MemoryGraph>>;

/// The id under which a worker's promotion of `request` from its tier
/// `tier` moves.
pub fn promotion_id(request: &str, tier: usize) -> String {
    format!("{request}#{tier}")
}

/// Inverse of `promotion_id`.
pub fn promotion_request(id: &str) -> &str {
    id.rsplit_once('#').map(|(r, _)| r).unwrap_or(id)
}

impl MemoryGraph {
    /// A bare graph over a tree with the given block size and KV curve.
    fn empty_with(block_size: u32, kv_bytes_at: KvBytesFn) -> Self {
        Self {
            radix: Arc::new(Mutex::new(Radix::new(block_size, kv_bytes_at))),
            flows: Flows::new(),
            vertex_index: HashMap::new(),
            adj: Vec::new(),
            stores: Vec::new(),
            tiers: Vec::new(),
            gpu_vertex: Vec::new(),
            node_of: Vec::new(),
            core_edge: None,
            handoff_paths: HashMap::new(),
            made_links: HashSet::new(),
            write_of: Vec::new(),
            evict_backed_first: Vec::new(),
            pending_writes: HashMap::new(),
            next_write_seq: 0,
            write_race_waits: 0,
        }
    }

    fn vertex(&mut self, key: VertexKey) -> VertexId {
        if let Some(&v) = self.vertex_index.get(&key) {
            return v;
        }
        let v = self.adj.len();
        self.adj.push(Vec::new());
        self.vertex_index.insert(key, v);
        v
    }

    /// One template link instance: an edge each way.
    fn link(&mut self, name: &str, a: VertexId, b: VertexId, capacity: f64, latency: f64) {
        if !self.made_links.insert((name.to_string(), a)) {
            return;
        }
        let ab = self.flows.add_edge(name.to_string(), capacity);
        let ba = self.flows.add_edge(name.to_string(), capacity);
        self.adj[a].push(Hop {
            to: b,
            edge: ab,
            latency,
        });
        self.adj[b].push(Hop {
            to: a,
            edge: ba,
            latency,
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn add_store(
        &mut self,
        name: &str,
        scope: Scope,
        owner: usize,
        capacity_blocks: u64,
        throughput: Option<f64>,
        transfer_bandwidth: Option<f64>,
        latency: f64,
        eviction: EvictionPolicy,
    ) -> StoreId {
        let vertex = self.vertex(VertexKey::Named(name.to_string(), owner));
        if let Some(pos) = self.stores.iter().position(|s| s.vertex == vertex) {
            return pos;
        }
        let throughput_edge = throughput.map(|b| self.flows.add_edge(format!("{name}:store"), b));
        let rid = self
            .radix
            .lock()
            .unwrap()
            .add_store(capacity_blocks.min(u32::MAX as u64) as u32, eviction);
        debug_assert_eq!(rid, self.stores.len(), "store ids agree with the tree");
        self.stores.push(Store {
            name: name.to_string(),
            scope,
            owner,
            capacity_blocks,
            vertex,
            throughput_edge,
            transfer_bandwidth,
            latency,
            eviction,
            next: None,
            num_evictions: 0,
            num_expired: 0,
            bytes_written: 0,
            bytes_read: 0,
            dead_bytes: 0,
        });
        self.stores.len() - 1
    }

    /// Shortest hop path from `from` to `to` over the directed edges.
    fn shortest_path(&self, from: VertexId, to: VertexId) -> Option<Path> {
        if from == to {
            return Some(Path::default());
        }
        let mut prev: HashMap<VertexId, (VertexId, EdgeId, f64)> = HashMap::new();
        let mut queue = VecDeque::from([from]);
        let mut seen = HashSet::from([from]);
        while let Some(v) = queue.pop_front() {
            for h in &self.adj[v] {
                if seen.insert(h.to) {
                    prev.insert(h.to, (v, h.edge, h.latency));
                    if h.to == to {
                        let mut edges = Vec::new();
                        let mut latency = 0.0;
                        let mut cur = to;
                        while cur != from {
                            let (p, e, l) = prev[&cur];
                            edges.push(e);
                            latency += l;
                            cur = p;
                        }
                        edges.reverse();
                        return Some(Path { edges, latency });
                    }
                    queue.push_back(h.to);
                }
            }
        }
        None
    }

    /// Instantiate the topology's memory from its pools' hardware
    /// `[memory]` templates and `[memory]` selections. Workers are
    /// numbered pool by pool in `pools` order; nodes likewise (pools never
    /// share a node). `bytes_per_block` quantises store capacities;
    /// `core_bw` bounds the hand-off core (`None` = unbounded).
    pub fn build(
        pools: &[&ClusterSpec],
        block_size: u32,
        kv_bytes_at: KvBytesFn,
        core_bw: Option<f64>,
    ) -> Result<Self, String> {
        let bytes_per_block = kv_bytes_at(block_size);
        let mut g = Self::empty_with(block_size, kv_bytes_at);
        g.core_edge = core_bw.map(|bw| g.flows.add_edge("core", bw));
        let total_nodes: u64 = pools
            .iter()
            .map(|cluster| {
                let (workers, tp) = cluster.graph_workers();
                let gpus_per_node = cluster.hardware.gpus_per_node().max(1) as u64;
                (workers as u64 * tp as u64).div_ceil(gpus_per_node)
            })
            .sum::<u64>()
            .max(1);
        let mut node_base = 0usize;
        for cluster in pools {
            let selection = &cluster.memory;
            let policies = selection.policies();
            let template = cluster.hardware.memory.as_ref();
            selection.validate(template)?;
            // Under DP-attention every rank of a replica is its own worker
            // in the graph: one GPU, its own port into the node's stores.
            let (num_workers, tp) = cluster.graph_workers();
            let gpus_per_node = cluster.hardware.gpus_per_node();
            // Workers packed node by node. A worker wider than a node spans
            // `nodes_per_worker` nodes and pools their node-scoped stores.
            let workers_per_node = (gpus_per_node / tp).max(1) as usize;
            let nodes_per_worker = tp.div_ceil(gpus_per_node).max(1) as u64;
            let first_worker = g.gpu_vertex.len();
            for i in 0..num_workers {
                let w = first_worker + i;
                let node = node_base + i / workers_per_node;
                g.node_of.push(node);
                let gv = g.vertex(VertexKey::Gpu(w));
                g.gpu_vertex.push(gv);
                g.tiers.push(Vec::new());
                g.write_of.push(policies.write);
                g.evict_backed_first.push(policies.hbm_evict_backed_first);
                let Some(t) = template else { continue };
                g.instantiate_worker(
                    t,
                    w,
                    node,
                    tp as f64,
                    nodes_per_worker as f64,
                    total_nodes as f64,
                    bytes_per_block,
                    &selection.tiers,
                    &selection.capacity,
                    policies.eviction,
                )?;
            }
            node_base += num_workers.div_ceil(workers_per_node).max(1);
        }
        Ok(g)
    }

    #[allow(clippy::too_many_arguments)]
    fn instantiate_worker(
        &mut self,
        t: &MemoryTemplate,
        w: WorkerId,
        node: usize,
        gpu_mult: f64,
        node_mult: f64,
        total_nodes: f64,
        bytes_per_block: u64,
        tiers: &[String],
        capacity: &std::collections::BTreeMap<String, f64>,
        eviction: EvictionPolicy,
    ) -> Result<(), String> {
        let gv = self.gpu_vertex[w];
        // Endpoint → (vertex, instance multiplicity) for this worker.
        let resolve = |g: &mut Self, name: &str| -> Result<(VertexId, f64), String> {
            Ok(match name {
                "gpu" => (gv, gpu_mult),
                "switch" => (g.vertex(VertexKey::Switch(node)), 1.0),
                "network" => (g.vertex(VertexKey::Network), 1.0),
                other => {
                    let scope = t
                        .store(other)
                        .map(|s| s.per)
                        .or_else(|| t.junction(other).map(|j| j.per))
                        .ok_or_else(|| format!("[memory] unknown endpoint `{other}`"))?;
                    match scope {
                        Scope::Gpu => (g.vertex(VertexKey::Named(other.to_string(), w)), gpu_mult),
                        Scope::Node => (
                            g.vertex(VertexKey::Named(other.to_string(), node)),
                            node_mult,
                        ),
                        Scope::Cluster => (g.vertex(VertexKey::Named(other.to_string(), 0)), 1.0),
                    }
                }
            })
        };
        // Stores: every one the template declares, capacity from the
        // selection's override or the template.
        for st in &t.stores {
            // A shipped cluster store is an opt-in example. Leaving it out
            // of `tiers` must not add zero-use stores or edges to existing
            // simulations and their summaries.
            if st.per == Scope::Cluster && !tiers.iter().any(|name| name == &st.name) {
                continue;
            }
            let cap_bytes = capacity.get(&st.name).copied().unwrap_or(st.capacity);
            let per_instance = (cap_bytes / bytes_per_block.max(1) as f64).floor() as u64;
            let (owner, mult, throughput, transfer_bandwidth) = match st.per {
                Scope::Gpu => (w, gpu_mult, st.bandwidth.map(|b| b * gpu_mult), None),
                Scope::Node => (node, node_mult, st.bandwidth.map(|b| b * node_mult), None),
                Scope::Cluster => (
                    0,
                    1.0,
                    st.aggregate_bandwidth
                        .or_else(|| st.bandwidth.map(|b| b * total_nodes)),
                    st.bandwidth.map(|b| b * st.stripe as f64),
                ),
            };
            self.add_store(
                &st.name,
                st.per,
                owner,
                (per_instance as f64 * mult) as u64,
                throughput,
                transfer_bandwidth,
                st.latency,
                eviction,
            );
        }
        // Links: one instance per instance of `from` in this worker's scope.
        for l in &t.links {
            let inactive_cluster_store = |endpoint: &str| {
                t.store(endpoint).is_some_and(|s| {
                    s.per == Scope::Cluster && !tiers.iter().any(|name| name == endpoint)
                })
            };
            if inactive_cluster_store(&l.from) || inactive_cluster_store(&l.to) {
                continue;
            }
            let (a, mult) = resolve(self, &l.from)?;
            let (b, _) = resolve(self, &l.to)?;
            self.link(&l.name, a, b, l.bandwidth * mult, l.latency);
        }
        // Tiers, closest first, with the promotion path from each and the
        // write path into each; each store's `next` is its successor.
        let mut prev: Option<StoreId> = None;
        for name in tiers {
            let (sv, _) = resolve(self, name)?;
            let store = self
                .stores
                .iter()
                .position(|s| s.vertex == sv)
                .ok_or_else(|| format!("[memory] tier `{name}` has no store instance"))?;
            let mut fetch = self.shortest_path(sv, gv).ok_or_else(|| {
                format!("[memory] tier `{name}` is not reachable from worker {w}'s gpu")
            })?;
            let mut write = self.shortest_path(gv, sv).ok_or_else(|| {
                format!("[memory] worker {w}'s gpu cannot reach tier `{name}` to write")
            })?;
            if let Some(e) = self.stores[store].throughput_edge {
                fetch.edges.insert(0, e);
                write.edges.push(e);
            }
            fetch.latency += self.stores[store].latency;
            write.latency += self.stores[store].latency;
            if let Some(p) = prev {
                if self.stores[p].next.is_none() && p != store {
                    self.stores[p].next = Some(store);
                }
            }
            prev = Some(store);
            self.tiers[w].push(Tier {
                store,
                fetch_path: fetch,
                write_path: write,
            });
        }
        Ok(())
    }

    /// A graph of `num_workers` workers on one node with the given tiers,
    /// closest first, each `(name, capacity_blocks, bandwidth, per)`: a
    /// `Gpu` store is private per worker, a `Node` store shared by all;
    /// every worker has its own link to each store. For tests and the
    /// hierarchy example. `Cluster` behaves like one topology-wide store.
    pub fn simple(num_workers: usize, tiers: &[(&str, u64, f64, Scope)]) -> Self {
        Self::simple_with(num_workers, tiers, 1, Arc::new(|t| t as u64))
    }

    /// `simple` over a tree with the given block size and KV curve.
    pub fn simple_with(
        num_workers: usize,
        tiers: &[(&str, u64, f64, Scope)],
        block_size: u32,
        kv_bytes_at: KvBytesFn,
    ) -> Self {
        let mut g = Self::empty_with(block_size, kv_bytes_at);
        for w in 0..num_workers {
            let gv = g.vertex(VertexKey::Gpu(w));
            g.gpu_vertex.push(gv);
            g.node_of.push(0);
            g.tiers.push(Vec::new());
            g.write_of.push(WritePolicy::default());
            g.evict_backed_first.push(false);
        }
        let mut prev: Vec<Option<StoreId>> = vec![None; num_workers];
        for &(name, capacity_blocks, bandwidth, per) in tiers {
            #[allow(clippy::needless_range_loop)]
            for w in 0..num_workers {
                let owner = match per {
                    Scope::Gpu => w,
                    Scope::Node => 0,
                    Scope::Cluster => 0,
                };
                let store = g.add_store(
                    name,
                    per,
                    owner,
                    capacity_blocks,
                    None,
                    None,
                    0.0,
                    EvictionPolicy::default(),
                );
                let sv = g.stores[store].vertex;
                let gv = g.gpu_vertex[w];
                g.link(&format!("{name}:link:{w}"), gv, sv, bandwidth, 0.0);
                let fetch = g.shortest_path(sv, gv).unwrap();
                let write = g.shortest_path(gv, sv).unwrap();
                if let Some(p) = prev[w] {
                    if g.stores[p].next.is_none() && p != store {
                        g.stores[p].next = Some(store);
                    }
                }
                prev[w] = Some(store);
                g.tiers[w].push(Tier {
                    store,
                    fetch_path: fetch,
                    write_path: write,
                });
            }
        }
        g
    }

    /// Set every worker's write policy and every store's eviction policy
    /// (tests and examples; configs set them through `[memory]`).
    pub fn with_policies(mut self, write: WritePolicy, eviction: EvictionPolicy) -> Self {
        for w in &mut self.write_of {
            *w = write;
        }
        let mut r = self.radix.lock().unwrap();
        for (i, st) in self.stores.iter_mut().enumerate() {
            st.eviction = eviction;
            r.set_store_eviction(i, eviction);
        }
        drop(r);
        self
    }

    /// Set every worker's HBM recycling preference (tests and examples).
    pub fn with_hbm_evict_backed_first(mut self, on: bool) -> Self {
        for e in &mut self.evict_backed_first {
            *e = on;
        }
        self
    }

    pub fn write_policy(&self, worker: WorkerId) -> WritePolicy {
        self.write_of.get(worker).copied().unwrap_or_default()
    }

    pub fn evict_backed_first(&self, worker: WorkerId) -> bool {
        self.evict_backed_first
            .get(worker)
            .copied()
            .unwrap_or(false)
    }

    /// Private per-worker tiers (see `simple`).
    pub fn private(num_workers: usize, tiers: &[(&str, u64, f64)]) -> Self {
        let spec: Vec<_> = tiers
            .iter()
            .map(|&(n, c, b)| (n, c, b, Scope::Gpu))
            .collect();
        Self::simple(num_workers, &spec)
    }

    /// Private per-worker tiers over a tree with the given block size and
    /// KV curve (what a manager's `with_private_tiers` builds).
    pub fn private_with(
        num_workers: usize,
        tiers: &[(&str, u64, f64)],
        block_size: u32,
        kv_bytes_at: KvBytesFn,
    ) -> Self {
        let spec: Vec<_> = tiers
            .iter()
            .map(|&(n, c, b)| (n, c, b, Scope::Gpu))
            .collect();
        Self::simple_with(num_workers, &spec, block_size, kv_bytes_at)
    }

    /// The KV tree.
    pub fn radix(&self) -> SharedRadix {
        self.radix.clone()
    }

    /// The store ids of `worker`'s tiers, closest first (indices into the
    /// tree's stores, equal to indices into `stores()`).
    pub fn store_ids_of(&self, worker: WorkerId) -> Vec<StoreId> {
        self.tiers
            .get(worker)
            .map(|ts| ts.iter().map(|t| t.store).collect())
            .unwrap_or_default()
    }

    /// Node-shared tiers (see `simple`).
    pub fn shared(num_workers: usize, tiers: &[(&str, u64, f64)]) -> Self {
        let spec: Vec<_> = tiers
            .iter()
            .map(|&(n, c, b)| (n, c, b, Scope::Node))
            .collect();
        Self::simple(num_workers, &spec)
    }

    pub fn shared_handle(self) -> SharedMemoryGraph {
        Arc::new(Mutex::new(self))
    }

    pub fn num_workers(&self) -> usize {
        self.tiers.len()
    }

    pub fn num_tiers(&self, worker: WorkerId) -> usize {
        self.tiers.get(worker).map_or(0, |t| t.len())
    }

    pub fn stores(&self) -> &[Store] {
        &self.stores
    }

    pub fn tiers(&self, worker: WorkerId) -> &[Tier] {
        &self.tiers[worker]
    }

    pub fn node_of(&self, worker: WorkerId) -> usize {
        self.node_of[worker]
    }

    pub fn flows(&self) -> &Flows {
        &self.flows
    }

    /// Which of `worker`'s tiers holds any of `span` (index into
    /// `tiers(worker)`), the closest first.
    pub fn tier_holding(&self, worker: WorkerId, span: Span) -> Option<usize> {
        let r = self.radix.lock().unwrap();
        self.tiers[worker]
            .iter()
            .position(|t| r.store_holds(t.store, span))
    }

    /// Whether any of `worker`'s tiers holds any of `span`.
    pub fn holds(&self, worker: WorkerId, span: Span) -> bool {
        self.tier_holding(worker, span).is_some()
    }

    /// Whether any of `worker`'s tiers holds the block of `hash` (scans
    /// the tree; tests and diagnostics).
    pub fn holds_hash(&self, worker: WorkerId, hash: u64) -> bool {
        let span = self.radix.lock().unwrap().span_of_hash(hash);
        span.is_some_and(|sp| self.holds(worker, sp))
    }

    /// Whether some tier of `worker` holds `span` (resident or arriving):
    /// its HBM blocks can be dropped without a write.
    pub fn is_backed(&self, worker: WorkerId, span: Span) -> bool {
        self.holds(worker, span)
    }

    /// Fresh blocks written into `worker`'s HBM in one allocation: under
    /// `write_through` they go to the first tier as one transfer.
    pub fn produced_batch(&mut self, worker: WorkerId, spans: &[Span]) {
        if matches!(self.write_policy(worker), WritePolicy::WriteThrough {}) {
            let items: Vec<(Span, Option<f64>)> = spans.iter().map(|&s| (s, None)).collect();
            self.write_batch(worker, &items);
        }
    }

    /// HBM hits of one allocation (`(span, hits so far)`): under
    /// `selective` those on their `min_hits`-th hit go to the first tier
    /// as one transfer.
    pub fn hit_batch(&mut self, worker: WorkerId, items: &[(Span, u32)]) {
        if let WritePolicy::Selective { min_hits } = self.write_policy(worker) {
            let n = min_hits.max(1);
            let batch: Vec<(Span, Option<f64>)> = items
                .iter()
                .filter(|&&(_, hits)| hits == n)
                .map(|&(s, _)| (s, None))
                .collect();
            self.write_batch(worker, &batch);
        }
    }

    /// Regions recycled from `worker`'s HBM in one allocation; the ones the
    /// write policy keeps go to the first tier as one transfer. Under
    /// `write_back` every region is written; under `live` only those whose
    /// session announced a re-entry; under the other policies unbacked
    /// regions are dropped.
    pub fn demote_batch(&mut self, worker: WorkerId, items: &[HbmEvicted]) {
        let batch: Vec<(Span, Option<f64>)> = match self.write_policy(worker) {
            WritePolicy::WriteBack {} => items.iter().map(|e| (e.span, e.next_arrival)).collect(),
            WritePolicy::Live {} => items
                .iter()
                .filter(|e| e.next_arrival.is_some())
                .map(|e| (e.span, e.next_arrival))
                .collect(),
            _ => Vec::new(),
        };
        self.write_batch(worker, &batch);
    }

    /// One recycled region (see `demote_batch`).
    pub fn demote(&mut self, worker: WorkerId, span: Span, next_arrival: Option<f64>) {
        self.demote_batch(worker, &[HbmEvicted { span, next_arrival }]);
    }

    /// Write `span` from `worker`'s GPU into its first tier (see
    /// `write_batch`).
    pub fn write(&mut self, worker: WorkerId, span: Span, next_arrival: Option<f64>) {
        self.write_batch(worker, &[(span, next_arrival)]);
    }

    /// Write a batch of regions from `worker`'s GPU into its first tier as
    /// one transfer, skipping what any tier already holds. Every range is
    /// arriving until the transfer lands.
    pub fn write_batch(&mut self, worker: WorkerId, items: &[(Span, Option<f64>)]) {
        if self.tiers[worker].is_empty() || items.is_empty() {
            return;
        }
        let now = self.flows.now();
        let tier = &self.tiers[worker][0];
        let (store, path) = (tier.store, tier.write_path.clone());
        let transfer_bandwidth = self.stores[store].transfer_bandwidth;
        let id = format!("w:{worker}:{store}:{}", self.next_write_seq);
        self.next_write_seq += 1;
        let mut total = 0u64;
        let mut entries: Vec<(StoreId, Span)> = Vec::new();
        let mut evicted = Vec::new();
        {
            let mut r = self.radix.lock().unwrap();
            let all_tiers: Vec<StoreId> = self.tiers[worker].iter().map(|t| t.store).collect();
            for &(span, _) in items {
                // Skip what any tier already holds; write the rest.
                let mut missing = vec![span];
                for &s in &all_tiers {
                    missing = missing
                        .into_iter()
                        .flat_map(|m| r.store_missing(s, m))
                        .collect();
                }
                for m in missing {
                    if m.is_empty() {
                        continue;
                    }
                    total += r.span_bytes(m);
                    entries.push((store, m));
                    evicted.extend(r.store_insert(store, m, Some(id.clone()), now));
                }
            }
        }
        if entries.is_empty() {
            return;
        }
        self.pending_writes.insert(id.clone(), entries);
        self.flows.submit_capped(
            id,
            Owner::Write,
            path.edges,
            total,
            path.latency,
            transfer_bandwidth,
            now,
        );
        self.cascade_batch(store, evicted);
    }

    /// An arriving range of write `id` was dropped (evicted, expired or
    /// removed) before landing: the transfer keeps moving for the rest of
    /// its batch and is cancelled once none remain.
    fn drop_pending(&mut self, id: &str, store: StoreId, span: Span) {
        let empty = match self.pending_writes.get_mut(id) {
            Some(v) => {
                v.retain(|&(s, sp)| {
                    !(s == store
                        && sp.node == span.node
                        && sp.start < span.end
                        && span.start < sp.end)
                });
                v.is_empty()
            }
            None => false,
        };
        if empty {
            self.pending_writes.remove(id);
            self.flows.cancel(id);
        }
    }

    /// Put `span` straight into `worker`'s tier `tier` as resident
    /// (pre-warming a store; evicts like a landing write would).
    pub fn plant(&mut self, worker: WorkerId, tier: usize, span: Span) {
        let now = self.flows.now();
        let store = self.tiers[worker][tier].store;
        let evicted = self
            .radix
            .lock()
            .unwrap()
            .store_insert(store, span, None, now);
        self.cascade_batch(store, evicted);
    }

    /// Account for regions `store` evicted; those going to the next tier
    /// move as one store → store transfer.
    fn cascade_batch(&mut self, store: StoreId, evicted: Vec<TierEvicted>) {
        if evicted.is_empty() {
            return;
        }
        let next = self.stores[store].next;
        let mut moving: Vec<TierEvicted> = Vec::new();
        let nodes: Vec<NodeId> = evicted.iter().map(|e| e.span.node).collect();
        for e in evicted {
            self.stores[store].num_evictions += e.span.len() as u64;
            self.stores[store].dead_bytes += e.dead_bytes;
            if let Some(id) = &e.write_id {
                self.drop_pending(&id.clone(), store, e.span);
            }
            if next.is_some() {
                moving.push(e);
            }
        }
        let (Some(next), false) = (next, moving.is_empty()) else {
            let mut r = self.radix.lock().unwrap();
            for n in nodes {
                r.prune_if_empty(n);
            }
            return;
        };
        let now = self.flows.now();
        let mut path = match self.shortest_path(self.stores[store].vertex, self.stores[next].vertex)
        {
            Some(p) => p,
            None => return,
        };
        if let Some(e) = self.stores[store].throughput_edge {
            path.edges.insert(0, e);
        }
        if let Some(e) = self.stores[next].throughput_edge {
            path.edges.push(e);
        }
        path.latency += self.stores[store].latency + self.stores[next].latency;
        let transfer_bandwidth = match (
            self.stores[store].transfer_bandwidth,
            self.stores[next].transfer_bandwidth,
        ) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        };
        let id = format!("c:{store}:{next}:{}", self.next_write_seq);
        self.next_write_seq += 1;
        let mut total = 0u64;
        let mut entries = Vec::with_capacity(moving.len());
        let mut evicted = Vec::new();
        {
            let mut r = self.radix.lock().unwrap();
            for e in moving {
                for m in r.store_missing(next, e.span) {
                    total += r.span_bytes(m);
                    entries.push((next, m));
                    evicted.extend(r.store_insert(next, m, Some(id.clone()), now));
                }
            }
        }
        {
            let mut r = self.radix.lock().unwrap();
            for n in nodes {
                r.prune_if_empty(n);
            }
        }
        if entries.is_empty() {
            return;
        }
        self.pending_writes.insert(id.clone(), entries);
        self.flows.submit_capped(
            id,
            Owner::Write,
            path.edges,
            total,
            path.latency,
            transfer_bandwidth,
            now,
        );
        self.cascade_batch(next, evicted);
    }

    /// `spans` were promoted from `worker`'s tiers back into its HBM in
    /// one transfer: each tier keeps its copies, marked read (and recently
    /// used under LRU / TTL).
    pub fn promoted_batch(&mut self, worker: WorkerId, spans: &[Span]) {
        let now = self.flows.now();
        let stores: Vec<StoreId> = self.tiers[worker].iter().map(|t| t.store).collect();
        let mut r = self.radix.lock().unwrap();
        for s in stores {
            let read = r.store_promoted(s, spans, now);
            self.stores[s].bytes_read += read;
        }
    }

    /// Remove `span` from whichever of `worker`'s tiers hold it.
    pub fn remove(&mut self, worker: WorkerId, span: Span) {
        let stores: Vec<StoreId> = self.tiers[worker].iter().map(|t| t.store).collect();
        for s in stores {
            let ids = self.radix.lock().unwrap().store_remove(s, span);
            for id in ids {
                self.drop_pending(&id, s, span);
            }
        }
    }

    /// Land finished writes and drop TTL-expired ranges.
    fn settle_writes(&mut self, now: f64) {
        if let Some(completed) = self.flows.take_completed(Owner::Write) {
            for id in completed {
                if let Some(entries) = self.pending_writes.remove(&id) {
                    let mut r = self.radix.lock().unwrap();
                    for (store, span) in entries {
                        r.store_landed(store, &[span]);
                        self.stores[store].bytes_written += r.span_bytes(span);
                    }
                }
            }
        }
        for s in 0..self.stores.len() {
            let EvictionPolicy::Ttl { seconds } = self.stores[s].eviction else {
                continue;
            };
            let expired = self.radix.lock().unwrap().store_expire(s, now, seconds);
            for e in &expired {
                self.stores[s].num_expired += e.span.len() as u64;
                self.stores[s].dead_bytes += e.dead_bytes;
                if let Some(id) = &e.write_id {
                    self.drop_pending(&id.clone(), s, e.span);
                }
            }
            let mut r = self.radix.lock().unwrap();
            for e in expired {
                r.prune_if_empty(e.span.node);
            }
        }
    }

    /// Time promoting `bytes` from `worker`'s tier `tier` into its HBM
    /// would take if started now, at the fetch path's current fair share
    /// (see [`Flows::estimate_new`]).
    pub fn estimate_promotion(&self, worker: WorkerId, tier: usize, bytes: u64) -> f64 {
        let path = &self.tiers[worker][tier].fetch_path;
        self.flows.estimate_new_capped(
            &path.edges,
            bytes as f64,
            path.latency,
            self.stores[self.tiers[worker][tier].store].transfer_bandwidth,
        )
    }

    /// Start moving `bytes` for `request` from `worker`'s tier `tier` into
    /// its HBM. The transfer's id is `promotion_id(request, tier)`. If any
    /// of `hashes` is still arriving in that store, the promotion waits for
    /// the latest of those writes before its bytes flow.
    pub fn submit_promotion(
        &mut self,
        worker: WorkerId,
        tier: usize,
        request: &str,
        bytes: u64,
        spans: &[Span],
        now: f64,
    ) {
        let path = self.tiers[worker][tier].fetch_path.clone();
        let store = self.tiers[worker][tier].store;
        let mut wait = 0.0_f64;
        {
            let r = self.radix.lock().unwrap();
            for &sp in spans {
                for id in r.store_arriving(store, sp) {
                    wait = wait.max(self.flows.estimate_remaining(&id));
                }
            }
        }
        if wait > 0.0 {
            self.write_race_waits += 1;
        }
        self.flows.submit_capped(
            promotion_id(request, tier),
            Owner::Worker(worker),
            path.edges,
            bytes,
            path.latency + wait,
            self.stores[store].transfer_bandwidth,
            now,
        );
    }

    /// The path a hand-off from `from` (prefill worker) to `to` (decode
    /// worker) takes: the shortest GPU-to-GPU route, through the core edge
    /// where it enters the network. Falls back to the core alone when the
    /// hardware gives the GPUs no route (no `[memory]` network links).
    pub fn handoff_path(&mut self, from: WorkerId, to: WorkerId) -> Result<Path, String> {
        if let Some(p) = self.handoff_paths.get(&(from, to)) {
            return Ok(p.clone());
        }
        let (a, b) = (self.gpu_vertex[from], self.gpu_vertex[to]);
        let network = self.vertex_index.get(&VertexKey::Network).copied();
        let path = if from == to {
            Path::default()
        } else {
            match self.shortest_path(a, b) {
                Some(mut p) => {
                    if let (Some(core), Some(net)) = (self.core_edge, network) {
                        if let Some(pos) = self.position_of_vertex(a, &p, net) {
                            p.edges.insert(pos, core);
                        }
                    }
                    p
                }
                None => match self.core_edge {
                    Some(core) => Path {
                        edges: vec![core],
                        latency: 0.0,
                    },
                    None => {
                        return Err(format!(
                            "no hand-off path from worker {from} to worker {to}: give the \
                             hardware [memory] links to \"network\" or set kv_link_bw"
                        ))
                    }
                },
            }
        };
        self.handoff_paths.insert((from, to), path.clone());
        Ok(path)
    }

    /// Number of edges of `path` (walked from `start`) traversed before
    /// arriving at `vertex`; `None` if the path never visits it.
    fn position_of_vertex(&self, start: VertexId, path: &Path, vertex: VertexId) -> Option<usize> {
        let mut cur = start;
        for (i, e) in path.edges.iter().enumerate() {
            if cur == vertex {
                return Some(i);
            }
            cur = self.adj[cur].iter().find(|h| h.edge == *e)?.to;
        }
        (cur == vertex).then_some(path.edges.len())
    }

    /// Start a hand-off of `bytes` for `request` from worker `from` to
    /// worker `to`.
    pub fn submit_handoff(
        &mut self,
        request: &str,
        from: WorkerId,
        to: WorkerId,
        bytes: u64,
        now: f64,
    ) -> Result<(), String> {
        let path = self.handoff_path(from, to)?;
        self.flows.submit(
            request.to_string(),
            Owner::Handoff,
            path.edges,
            bytes,
            path.latency,
            now,
        );
        Ok(())
    }

    /// Advance every transfer to `now`; completions are queued per owner.
    /// Writes that landed become resident; TTL-expired entries are dropped.
    pub fn advance(&mut self, now: f64) -> Vec<(Owner, String)> {
        let done = self.flows.advance(now);
        self.settle_writes(now);
        done
    }

    pub fn take_completed(&mut self, owner: Owner) -> Option<HashSet<String>> {
        self.flows.take_completed(owner)
    }

    pub fn owners_with_completions(&self) -> Vec<Owner> {
        self.flows.owners_with_completions()
    }

    pub fn next_completion_delay(&mut self) -> Option<f64> {
        self.flows.next_completion_delay()
    }

    /// Projected remaining time for `request`'s promotions on `worker`,
    /// tiers taken serially.
    pub fn estimate_promotion_remaining(&mut self, worker: WorkerId, request: &str) -> f64 {
        let n = self.num_tiers(worker);
        (0..n)
            .map(|i| self.flows.estimate_remaining(&promotion_id(request, i)))
            .sum()
    }

    pub fn estimate_remaining(&mut self, id: &str) -> f64 {
        self.flows.estimate_remaining(id)
    }

    /// Per-store occupancy in blocks, by store name (summed over instances).
    pub fn occupancy_by_name(&self) -> HashMap<String, u64> {
        let r = self.radix.lock().unwrap();
        let mut out = HashMap::new();
        for (i, s) in self.stores.iter().enumerate() {
            *out.entry(s.name.clone()).or_insert(0) += r.store_held(i) as u64;
        }
        out
    }

    /// Per-store-name totals over instances: `(capacity_blocks, held
    /// blocks, bytes_written, bytes_read, dead_bytes, evictions, expired)`.
    pub fn store_totals(&self) -> Vec<(String, StoreTotals)> {
        let r = self.radix.lock().unwrap();
        let mut out: Vec<(String, StoreTotals)> = Vec::new();
        for (i, s) in self.stores.iter().enumerate() {
            let t = match out.iter_mut().find(|(n, _)| *n == s.name) {
                Some((_, t)) => t,
                None => {
                    out.push((s.name.clone(), StoreTotals::default()));
                    &mut out.last_mut().unwrap().1
                }
            };
            t.instances += 1;
            t.capacity_blocks += s.capacity_blocks;
            t.held_blocks += r.store_held(i) as u64;
            t.bytes_written += s.bytes_written;
            t.bytes_read += s.bytes_read;
            t.dead_bytes += s.dead_bytes;
            t.evictions += s.num_evictions;
            t.expired += s.num_expired;
        }
        out
    }

    /// Per-edge-name totals over instances: `(instances, capacity per
    /// instance, bytes moved)`.
    pub fn edge_totals(&self) -> Vec<(String, EdgeTotals)> {
        let mut out: Vec<(String, EdgeTotals)> = Vec::new();
        for e in self.flows.edges() {
            let t = match out.iter_mut().find(|(n, _)| *n == e.name) {
                Some((_, t)) => t,
                None => {
                    out.push((e.name.clone(), EdgeTotals::default()));
                    &mut out.last_mut().unwrap().1
                }
            };
            t.instances += 1;
            t.capacity = e.capacity;
            t.bytes_moved += e.bytes_moved;
        }
        out
    }
}

/// Totals over the instances of one store name.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StoreTotals {
    pub instances: u64,
    pub capacity_blocks: u64,
    pub held_blocks: u64,
    pub bytes_written: u64,
    pub bytes_read: u64,
    pub dead_bytes: u64,
    pub evictions: u64,
    pub expired: u64,
}

/// Totals over the instances of one edge name (both directions).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EdgeTotals {
    pub instances: u64,
    /// Capacity of one instance, bytes/s.
    pub capacity: f64,
    pub bytes_moved: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HardwareConfig, MemoryConfig, MemoryTemplate, ParallelConfig};

    fn hardware(memory: &str, gpus_per_node: u32) -> HardwareConfig {
        let mut m: MemoryTemplate = toml::from_str(memory).unwrap();
        m.gpus_per_node = Some(gpus_per_node);
        m.validate().unwrap();
        HardwareConfig {
            name: "test".into(),
            flops_fp4: None,
            flops_fp8: None,
            flops_bf16: Some(1e15),
            flops_fp16: Some(1e15),
            memory_bandwidth: 1e12,
            memory_capacity: 80_000_000_000,
            memory: Some(m),
            fabric: None,
        }
    }

    const MEM: &str = r#"
[[stores]]
name = "host"
per = "node"
capacity = 1000
[[stores]]
name = "local"
per = "gpu"
capacity = 100
[[stores]]
name = "nvme"
per = "node"
capacity = 10000
bandwidth = 5
[[junctions]]
name = "port"
per = "gpu"
[[links]]
name = "pcie"
from = "gpu"
to = "port"
bandwidth = 10
[[links]]
name = "port-host"
from = "port"
to = "host"
bandwidth = 1000
[[links]]
name = "port-nvme"
from = "port"
to = "nvme"
bandwidth = 1000
[[links]]
name = "c2c"
from = "gpu"
to = "local"
bandwidth = 20
[[links]]
name = "nic"
from = "gpu"
to = "network"
bandwidth = 3
"#;

    const CLUSTER_MEM: &str = r#"
gpus_per_node = 1
[[stores]]
name = "pool"
per = "cluster"
capacity = 1000
bandwidth = 10
stripe = 2
latency = 2
[[links]]
name = "nic"
from = "gpu"
to = "network"
bandwidth = 100
latency = 0.5
[[links]]
name = "core"
from = "network"
to = "pool"
bandwidth = 100
latency = 0.25
"#;

    fn cluster(tiers: &[&str], num_workers: u32, tp: u32, gpus_per_node: u32) -> ClusterSpec {
        let sel: MemoryConfig = toml::from_str(&format!(
            "tiers = [{}]",
            tiers
                .iter()
                .map(|t| format!("\"{t}\""))
                .collect::<Vec<_>>()
                .join(", ")
        ))
        .unwrap();
        ClusterSpec {
            hardware: hardware(MEM, gpus_per_node),
            parallel: ParallelConfig {
                tp,
                ep: 1,
                dp_attention: false,
            },
            num_workers,
            memory: sel,
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    /// Build with 10 bytes per block (block size 1, 10 B per token).
    fn build10(pools: &[&ClusterSpec], core: Option<f64>) -> Result<MemoryGraph, String> {
        MemoryGraph::build(pools, 1, Arc::new(|t| 10 * t as u64), core)
    }

    /// The one-block span of `hash` (a root child of its own), inserting it
    /// if new.
    fn sp(g: &MemoryGraph, hash: u64) -> Span {
        let mut r = g.radix.lock().unwrap();
        let path = r.insert(&[hash]);
        Span {
            node: path.segs[0].node,
            start: 0,
            end: 1,
        }
    }

    fn holds(g: &MemoryGraph, w: WorkerId, hash: u64) -> bool {
        g.holds(w, sp(g, hash))
    }

    fn tier_holding(g: &MemoryGraph, w: WorkerId, hash: u64) -> Option<usize> {
        g.tier_holding(w, sp(g, hash))
    }

    fn store_resident(g: &MemoryGraph, s: StoreId, hash: u64) -> bool {
        g.radix.lock().unwrap().store_resident_hash(s, hash)
    }

    fn store_contains(g: &MemoryGraph, s: StoreId, hash: u64) -> bool {
        g.radix.lock().unwrap().store_contains_hash(s, hash)
    }

    fn set_outlook(g: &MemoryGraph, hash: u64, t: Option<f64>) {
        let mut r = g.radix.lock().unwrap();
        let path = r.insert(&[hash]);
        r.set_outlook(&path, t, 1);
    }

    #[test]
    fn instantiates_per_gpu_and_per_node_stores_with_paths() {
        // 4 workers of tp=2 on 4-GPU nodes: two workers per node, two nodes.
        let c = cluster(&["local", "host", "nvme"], 4, 2, 4);
        let g = build10(&[&c], None).unwrap();
        assert_eq!(g.num_workers(), 4);
        assert_eq!(g.node_of(0), 0);
        assert_eq!(g.node_of(1), 0);
        assert_eq!(g.node_of(2), 1);
        // 4 private + 2 host + 2 nvme store instances.
        assert_eq!(g.stores().len(), 8);
        // Private: 100 bytes / 10 per block × tp 2 = 20 blocks; one hop.
        let t0 = &g.tiers(0)[0];
        assert_eq!(g.stores()[t0.store].capacity_blocks, 20);
        assert_eq!(t0.fetch_path.edges.len(), 1);
        assert_eq!(g.flows().edges()[t0.fetch_path.edges[0]].capacity, 40.0);
        // Node store: 100 blocks, shared by workers 0 and 1, not 2; two
        // hops (host → port → gpu).
        let host0 = g.tiers(0)[1].store;
        assert_eq!(g.tiers(1)[1].store, host0);
        assert_ne!(g.tiers(2)[1].store, host0);
        assert_eq!(g.stores()[host0].capacity_blocks, 100);
        assert_eq!(g.tiers(0)[1].fetch_path.edges.len(), 2);
        // The nvme path starts with the store's own throughput edge (5 per
        // node instance) and then shares the same port edge as host.
        let nv = &g.tiers(0)[2].fetch_path;
        assert_eq!(nv.edges.len(), 3);
        assert_eq!(g.flows().edges()[nv.edges[0]].capacity, 5.0);
        assert_eq!(nv.edges[2], g.tiers(0)[1].fetch_path.edges[1]);
    }

    #[test]
    fn cluster_store_is_shared_across_nodes_and_routes_over_nic_and_core() {
        let mut c = cluster(&["pool"], 4, 1, 1);
        c.hardware.memory = Some(toml::from_str(CLUSTER_MEM).unwrap());
        c.memory = toml::from_str("tiers = [\"pool\"]\n[capacity]\npool = 500").unwrap();
        let g = build10(&[&c], None).unwrap();

        assert_eq!(g.stores().len(), 1, "one store for the topology");
        assert_eq!(g.stores()[0].scope, Scope::Cluster);
        assert_eq!(g.stores()[0].owner, 0);
        assert_eq!(g.stores()[0].capacity_blocks, 50, "capacity override");
        assert_eq!(g.tiers(0)[0].store, g.tiers(3)[0].store);
        assert_ne!(g.node_of(0), g.node_of(3));

        let path = &g.tiers(3)[0].fetch_path;
        let names: Vec<&str> = path
            .edges
            .iter()
            .map(|&e| g.flows().edges()[e].name.as_str())
            .collect();
        assert_eq!(names, ["pool:store", "core", "nic"]);
        assert!(close(path.latency, 2.75), "store and link latency");
        assert_eq!(g.stores()[0].transfer_bandwidth, Some(20.0));
        assert_eq!(g.tiers(3)[0].write_path.edges.len(), 3);
        assert!(close(g.tiers(3)[0].write_path.latency, 2.75));
        // Four one-GPU nodes contribute 10 B/s each to the shared edge.
        assert_eq!(
            g.flows().edges()[g.stores()[0].throughput_edge.unwrap()].capacity,
            40.0
        );
        // One access can stripe over two bandwidth units, but no more.
        assert!(close(g.estimate_promotion(3, 0, 100), 2.75 + 5.0));
    }

    #[test]
    fn cluster_stripes_are_per_transfer_and_share_the_aggregate() {
        let mut c = cluster(&["pool"], 4, 1, 1);
        c.hardware.memory = Some(toml::from_str(CLUSTER_MEM).unwrap());
        let mut g = build10(&[&c], None).unwrap();
        for w in 0..3 {
            g.submit_promotion(w, 0, &format!("p{w}"), 100, &[], 0.0);
        }
        // stripe = 2 caps each transfer at 20 B/s; three transfers also
        // share the 4-node aggregate (40 B/s), so each receives 40/3.
        for w in 0..3 {
            assert!(close(
                g.estimate_promotion_remaining(w, &format!("p{w}")),
                2.75 + 7.5
            ));
        }

        let explicit = CLUSTER_MEM.replace("stripe = 2", "stripe = 2\naggregate_bandwidth = 25");
        let mut c = cluster(&["pool"], 2, 1, 1);
        c.hardware.memory = Some(toml::from_str(&explicit).unwrap());
        let mut g = build10(&[&c], None).unwrap();
        g.submit_promotion(0, 0, "a", 100, &[], 0.0);
        g.submit_promotion(1, 0, "b", 100, &[], 0.0);
        // Each transfer could use 20, but the explicit 25 aggregate splits
        // to 12.5 under contention.
        assert!(close(g.estimate_promotion_remaining(0, "a"), 2.75 + 8.0));
        assert!(close(g.estimate_promotion_remaining(1, "b"), 10.75));
    }

    #[test]
    fn empty_selection_builds_a_bare_graph_and_missing_template_errors() {
        let g = build10(&[&cluster(&[], 2, 1, 4)], None).unwrap();
        assert_eq!(g.num_workers(), 2);
        assert_eq!(g.num_tiers(0), 0);

        let mut inactive = cluster(&[], 2, 1, 1);
        inactive.hardware.memory = Some(toml::from_str(CLUSTER_MEM).unwrap());
        let g = build10(&[&inactive], None).unwrap();
        assert!(g.stores().is_empty(), "unselected cluster store is absent");
        assert!(
            g.flows().edges().iter().all(|e| e.name == "nic"),
            "its network and throughput edges are absent too"
        );

        let mut c = cluster(&["host"], 2, 1, 4);
        c.hardware.memory = None;
        assert!(build10(&[&c], None).is_err());
    }

    #[test]
    fn shared_store_is_visible_across_workers_and_cascades() {
        let mut g = MemoryGraph::shared(2, &[("host", 2, 1.0)]);
        let (s1, s2) = (sp(&g, 1), sp(&g, 2));
        g.plant(0, 0, s1);
        g.plant(0, 0, s2);
        assert!(holds(&g, 1, 1));
        assert_eq!(tier_holding(&g, 1, 2), Some(0));
        // Third insert evicts the oldest (1) off the bottom.
        let s3 = sp(&g, 3);
        g.plant(1, 0, s3);
        assert!(!holds(&g, 0, 1));
        assert!(holds(&g, 0, 3));
        assert_eq!(g.stores()[0].num_evictions, 1);
        // Removal takes it away for everyone.
        g.remove(1, s2);
        assert!(!holds(&g, 0, 2));
    }

    #[test]
    fn private_stores_do_not_leak_between_workers() {
        let mut g = MemoryGraph::private(2, &[("local", 4, 1.0), ("host", 4, 1.0)]);
        let s7 = sp(&g, 7);
        g.plant(0, 0, s7);
        assert!(holds(&g, 0, 7));
        assert!(!holds(&g, 1, 7));
        // Cascade: fill tier 0 with 4 more, 7 falls to tier 1.
        for h in 10..14 {
            let s = sp(&g, h);
            g.plant(0, 0, s);
        }
        assert_eq!(tier_holding(&g, 0, 7), Some(1));
        assert_eq!(tier_holding(&g, 0, 13), Some(0));
    }

    #[test]
    fn promotions_share_the_port_and_the_drive() {
        // One worker, host and nvme both behind the 10-wide port; nvme's
        // drive is 5. A host promotion alone runs at 10; with an nvme
        // promotion in flight the port is split 5/5 (the drive is not
        // binding); once it completes host gets 10 back.
        let c = cluster(&["host", "nvme"], 1, 1, 4);
        let mut g = build10(&[&c], None).unwrap();
        g.submit_promotion(0, 0, "a", 100, &[], 0.0);
        assert!(close(g.estimate_promotion_remaining(0, "a"), 10.0));
        g.submit_promotion(0, 1, "b", 25, &[], 0.0);
        assert!(close(g.estimate_remaining(&promotion_id("a", 0)), 20.0));
        assert!(close(g.estimate_remaining(&promotion_id("b", 1)), 5.0));
        let done = g.advance(5.0);
        assert_eq!(done.len(), 1);
        assert_eq!(g.take_completed(Owner::Worker(0)).unwrap().len(), 1);
        // a: 75 left at 10 → 7.5 s.
        assert!(close(g.estimate_promotion_remaining(0, "a"), 7.5));
        // Two nvme promotions: the 5-wide drive binds them at 2.5 each.
        g.submit_promotion(0, 1, "c", 25, &[], 5.0);
        g.submit_promotion(0, 1, "d", 25, &[], 5.0);
        assert!(close(g.estimate_remaining(&promotion_id("c", 1)), 10.0));
        // ...and the port's remaining 5 goes to a (75 left → 15 s).
        assert!(close(g.estimate_promotion_remaining(0, "a"), 15.0));
    }

    /// One worker; tier 0 holds 2 blocks at 10 B/s, tier 1 holds 4 at 5
    /// B/s; 100 bytes per block.
    fn two_tier_private(write: WritePolicy, eviction: EvictionPolicy) -> MemoryGraph {
        MemoryGraph::private_with(
            1,
            &[("t0", 2, 10.0), ("t1", 4, 5.0)],
            1,
            Arc::new(|t| 100 * t as u64),
        )
        .with_policies(write, eviction)
    }

    #[test]
    fn write_back_writes_on_eviction_and_promotions_wait_for_arrival() {
        let mut g = two_tier_private(WritePolicy::WriteBack {}, EvictionPolicy::Fifo {});
        let (s1, s2) = (sp(&g, 1), sp(&g, 2));
        // Production writes nothing under write-back.
        g.produced_batch(0, &[s1]);
        assert!(!holds(&g, 0, 1));
        // Eviction from HBM starts a 100-byte write at 10 B/s: arriving now,
        // resident at t = 10.
        g.demote(0, s1, None);
        assert!(holds(&g, 0, 1));
        assert!(!store_resident(&g, 0, 1));
        assert!(!g.is_backed(0, s2));
        assert!(g.is_backed(0, s1));
        // A promotion of block 1 submitted at t = 4 waits the remaining 6 s
        // of the write before its own 100 bytes (10 s at 10 B/s).
        g.advance(4.0);
        g.submit_promotion(0, 0, "r", 100, &[s1], 4.0);
        assert_eq!(g.write_race_waits, 1);
        assert!(close(g.estimate_promotion_remaining(0, "r"), 6.0 + 10.0));
        g.advance(10.0);
        assert!(store_resident(&g, 0, 1));
        assert_eq!(g.stores()[0].bytes_written, 100);
        // The write's bytes moved on the write path, the promotion's on the
        // fetch path (different edges, same link).
        let done = g.advance(20.0);
        assert_eq!(done.len(), 1);
        assert_eq!(g.take_completed(Owner::Worker(0)).unwrap().len(), 1);
        // Promotion marks the range read; the tier keeps its copy.
        g.promoted_batch(0, &[s1]);
        assert!(holds(&g, 0, 1));
        assert_eq!(g.stores()[0].bytes_read, 100);
        // Re-eviction of a backed block writes nothing.
        let before = g.flows().bytes_submitted_write;
        g.demote(0, s1, None);
        assert_eq!(g.flows().bytes_submitted_write, before);
    }

    #[test]
    fn write_through_writes_on_production_and_selective_on_the_nth_hit() {
        let mut g = two_tier_private(WritePolicy::WriteThrough {}, EvictionPolicy::Fifo {});
        let s7 = sp(&g, 7);
        g.produced_batch(0, &[s7]);
        assert!(holds(&g, 0, 7));
        assert!(close(g.flows().bytes_submitted_write, 100.0));
        g.demote(0, s7, None); // already backed: nothing more
        assert!(close(g.flows().bytes_submitted_write, 100.0));

        let mut g = two_tier_private(
            WritePolicy::Selective { min_hits: 2 },
            EvictionPolicy::Fifo {},
        );
        let s7 = sp(&g, 7);
        g.produced_batch(0, &[s7]);
        g.hit_batch(0, &[(s7, 1)]);
        assert!(!holds(&g, 0, 7));
        g.hit_batch(0, &[(s7, 2)]);
        assert!(holds(&g, 0, 7));
        g.hit_batch(0, &[(s7, 3)]);
        assert!(close(g.flows().bytes_submitted_write, 100.0));
        // An unbacked block evicted under selective is dropped.
        let s8 = sp(&g, 8);
        g.demote(0, s8, None);
        assert!(!holds(&g, 0, 8));
    }

    #[test]
    fn a_full_store_cascades_into_the_next_as_a_transfer_and_counts_dead_bytes() {
        let mut g = two_tier_private(WritePolicy::WriteBack {}, EvictionPolicy::Fifo {});
        // Fill tier 0 (2 blocks) then a third demotion evicts the oldest
        // into tier 1 over the store → store path (10 → gpu → 5: 5 B/s).
        for h in [1, 2] {
            let s = sp(&g, h);
            g.plant(0, 0, s);
        }
        let s3 = sp(&g, 3);
        g.demote(0, s3, None);
        assert_eq!(tier_holding(&g, 0, 1), Some(1));
        assert!(!store_resident(&g, 1, 1));
        assert_eq!(tier_holding(&g, 0, 3), Some(0));
        assert_eq!(g.stores()[0].num_evictions, 1);
        // Block 1 carries 100 bytes: lands at 20 s over the 5 B/s drive.
        g.advance(20.0);
        assert!(store_resident(&g, 1, 1));
        // Fill tier 1 (4) and push one off the bottom unread: dead bytes.
        for h in [11, 12, 13, 14] {
            let s = sp(&g, h);
            g.plant(0, 1, s);
        }
        assert!(!holds(&g, 0, 1));
        assert_eq!(g.stores()[1].dead_bytes, 100);
        assert_eq!(g.stores()[1].num_evictions, 1);
        let totals = g.store_totals();
        assert_eq!(totals[1].0, "t1");
        assert_eq!(totals[1].1.evictions, 1);
    }

    #[test]
    fn lru_refreshes_on_promotion_and_ttl_expires_and_cancels_writes() {
        let mut g = two_tier_private(WritePolicy::WriteBack {}, EvictionPolicy::Lru {});
        let (s1, s2, s3) = (sp(&g, 1), sp(&g, 2), sp(&g, 3));
        g.plant(0, 0, s1);
        g.plant(0, 0, s2);
        // Under LRU, promoting 1 makes 2 the victim.
        g.promoted_batch(0, &[s1]);
        g.plant(0, 0, s3);
        assert!(holds(&g, 0, 1));
        assert_eq!(tier_holding(&g, 0, 2), Some(1)); // cascaded to t1
                                                     // FIFO would have evicted 1 instead.
        let mut f = two_tier_private(WritePolicy::WriteBack {}, EvictionPolicy::Fifo {});
        let (f1, f2, f3) = (sp(&f, 1), sp(&f, 2), sp(&f, 3));
        f.plant(0, 0, f1);
        f.plant(0, 0, f2);
        f.promoted_batch(0, &[f1]);
        f.plant(0, 0, f3);
        assert_eq!(tier_holding(&f, 0, 1), Some(1));
        assert_eq!(tier_holding(&f, 0, 2), Some(0));

        // TTL: a range untouched for > 5 s is dropped on advance, an
        // arriving write is cancelled.
        let mut t = two_tier_private(
            WritePolicy::WriteBack {},
            EvictionPolicy::Ttl { seconds: 5.0 },
        );
        let (t1, t2) = (sp(&t, 1), sp(&t, 2));
        t.plant(0, 0, t1);
        t.advance(3.0);
        t.demote(0, t2, None); // 10 s write, arriving
        assert!(holds(&t, 0, 1) && holds(&t, 0, 2));
        t.advance(6.0);
        assert!(!holds(&t, 0, 1), "expired");
        assert!(holds(&t, 0, 2), "touched at 3");
        assert_eq!(t.stores()[0].num_expired, 1);
        t.advance(9.0);
        assert!(!holds(&t, 0, 2), "expired while arriving");
        assert_eq!(t.flows().num_in_flight(), 0, "write cancelled");
    }

    #[test]
    fn outlook_eviction_drops_dead_then_farthest_and_live_writes_only_announced() {
        // Store t0 holds 2 blocks. Under `outlook` the victim is a block
        // with no re-entry announced, then the farthest re-entry.
        let mut g = two_tier_private(WritePolicy::Live {}, EvictionPolicy::Outlook {});
        let spans: Vec<Span> = (1..=6).map(|h| sp(&g, h)).collect();
        g.demote(0, spans[0], Some(50.0)); // re-entry at 50
        g.demote(0, spans[1], Some(500.0)); // re-entry at 500
        g.demote(0, spans[2], None); // trajectory over: not written at all
        assert!(holds(&g, 0, 1) && holds(&g, 0, 2));
        assert!(!holds(&g, 0, 3), "live: no announced re-entry, no write");
        // Outlook keys come from the node marks: announce them.
        set_outlook(&g, 1, Some(50.0));
        set_outlook(&g, 2, Some(500.0));
        // A third announced block: the farthest (2 @ 500) cascades to t1.
        set_outlook(&g, 4, Some(100.0));
        g.demote(0, spans[3], Some(100.0));
        assert_eq!(tier_holding(&g, 0, 1), Some(0));
        assert_eq!(tier_holding(&g, 0, 4), Some(0));
        assert_eq!(tier_holding(&g, 0, 2), Some(1));
        // The re-entry of 1 moves out past 4's: now 1 is the victim.
        set_outlook(&g, 1, Some(1000.0));
        set_outlook(&g, 5, Some(60.0));
        g.demote(0, spans[4], Some(60.0));
        assert_eq!(tier_holding(&g, 0, 1), Some(1));
        assert!(holds(&g, 0, 4) && holds(&g, 0, 5));
        // Ending 5's trajectory makes it the victim before any announced one.
        set_outlook(&g, 5, None);
        set_outlook(&g, 6, Some(60.0));
        g.demote(0, spans[5], Some(60.0));
        assert!(!store_contains(&g, 0, 5));
        assert!(holds(&g, 0, 4) && holds(&g, 0, 6));
        // write_back with outlook eviction still writes everything.
        let mut w = two_tier_private(WritePolicy::WriteBack {}, EvictionPolicy::Outlook {});
        let w3 = sp(&w, 3);
        w.demote(0, w3, None);
        assert!(holds(&w, 0, 3));
    }

    #[test]
    fn handoff_routes_through_the_network_and_the_core() {
        // Two pools of one worker each, on different nodes; nic 3 each way,
        // core 2 → the core binds.
        let p = cluster(&[], 1, 1, 4);
        let d = cluster(&[], 1, 1, 4);
        let mut g = build10(&[&p, &d], Some(2.0)).unwrap();
        let path = g.handoff_path(0, 1).unwrap();
        // gpu0 → network (nic), core, network → gpu1 (nic).
        assert_eq!(path.edges.len(), 3);
        g.submit_handoff("h", 0, 1, 20, 0.0).unwrap();
        assert!(close(g.estimate_remaining("h"), 10.0));
        // Without a core the NIC binds at 3.
        let mut g2 = build10(&[&p, &d], None).unwrap();
        g2.submit_handoff("h", 0, 1, 30, 0.0).unwrap();
        assert!(close(g2.estimate_remaining("h"), 10.0));
        // Hardware without any network link: the core alone.
        let mut bare = cluster(&[], 1, 1, 4);
        bare.hardware.memory = None;
        let mut g3 = build10(&[&bare, &bare], Some(4.0)).unwrap();
        assert_eq!(g3.handoff_path(0, 1).unwrap().edges.len(), 1);
        g3.submit_handoff("h", 0, 1, 40, 0.0).unwrap();
        assert!(close(g3.estimate_remaining("h"), 10.0));
        // And neither: an error.
        let mut g4 = build10(&[&bare, &bare], None).unwrap();
        assert!(g4.handoff_path(0, 1).is_err());
    }
}
