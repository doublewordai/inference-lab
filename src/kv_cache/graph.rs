//! The KV memory of a topology beyond HBM, as a graph: stores and links
//! instantiated from each pool's hardware `[memory]` template per GPU or
//! per node, joined across nodes by the scale-out network.
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
//! neighbours can promote. A promotion moves bytes along the shortest path
//! from the store to the worker's GPU; a prefill → decode hand-off along
//! the shortest path from one GPU to the other (through the network when
//! they are on different nodes, with an optional core capacity).
//!
//! The graph is shared by the topology's workers behind a mutex; the
//! engine is single-threaded, so the lock only serialises.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use super::flows::{EdgeId, Flows, Owner};
use super::free_queue::outlook_key;
use crate::config::{ClusterSpec, EvictionPolicy, MemoryTemplate, Scope, WritePolicy};

pub type StoreId = usize;
pub type WorkerId = usize;
pub type VertexId = usize;

/// State of a block in a store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryState {
    /// The write has landed; promotable.
    Resident,
    /// The write is in flight; a promotion waits for it.
    Arriving,
}

#[derive(Debug, Clone)]
struct EntryInfo {
    state: EntryState,
    bytes: u64,
    /// Primary recycling key: `0` except under `outlook` eviction, where it
    /// orders blocks farthest-re-entry first (`outlook_key`).
    key: u64,
    /// Announced re-entry, if known; carried down a cascade.
    outlook: Option<f64>,
    /// Position in the recycling order (higher = more recent).
    seq: u64,
    /// Last insert / promotion time.
    touched: f64,
    /// Ever promoted from this store.
    read: bool,
    /// Transfer id of the write, while arriving.
    write_id: Option<String>,
}

/// An entry a full store pushed out on insert.
#[derive(Debug)]
struct Evicted {
    hash: u64,
    bytes: u64,
    /// Transfer id of its write, if still arriving.
    write_id: Option<String>,
    outlook: Option<f64>,
}

/// One instance of a store: block hashes with a capacity, an eviction
/// order, and byte accounting.
#[derive(Debug)]
pub struct Store {
    pub name: String,
    pub scope: Scope,
    /// Node (for `Node` scope) or worker (for `Gpu` scope) this instance
    /// belongs to.
    pub owner: usize,
    pub capacity_blocks: u64,
    pub vertex: VertexId,
    /// The store's own throughput, if bounded.
    pub throughput_edge: Option<EdgeId>,
    pub eviction: EvictionPolicy,
    /// The store a full instance evicts into (the next tier), if any.
    pub next: Option<StoreId>,
    entries: HashMap<u64, EntryInfo>,
    /// Recycling order: smallest `(key, seq)` first.
    order: BTreeMap<(u64, u64), u64>,
    next_seq: u64,
    pub num_evictions: u64,
    /// Entries dropped by TTL.
    pub num_expired: u64,
    /// Bytes whose write landed here.
    pub bytes_written: u64,
    /// Bytes promoted from here.
    pub bytes_read: u64,
    /// Bytes evicted or expired without ever being promoted.
    pub dead_bytes: u64,
}

impl Store {
    pub fn contains(&self, hash: u64) -> bool {
        self.entries.contains_key(&hash)
    }

    pub fn is_resident(&self, hash: u64) -> bool {
        self.entries
            .get(&hash)
            .is_some_and(|e| e.state == EntryState::Resident)
    }

    fn write_id_of(&self, hash: u64) -> Option<&str> {
        self.entries.get(&hash).and_then(|e| e.write_id.as_deref())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Bytes held (resident + arriving).
    pub fn bytes_held(&self) -> u64 {
        self.entries.values().map(|e| e.bytes).sum()
    }

    fn bump(&mut self, hash: u64, now: f64) {
        if let Some(e) = self.entries.get_mut(&hash) {
            self.order.remove(&(e.key, e.seq));
            e.seq = self.next_seq;
            e.touched = now;
            self.next_seq += 1;
            self.order.insert((e.key, e.seq), hash);
        }
    }

    /// The primary recycling key of an entry with `outlook` under this
    /// store's policy.
    fn key_for(&self, outlook: Option<f64>) -> u64 {
        match self.eviction {
            EvictionPolicy::Outlook {} => outlook_key(outlook),
            _ => 0,
        }
    }

    /// `hash`'s announced re-entry changed: re-order it under `outlook`
    /// eviction (other policies ignore it).
    fn set_outlook(&mut self, hash: u64, outlook: Option<f64>) {
        let key = self.key_for(outlook);
        if let Some(e) = self.entries.get_mut(&hash) {
            e.outlook = outlook;
            if e.key == key {
                return;
            }
            self.order.remove(&(e.key, e.seq));
            e.key = key;
            self.order.insert((e.key, e.seq), hash);
        }
    }

    /// Insert `hash`. Returns the evicted entry `(hash, bytes, write id if
    /// it was still arriving)` if the store was full (or has no capacity).
    fn insert(
        &mut self,
        hash: u64,
        bytes: u64,
        state: EntryState,
        write_id: Option<String>,
        outlook: Option<f64>,
        now: f64,
    ) -> Option<Evicted> {
        if self.capacity_blocks == 0 {
            return Some(Evicted {
                hash,
                bytes,
                write_id,
                outlook,
            });
        }
        if self.entries.contains_key(&hash) {
            return None;
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        let key = self.key_for(outlook);
        self.entries.insert(
            hash,
            EntryInfo {
                state,
                bytes,
                key,
                outlook,
                seq,
                touched: now,
                read: false,
                write_id,
            },
        );
        self.order.insert((key, seq), hash);
        if self.entries.len() as u64 > self.capacity_blocks {
            let (&oldest_key, &oldest) = self.order.iter().next().unwrap();
            self.order.remove(&oldest_key);
            let e = self.entries.remove(&oldest).unwrap();
            self.num_evictions += 1;
            if !e.read {
                self.dead_bytes += e.bytes;
            }
            Some(Evicted {
                hash: oldest,
                bytes: e.bytes,
                write_id: e.write_id,
                outlook: e.outlook,
            })
        } else {
            None
        }
    }

    /// The write for `hash` landed.
    fn landed(&mut self, hash: u64) {
        if let Some(e) = self.entries.get_mut(&hash) {
            e.state = EntryState::Resident;
            e.write_id = None;
            self.bytes_written += e.bytes;
        }
    }

    /// `hash` was promoted from here: mark read; refresh recency under
    /// LRU / TTL.
    fn promoted(&mut self, hash: u64, now: f64) {
        let refresh = !matches!(self.eviction, EvictionPolicy::Fifo {});
        if let Some(e) = self.entries.get_mut(&hash) {
            e.read = true;
            self.bytes_read += e.bytes;
        }
        if refresh {
            self.bump(hash, now);
        }
    }

    /// Remove a specific hash. Returns its info if present.
    fn remove(&mut self, hash: u64) -> Option<EntryInfo> {
        let e = self.entries.remove(&hash)?;
        self.order.remove(&(e.key, e.seq));
        Some(e)
    }

    /// Drop entries untouched for longer than the TTL. Returns `(hash,
    /// write id)` for those still arriving.
    fn expire(&mut self, now: f64) -> Vec<(u64, String)> {
        let EvictionPolicy::Ttl { seconds } = self.eviction else {
            return Vec::new();
        };
        let mut cancelled = Vec::new();
        while let Some((&key, &hash)) = self.order.first_key_value() {
            let e = &self.entries[&hash];
            if now - e.touched <= seconds {
                break;
            }
            self.order.remove(&key);
            let e = self.entries.remove(&hash).unwrap();
            self.num_expired += 1;
            if !e.read {
                self.dead_bytes += e.bytes;
            }
            if let Some(id) = e.write_id {
                cancelled.push((hash, id));
            }
        }
        cancelled
    }
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
    /// Store or junction: name and scope key (worker for `Gpu` scope,
    /// node for `Node` scope).
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
    /// In-flight writes: transfer id → the (destination store, hash)
    /// entries it carries that have not landed or been dropped. A batch of
    /// blocks written together moves as one transfer.
    pending_writes: HashMap<String, Vec<(StoreId, u64)>>,
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
    fn empty() -> Self {
        Self {
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
        eviction: EvictionPolicy,
    ) -> StoreId {
        let vertex = self.vertex(VertexKey::Named(name.to_string(), owner));
        if let Some(pos) = self.stores.iter().position(|s| s.vertex == vertex) {
            return pos;
        }
        let throughput_edge = throughput.map(|b| self.flows.add_edge(format!("{name}:store"), b));
        self.stores.push(Store {
            name: name.to_string(),
            scope,
            owner,
            capacity_blocks,
            vertex,
            throughput_edge,
            eviction,
            next: None,
            entries: HashMap::new(),
            order: BTreeMap::new(),
            next_seq: 0,
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
        bytes_per_block: u64,
        core_bw: Option<f64>,
    ) -> Result<Self, String> {
        let mut g = Self::empty();
        g.core_edge = core_bw.map(|bw| g.flows.add_edge("core", bw));
        let mut node_base = 0usize;
        for cluster in pools {
            let selection = &cluster.memory;
            let policies = selection.policies();
            let template = cluster.hardware.memory.as_ref();
            selection.validate(template)?;
            let num_workers = cluster.num_workers.max(1) as usize;
            let tp = cluster.parallel.tp.max(1);
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
                    }
                }
            })
        };
        // Stores: every one the template declares, capacity from the
        // selection's override or the template.
        for st in &t.stores {
            let cap_bytes = capacity.get(&st.name).copied().unwrap_or(st.capacity);
            let per_instance = (cap_bytes / bytes_per_block.max(1) as f64).floor() as u64;
            let (owner, mult) = match st.per {
                Scope::Gpu => (w, gpu_mult),
                Scope::Node => (node, node_mult),
            };
            self.add_store(
                &st.name,
                st.per,
                owner,
                (per_instance as f64 * mult) as u64,
                st.bandwidth.map(|b| b * mult),
                eviction,
            );
        }
        // Links: one instance per instance of `from` in this worker's scope.
        for l in &t.links {
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
    /// hierarchy example.
    pub fn simple(num_workers: usize, tiers: &[(&str, u64, f64, Scope)]) -> Self {
        let mut g = Self::empty();
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
                };
                let store = g.add_store(
                    name,
                    per,
                    owner,
                    capacity_blocks,
                    None,
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
        for st in &mut self.stores {
            st.eviction = eviction;
        }
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

    /// Index of the first of `worker`'s tiers holding `hash`.
    pub fn tier_holding(&self, worker: WorkerId, hash: u64) -> Option<usize> {
        self.tiers[worker]
            .iter()
            .position(|t| self.stores[t.store].contains(hash))
    }

    /// Whether any of `worker`'s tiers holds `hash`.
    pub fn holds(&self, worker: WorkerId, hash: u64) -> bool {
        self.tier_holding(worker, hash).is_some()
    }

    /// Whether some tier of `worker` holds `hash` (resident or arriving):
    /// its HBM block can be dropped without a write.
    pub fn is_backed(&self, worker: WorkerId, hash: u64) -> bool {
        self.holds(worker, hash)
    }

    /// A fresh block of `bytes` for `hash` was just written into
    /// `worker`'s HBM. Under `write_through` it starts its way to the
    /// first tier now.
    pub fn produced(&mut self, worker: WorkerId, hash: u64, bytes: u64) {
        self.produced_batch(worker, &[(hash, bytes)]);
    }

    /// Fresh blocks written into `worker`'s HBM in one allocation: under
    /// `write_through` they go to the first tier as one transfer.
    pub fn produced_batch(&mut self, worker: WorkerId, items: &[(u64, u64)]) {
        if matches!(self.write_policy(worker), WritePolicy::WriteThrough {}) {
            let batch: Vec<(u64, u64, Option<f64>)> =
                items.iter().map(|&(h, b)| (h, b, None)).collect();
            self.write_batch(worker, &batch);
        }
    }

    /// `hash` took its `hits`-th hit in `worker`'s HBM. Under `selective`
    /// the `min_hits`-th hit starts its write.
    pub fn hit(&mut self, worker: WorkerId, hash: u64, bytes: u64, hits: u32) {
        self.hit_batch(worker, &[(hash, bytes, hits)]);
    }

    /// HBM hits of one allocation (`(hash, bytes, hits so far)`): under
    /// `selective` those on their `min_hits`-th hit go to the first tier
    /// as one transfer.
    pub fn hit_batch(&mut self, worker: WorkerId, items: &[(u64, u64, u32)]) {
        if let WritePolicy::Selective { min_hits } = self.write_policy(worker) {
            let n = min_hits.max(1);
            let batch: Vec<(u64, u64, Option<f64>)> = items
                .iter()
                .filter(|&&(_, _, hits)| hits == n)
                .map(|&(h, b, _)| (h, b, None))
                .collect();
            self.write_batch(worker, &batch);
        }
    }

    /// `worker`'s HBM recycled the block holding `hash`, whose announced
    /// re-entry (if any) is `outlook`. Under `write_back` a block no tier
    /// holds is written to the first tier now; under `live` only if a
    /// re-entry is announced; under the other policies an unbacked block
    /// is dropped.
    pub fn demote(&mut self, worker: WorkerId, hash: u64, bytes: u64, outlook: Option<f64>) {
        self.demote_batch(worker, &[(hash, bytes, outlook)]);
    }

    /// Blocks recycled from `worker`'s HBM in one allocation (`(hash,
    /// bytes, outlook)`); the ones the write policy keeps go to the first
    /// tier as one transfer.
    pub fn demote_batch(&mut self, worker: WorkerId, items: &[(u64, u64, Option<f64>)]) {
        let batch: Vec<(u64, u64, Option<f64>)> = match self.write_policy(worker) {
            WritePolicy::WriteBack {} => items.to_vec(),
            WritePolicy::Live {} => items.iter().filter(|i| i.2.is_some()).copied().collect(),
            _ => Vec::new(),
        };
        self.write_batch(worker, &batch);
    }

    /// `hash`'s announced re-entry changed (or ended): whichever of
    /// `worker`'s tiers holds it re-orders it under `outlook` eviction.
    pub fn set_outlook(&mut self, worker: WorkerId, hash: u64, outlook: Option<f64>) {
        if let Some(i) = self.tier_holding(worker, hash) {
            let store = self.tiers[worker][i].store;
            self.stores[store].set_outlook(hash, outlook);
        }
    }

    /// Start writing `hash` (`bytes`) from `worker`'s GPU into its first
    /// tier, unless a tier already holds it. The entry is `Arriving` until
    /// the transfer lands; `outlook` is its announced re-entry, if known.
    pub fn write(&mut self, worker: WorkerId, hash: u64, bytes: u64, outlook: Option<f64>) {
        self.write_batch(worker, &[(hash, bytes, outlook)]);
    }

    /// Write a batch of blocks (`(hash, bytes, outlook)`) from `worker`'s
    /// GPU into its first tier as one transfer, skipping any a tier
    /// already holds. Every entry is `Arriving` until the transfer lands.
    pub fn write_batch(&mut self, worker: WorkerId, items: &[(u64, u64, Option<f64>)]) {
        if self.tiers[worker].is_empty() {
            return;
        }
        let items: Vec<&(u64, u64, Option<f64>)> = items
            .iter()
            .filter(|(h, _, _)| !self.holds(worker, *h))
            .collect();
        if items.is_empty() {
            return;
        }
        let now = self.flows.now();
        let tier = &self.tiers[worker][0];
        let (store, path) = (tier.store, tier.write_path.clone());
        let id = format!("w:{worker}:{store}:{}", self.next_write_seq);
        self.next_write_seq += 1;
        let mut total = 0u64;
        let mut entries = Vec::with_capacity(items.len());
        let mut evicted = Vec::new();
        for &&(hash, bytes, outlook) in &items {
            if let Some(e) = self.stores[store].insert(
                hash,
                bytes,
                EntryState::Arriving,
                Some(id.clone()),
                outlook,
                now,
            ) {
                evicted.push(e);
            }
            // The store may have evicted an earlier entry of this same
            // batch; only entries still present ride the transfer.
            total += bytes;
            entries.push((store, hash));
        }
        self.pending_writes.insert(id.clone(), entries);
        self.flows
            .submit(id, Owner::Write, path.edges, total, path.latency, now);
        self.cascade_batch(store, evicted);
    }

    /// An arriving entry of write `id` was dropped (evicted, expired or
    /// removed) before landing: the transfer keeps moving for the rest of
    /// its batch and is cancelled once none remain.
    fn drop_pending(&mut self, id: &str, store: StoreId, hash: u64) {
        let empty = match self.pending_writes.get_mut(id) {
            Some(v) => {
                v.retain(|&(s, h)| !(s == store && h == hash));
                v.is_empty()
            }
            None => false,
        };
        if empty {
            self.pending_writes.remove(id);
            self.flows.cancel(id);
        }
    }

    /// Put `hash` straight into `worker`'s tier `tier` as resident
    /// (pre-warming a store; evicts like a landing write would).
    pub fn plant(&mut self, worker: WorkerId, tier: usize, hash: u64) {
        let now = self.flows.now();
        let store = self.tiers[worker][tier].store;
        let bytes = self.default_block_bytes();
        let evicted = self.stores[store].insert(hash, bytes, EntryState::Resident, None, None, now);
        self.cascade(store, evicted);
    }

    /// Bytes a planted block is taken to hold: the mean of what the store
    /// holds, else 1 (only affects cascade transfer sizes in tests).
    fn default_block_bytes(&self) -> u64 {
        1
    }

    /// An entry `store` evicted on insert: drop its write if it was still
    /// arriving, then move it to `store`'s next tier as a store → store
    /// transfer, or let it go.
    fn cascade(&mut self, store: StoreId, evicted: Option<Evicted>) {
        if let Some(e) = evicted {
            self.cascade_batch(store, vec![e]);
        }
    }

    /// Entries `store` evicted while inserting a batch: those going to the
    /// next tier move as one store → store transfer.
    fn cascade_batch(&mut self, store: StoreId, evicted: Vec<Evicted>) {
        if evicted.is_empty() {
            return;
        }
        let next = self.stores[store].next;
        let mut moving: Vec<Evicted> = Vec::new();
        for e in evicted {
            if let Some(id) = &e.write_id {
                self.drop_pending(&id.clone(), store, e.hash);
            }
            if next.is_some_and(|n| !self.stores[n].contains(e.hash)) {
                moving.push(e);
            }
        }
        let (Some(next), false) = (next, moving.is_empty()) else {
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
        let id = format!("c:{store}:{next}:{}", self.next_write_seq);
        self.next_write_seq += 1;
        let mut total = 0u64;
        let mut entries = Vec::with_capacity(moving.len());
        let mut evicted = Vec::new();
        for e in moving {
            if let Some(ev) = self.stores[next].insert(
                e.hash,
                e.bytes,
                EntryState::Arriving,
                Some(id.clone()),
                e.outlook,
                now,
            ) {
                evicted.push(ev);
            }
            total += e.bytes;
            entries.push((next, e.hash));
        }
        self.pending_writes.insert(id.clone(), entries);
        self.flows
            .submit(id, Owner::Write, path.edges, total, path.latency, now);
        self.cascade_batch(next, evicted);
    }

    /// `hash` was promoted from `worker`'s tiers back into its HBM: the
    /// tier keeps its copy (KV is immutable); mark it read and, under LRU /
    /// TTL, recently used.
    pub fn promoted(&mut self, worker: WorkerId, hash: u64) {
        let now = self.flows.now();
        if let Some(i) = self.tier_holding(worker, hash) {
            let store = self.tiers[worker][i].store;
            self.stores[store].promoted(hash, now);
        }
    }

    /// Remove `hash` from whichever of `worker`'s tiers holds it.
    pub fn remove(&mut self, worker: WorkerId, hash: u64) {
        for i in 0..self.tiers[worker].len() {
            let store = self.tiers[worker][i].store;
            if let Some(e) = self.stores[store].remove(hash) {
                if let Some(id) = e.write_id {
                    self.drop_pending(&id, store, hash);
                }
                return;
            }
        }
    }

    /// Land finished writes and drop TTL-expired entries.
    fn settle_writes(&mut self, now: f64) {
        for id in self.flows.take_completed(Owner::Write) {
            if let Some(entries) = self.pending_writes.remove(&id) {
                for (store, hash) in entries {
                    self.stores[store].landed(hash);
                }
            }
        }
        for s in 0..self.stores.len() {
            for (hash, id) in self.stores[s].expire(now) {
                self.drop_pending(&id, s, hash);
            }
        }
    }

    /// Time promoting `bytes` from `worker`'s tier `tier` into its HBM
    /// would take if started now, at the fetch path's current fair share
    /// (see [`Flows::estimate_new`]).
    pub fn estimate_promotion(&self, worker: WorkerId, tier: usize, bytes: u64) -> f64 {
        let path = &self.tiers[worker][tier].fetch_path;
        self.flows
            .estimate_new(&path.edges, bytes as f64, path.latency)
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
        hashes: &[u64],
        now: f64,
    ) {
        let path = self.tiers[worker][tier].fetch_path.clone();
        let store = self.tiers[worker][tier].store;
        let mut wait = 0.0_f64;
        for &h in hashes {
            if let Some(id) = self.stores[store].write_id_of(h) {
                wait = wait.max(self.flows.estimate_remaining(id));
            }
        }
        if wait > 0.0 {
            self.write_race_waits += 1;
        }
        self.flows.submit(
            promotion_id(request, tier),
            Owner::Worker(worker),
            path.edges,
            bytes,
            path.latency + wait,
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

    pub fn take_completed(&mut self, owner: Owner) -> HashSet<String> {
        self.flows.take_completed(owner)
    }

    pub fn owners_with_completions(&self) -> Vec<Owner> {
        self.flows.owners_with_completions()
    }

    pub fn next_completion_delay(&self) -> Option<f64> {
        self.flows.next_completion_delay()
    }

    /// Projected remaining time for `request`'s promotions on `worker`,
    /// tiers taken serially.
    pub fn estimate_promotion_remaining(&self, worker: WorkerId, request: &str) -> f64 {
        (0..self.num_tiers(worker))
            .map(|i| self.flows.estimate_remaining(&promotion_id(request, i)))
            .sum()
    }

    pub fn estimate_remaining(&self, id: &str) -> f64 {
        self.flows.estimate_remaining(id)
    }

    /// Per-store occupancy in blocks, by store name (summed over instances).
    pub fn occupancy_by_name(&self) -> HashMap<String, u64> {
        let mut out = HashMap::new();
        for s in &self.stores {
            *out.entry(s.name.clone()).or_insert(0) += s.len() as u64;
        }
        out
    }

    /// Per-store-name totals over instances: `(capacity_blocks, held
    /// blocks, bytes_written, bytes_read, dead_bytes, evictions, expired)`.
    pub fn store_totals(&self) -> Vec<(String, StoreTotals)> {
        let mut out: Vec<(String, StoreTotals)> = Vec::new();
        for s in &self.stores {
            let t = match out.iter_mut().find(|(n, _)| *n == s.name) {
                Some((_, t)) => t,
                None => {
                    out.push((s.name.clone(), StoreTotals::default()));
                    &mut out.last_mut().unwrap().1
                }
            };
            t.instances += 1;
            t.capacity_blocks += s.capacity_blocks;
            t.held_blocks += s.len() as u64;
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

    #[test]
    fn instantiates_per_gpu_and_per_node_stores_with_paths() {
        // 4 workers of tp=2 on 4-GPU nodes: two workers per node, two nodes.
        let c = cluster(&["local", "host", "nvme"], 4, 2, 4);
        let g = MemoryGraph::build(&[&c], 10, None).unwrap();
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
    fn empty_selection_builds_a_bare_graph_and_missing_template_errors() {
        let g = MemoryGraph::build(&[&cluster(&[], 2, 1, 4)], 10, None).unwrap();
        assert_eq!(g.num_workers(), 2);
        assert_eq!(g.num_tiers(0), 0);
        let mut c = cluster(&["host"], 2, 1, 4);
        c.hardware.memory = None;
        assert!(MemoryGraph::build(&[&c], 10, None).is_err());
    }

    #[test]
    fn shared_store_is_visible_across_workers_and_cascades() {
        let mut g = MemoryGraph::shared(2, &[("host", 2, 1.0)]);
        g.plant(0, 0, 1);
        g.plant(0, 0, 2);
        assert!(g.holds(1, 1));
        assert_eq!(g.tier_holding(1, 2), Some(0));
        // Third insert evicts the oldest (1) off the bottom.
        g.plant(1, 0, 3);
        assert!(!g.holds(0, 1));
        assert!(g.holds(0, 3));
        assert_eq!(g.stores()[0].num_evictions, 1);
        // Promotion removes it for everyone.
        g.remove(1, 2);
        assert!(!g.holds(0, 2));
    }

    #[test]
    fn private_stores_do_not_leak_between_workers() {
        let mut g = MemoryGraph::private(2, &[("local", 4, 1.0), ("host", 4, 1.0)]);
        g.plant(0, 0, 7);
        assert!(g.holds(0, 7));
        assert!(!g.holds(1, 7));
        // Cascade: fill tier 0 with 4 more, 7 falls to tier 1.
        for h in 10..14 {
            g.plant(0, 0, h);
        }
        assert_eq!(g.tier_holding(0, 7), Some(1));
        assert_eq!(g.tier_holding(0, 13), Some(0));
    }

    #[test]
    fn promotions_share_the_port_and_the_drive() {
        // One worker, host and nvme both behind the 10-wide port; nvme's
        // drive is 5. A host promotion alone runs at 10; with an nvme
        // promotion in flight the port is split 5/5 (the drive is not
        // binding); once it completes host gets 10 back.
        let c = cluster(&["host", "nvme"], 1, 1, 4);
        let mut g = MemoryGraph::build(&[&c], 10, None).unwrap();
        g.submit_promotion(0, 0, "a", 100, &[], 0.0);
        assert!(close(g.estimate_promotion_remaining(0, "a"), 10.0));
        g.submit_promotion(0, 1, "b", 25, &[], 0.0);
        assert!(close(g.estimate_remaining(&promotion_id("a", 0)), 20.0));
        assert!(close(g.estimate_remaining(&promotion_id("b", 1)), 5.0));
        let done = g.advance(5.0);
        assert_eq!(done.len(), 1);
        assert_eq!(g.take_completed(Owner::Worker(0)).len(), 1);
        // a: 75 left at 10 → 7.5 s.
        assert!(close(g.estimate_promotion_remaining(0, "a"), 7.5));
        // Two nvme promotions: the 5-wide drive binds them at 2.5 each.
        g.submit_promotion(0, 1, "c", 25, &[], 5.0);
        g.submit_promotion(0, 1, "d", 25, &[], 5.0);
        assert!(close(g.estimate_remaining(&promotion_id("c", 1)), 10.0));
        // ...and the port's remaining 5 goes to a (75 left → 15 s).
        assert!(close(g.estimate_promotion_remaining(0, "a"), 15.0));
    }

    fn two_tier_private(write: WritePolicy, eviction: EvictionPolicy) -> MemoryGraph {
        // One worker; tier 0 holds 2 blocks at 10 B/s, tier 1 holds 4 at 5 B/s.
        MemoryGraph::private(1, &[("t0", 2, 10.0), ("t1", 4, 5.0)]).with_policies(write, eviction)
    }

    #[test]
    fn write_back_writes_on_eviction_and_promotions_wait_for_arrival() {
        let mut g = two_tier_private(WritePolicy::WriteBack {}, EvictionPolicy::Fifo {});
        // Production writes nothing under write-back.
        g.produced(0, 1, 100);
        assert!(!g.holds(0, 1));
        // Eviction from HBM starts a 100-byte write at 10 B/s: arriving now,
        // resident at t = 10.
        g.demote(0, 1, 100, None);
        assert!(g.holds(0, 1));
        assert!(!g.stores()[0].is_resident(1));
        assert!(!g.is_backed(0, 2));
        assert!(g.is_backed(0, 1));
        // A promotion of hash 1 submitted at t = 4 waits the remaining 6 s
        // of the write before its own 100 bytes (10 s at 10 B/s).
        g.advance(4.0);
        g.submit_promotion(0, 0, "r", 100, &[1], 4.0);
        assert_eq!(g.write_race_waits, 1);
        assert!(close(g.estimate_promotion_remaining(0, "r"), 6.0 + 10.0));
        g.advance(10.0);
        assert!(g.stores()[0].is_resident(1));
        assert_eq!(g.stores()[0].bytes_written, 100);
        // The write's bytes moved on the write path, the promotion's on the
        // fetch path (different edges, same link).
        let done = g.advance(20.0);
        assert_eq!(done.len(), 1);
        assert_eq!(g.take_completed(Owner::Worker(0)).len(), 1);
        // Promotion marks the entry read; the tier keeps its copy.
        g.promoted(0, 1);
        assert!(g.holds(0, 1));
        assert_eq!(g.stores()[0].bytes_read, 100);
        // Re-eviction of a backed block writes nothing.
        let before = g.flows().bytes_submitted_write;
        g.demote(0, 1, 100, None);
        assert_eq!(g.flows().bytes_submitted_write, before);
    }

    #[test]
    fn write_through_writes_on_production_and_selective_on_the_nth_hit() {
        let mut g = two_tier_private(WritePolicy::WriteThrough {}, EvictionPolicy::Fifo {});
        g.produced(0, 7, 50);
        assert!(g.holds(0, 7));
        assert!(close(g.flows().bytes_submitted_write, 50.0));
        g.demote(0, 7, 50, None); // already backed: nothing more
        assert!(close(g.flows().bytes_submitted_write, 50.0));

        let mut g = two_tier_private(
            WritePolicy::Selective { min_hits: 2 },
            EvictionPolicy::Fifo {},
        );
        g.produced(0, 7, 50);
        g.hit(0, 7, 50, 1);
        assert!(!g.holds(0, 7));
        g.hit(0, 7, 50, 2);
        assert!(g.holds(0, 7));
        g.hit(0, 7, 50, 3);
        assert!(close(g.flows().bytes_submitted_write, 50.0));
        // An unbacked block evicted under selective is dropped.
        g.demote(0, 8, 50, None);
        assert!(!g.holds(0, 8));
    }

    #[test]
    fn a_full_store_cascades_into_the_next_as_a_transfer_and_counts_dead_bytes() {
        let mut g = two_tier_private(WritePolicy::WriteBack {}, EvictionPolicy::Fifo {});
        // Fill tier 0 (2 blocks) then a third demotion evicts the oldest
        // into tier 1 over the store → store path (10 → gpu → 5: 5 B/s).
        for h in [1, 2] {
            g.plant(0, 0, h);
        }
        g.demote(0, 3, 20, None);
        assert_eq!(g.tier_holding(0, 1), Some(1));
        assert!(!g.stores()[1].is_resident(1));
        assert_eq!(g.tier_holding(0, 3), Some(0));
        assert_eq!(g.stores()[0].num_evictions, 1);
        // Planted block 1 carries 1 byte: lands at 1/5 s.
        g.advance(1.0);
        assert!(g.stores()[1].is_resident(1));
        // Fill tier 1 (4) and push one off the bottom unread: dead bytes.
        for h in [11, 12, 13, 14] {
            g.plant(0, 1, h);
        }
        assert!(!g.holds(0, 1));
        assert_eq!(g.stores()[1].dead_bytes, 1);
        assert_eq!(g.stores()[1].num_evictions, 1);
        let totals = g.store_totals();
        assert_eq!(totals[1].0, "t1");
        assert_eq!(totals[1].1.evictions, 1);
    }

    #[test]
    fn lru_refreshes_on_promotion_and_ttl_expires_and_cancels_writes() {
        let mut g = two_tier_private(WritePolicy::WriteBack {}, EvictionPolicy::Lru {});
        g.plant(0, 0, 1);
        g.plant(0, 0, 2);
        // Under LRU, promoting 1 makes 2 the victim.
        g.promoted(0, 1);
        g.plant(0, 0, 3);
        assert!(g.holds(0, 1));
        assert_eq!(g.tier_holding(0, 2), Some(1)); // cascaded to t1
                                                   // FIFO would have evicted 1 instead.
        let mut f = two_tier_private(WritePolicy::WriteBack {}, EvictionPolicy::Fifo {});
        f.plant(0, 0, 1);
        f.plant(0, 0, 2);
        f.promoted(0, 1);
        f.plant(0, 0, 3);
        assert_eq!(f.tier_holding(0, 1), Some(1));
        assert_eq!(f.tier_holding(0, 2), Some(0));

        // TTL: an entry untouched for > 5 s is dropped on advance, an
        // arriving write is cancelled.
        let mut t = two_tier_private(
            WritePolicy::WriteBack {},
            EvictionPolicy::Ttl { seconds: 5.0 },
        );
        t.plant(0, 0, 1);
        t.advance(3.0);
        t.demote(0, 2, 100, None); // 10 s write, arriving
        assert!(t.holds(0, 1) && t.holds(0, 2));
        t.advance(6.0);
        assert!(!t.holds(0, 1), "expired");
        assert!(t.holds(0, 2), "touched at 3");
        assert_eq!(t.stores()[0].num_expired, 1);
        t.advance(9.0);
        assert!(!t.holds(0, 2), "expired while arriving");
        assert_eq!(t.flows().num_in_flight(), 0, "write cancelled");
    }

    #[test]
    fn outlook_eviction_drops_dead_then_farthest_and_live_writes_only_announced() {
        // Store t0 holds 2 blocks. Under `outlook` the victim is a block
        // with no re-entry announced, then the farthest re-entry.
        let mut g = two_tier_private(WritePolicy::Live {}, EvictionPolicy::Outlook {});
        g.demote(0, 1, 100, Some(50.0)); // re-entry at 50
        g.demote(0, 2, 100, Some(500.0)); // re-entry at 500
        g.demote(0, 3, 100, None); // trajectory over: not written at all
        assert!(g.holds(0, 1) && g.holds(0, 2));
        assert!(!g.holds(0, 3), "live: no announced re-entry, no write");
        // A third announced block: the farthest (2 @ 500) cascades to t1.
        g.demote(0, 4, 100, Some(100.0));
        assert_eq!(g.tier_holding(0, 1), Some(0));
        assert_eq!(g.tier_holding(0, 4), Some(0));
        assert_eq!(g.tier_holding(0, 2), Some(1));
        // The re-entry of 1 moves out past 4's: now 1 is the victim.
        g.set_outlook(0, 1, Some(1000.0));
        g.demote(0, 5, 100, Some(60.0));
        assert_eq!(g.tier_holding(0, 1), Some(1));
        assert!(g.holds(0, 4) && g.holds(0, 5));
        // Ending 5's trajectory makes it the victim before any announced one.
        g.set_outlook(0, 5, None);
        g.demote(0, 6, 100, Some(60.0));
        assert!(!g.stores()[0].contains(5));
        assert!(g.holds(0, 4) && g.holds(0, 6));
        // write_back with outlook eviction still writes everything.
        let mut w = two_tier_private(WritePolicy::WriteBack {}, EvictionPolicy::Outlook {});
        w.demote(0, 3, 100, None);
        assert!(w.holds(0, 3));
    }

    #[test]
    fn handoff_routes_through_the_network_and_the_core() {
        // Two pools of one worker each, on different nodes; nic 3 each way,
        // core 2 → the core binds.
        let p = cluster(&[], 1, 1, 4);
        let d = cluster(&[], 1, 1, 4);
        let mut g = MemoryGraph::build(&[&p, &d], 10, Some(2.0)).unwrap();
        let path = g.handoff_path(0, 1).unwrap();
        // gpu0 → network (nic), core, network → gpu1 (nic).
        assert_eq!(path.edges.len(), 3);
        g.submit_handoff("h", 0, 1, 20, 0.0).unwrap();
        assert!(close(g.estimate_remaining("h"), 10.0));
        // Without a core the NIC binds at 3.
        let mut g2 = MemoryGraph::build(&[&p, &d], 10, None).unwrap();
        g2.submit_handoff("h", 0, 1, 30, 0.0).unwrap();
        assert!(close(g2.estimate_remaining("h"), 10.0));
        // Hardware without any network link: the core alone.
        let mut bare = cluster(&[], 1, 1, 4);
        bare.hardware.memory = None;
        let mut g3 = MemoryGraph::build(&[&bare, &bare], 10, Some(4.0)).unwrap();
        assert_eq!(g3.handoff_path(0, 1).unwrap().edges.len(), 1);
        g3.submit_handoff("h", 0, 1, 40, 0.0).unwrap();
        assert!(close(g3.estimate_remaining("h"), 10.0));
        // And neither: an error.
        let mut g4 = MemoryGraph::build(&[&bare, &bare], 10, None).unwrap();
        assert!(g4.handoff_path(0, 1).is_err());
    }
}
