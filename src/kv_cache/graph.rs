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

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use super::flows::{EdgeId, Flows, Owner};
use crate::config::{ClusterSpec, MemoryTemplate, Scope};

pub type StoreId = usize;
pub type WorkerId = usize;
pub type VertexId = usize;

/// One instance of a store: a set of block hashes with a capacity.
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
    members: HashSet<u64>,
    /// Insertion order; front is oldest.
    order: VecDeque<u64>,
    pub num_evictions: u64,
}

impl Store {
    pub fn contains(&self, hash: u64) -> bool {
        self.members.contains(&hash)
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Insert a hash. Returns `Some(evicted)` if at capacity.
    fn insert(&mut self, hash: u64) -> Option<u64> {
        if self.capacity_blocks == 0 {
            return Some(hash);
        }
        if self.members.contains(&hash) {
            return None;
        }
        self.members.insert(hash);
        self.order.push_back(hash);
        if self.members.len() as u64 > self.capacity_blocks {
            let oldest = self.order.pop_front().unwrap();
            self.members.remove(&oldest);
            self.num_evictions += 1;
            Some(oldest)
        } else {
            None
        }
    }

    /// Remove a specific hash (on promotion back to HBM).
    fn remove(&mut self, hash: u64) -> bool {
        if self.members.remove(&hash) {
            if let Some(pos) = self.order.iter().position(|&h| h == hash) {
                self.order.remove(pos);
            }
            true
        } else {
            false
        }
    }
}

/// A route through the graph: edges in order and the latency paid before
/// bytes flow.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Path {
    pub edges: Vec<EdgeId>,
    pub latency: f64,
}

/// One worker's access to one store: the store and the path a promotion
/// from it takes into the worker's HBM.
#[derive(Debug, Clone)]
pub struct Tier {
    pub store: StoreId,
    pub fetch_path: Path,
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

    fn add_store(
        &mut self,
        name: &str,
        scope: Scope,
        owner: usize,
        capacity_blocks: u64,
        throughput: Option<f64>,
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
            members: HashSet::new(),
            order: VecDeque::new(),
            num_evictions: 0,
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
            );
        }
        // Links: one instance per instance of `from` in this worker's scope.
        for l in &t.links {
            let (a, mult) = resolve(self, &l.from)?;
            let (b, _) = resolve(self, &l.to)?;
            self.link(&l.name, a, b, l.bandwidth * mult, l.latency);
        }
        // Tiers, closest first, with the promotion path from each.
        for name in tiers {
            let (sv, _) = resolve(self, name)?;
            let store = self
                .stores
                .iter()
                .position(|s| s.vertex == sv)
                .ok_or_else(|| format!("[memory] tier `{name}` has no store instance"))?;
            let mut path = self.shortest_path(sv, gv).ok_or_else(|| {
                format!("[memory] tier `{name}` is not reachable from worker {w}'s gpu")
            })?;
            if let Some(e) = self.stores[store].throughput_edge {
                path.edges.insert(0, e);
            }
            self.tiers[w].push(Tier {
                store,
                fetch_path: path,
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
        }
        for &(name, capacity_blocks, bandwidth, per) in tiers {
            for w in 0..num_workers {
                let owner = match per {
                    Scope::Gpu => w,
                    Scope::Node => 0,
                };
                let store = g.add_store(name, per, owner, capacity_blocks, None);
                let sv = g.stores[store].vertex;
                let gv = g.gpu_vertex[w];
                g.link(&format!("{name}:link:{w}"), gv, sv, bandwidth, 0.0);
                let path = g.shortest_path(sv, gv).unwrap();
                g.tiers[w].push(Tier {
                    store,
                    fetch_path: path,
                });
            }
        }
        g
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

    /// Push a hash evicted from `worker`'s HBM down its tiers, starting at
    /// the first. If a store evicts to make room, the eviction cascades to
    /// the next tier. Hashes that fall off the bottom are dropped.
    pub fn demote(&mut self, worker: WorkerId, hash: u64) {
        self.plant(worker, 0, hash);
    }

    /// Put `hash` straight into `worker`'s tier `tier` (pre-warming a
    /// store; evicts like a demotion into that tier would).
    pub fn plant(&mut self, worker: WorkerId, tier: usize, hash: u64) {
        let mut hash = hash;
        for i in tier..self.tiers[worker].len() {
            let store = self.tiers[worker][i].store;
            match self.stores[store].insert(hash) {
                None => return,
                Some(evicted) => hash = evicted,
            }
        }
    }

    /// Remove `hash` from whichever of `worker`'s tiers holds it (it was
    /// promoted back to HBM).
    pub fn remove(&mut self, worker: WorkerId, hash: u64) {
        for i in 0..self.tiers[worker].len() {
            let store = self.tiers[worker][i].store;
            if self.stores[store].remove(hash) {
                return;
            }
        }
    }

    /// Start moving `bytes` for `request` from `worker`'s tier `tier` into
    /// its HBM. The transfer's id is `promotion_id(request, tier)`.
    pub fn submit_promotion(
        &mut self,
        worker: WorkerId,
        tier: usize,
        request: &str,
        bytes: u64,
        now: f64,
    ) {
        let path = self.tiers[worker][tier].fetch_path.clone();
        self.flows.submit(
            promotion_id(request, tier),
            Owner::Worker(worker),
            path.edges,
            bytes,
            path.latency,
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
    pub fn advance(&mut self, now: f64) -> Vec<(Owner, String)> {
        self.flows.advance(now)
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
        g.demote(0, 1);
        g.demote(0, 2);
        assert!(g.holds(1, 1));
        assert_eq!(g.tier_holding(1, 2), Some(0));
        // Third insert evicts the oldest (1) off the bottom.
        g.demote(1, 3);
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
        g.demote(0, 7);
        assert!(g.holds(0, 7));
        assert!(!g.holds(1, 7));
        // Cascade: fill tier 0 with 4 more, 7 falls to tier 1.
        for h in 10..14 {
            g.demote(0, h);
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
        g.submit_promotion(0, 0, "a", 100, 0.0);
        assert!(close(g.estimate_promotion_remaining(0, "a"), 10.0));
        g.submit_promotion(0, 1, "b", 25, 0.0);
        assert!(close(g.estimate_remaining(&promotion_id("a", 0)), 20.0));
        assert!(close(g.estimate_remaining(&promotion_id("b", 1)), 5.0));
        let done = g.advance(5.0);
        assert_eq!(done.len(), 1);
        assert_eq!(g.take_completed(Owner::Worker(0)).len(), 1);
        // a: 75 left at 10 → 7.5 s.
        assert!(close(g.estimate_promotion_remaining(0, "a"), 7.5));
        // Two nvme promotions: the 5-wide drive binds them at 2.5 each.
        g.submit_promotion(0, 1, "c", 25, 5.0);
        g.submit_promotion(0, 1, "d", 25, 5.0);
        assert!(close(g.estimate_remaining(&promotion_id("c", 1)), 10.0));
        // ...and the port's remaining 5 goes to a (75 left → 15 s).
        assert!(close(g.estimate_promotion_remaining(0, "a"), 15.0));
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
