//! The KV memory of a worker pool beyond HBM: stores instantiated from the
//! hardware's `[memory]` template per GPU or per node, and each worker's
//! link to each store it can use.
//!
//! A worker (a `tp`-GPU replica) sees an ordered list of tiers, closest
//! first: the deployment's `[memory] tiers`. A `per = "gpu"` store is
//! private to the worker (its `tp` GPUs' instances pooled); a `per =
//! "node"` store is one instance shared by every worker on that node —
//! what one worker demotes, its neighbours can promote. Every worker
//! reaches a store over its own link (its GPUs' C2C / PCIe ports pooled),
//! so a shared store is contended for capacity, not yet for bandwidth.
//! Multi-hop paths and shared links (NVLink to a peer's HBM, a node's NIC,
//! an NVMe drive's own throughput) are the next step.
//!
//! The graph is shared by the pool's workers behind a mutex; the engine is
//! single-threaded, so the lock only serialises.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use super::link::Link;
use crate::config::{ClusterSpec, Scope};

pub type StoreId = usize;
pub type WorkerId = usize;

/// One instance of a store: a set of block hashes with a capacity.
#[derive(Debug)]
pub struct Store {
    pub name: String,
    pub scope: Scope,
    /// Node (for `Node` scope) or worker (for `Gpu` scope) this instance
    /// belongs to.
    pub owner: usize,
    pub capacity_blocks: u64,
    members: HashSet<u64>,
    /// Insertion order; front is oldest.
    order: VecDeque<u64>,
    pub num_evictions: u64,
}

impl Store {
    fn new(name: String, scope: Scope, owner: usize, capacity_blocks: u64) -> Self {
        Self {
            name,
            scope,
            owner,
            capacity_blocks,
            members: HashSet::new(),
            order: VecDeque::new(),
            num_evictions: 0,
        }
    }

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

/// One worker's access to one store: the store and the worker's own link
/// to it (bandwidth-shared among that worker's transfers on this tier).
#[derive(Debug)]
pub struct Tier {
    pub store: StoreId,
    pub link: Link,
}

/// The pool's KV memory beyond HBM.
#[derive(Debug)]
pub struct MemoryGraph {
    stores: Vec<Store>,
    /// `tiers[worker]`: that worker's tiers, closest first.
    tiers: Vec<Vec<Tier>>,
    /// Node each worker sits on.
    node_of: Vec<usize>,
}

/// A graph shared by a pool's workers.
pub type SharedMemoryGraph = Arc<Mutex<MemoryGraph>>;

impl MemoryGraph {
    /// Instantiate the pool's tiers from its hardware's `[memory]` template
    /// and its `[memory]` selection. `bytes_per_block` quantises store
    /// capacities. Returns `None` when the deployment uses no tiers.
    pub fn build(cluster: &ClusterSpec, bytes_per_block: u64) -> Result<Option<Self>, String> {
        let selection = &cluster.memory;
        if selection.is_empty() {
            return Ok(None);
        }
        let template = cluster.hardware.memory.as_ref().ok_or_else(|| {
            "[memory] tiers set but the hardware has no [memory] block".to_string()
        })?;
        selection.validate(Some(template))?;
        let num_workers = cluster.num_workers.max(1) as usize;
        let tp = cluster.parallel.tp.max(1);
        let gpus_per_node = cluster.hardware.gpus_per_node();
        // Workers packed node by node. A worker wider than a node spans
        // `nodes_per_worker` nodes and pools their node-scoped stores.
        let workers_per_node = (gpus_per_node / tp).max(1) as usize;
        let nodes_per_worker = tp.div_ceil(gpus_per_node).max(1) as u64;
        let node_of: Vec<usize> = (0..num_workers).map(|w| w / workers_per_node).collect();
        let num_nodes = node_of.last().map(|n| n + 1).unwrap_or(0);

        let mut graph = MemoryGraph {
            stores: Vec::new(),
            tiers: (0..num_workers).map(|_| Vec::new()).collect(),
            node_of,
        };
        for name in &selection.tiers {
            let st = template.store(name).expect("validated");
            let link = template.gpu_link_to(name).expect("validated");
            let cap_bytes = selection.capacity.get(name).copied().unwrap_or(st.capacity);
            let per_instance_blocks = (cap_bytes / bytes_per_block.max(1) as f64).floor() as u64;
            // Each worker's link is its `tp` GPUs' ports pooled.
            let worker_bw = link.bandwidth * tp as f64;
            match st.per {
                Scope::Gpu => {
                    for w in 0..num_workers {
                        let id = graph.stores.len();
                        graph.stores.push(Store::new(
                            name.clone(),
                            Scope::Gpu,
                            w,
                            per_instance_blocks * tp as u64,
                        ));
                        graph.tiers[w].push(Tier {
                            store: id,
                            link: Link::new(worker_bw),
                        });
                    }
                }
                Scope::Node => {
                    let first = graph.stores.len();
                    for n in 0..num_nodes {
                        graph.stores.push(Store::new(
                            name.clone(),
                            Scope::Node,
                            n,
                            per_instance_blocks * nodes_per_worker,
                        ));
                    }
                    for w in 0..num_workers {
                        graph.tiers[w].push(Tier {
                            store: first + graph.node_of[w],
                            link: Link::new(worker_bw),
                        });
                    }
                }
            }
        }
        Ok(Some(graph))
    }

    /// A graph of private per-worker tiers, one instance and link per
    /// worker per `(name, capacity_blocks, bandwidth)`; closest first.
    /// What a hardware `[memory]` of `per = "gpu"` stores builds.
    pub fn private(num_workers: usize, tiers: &[(&str, u64, f64)]) -> Self {
        let mut graph = MemoryGraph {
            stores: Vec::new(),
            tiers: (0..num_workers).map(|_| Vec::new()).collect(),
            node_of: vec![0; num_workers],
        };
        for &(name, capacity_blocks, bandwidth) in tiers {
            for w in 0..num_workers {
                let id = graph.stores.len();
                graph
                    .stores
                    .push(Store::new(name.to_string(), Scope::Gpu, w, capacity_blocks));
                graph.tiers[w].push(Tier {
                    store: id,
                    link: Link::new(bandwidth),
                });
            }
        }
        graph
    }

    /// A graph of node-shared tiers: `num_workers` workers on one node,
    /// one instance per `(name, capacity_blocks, bandwidth)` shared by all
    /// of them, each with its own link.
    pub fn shared(num_workers: usize, tiers: &[(&str, u64, f64)]) -> Self {
        let mut graph = MemoryGraph {
            stores: Vec::new(),
            tiers: (0..num_workers).map(|_| Vec::new()).collect(),
            node_of: vec![0; num_workers],
        };
        for &(name, capacity_blocks, bandwidth) in tiers {
            let id = graph.stores.len();
            graph.stores.push(Store::new(
                name.to_string(),
                Scope::Node,
                0,
                capacity_blocks,
            ));
            for w in 0..num_workers {
                graph.tiers[w].push(Tier {
                    store: id,
                    link: Link::new(bandwidth),
                });
            }
        }
        graph
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
        let mut hash = hash;
        for i in 0..self.tiers[worker].len() {
            let store = self.tiers[worker][i].store;
            match self.stores[store].insert(hash) {
                None => return,
                Some(evicted) => hash = evicted,
            }
        }
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

    /// Start moving `bytes` for `id` from `worker`'s tier `tier` into its
    /// HBM.
    pub fn submit(&mut self, worker: WorkerId, tier: usize, id: String, bytes: u64, now: f64) {
        self.tiers[worker][tier].link.submit(id, bytes, now);
    }

    /// Advance `worker`'s tier links to `now`; returns, per tier, the ids
    /// whose transfer on that tier completed.
    pub fn advance(&mut self, worker: WorkerId, now: f64) -> Vec<HashSet<String>> {
        self.tiers[worker]
            .iter_mut()
            .map(|t| t.link.advance(now))
            .collect()
    }

    /// Projected remaining time for `id` on `worker`'s tiers, tiers taken
    /// serially.
    pub fn estimate_remaining(&self, worker: WorkerId, id: &str) -> f64 {
        self.tiers[worker]
            .iter()
            .map(|t| t.link.estimate_remaining(id))
            .sum()
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
[[links]]
name = "pcie"
from = "gpu"
to = "host"
bandwidth = 10
[[links]]
name = "c2c"
from = "gpu"
to = "local"
bandwidth = 20
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

    #[test]
    fn instantiates_per_gpu_and_per_node_stores() {
        // 4 workers of tp=2 on 4-GPU nodes: two workers per node, two nodes.
        let g = MemoryGraph::build(&cluster(&["local", "host"], 4, 2, 4), 10)
            .unwrap()
            .unwrap();
        assert_eq!(g.num_workers(), 4);
        assert_eq!(g.node_of(0), 0);
        assert_eq!(g.node_of(1), 0);
        assert_eq!(g.node_of(2), 1);
        // 4 private stores + 2 node stores.
        assert_eq!(g.stores().len(), 6);
        // Private: 100 bytes / 10 per block × tp 2 = 20 blocks; link 20 × 2.
        let t0 = &g.tiers(0)[0];
        assert_eq!(g.stores()[t0.store].capacity_blocks, 20);
        assert_eq!(t0.link.bandwidth(), 40.0);
        // Node store: 100 blocks, shared by workers 0 and 1, not 2.
        let host0 = g.tiers(0)[1].store;
        assert_eq!(g.tiers(1)[1].store, host0);
        assert_ne!(g.tiers(2)[1].store, host0);
        assert_eq!(g.stores()[host0].capacity_blocks, 100);
    }

    #[test]
    fn empty_selection_builds_nothing_and_missing_template_errors() {
        assert!(MemoryGraph::build(&cluster(&[], 2, 1, 4), 10)
            .unwrap()
            .is_none());
        let mut c = cluster(&["host"], 2, 1, 4);
        c.hardware.memory = None;
        assert!(MemoryGraph::build(&c, 10).is_err());
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
}
