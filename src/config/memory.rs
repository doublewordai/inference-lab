//! KV memory beyond HBM: the stores a topology offers and the links that reach
//! them (on the hardware), and which of those a deployment uses as KV
//! tiers (on the deployment).
//!
//! Hardware side — a template instantiated per GPU, per node, or once for
//! the whole cluster:
//!
//! ```toml
//! # catalog/hardware/gh200.toml
//! [memory]
//! [[memory.stores]]
//! name = "grace_dram"; per = "gpu"; capacity = 480e9      # one per superchip
//! [[memory.stores]]
//! name = "nvme"; per = "node"; capacity = 8e12
//! [[memory.stores]]
//! name = "cluster_nvme"; per = "cluster"; capacity = 120e12
//! bandwidth = 56e9; stripe = 4; latency = 2e-3
//! [[memory.links]]
//! name = "c2c"; from = "gpu"; to = "grace_dram"; bandwidth = 450e9
//! [[memory.links]]
//! name = "pcie"; from = "gpu"; to = "nvme"; bandwidth = 64e9
//! [[memory.links]]
//! name = "cluster-storage"; from = "network"; to = "cluster_nvme"; bandwidth = 1e12
//! ```
//!
//! Endpoints are `"gpu"` (one instance per GPU), `"switch"` (the node's
//! scale-up fabric), `"network"` (the scale-out core), a store, or a
//! junction — a named point with no capacity of its own, declared with
//! `[[memory.junctions]] name = "pcie"; per = "gpu"`, so that several links
//! can share one port (`gpu → pcie`, then `pcie → host_dram` and `pcie →
//! nvme`). Every link is full duplex: a template link instantiates one edge
//! each way at `bandwidth`. A transfer's path is the shortest hop path
//! between its ends; its rate is its max-min fair share on every edge of
//! the path (see `kv_cache::flows`). A cluster store has one shared aggregate
//! throughput edge (`aggregate_bandwidth`, or `nodes × bandwidth`) and a
//! private `stripe × bandwidth` cap on each transfer.
//!
//! Deployment side — which stores hold evicted KV, closest first, how much
//! of each they may use, and the write and eviction policies:
//!
//! ```toml
//! [memory]
//! tiers = ["grace_dram", "nvme"]
//! write = { policy = "write_through" }        # write_back | write_through | selective
//! eviction = { policy = "ttl", seconds = 3600 }   # fifo | lru | ttl
//! backup = "on_evict"                              # on_evict | on_land
//! hit_refresh = "first_tier"                       # first_tier | none
//! hbm_evict_backed_first = true
//! [memory.capacity]
//! grace_dram = 200e9
//! ```
//!
//! Tiers are inclusive: a block promoted back to HBM keeps its tier copy
//! (KV is immutable), so its next eviction from HBM is a free drop. Under
//! `write_back` a block no tier holds is written when its HBM block is
//! recycled; under `write_through` every fresh block is written as it is
//! produced; under `selective` a block is written on its `min_hits`-th HBM
//! hit and dropped otherwise. Writes are transfers GPU → store on the same
//! graph as promotions; a block is promotable once its write has landed
//! (a promotion of a block still arriving waits for it). A full store
//! evicts into the next tier — a store → store transfer — or drops. Under
//! `backup = "on_land"` a block is also forwarded to the next tier as soon
//! as its write lands (SGLang HiCache backs a node up to its storage
//! backend the moment its device → host copy completes), so a private host
//! tier's contents reach a shared storage tier within a step rather than
//! when the host tier evicts them; `on_evict` (default) forwards only on
//! eviction. Under `hit_refresh = "first_tier"` (default) a prefix hit in
//! HBM re-stamps the first tier's copy as recently used, so that tier ages
//! with HBM the way HiCache's host tier does (one radix tree, one
//! `last_access_time`); lower tiers see only the references that reach them.
//!
//! `kind = "peer_hbm"` is the one virtual tier: it names KV already
//! resident in a sibling worker's HBM on the same node. It has no capacity
//! or write path; promotions traverse GPU → switch → GPU. A store's `pin`
//! controls whether its source survives until a fetch drains (default true
//! for peer HBM, false for capacity-bearing stores). An unpinned fetch
//! rechecks its source on completion and lands only the surviving prefix.
//!
//! With no `[memory]` on the deployment there is no tiering (HBM only).

use std::collections::BTreeMap;

use serde::Deserialize;

/// How many instances of a store or junction the topology has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    /// One per GPU, private to it (Grace LPDDR behind its superchip's C2C,
    /// a per-GPU NVMe drive).
    Gpu,
    /// One per node, shared by the node's GPUs (host DRAM over PCIe, a
    /// node-local NVMe pool).
    Node,
    /// One for the whole topology, reachable through the scale-out network.
    Cluster,
}

/// What a hardware memory entry represents.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreKind {
    /// A capacity-bearing store whose contents live in the radix tree.
    #[default]
    Store,
    /// The HBM already resident on another worker in the same node. This is
    /// a virtual, read-only tier: it has no capacity or contents of its own.
    PeerHbm,
}

/// When a block's KV is written to the first tier below HBM.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case", deny_unknown_fields)]
pub enum WritePolicy {
    /// At eviction from HBM: a block that no tier holds is written out
    /// when its HBM block is recycled. Writes only what is evicted; the
    /// block reaches the tier late.
    WriteBack {},
    /// At production: every fresh block is written as soon as it is
    /// computed. Tier ingress is the whole KV production rate; eviction
    /// from HBM is then free.
    WriteThrough {},
    /// At the `min_hits`-th HBM hit: only blocks that prove reusable are
    /// written; the rest are dropped on eviction (SGLang HiCache's
    /// `write_through_selective`).
    Selective {
        #[serde(default = "default_min_hits")]
        min_hits: u32,
    },
    /// At eviction from HBM, only when the block's session has announced
    /// a re-entry (an [`Outlook`](crate::request::Outlook)): blocks whose
    /// trajectory is over are dropped. Needs a session workload; on other
    /// workloads no block ever has an outlook and nothing is written.
    Live {},
}

fn default_min_hits() -> u32 {
    1
}

impl Default for WritePolicy {
    fn default() -> Self {
        WritePolicy::WriteBack {}
    }
}

/// Whether an HBM prefix hit re-stamps the tier copies of those blocks as
/// recently used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HitRefresh {
    /// The first tier below HBM ages with HBM: a hit on a resident prefix
    /// counts as a use of its tier copy (HiCache's device and host tiers
    /// share one radix tree and one `last_access_time`). Lower tiers see
    /// only what reaches them.
    #[default]
    FirstTier,
    /// Tier copies are re-stamped only when promoted from.
    None,
}

impl HitRefresh {
    pub fn name(&self) -> &'static str {
        match self {
            HitRefresh::FirstTier => "first_tier",
            HitRefresh::None => "none",
        }
    }
}

/// When a tier forwards a block to the tier below it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackupPolicy {
    /// Only when the tier evicts it (a store → store transfer of what would
    /// otherwise be dropped).
    #[default]
    OnEvict,
    /// As soon as its write into the tier lands: every tier below the first
    /// receives a copy within a transfer of production (HiCache's storage
    /// backup after the device → host DMA).
    OnLand,
}

impl BackupPolicy {
    pub fn name(&self) -> &'static str {
        match self {
            BackupPolicy::OnEvict => "on_evict",
            BackupPolicy::OnLand => "on_land",
        }
    }
}

impl WritePolicy {
    pub fn name(&self) -> &'static str {
        match self {
            WritePolicy::WriteBack {} => "write_back",
            WritePolicy::WriteThrough {} => "write_through",
            WritePolicy::Selective { .. } => "selective",
            WritePolicy::Live {} => "live",
        }
    }
}

/// Which block a store recycles first when full, and whether it drops
/// blocks that have gone untouched.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case", deny_unknown_fields)]
pub enum EvictionPolicy {
    /// Least recently inserted.
    Fifo {},
    /// Least recently inserted or promoted from.
    Lru {},
    /// Least recently used, and any block untouched for `seconds` is
    /// dropped whether or not the store is full.
    Ttl { seconds: f64 },
    /// Blocks with no announced re-entry first (least recently inserted
    /// among them), then the block whose re-entry is farthest away
    /// (Belady's rule over the sessions' outlooks).
    Outlook {},
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        EvictionPolicy::Fifo {}
    }
}

impl EvictionPolicy {
    pub fn name(&self) -> &'static str {
        match self {
            EvictionPolicy::Fifo {} => "fifo",
            EvictionPolicy::Lru {} => "lru",
            EvictionPolicy::Ttl { .. } => "ttl",
            EvictionPolicy::Outlook {} => "outlook",
        }
    }
}

/// Which free HBM block a worker recycles first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case", deny_unknown_fields)]
pub enum HbmEviction {
    /// Least recently freed (the free queue's order).
    Lru {},
    /// Blocks with no announced re-entry first (least recently freed
    /// among them), then the farthest re-entry first, each sequence tail
    /// first so what survives is a prefix.
    Outlook {},
}

impl Default for HbmEviction {
    fn default() -> Self {
        HbmEviction::Lru {}
    }
}

impl HbmEviction {
    pub fn name(&self) -> &'static str {
        match self {
            HbmEviction::Lru {} => "lru",
            HbmEviction::Outlook {} => "outlook",
        }
    }
}

/// Where a re-entry's cached prefix comes from when a tier holds it.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case", deny_unknown_fields)]
pub enum SourcePolicy {
    /// Always promote from the tier (a prefix-cache hit is a hit).
    Promote {},
    /// Promote only if the transfer, at the fetch path's current fair
    /// share, beats recomputing the prefix at the worker's roofline
    /// prefill rate; otherwise recompute it (the tier keeps its copy).
    MinTime {},
}

impl Default for SourcePolicy {
    fn default() -> Self {
        SourcePolicy::Promote {}
    }
}

impl SourcePolicy {
    pub fn name(&self) -> &'static str {
        match self {
            SourcePolicy::Promote {} => "promote",
            SourcePolicy::MinTime {} => "min_time",
        }
    }
}

/// Whether a worker pulls a demoted prefix back before its announced
/// re-entry.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case", deny_unknown_fields)]
pub enum PrefetchPolicy {
    /// Never; a re-entry promotes on arrival.
    None {},
    /// When a block with an outlook leaves HBM, schedule its promotion so
    /// it lands (at the path's current fair share) `lead` seconds before
    /// the re-entry; a block already overdue starts at once.
    Outlook {
        #[serde(default)]
        lead: f64,
    },
}

impl Default for PrefetchPolicy {
    fn default() -> Self {
        PrefetchPolicy::None {}
    }
}

impl PrefetchPolicy {
    pub fn name(&self) -> &'static str {
        match self {
            PrefetchPolicy::None {} => "none",
            PrefetchPolicy::Outlook { .. } => "outlook",
        }
    }
}

/// A named bundle of movement policies. Explicit `[memory]` fields
/// override the preset's choices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryPreset {
    /// Decides only from what has already happened, the way shipped
    /// stacks do: promote on any hit, LRU in HBM and in every tier,
    /// write on the first HBM hit (SGLang HiCache's selective
    /// write-through), no prefetch, recycle backed blocks first.
    Reactive,
    /// Knows every session's next re-entry: fetch or recompute by time,
    /// Belady eviction in HBM and the tiers, write only live trajectories,
    /// prefetch to land on arrival.
    Oracle,
}

impl MemoryPreset {
    pub fn name(&self) -> &'static str {
        match self {
            MemoryPreset::Reactive => "reactive",
            MemoryPreset::Oracle => "oracle",
        }
    }
}

/// The movement policies a deployment runs, every choice made.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryPolicies {
    pub source: SourcePolicy,
    pub hbm_eviction: HbmEviction,
    pub write: WritePolicy,
    pub eviction: EvictionPolicy,
    pub prefetch: PrefetchPolicy,
    pub backup: BackupPolicy,
    pub hit_refresh: HitRefresh,
    pub hbm_evict_backed_first: bool,
}

impl Default for MemoryPolicies {
    fn default() -> Self {
        Self {
            source: SourcePolicy::Promote {},
            hbm_eviction: HbmEviction::Lru {},
            write: WritePolicy::WriteBack {},
            eviction: EvictionPolicy::Fifo {},
            prefetch: PrefetchPolicy::None {},
            backup: BackupPolicy::OnEvict,
            hit_refresh: HitRefresh::FirstTier,
            hbm_evict_backed_first: false,
        }
    }
}

impl MemoryPolicies {
    pub fn preset(preset: MemoryPreset) -> Self {
        match preset {
            MemoryPreset::Reactive => Self {
                source: SourcePolicy::Promote {},
                hbm_eviction: HbmEviction::Lru {},
                write: WritePolicy::Selective { min_hits: 1 },
                eviction: EvictionPolicy::Lru {},
                prefetch: PrefetchPolicy::None {},
                backup: BackupPolicy::OnEvict,
                hit_refresh: HitRefresh::FirstTier,
                hbm_evict_backed_first: true,
            },
            MemoryPreset::Oracle => Self {
                source: SourcePolicy::MinTime {},
                hbm_eviction: HbmEviction::Outlook {},
                write: WritePolicy::Live {},
                eviction: EvictionPolicy::Outlook {},
                prefetch: PrefetchPolicy::Outlook { lead: 0.0 },
                backup: BackupPolicy::OnEvict,
                hit_refresh: HitRefresh::FirstTier,
                hbm_evict_backed_first: true,
            },
        }
    }

    /// Whether any policy reads the sessions' outlooks.
    pub fn uses_outlook(&self) -> bool {
        matches!(self.hbm_eviction, HbmEviction::Outlook {})
            || matches!(self.write, WritePolicy::Live {})
            || matches!(self.eviction, EvictionPolicy::Outlook {})
            || matches!(self.prefetch, PrefetchPolicy::Outlook { .. })
    }
}

/// A store a hardware topology offers for KV beyond HBM.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoreTemplate {
    pub name: String,
    pub per: Scope,
    /// Normal capacity-bearing store, or the virtual peer-HBM tier.
    #[serde(default)]
    pub kind: StoreKind,
    /// Bytes per instance.
    #[serde(default)]
    pub capacity: f64,
    /// The store's own throughput per instance, bytes/s (an NVMe pool's
    /// aggregate drive rate). For a cluster store this is one node's rate;
    /// `stripe` and `aggregate_bandwidth` turn it into the two access limits.
    /// Unset: unbounded — only the links limit.
    #[serde(default)]
    pub bandwidth: Option<f64>,
    /// Number of this cluster store's bandwidth units one transfer may use
    /// in parallel. The transfer is still bounded by the shared aggregate
    /// throughput. Defaults to one.
    #[serde(default = "default_stripe")]
    pub stripe: u32,
    /// Explicit shared throughput across every transfer to or from a cluster
    /// store, bytes/s. By default this is `nodes × bandwidth`.
    #[serde(default)]
    pub aggregate_bandwidth: Option<f64>,
    /// Fixed access cost per fetch or write, seconds.
    #[serde(default)]
    pub latency: f64,
    /// Keep a promotion's source resident until the transfer drains. The
    /// virtual peer-HBM tier defaults to pinned; normal stores preserve the
    /// historical unpinned behaviour.
    #[serde(default)]
    pub pin: Option<bool>,
}

impl StoreTemplate {
    pub fn pins_fetches(&self) -> bool {
        self.pin.unwrap_or(matches!(self.kind, StoreKind::PeerHbm))
    }
}

fn default_stripe() -> u32 {
    1
}

/// A named point in the graph with no capacity of its own, so that several
/// links can share one (a GPU's PCIe port feeding host DRAM and NVMe).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JunctionTemplate {
    pub name: String,
    pub per: Scope,
}

/// A graph link. `from` is `"gpu"` (one instance per GPU, its own port),
/// `"network"`, a store or a junction; `to` is a store, a junction, `"switch"`
/// (the node's scale-up fabric) or `"network"` (the scale-out core). One
/// instance per instance of `from`, full duplex at `bandwidth` each way.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkTemplate {
    pub name: String,
    pub from: String,
    pub to: String,
    /// Bytes/s per direction, per instance.
    pub bandwidth: f64,
    /// Fixed cost per transfer, seconds.
    #[serde(default)]
    pub latency: f64,
}

/// The KV memory a hardware class offers: stores and the links that reach
/// them. Ships with the hardware preset; a deployment picks tiers from it.
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct MemoryTemplate {
    /// GPUs sharing one node's `per = "node"` stores. Defaults to the
    /// fabric's `gpus_per_node`, else 1.
    #[serde(default)]
    pub gpus_per_node: Option<u32>,
    #[serde(default)]
    pub stores: Vec<StoreTemplate>,
    #[serde(default)]
    pub junctions: Vec<JunctionTemplate>,
    #[serde(default)]
    pub links: Vec<LinkTemplate>,
}

impl MemoryTemplate {
    pub fn store(&self, name: &str) -> Option<&StoreTemplate> {
        self.stores.iter().find(|s| s.name == name)
    }

    pub fn normal_store(&self, name: &str) -> Option<&StoreTemplate> {
        self.store(name)
            .filter(|s| matches!(s.kind, StoreKind::Store))
    }

    pub fn junction(&self, name: &str) -> Option<&JunctionTemplate> {
        self.junctions.iter().find(|j| j.name == name)
    }

    /// The direct GPU → `store` link, if the template has one.
    pub fn gpu_link_to(&self, store: &str) -> Option<&LinkTemplate> {
        self.links.iter().find(|l| l.from == "gpu" && l.to == store)
    }

    /// Whether a GPU can reach `target` over the template's links (any
    /// number of hops, following links in their declared direction).
    pub fn gpu_reaches(&self, target: &str) -> bool {
        if self
            .store(target)
            .is_some_and(|s| matches!(s.kind, StoreKind::PeerHbm))
        {
            return self
                .links
                .iter()
                .any(|l| l.from == "gpu" && l.to == "switch");
        }
        let mut frontier = vec!["gpu".to_string()];
        let mut seen = std::collections::HashSet::new();
        while let Some(v) = frontier.pop() {
            if v == target {
                return true;
            }
            if !seen.insert(v.clone()) {
                continue;
            }
            for l in self.links.iter().filter(|l| l.from == v) {
                frontier.push(l.to.clone());
            }
        }
        false
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut seen = std::collections::HashSet::new();
        for s in &self.stores {
            if !seen.insert(s.name.as_str()) {
                return Err(format!("[memory] store `{}` declared twice", s.name));
            }
            match s.kind {
                StoreKind::Store => {
                    if s.name == "peer_hbm" {
                        return Err(
                            "[memory] `peer_hbm` is reserved for kind = \"peer_hbm\"".to_string()
                        );
                    }
                    if s.capacity <= 0.0 {
                        return Err(format!("[memory] store `{}` needs capacity > 0", s.name));
                    }
                    if s.bandwidth.is_some_and(|b| b <= 0.0) {
                        return Err(format!("[memory] store `{}` needs bandwidth > 0", s.name));
                    }
                }
                StoreKind::PeerHbm => {
                    if s.name != "peer_hbm" {
                        return Err(format!(
                            "[memory] peer_hbm kind must use the reserved name `peer_hbm`, got `{}`",
                            s.name
                        ));
                    }
                    if s.per != Scope::Node {
                        return Err("[memory] `peer_hbm` must be per = \"node\"".to_string());
                    }
                    if s.capacity != 0.0 || s.bandwidth.is_some() {
                        return Err(
                            "[memory] `peer_hbm` has no capacity or bandwidth of its own"
                                .to_string(),
                        );
                    }
                }
            }
            if s.stripe == 0 {
                return Err(format!("[memory] store `{}` needs stripe > 0", s.name));
            }
            if s.aggregate_bandwidth.is_some_and(|b| b <= 0.0) {
                return Err(format!(
                    "[memory] store `{}` needs aggregate_bandwidth > 0",
                    s.name
                ));
            }
            if s.latency < 0.0 {
                return Err(format!("[memory] store `{}` needs latency >= 0", s.name));
            }
            if s.per != Scope::Cluster && (s.stripe != 1 || s.aggregate_bandwidth.is_some()) {
                return Err(format!(
                    "[memory] store `{}` uses stripe/aggregate_bandwidth but is not per = \"cluster\"",
                    s.name
                ));
            }
        }
        for j in &self.junctions {
            if ["gpu", "switch", "network"].contains(&j.name.as_str())
                || !seen.insert(j.name.as_str())
            {
                return Err(format!(
                    "[memory] junction `{}` clashes with a store or a reserved name",
                    j.name
                ));
            }
        }
        for l in &self.links {
            let from_ok = l.from == "gpu"
                || l.from == "network"
                || self.normal_store(&l.from).is_some()
                || self.junction(&l.from).is_some();
            if !from_ok {
                return Err(format!(
                    "[memory] link `{}`: from must be \"gpu\", \"network\", a store or a junction, got \"{}\"",
                    l.name, l.from
                ));
            }
            let to_ok = l.to == "switch"
                || l.to == "network"
                || self.normal_store(&l.to).is_some()
                || self.junction(&l.to).is_some();
            if !to_ok {
                return Err(format!(
                    "[memory] link `{}`: to must be a store, a junction, \"switch\" or \"network\", got \"{}\"",
                    l.name, l.to
                ));
            }
            if l.from == l.to {
                return Err(format!(
                    "[memory] link `{}` joins `{}` to itself",
                    l.name, l.from
                ));
            }
            if l.bandwidth <= 0.0 {
                return Err(format!("[memory] link `{}` needs bandwidth > 0", l.name));
            }
        }
        Ok(())
    }
}

/// Which of the hardware's stores a deployment uses as KV tiers.
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct MemoryConfig {
    /// Store names, closest to HBM first. KV evicted from HBM falls through
    /// them in this order and is promoted back over each store's link
    /// instead of being recomputed. Empty: no tiering.
    #[serde(default)]
    pub tiers: Vec<String>,
    /// Per-store cap on the bytes (per instance) given to KV; defaults to
    /// the store's full capacity.
    #[serde(default)]
    pub capacity: BTreeMap<String, f64>,
    /// A named bundle of the policies below; explicit fields override it.
    /// Default: `promote` / `lru` / `write_back` / `fifo` / no prefetch /
    /// no backed-first recycling.
    #[serde(default)]
    pub preset: Option<MemoryPreset>,
    /// Where a re-entry's tier-held prefix comes from: `promote` or
    /// `min_time` (fetch vs recompute by time).
    #[serde(default)]
    pub source: Option<SourcePolicy>,
    /// Which free HBM block is recycled first: `lru` or `outlook`.
    #[serde(default)]
    pub hbm_eviction: Option<HbmEviction>,
    /// When KV is written to the first tier.
    #[serde(default)]
    pub write: Option<WritePolicy>,
    /// How every tier picks what to recycle.
    #[serde(default)]
    pub eviction: Option<EvictionPolicy>,
    /// Whether demoted prefixes are pulled back ahead of their re-entry.
    #[serde(default)]
    pub prefetch: Option<PrefetchPolicy>,
    /// When a tier forwards a block to the tier below: on eviction, or as
    /// soon as the block lands.
    #[serde(default)]
    pub backup: Option<BackupPolicy>,
    /// Whether an HBM prefix hit re-stamps the first tier's copies as used.
    #[serde(default)]
    pub hit_refresh: Option<HitRefresh>,
    /// When HBM must recycle a block, prefer one whose KV a tier already
    /// holds (dropping it is free) over the policy's first choice,
    /// looking a bounded distance up the free queue.
    #[serde(default)]
    pub hbm_evict_backed_first: Option<bool>,
}

impl MemoryConfig {
    /// The policies this deployment runs: the preset's choices (or the
    /// defaults) with every explicit field applied over them.
    pub fn policies(&self) -> MemoryPolicies {
        let mut p = self.preset.map(MemoryPolicies::preset).unwrap_or_default();
        if let Some(v) = self.source {
            p.source = v;
        }
        if let Some(v) = self.hbm_eviction {
            p.hbm_eviction = v;
        }
        if let Some(v) = self.write {
            p.write = v;
        }
        if let Some(v) = self.eviction {
            p.eviction = v;
        }
        if let Some(v) = self.prefetch {
            p.prefetch = v;
        }
        if let Some(v) = self.backup {
            p.backup = v;
        }
        if let Some(v) = self.hit_refresh {
            p.hit_refresh = v;
        }
        if let Some(v) = self.hbm_evict_backed_first {
            p.hbm_evict_backed_first = v;
        }
        p
    }
}

impl MemoryConfig {
    pub fn is_empty(&self) -> bool {
        self.tiers.is_empty()
    }

    /// Check every tier names a store the template has, reachable from a
    /// GPU over its links, and every capacity override names a tier.
    pub fn validate(&self, template: Option<&MemoryTemplate>) -> Result<(), String> {
        if self.tiers.is_empty() {
            if let Some(name) = self.capacity.keys().next() {
                return Err(format!(
                    "[memory] capacity override for `{name}` but no tiers"
                ));
            }
            return Ok(());
        }
        let Some(template) = template else {
            return Err(format!(
                "[memory] tiers {:?} but the hardware declares no [memory] stores",
                self.tiers
            ));
        };
        let mut seen = std::collections::HashSet::new();
        for name in &self.tiers {
            if !seen.insert(name.as_str()) {
                return Err(format!("[memory] tier `{name}` listed twice"));
            }
            if template.store(name).is_none() {
                let have: Vec<&str> = template.stores.iter().map(|s| s.name.as_str()).collect();
                return Err(format!(
                    "[memory] tier `{name}` is not a store of this hardware (have: {})",
                    have.join(", ")
                ));
            }
            if !template.gpu_reaches(name) {
                return Err(format!(
                    "[memory] tier `{name}` is not reachable from a gpu over this hardware's links"
                ));
            }
        }
        for (name, cap) in &self.capacity {
            if !self.tiers.iter().any(|t| t == name) {
                return Err(format!(
                    "[memory] capacity override for `{name}`, which is not a tier"
                ));
            }
            if *cap <= 0.0 {
                return Err(format!("[memory] capacity for `{name}` must be > 0"));
            }
            if template
                .store(name)
                .is_some_and(|s| matches!(s.kind, StoreKind::PeerHbm))
            {
                return Err("[memory] `peer_hbm` has no capacity to override".to_string());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEMPLATE: &str = r#"
gpus_per_node = 4
[[stores]]
name = "grace_dram"
per = "gpu"
capacity = 480e9
[[stores]]
name = "nvme"
per = "node"
capacity = 8e12
[[links]]
name = "c2c"
from = "gpu"
to = "grace_dram"
bandwidth = 450e9
[[links]]
name = "pcie"
from = "gpu"
to = "nvme"
bandwidth = 64e9
[[links]]
name = "nvlink"
from = "gpu"
to = "switch"
bandwidth = 450e9
"#;

    #[test]
    fn parses_and_validates_a_template() {
        let t: MemoryTemplate = toml::from_str(TEMPLATE).unwrap();
        t.validate().unwrap();
        assert_eq!(t.gpus_per_node, Some(4));
        assert_eq!(t.store("nvme").unwrap().per, Scope::Node);
        assert_eq!(t.gpu_link_to("grace_dram").unwrap().bandwidth, 450e9);
        assert!(t.gpu_link_to("nope").is_none());
    }

    #[test]
    fn parses_cluster_store_controls_and_their_defaults() {
        let t: MemoryTemplate = toml::from_str(
            r#"
[[stores]]
name = "pool"
per = "cluster"
capacity = 120e12
bandwidth = 56e9
stripe = 4
aggregate_bandwidth = 200e9
latency = 2e-3
[[links]]
name = "pool-core"
from = "network"
to = "pool"
bandwidth = 1e12
"#,
        )
        .unwrap();
        t.validate().unwrap();
        let pool = t.store("pool").unwrap();
        assert_eq!(pool.per, Scope::Cluster);
        assert_eq!(pool.stripe, 4);
        assert_eq!(pool.aggregate_bandwidth, Some(200e9));
        assert_eq!(pool.latency, 2e-3);

        let defaults: MemoryTemplate =
            toml::from_str("[[stores]]\nname = \"pool\"\nper = \"cluster\"\ncapacity = 1.0")
                .unwrap();
        let pool = defaults.store("pool").unwrap();
        assert_eq!(pool.stripe, 1);
        assert_eq!(pool.aggregate_bandwidth, None);
        assert_eq!(pool.latency, 0.0);
    }

    #[test]
    fn peer_hbm_is_virtual_node_scoped_and_pinned_by_default() {
        let text = format!(
            "{TEMPLATE}\n[[stores]]\nname = \"peer_hbm\"\nper = \"node\"\nkind = \"peer_hbm\"\n"
        );
        let t: MemoryTemplate = toml::from_str(&text).unwrap();
        t.validate().unwrap();
        let peer = t.store("peer_hbm").unwrap();
        assert_eq!(peer.kind, StoreKind::PeerHbm);
        assert!(peer.pins_fetches());
        assert!(t.gpu_reaches("peer_hbm"));

        let unpinned = text.replace("kind = \"peer_hbm\"", "kind = \"peer_hbm\"\npin = false");
        let t: MemoryTemplate = toml::from_str(&unpinned).unwrap();
        assert!(!t.store("peer_hbm").unwrap().pins_fetches());

        let private = text.replace("per = \"node\"\nkind", "per = \"gpu\"\nkind");
        assert!(toml::from_str::<MemoryTemplate>(&private)
            .unwrap()
            .validate()
            .unwrap_err()
            .contains("per = \"node\""));
    }

    #[test]
    fn rejects_bad_endpoints_and_duplicates() {
        let bad = TEMPLATE.replace("to = \"nvme\"", "to = \"ssd\"");
        let t: MemoryTemplate = toml::from_str(&bad).unwrap();
        assert!(t.validate().unwrap_err().contains("ssd"));
        let dup =
            format!("{TEMPLATE}\n[[stores]]\nname = \"nvme\"\nper = \"node\"\ncapacity = 1.0\n");
        let t: MemoryTemplate = toml::from_str(&dup).unwrap();
        assert!(t.validate().unwrap_err().contains("twice"));
        assert!(toml::from_str::<MemoryTemplate>(
            "[[stores]]\nname = \"x\"\nper = \"rack\"\ncapacity = 1.0"
        )
        .is_err());

        for bad in ["stripe = 0", "aggregate_bandwidth = 0", "latency = -1"] {
            let raw = format!("[[stores]]\nname = \"x\"\nper = \"cluster\"\ncapacity = 1.0\n{bad}");
            let t: MemoryTemplate = toml::from_str(&raw).unwrap();
            assert!(t.validate().is_err(), "{bad}");
        }
    }

    #[test]
    fn parses_write_and_eviction_policies() {
        let c: MemoryConfig = toml::from_str(
            "tiers = [\"nvme\"]\nwrite = { policy = \"selective\", min_hits = 2 }\neviction = { policy = \"ttl\", seconds = 60 }\nhbm_evict_backed_first = true",
        )
        .unwrap();
        let p = c.policies();
        assert_eq!(p.write, WritePolicy::Selective { min_hits: 2 });
        assert_eq!(p.eviction, EvictionPolicy::Ttl { seconds: 60.0 });
        assert!(p.hbm_evict_backed_first);
        assert_eq!(p.source, SourcePolicy::Promote {});
        assert_eq!(p.hbm_eviction, HbmEviction::Lru {});
        assert_eq!(p.prefetch, PrefetchPolicy::None {});
        let d: MemoryConfig = toml::from_str("tiers = [\"nvme\"]").unwrap();
        assert_eq!(d.policies(), MemoryPolicies::default());
        assert!(toml::from_str::<MemoryConfig>("write = { policy = \"nope\" }").is_err());
    }

    #[test]
    fn presets_bundle_policies_and_explicit_fields_override() {
        let o: MemoryConfig = toml::from_str("tiers = [\"nvme\"]\npreset = \"oracle\"").unwrap();
        let p = o.policies();
        assert_eq!(p, MemoryPolicies::preset(MemoryPreset::Oracle));
        assert_eq!(p.source, SourcePolicy::MinTime {});
        assert_eq!(p.hbm_eviction, HbmEviction::Outlook {});
        assert_eq!(p.write, WritePolicy::Live {});
        assert_eq!(p.eviction, EvictionPolicy::Outlook {});
        assert_eq!(p.prefetch, PrefetchPolicy::Outlook { lead: 0.0 });
        assert!(p.uses_outlook());

        let r: MemoryConfig = toml::from_str(
            "tiers = [\"nvme\"]\npreset = \"reactive\"\nsource = { policy = \"min_time\" }\nprefetch = { policy = \"outlook\", lead = 2.5 }",
        )
        .unwrap();
        let p = r.policies();
        assert_eq!(p.source, SourcePolicy::MinTime {});
        assert_eq!(p.prefetch, PrefetchPolicy::Outlook { lead: 2.5 });
        assert_eq!(p.write, WritePolicy::Selective { min_hits: 1 });
        assert_eq!(p.eviction, EvictionPolicy::Lru {});
        assert!(!MemoryPolicies::preset(MemoryPreset::Reactive).uses_outlook());
        assert!(toml::from_str::<MemoryConfig>("preset = \"dynamo\"").is_err());
    }

    #[test]
    fn deployment_side_checks_against_the_template() {
        let t: MemoryTemplate = toml::from_str(TEMPLATE).unwrap();
        let ok: MemoryConfig =
            toml::from_str("tiers = [\"grace_dram\", \"nvme\"]\n[capacity]\ngrace_dram = 200e9")
                .unwrap();
        ok.validate(Some(&t)).unwrap();
        let none = MemoryConfig::default();
        none.validate(None).unwrap();
        none.validate(Some(&t)).unwrap();
        let missing: MemoryConfig = toml::from_str("tiers = [\"host_dram\"]").unwrap();
        assert!(missing
            .validate(Some(&t))
            .unwrap_err()
            .contains("host_dram"));
        assert!(ok.validate(None).unwrap_err().contains("no [memory]"));
        let bad_cap: MemoryConfig =
            toml::from_str("tiers = [\"nvme\"]\n[capacity]\ngrace_dram = 1e9").unwrap();
        assert!(bad_cap
            .validate(Some(&t))
            .unwrap_err()
            .contains("not a tier"));
        let unreachable = TEMPLATE.replace("to = \"nvme\"", "to = \"switch\"");
        let t2: MemoryTemplate = toml::from_str(&unreachable).unwrap();
        let nv: MemoryConfig = toml::from_str("tiers = [\"nvme\"]").unwrap();
        assert!(nv
            .validate(Some(&t2))
            .unwrap_err()
            .contains("not reachable"));
        // Reachable through a junction is fine.
        let via = format!(
            "{unreachable}\n[[junctions]]\nname = \"port\"\nper = \"gpu\"\n\
             [[links]]\nname = \"a\"\nfrom = \"gpu\"\nto = \"port\"\nbandwidth = 1.0\n\
             [[links]]\nname = \"b\"\nfrom = \"port\"\nto = \"nvme\"\nbandwidth = 1.0\n"
        );
        let t3: MemoryTemplate = toml::from_str(&via).unwrap();
        t3.validate().unwrap();
        assert!(t3.gpu_reaches("nvme"));
        nv.validate(Some(&t3)).unwrap();
    }
}
