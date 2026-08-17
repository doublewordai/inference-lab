//! KV memory beyond HBM: the stores a node offers and the links that reach
//! them (on the hardware), and which of those a deployment uses as KV
//! tiers (on the deployment).
//!
//! Hardware side — a template instantiated per GPU or per node:
//!
//! ```toml
//! # catalog/hardware/gh200.toml
//! [memory]
//! [[memory.stores]]
//! name = "grace_dram"; per = "gpu"; capacity = 480e9      # one per superchip
//! [[memory.stores]]
//! name = "nvme"; per = "node"; capacity = 8e12
//! [[memory.links]]
//! name = "c2c"; from = "gpu"; to = "grace_dram"; bandwidth = 450e9
//! [[memory.links]]
//! name = "pcie"; from = "gpu"; to = "nvme"; bandwidth = 64e9
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
//! the path (see `kv_cache::flows`).
//!
//! Deployment side — which stores hold evicted KV, closest first, how much
//! of each they may use, and the write and eviction policies:
//!
//! ```toml
//! [memory]
//! tiers = ["grace_dram", "nvme"]
//! write = { policy = "write_through" }        # write_back | write_through | selective
//! eviction = { policy = "ttl", seconds = 3600 }   # fifo | lru | ttl
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
//! evicts into the next tier — a store → store transfer — or drops.
//!
//! With no `[memory]` on the deployment there is no tiering (HBM only).

use std::collections::BTreeMap;

use serde::Deserialize;

/// How many instances of a store or link a node has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    /// One per GPU, private to it (Grace LPDDR behind its superchip's C2C,
    /// a per-GPU NVMe drive).
    Gpu,
    /// One per node, shared by the node's GPUs (host DRAM over PCIe, a
    /// node-local NVMe pool).
    Node,
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
                hbm_evict_backed_first: true,
            },
            MemoryPreset::Oracle => Self {
                source: SourcePolicy::MinTime {},
                hbm_eviction: HbmEviction::Outlook {},
                write: WritePolicy::Live {},
                eviction: EvictionPolicy::Outlook {},
                prefetch: PrefetchPolicy::Outlook { lead: 0.0 },
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

/// A store a node offers for KV beyond HBM.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoreTemplate {
    pub name: String,
    pub per: Scope,
    /// Bytes per instance.
    pub capacity: f64,
    /// The store's own throughput per instance, bytes/s (an NVMe pool's
    /// aggregate drive rate), shared by every transfer in or out of it.
    /// Unset: unbounded — only the links limit.
    #[serde(default)]
    pub bandwidth: Option<f64>,
}

/// A named point on the node with no capacity of its own, so that several
/// links can share one (a GPU's PCIe port feeding host DRAM and NVMe).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JunctionTemplate {
    pub name: String,
    pub per: Scope,
}

/// A link on the node. `from` is `"gpu"` (one instance per GPU, its own
/// port), a store or a junction; `to` is a store, a junction, `"switch"`
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

/// The KV memory a node class offers: stores and the links that reach
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
            if s.capacity <= 0.0 {
                return Err(format!("[memory] store `{}` needs capacity > 0", s.name));
            }
            if s.bandwidth.is_some_and(|b| b <= 0.0) {
                return Err(format!("[memory] store `{}` needs bandwidth > 0", s.name));
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
                || self.store(&l.from).is_some()
                || self.junction(&l.from).is_some();
            if !from_ok {
                return Err(format!(
                    "[memory] link `{}`: from must be \"gpu\", a store or a junction, got \"{}\"",
                    l.name, l.from
                ));
            }
            let to_ok = l.to == "switch"
                || l.to == "network"
                || self.store(&l.to).is_some()
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
