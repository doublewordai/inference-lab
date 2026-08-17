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
//! Deployment side — which stores hold evicted KV, closest first, and how
//! much of each they may use:
//!
//! ```toml
//! [memory]
//! tiers = ["grace_dram", "nvme"]
//! [memory.capacity]
//! grace_dram = 200e9
//! ```
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

/// Which block a store recycles first when full.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvictionPolicy {
    /// Least recently inserted (a demoted block is not touched again until
    /// it is promoted out, so insertion order is recency order).
    #[default]
    Fifo,
}

/// A store a node offers for KV beyond HBM.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoreTemplate {
    pub name: String,
    pub per: Scope,
    /// Bytes per instance.
    pub capacity: f64,
    #[serde(default)]
    pub eviction: EvictionPolicy,
}

/// A link on the node. `from` is `"gpu"` (one instance per GPU, its own
/// port) or `"node"` (one per node); `to` is a store name, `"switch"` (the
/// scale-up fabric) or `"network"` (the scale-out NIC).
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
    pub links: Vec<LinkTemplate>,
}

impl MemoryTemplate {
    pub fn store(&self, name: &str) -> Option<&StoreTemplate> {
        self.stores.iter().find(|s| s.name == name)
    }

    /// The direct GPU → `store` link, if the template has one.
    pub fn gpu_link_to(&self, store: &str) -> Option<&LinkTemplate> {
        self.links.iter().find(|l| l.from == "gpu" && l.to == store)
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
        }
        for l in &self.links {
            if l.from != "gpu" && l.from != "node" {
                return Err(format!(
                    "[memory] link `{}`: from must be \"gpu\" or \"node\", got \"{}\"",
                    l.name, l.from
                ));
            }
            let to_ok = l.to == "switch" || l.to == "network" || self.store(&l.to).is_some();
            if !to_ok {
                return Err(format!(
                    "[memory] link `{}`: to must be a store name, \"switch\" or \"network\", got \"{}\"",
                    l.name, l.to
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
}

impl MemoryConfig {
    pub fn is_empty(&self) -> bool {
        self.tiers.is_empty()
    }

    /// Check every tier names a store the template has, reachable from a
    /// GPU by a direct link, and every capacity override names a tier.
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
            if template.gpu_link_to(name).is_none() {
                return Err(format!(
                    "[memory] tier `{name}` has no direct gpu → {name} link on this hardware"
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
            .contains("no direct gpu"));
    }
}
