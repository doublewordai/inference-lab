//! Model config files.
//!
//! A model config file describes one model, the engine arguments it runs
//! with, and every hardware it is deployed on:
//!
//! ```toml
//! model = "gpt-oss-120b"            # catalog name, or an inline [model] table
//!
//! [scheduler]                       # engine args shared by every hardware entry
//! max_num_batched_tokens = 8192
//! max_num_seqs = 4096
//! policy = "priority"
//! enable_chunked_prefill = true
//! block_size = 64
//!
//! [hardware.b200]                   # key = catalog hardware name
//! tp = 2
//!
//! [hardware.gh200]
//! tp = 4
//! scheduler = { max_num_batched_tokens = 4096 }   # per-hardware override
//! ```
//!
//! Each `[hardware.<name>]` entry may set `tp`, `ep`, `dp_attention`,
//! `replicas`, a partial `scheduler` override (merged over the shared
//! block), a `speculative` block (replacing the shared one), and a `router`
//! table (replacing the shared `[router]`). `spec` overrides the hardware
//! itself: another catalog name, or an inline hardware table.
//!
//! `replicas` (default 1) is how many identical workers of this deployment
//! run behind the router; `[router]` picks the policy that spreads requests
//! across them, and `[decode_router]` (default: `[router]`) the policy that
//! spreads hand-offs across a disaggregated decode pool (see
//! [`super::router`]). `[memory]` (shared, or per entry) picks which of the
//! hardware's stores hold evicted KV (see [`super::memory`]).
//!
//! [`ModelConfig::deployment`] resolves one entry into a [`Deployment`]: the
//! model on that hardware, with no workload. A [`Deployment`] plus a
//! [`WorkloadConfig`] is a [`Config`], which is what a simulation runs.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::Path;

use serde::Deserialize;
use toml::{Table, Value};

use super::{
    hardware_ref, model_ref, ClusterSpec, Config, FaultConfig, HardwareConfig, MemoryConfig,
    ModelSpec, ParallelConfig, RouterConfig, SchedulerConfig, SpeculativeConfig, TimeCorrection,
    WorkloadConfig,
};

/// A model on one hardware: everything a simulation needs except the
/// workload.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Deployment {
    #[serde(deserialize_with = "hardware_ref")]
    pub hardware: HardwareConfig,
    #[serde(default)]
    pub parallel: ParallelConfig,
    #[serde(deserialize_with = "model_ref")]
    pub model: ModelSpec,
    pub scheduler: SchedulerConfig,
    /// Identical replicas behind the router. Defaults to 1.
    #[serde(default = "default_replicas")]
    pub replicas: u32,
    #[serde(default)]
    pub router: RouterConfig,
    /// Disaggregated topologies: router for the decode pool. Defaults to
    /// `router`.
    #[serde(default)]
    pub decode_router: Option<RouterConfig>,
    /// KV tiers beyond HBM, chosen from the hardware's `[memory]` stores.
    #[serde(default)]
    pub memory: MemoryConfig,
    /// Optional step-time calibration against a measured engine
    /// (`t = alpha × t_roofline + beta`); absent = pure roofline.
    #[serde(default)]
    pub time_correction: Option<TimeCorrection>,
    #[serde(default)]
    pub speculative: Option<SpeculativeConfig>,
    #[serde(default)]
    pub fault: Option<FaultConfig>,
}

fn default_replicas() -> u32 {
    1
}

impl Deployment {
    /// Router for the decode pool of a disaggregated topology: the
    /// `decode_router` block, or `router` when there is none.
    pub fn decode_router(&self) -> &RouterConfig {
        self.decode_router.as_ref().unwrap_or(&self.router)
    }

    /// This deployment's hardware and parallel layout as one worker pool of
    /// `replicas` workers.
    pub fn cluster(&self) -> ClusterSpec {
        ClusterSpec {
            hardware: self.hardware.clone(),
            parallel: self.parallel.clone(),
            num_workers: self.replicas.max(1),
            memory: self.memory.clone(),
        }
    }

    /// Pair with a workload to get a runnable simulation config.
    pub fn with_workload(self, workload: WorkloadConfig) -> Config {
        Config::new(self, workload)
    }
}

/// Error resolving a deployment from a model config file.
#[derive(Debug)]
pub struct DeploymentError(String);

impl fmt::Display for DeploymentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DeploymentError {}

/// One `[hardware.<name>]` entry, as written.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct HardwareEntry {
    /// Catalog name or inline table for the hardware; defaults to the entry key.
    #[serde(default)]
    spec: Option<Value>,
    #[serde(default)]
    tp: Option<u32>,
    #[serde(default)]
    ep: Option<u32>,
    #[serde(default)]
    dp_attention: Option<bool>,
    /// Identical replicas of this deployment. Defaults to 1.
    #[serde(default)]
    replicas: Option<u32>,
    /// Partial `[scheduler]` override, merged over the shared block.
    #[serde(default)]
    scheduler: Option<Table>,
    /// Replaces the shared `[speculative]` block for this hardware.
    #[serde(default)]
    speculative: Option<Table>,
    /// Replaces the shared `[router]` block for this hardware.
    #[serde(default)]
    router: Option<Table>,
    /// Replaces the shared `[decode_router]` block for this hardware.
    #[serde(default)]
    decode_router: Option<Table>,
    /// Replaces the shared `[memory]` block for this hardware.
    #[serde(default)]
    memory: Option<Table>,
    /// Step-time calibration for this hardware (`{ alpha, beta }`).
    #[serde(default)]
    time_correction: Option<Table>,
}

/// A parsed model config file. Resolve a hardware entry with
/// [`ModelConfig::deployment`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    model: Value,
    scheduler: Table,
    #[serde(default)]
    speculative: Option<Table>,
    #[serde(default)]
    router: Option<Table>,
    #[serde(default)]
    decode_router: Option<Table>,
    #[serde(default)]
    memory: Option<Table>,
    #[serde(default)]
    fault: Option<Table>,
    hardware: BTreeMap<String, HardwareEntry>,
}

impl ModelConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let contents =
            fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
        Self::from_toml(&contents).map_err(|e| format!("{}: {e}", path.display()).into())
    }

    pub fn from_toml(src: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let cfg: ModelConfig = toml::from_str(src)?;
        if cfg.hardware.is_empty() {
            return Err("no [hardware.<name>] entries".into());
        }
        Ok(cfg)
    }

    /// Hardware entry names, sorted.
    pub fn hardware_names(&self) -> Vec<&str> {
        self.hardware.keys().map(String::as_str).collect()
    }

    /// Resolve the deployment for `hardware`. `None` selects the entry when
    /// the file has exactly one.
    pub fn deployment(&self, hardware: Option<&str>) -> Result<Deployment, DeploymentError> {
        let names = self.hardware_names().join(", ");
        let name = match hardware {
            Some(h) => h,
            None if self.hardware.len() == 1 => self.hardware.keys().next().unwrap(),
            None => {
                return Err(DeploymentError(format!(
                    "config has hardware entries {names}; choose one"
                )))
            }
        };
        let entry = self.hardware.get(name).ok_or_else(|| {
            DeploymentError(format!("no [hardware.{name}] entry (have: {names})"))
        })?;

        let mut merged = Table::new();
        merged.insert(
            "hardware".into(),
            entry
                .spec
                .clone()
                .unwrap_or_else(|| Value::String(name.to_string())),
        );
        let mut parallel = Table::new();
        if let Some(tp) = entry.tp {
            parallel.insert("tp".into(), Value::Integer(tp as i64));
        }
        if let Some(ep) = entry.ep {
            parallel.insert("ep".into(), Value::Integer(ep as i64));
        }
        if let Some(dp) = entry.dp_attention {
            parallel.insert("dp_attention".into(), Value::Boolean(dp));
        }
        merged.insert("parallel".into(), Value::Table(parallel));
        merged.insert("model".into(), self.model.clone());
        let mut scheduler = self.scheduler.clone();
        if let Some(over) = &entry.scheduler {
            for (k, v) in over {
                scheduler.insert(k.clone(), v.clone());
            }
        }
        merged.insert("scheduler".into(), Value::Table(scheduler));
        if let Some(replicas) = entry.replicas {
            merged.insert("replicas".into(), Value::Integer(replicas as i64));
        }
        if let Some(spec) = entry.speculative.as_ref().or(self.speculative.as_ref()) {
            merged.insert("speculative".into(), Value::Table(spec.clone()));
        }
        if let Some(router) = entry.router.as_ref().or(self.router.as_ref()) {
            merged.insert("router".into(), Value::Table(router.clone()));
        }
        if let Some(router) = entry.decode_router.as_ref().or(self.decode_router.as_ref()) {
            merged.insert("decode_router".into(), Value::Table(router.clone()));
        }
        if let Some(memory) = entry.memory.as_ref().or(self.memory.as_ref()) {
            merged.insert("memory".into(), Value::Table(memory.clone()));
        }
        if let Some(tc) = &entry.time_correction {
            merged.insert("time_correction".into(), Value::Table(tc.clone()));
        }
        if let Some(fault) = &self.fault {
            merged.insert("fault".into(), Value::Table(fault.clone()));
        }
        let deployment = Deployment::deserialize(Value::Table(merged))
            .map_err(|e| DeploymentError(format!("[hardware.{name}]: {e}")))?;
        deployment
            .cluster()
            .validate()
            .map_err(|e| DeploymentError(format!("[hardware.{name}]: {e}")))?;
        if let Some(tc) = &deployment.time_correction {
            tc.validate()
                .map_err(|e| DeploymentError(format!("[hardware.{name}]: {e}")))?;
        }
        Ok(deployment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILE: &str = r#"
model = "deepseek-v4-flash"

[scheduler]
max_num_batched_tokens = 8192
max_num_seqs = 256
policy = "fcfs"
enable_chunked_prefill = true
block_size = 64

[speculative]
gamma = 4
policy = "goodput_budget"
[speculative.acceptance]
kind = "constant"
alpha = 0.75

[router]
policy = "least_loaded"

[decode_router]
policy = "kv_aware_decode"

[hardware.b200]
tp = 4
replicas = 4

[hardware.gh200]
tp = 4
scheduler = { max_num_batched_tokens = 4096 }
router = { policy = "prefix_affinity", max_load_ratio = 1.5 }
time_correction = { alpha = 1.5, beta = 0.002 }
[hardware.gh200.speculative]
gamma = 2
policy = "goodput_budget"
[hardware.gh200.speculative.acceptance]
kind = "constant"
alpha = 0.5
"#;

    #[test]
    fn resolves_each_hardware_entry() {
        let cfg = ModelConfig::from_toml(FILE).unwrap();
        assert_eq!(cfg.hardware_names(), vec!["b200", "gh200"]);

        let b200 = cfg.deployment(Some("b200")).unwrap();
        assert_eq!(b200.hardware.name, "B200");
        assert_eq!(b200.parallel.tp, 4);
        assert_eq!(b200.parallel.ep, 1);
        assert_eq!(b200.scheduler.max_num_batched_tokens, 8192);
        assert_eq!(b200.speculative.as_ref().unwrap().gamma, 4);
        assert_eq!(b200.replicas, 4);
        assert_eq!(b200.router, RouterConfig::LeastLoaded {});
        assert_eq!(
            *b200.decode_router(),
            RouterConfig::KvAwareDecode { load_weight: 64.0 }
        );
        assert_eq!(b200.cluster().num_workers, 4);
        assert_eq!(b200.time_correction, None);

        let gh = cfg.deployment(Some("gh200")).unwrap();
        assert_eq!(gh.replicas, 1);
        assert_eq!(
            gh.time_correction,
            Some(TimeCorrection {
                alpha: 1.5,
                beta: 0.002
            })
        );
        assert_eq!(
            gh.router,
            RouterConfig::PrefixAffinity {
                max_load_ratio: Some(1.5)
            }
        );
        // No per-entry decode_router: the shared block applies.
        assert_eq!(
            *gh.decode_router(),
            RouterConfig::KvAwareDecode { load_weight: 64.0 }
        );
        assert_eq!(gh.hardware.name, "GH200");
        assert_eq!(gh.scheduler.max_num_batched_tokens, 4096);
        assert_eq!(gh.scheduler.max_num_seqs, 256);
        assert_eq!(gh.speculative.as_ref().unwrap().gamma, 2);
    }

    #[test]
    fn a_single_entry_needs_no_name() {
        let single = r#"
model = "deepseek-v4-flash"
[scheduler]
max_num_batched_tokens = 8192
max_num_seqs = 256
policy = "fcfs"
enable_chunked_prefill = true
block_size = 64
[hardware.b300]
tp = 8
"#;
        let cfg = ModelConfig::from_toml(single).unwrap();
        assert_eq!(cfg.deployment(None).unwrap().hardware.name, "B300");
    }

    #[test]
    fn spec_overrides_the_key() {
        let src = r#"
model = "deepseek-v4-flash"
[scheduler]
max_num_batched_tokens = 8192
max_num_seqs = 256
policy = "fcfs"
enable_chunked_prefill = true
block_size = 64
[hardware.isambard]
spec = "gh200"
tp = 4
[hardware.custom]
tp = 1
[hardware.custom.spec]
name = "Custom"
flops_bf16 = 1e15
memory_bandwidth = 1e12
memory_capacity = 80000000000
"#;
        let cfg = ModelConfig::from_toml(src).unwrap();
        assert_eq!(
            cfg.deployment(Some("isambard")).unwrap().hardware.name,
            "GH200"
        );
        assert_eq!(
            cfg.deployment(Some("custom")).unwrap().hardware.name,
            "Custom"
        );
    }

    #[test]
    fn errors_name_the_choices() {
        let cfg = ModelConfig::from_toml(FILE).unwrap();
        let e = cfg.deployment(None).unwrap_err().to_string();
        assert!(e.contains("b200, gh200"), "{e}");
        let e = cfg.deployment(Some("h100")).unwrap_err().to_string();
        assert!(e.contains("no [hardware.h100]"), "{e}");
    }

    #[test]
    fn rejects_unknown_fields_and_missing_hardware() {
        let e = ModelConfig::from_toml(&FILE.replace(
            "replicas = 4\n\n[hardware.gh",
            "replicas = 4\nnodes = 2\n\n[hardware.gh",
        ))
        .unwrap_err()
        .to_string();
        assert!(e.contains("nodes"), "{e}");
        let e = ModelConfig::from_toml(&format!("{FILE}\n[workload]\nseed = 1\n"))
            .unwrap_err()
            .to_string();
        assert!(e.contains("workload"), "{e}");
        let no_hw: String = FILE
            .lines()
            .take_while(|l| !l.starts_with("[hardware"))
            .collect::<Vec<_>>()
            .join("\n");
        let e = ModelConfig::from_toml(&no_hw).unwrap_err().to_string();
        assert!(e.contains("hardware"), "{e}");
    }

    #[test]
    fn scheduler_override_is_validated() {
        let src = FILE.replace(
            "scheduler = { max_num_batched_tokens = 4096 }",
            "scheduler = { max_num_batched_tokens = 4096, bogus = 1 }",
        );
        let cfg = ModelConfig::from_toml(&src).unwrap();
        let e = cfg.deployment(Some("gh200")).unwrap_err().to_string();
        assert!(e.contains("bogus"), "{e}");
    }
}
