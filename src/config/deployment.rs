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
//! [hardware.gh200-120]
//! tp = 4
//! scheduler = { max_num_batched_tokens = 4096 }   # per-hardware override
//! ```
//!
//! Each `[hardware.<name>]` entry may set `tp`, `ep`, `dp_attention`, a
//! partial `scheduler` override (merged over the shared block), and a
//! `speculative` block (replacing the shared one). `spec` overrides the
//! hardware itself: another catalog name, or an inline hardware table.
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
    hardware_ref, model_ref, ClusterSpec, Config, FaultConfig, HardwareConfig, ModelSpec,
    ParallelConfig, SchedulerConfig, SpeculativeConfig, WorkloadConfig,
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
    #[serde(default)]
    pub speculative: Option<SpeculativeConfig>,
    #[serde(default)]
    pub fault: Option<FaultConfig>,
}

impl Deployment {
    /// This deployment's hardware and parallel layout as one worker pool
    /// (no collective-comms model, one worker).
    pub fn cluster(&self) -> ClusterSpec {
        ClusterSpec {
            hardware: self.hardware.clone(),
            parallel: self.parallel.clone(),
            comms: None,
            num_workers: 1,
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
    /// Partial `[scheduler]` override, merged over the shared block.
    #[serde(default)]
    scheduler: Option<Table>,
    /// Replaces the shared `[speculative]` block for this hardware.
    #[serde(default)]
    speculative: Option<Table>,
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
        if let Some(spec) = entry.speculative.as_ref().or(self.speculative.as_ref()) {
            merged.insert("speculative".into(), Value::Table(spec.clone()));
        }
        if let Some(fault) = &self.fault {
            merged.insert("fault".into(), Value::Table(fault.clone()));
        }
        Deployment::deserialize(Value::Table(merged))
            .map_err(|e| DeploymentError(format!("[hardware.{name}]: {e}")))
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

[hardware.b200]
tp = 4

[hardware.gh200-120]
tp = 4
scheduler = { max_num_batched_tokens = 4096 }
[hardware.gh200-120.speculative]
gamma = 2
policy = "goodput_budget"
[hardware.gh200-120.speculative.acceptance]
kind = "constant"
alpha = 0.5
"#;

    #[test]
    fn resolves_each_hardware_entry() {
        let cfg = ModelConfig::from_toml(FILE).unwrap();
        assert_eq!(cfg.hardware_names(), vec!["b200", "gh200-120"]);

        let b200 = cfg.deployment(Some("b200")).unwrap();
        assert_eq!(b200.hardware.name, "B200");
        assert_eq!(b200.parallel.tp, 4);
        assert_eq!(b200.parallel.ep, 1);
        assert_eq!(b200.scheduler.max_num_batched_tokens, 8192);
        assert_eq!(b200.speculative.as_ref().unwrap().gamma, 4);

        let gh = cfg.deployment(Some("gh200-120")).unwrap();
        assert_eq!(gh.hardware.name, "GH200-120");
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
spec = "gh200-120"
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
            "GH200-120"
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
        assert!(e.contains("b200, gh200-120"), "{e}");
        let e = cfg.deployment(Some("h100")).unwrap_err().to_string();
        assert!(e.contains("no [hardware.h100]"), "{e}");
    }

    #[test]
    fn rejects_unknown_fields_and_missing_hardware() {
        let e = ModelConfig::from_toml(&FILE.replace(
            "tp = 4\n\n[hardware.gh",
            "tp = 4\nnodes = 2\n\n[hardware.gh",
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
        let e = cfg.deployment(Some("gh200-120")).unwrap_err().to_string();
        assert!(e.contains("bogus"), "{e}");
    }
}
