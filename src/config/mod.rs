pub mod deployment;
pub mod hardware;
pub mod model;
pub mod parallel;
pub mod router;
pub mod scheduler;
pub mod speculative;
pub mod topology;
pub mod workload;

pub use deployment::{Deployment, DeploymentError, ModelConfig};
pub use hardware::{FabricConfig, FabricLink, HardwareConfig, KVTier, Precision};
pub use model::{
    expected_distinct_experts, History, Indexer, LayerClass, ModelSpec, Routing, WeightStream,
};
pub use parallel::ParallelConfig;
pub use router::RouterConfig;
pub use scheduler::SchedulerConfig;
pub use speculative::{
    AcceptanceModel, DrafterCost, GammaPolicy, MeasuredCostConfig, SpeculativeConfig,
    SwitchConstraints, TraceBank, TraceRound,
};
pub use topology::{ClusterSpec, DisaggTopology};
pub use workload::{ArrivalPattern, LengthDistribution, RateSchedule, WorkloadConfig};

#[cfg(test)]
use crate::scheduler::SchedulingPolicy;
use serde::de::{self, Deserializer};
use serde::Deserialize;

/// A config field that is either a catalog name or an inline table.
#[derive(Deserialize)]
#[serde(untagged)]
enum NameOrInline<T> {
    Name(String),
    Inline(T),
}

/// Deserialise a hardware spec given as a catalog name or an inline table.
pub(crate) fn hardware_ref<'de, D: Deserializer<'de>>(d: D) -> Result<HardwareConfig, D::Error> {
    match NameOrInline::<HardwareConfig>::deserialize(d)? {
        NameOrInline::Name(n) => crate::catalog::hardware(&n).map_err(de::Error::custom),
        NameOrInline::Inline(h) => Ok(h),
    }
}

/// Deserialise a model spec given as a catalog name or an inline table.
pub(crate) fn model_ref<'de, D: Deserializer<'de>>(d: D) -> Result<ModelSpec, D::Error> {
    let spec = match NameOrInline::<ModelSpec>::deserialize(d)? {
        NameOrInline::Name(n) => crate::catalog::model(&n).map_err(de::Error::custom)?,
        NameOrInline::Inline(m) => m,
    };
    spec.validate().map_err(de::Error::custom)?;
    Ok(spec)
}

/// A runnable simulation: one deployment plus one workload. Built from a
/// model config file and a workload file (see [`deployment`]), or as JSON
/// through the wasm API.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Catalog name (`hardware = "b200"`) or inline `[hardware]` table.
    #[serde(deserialize_with = "hardware_ref")]
    pub hardware: HardwareConfig,
    #[serde(default)]
    pub parallel: ParallelConfig,
    /// Catalog name (`model = "deepseek-v4-flash"`) or inline `[model]` table.
    #[serde(deserialize_with = "model_ref")]
    pub model: ModelSpec,
    pub scheduler: SchedulerConfig,
    /// Identical replicas of this deployment fronted by `router`. Defaults
    /// to 1.
    #[serde(default = "default_replicas")]
    pub replicas: u32,
    /// How requests are spread across the replicas.
    #[serde(default)]
    pub router: RouterConfig,
    /// Disaggregated topologies: how hand-offs are spread across the decode
    /// pool. Defaults to `router`.
    #[serde(default)]
    pub decode_router: Option<RouterConfig>,
    pub workload: WorkloadConfig,
    /// Optional speculative decoding. When set, decode steps verify `gamma + 1`
    /// tokens and advance by `accepted + 1` per the acceptance model.
    #[serde(default)]
    pub speculative: Option<SpeculativeConfig>,
    /// Serve-mode only: static mid-stream fault injection applied to every streaming
    /// chat completion on this model (the fallback trigger for clients that can't set
    /// the `x-inference-lab-fault` header). Validated by `serve::fault` at server
    /// startup; inert outside `serve`.
    #[serde(default)]
    pub fault: Option<FaultConfig>,
}

fn default_replicas() -> u32 {
    1
}

/// TOML shape of `[fault]` (see `serve::fault` for the mode list and semantics).
/// Kept as plain strings/options here so the core config crate stays serve-agnostic;
/// `serve::fault::FaultSpec::from_config` validates and applies defaults.
#[derive(Debug, Clone, Deserialize)]
pub struct FaultConfig {
    /// Death mode name, e.g. "cut_mid_frame" — same names the header accepts.
    pub mode: String,
    /// Content-bearing delta frames emitted before the fault fires.
    #[serde(default)]
    pub after_chunks: Option<u32>,
    /// Fixed pacing between emitted frames.
    #[serde(default)]
    pub delay_ms: Option<u64>,
    /// `cut_mid_frame` variant: cut inside a multi-byte UTF-8 character.
    #[serde(default)]
    pub utf8: Option<bool>,
}

impl Config {
    /// A simulation: a deployment (model on hardware) plus the workload
    /// offered to it.
    pub fn new(deployment: Deployment, workload: WorkloadConfig) -> Self {
        let Deployment {
            hardware,
            parallel,
            model,
            scheduler,
            replicas,
            router,
            decode_router,
            speculative,
            fault,
        } = deployment;
        let mut config = Config {
            hardware,
            parallel,
            model,
            scheduler,
            replicas,
            router,
            decode_router,
            workload,
            speculative,
            fault,
        };
        config.finalize();
        config
    }

    /// Fill in derived fields after deserialization. Public so wasm.rs can
    /// call it after `serde_json::from_str`.
    pub fn finalize(&mut self) {
        let max_model_len = self
            .scheduler
            .max_model_len
            .unwrap_or(self.model.max_seq_len);
        self.scheduler.set_default_prefill_threshold(max_model_len);
    }

    /// Router for the decode pool of a disaggregated topology: the
    /// `decode_router` block, or `router` when there is none.
    pub fn decode_router(&self) -> &RouterConfig {
        self.decode_router.as_ref().unwrap_or(&self.router)
    }

    /// The worker pool this config describes: its hardware and parallel
    /// layout as one `ClusterSpec` of `replicas` workers.
    pub fn cluster(&self) -> ClusterSpec {
        ClusterSpec {
            hardware: self.hardware.clone(),
            parallel: self.parallel.clone(),
            num_workers: self.replicas.max(1),
        }
    }

    /// Get a default configuration for testing: a 7B-class bf16 dense model
    /// (32 layers, hidden 4096, 32 heads / 8 KV heads) on a 1 PFLOP/s,
    /// 1 TB/s test GPU with a 60 GB KV cache.
    #[cfg(test)]
    pub fn test_default() -> Self {
        let hardware = HardwareConfig {
            name: "Test GPU".to_string(),
            flops_fp4: None,
            flops_fp8: None,
            flops_bf16: Some(1e15),
            flops_fp16: Some(1e15),
            memory_bandwidth: 1e12,
            memory_capacity: 80_000_000_000,
            kv_tiers: Vec::new(),
            fabric: None,
        };
        let parallel = ParallelConfig::default();

        let model = ModelSpec {
            name: "Test Model".to_string(),
            hidden_dim: 4096,
            max_seq_len: 2048,
            attention_precision: Precision::Bf16,
            activation_bytes: 2,
            weights: vec![WeightStream {
                precision: Precision::Bf16,
                active_params: 7_000_000_000,
                resident_params: 7_000_000_000,
                routing: None,
            }],
            layers: vec![LayerClass::Attention {
                count: 32,
                heads: 32,
                head_dim: 128,
                kv_heads: 8,
                kv_shared: false,
                window: 0,
                kv_precision: Precision::Bf16,
            }],
        };

        let mut scheduler = SchedulerConfig {
            max_num_batched_tokens: 2048,
            max_num_seqs: 128,
            policy: SchedulingPolicy::FCFS,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
            max_num_partial_prefills: 1,
            block_size: 16,
            gpu_memory_utilization: 0.9,
            kv_cache_capacity: 60_000_000_000,
            max_model_len: None,
            enable_preemption_free: false,
            enable_cascade_attention: false,
        };
        scheduler.set_default_prefill_threshold(model.max_seq_len);

        let workload = WorkloadConfig {
            dataset_path: None,
            arrival_pattern: ArrivalPattern::Poisson,
            arrival_rate: 1.0,
            rate_schedule: None,
            num_concurrent_users: None,
            input_len_dist: LengthDistribution::Fixed { value: 100 },
            output_len_dist: LengthDistribution::Fixed { value: 50 },
            num_requests: Some(10),
            duration_secs: None,
            seed: 42,
            closed_loop_jitter_secs: None,
        };

        Config {
            hardware,
            parallel,
            model,
            scheduler,
            replicas: 1,
            router: RouterConfig::default(),
            decode_router: None,
            workload,
            speculative: None,
            fault: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toml_files(dir: &str) -> Vec<std::path::PathBuf> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut paths: Vec<_> = std::fs::read_dir(root.join(dir))
            .unwrap_or_else(|e| panic!("{dir}: {e}"))
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("toml"))
            .collect();
        paths.sort();
        paths
    }

    /// Every model config shipped under configs/ must parse and every one of
    /// its hardware entries must resolve; every workload under workloads/
    /// must parse. Unknown fields are rejected, so this catches drift.
    #[test]
    fn shipped_configs_resolve() {
        let configs = toml_files("configs");
        let workloads = toml_files("workloads");
        assert!(!configs.is_empty() && !workloads.is_empty());
        let mut failures = Vec::new();
        for p in &configs {
            match ModelConfig::from_file(p) {
                Err(e) => failures.push(e.to_string()),
                Ok(cfg) => {
                    for hw in cfg.hardware_names() {
                        if let Err(e) = cfg.deployment(Some(hw)) {
                            failures.push(format!("{}: {e}", p.display()));
                        }
                    }
                }
            }
        }
        for p in &workloads {
            if let Err(e) = WorkloadConfig::from_file(p) {
                failures.push(e.to_string());
            }
        }
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }

    #[test]
    fn test_config_creation() {
        let config = Config::test_default();
        assert!(config.model.kv_storage_bytes(1) > 0);
        assert!(config.model.validate().is_ok());
    }

    #[test]
    fn hardware_and_model_resolve_from_the_catalog_by_name() {
        let toml_src = r#"
hardware = "b200"
model = "deepseek-v4-flash"

[scheduler]
max_num_batched_tokens = 8192
max_num_seqs = 256
policy = "fcfs"
enable_chunked_prefill = true
block_size = 64
"#;
        let dep: Deployment = toml::from_str(toml_src).unwrap();
        assert_eq!(dep.hardware.name, "B200");
        assert_eq!(dep.model.name, "DeepSeek-V4-Flash");
        let bad = toml_src.replace("\"b200\"", "\"b2000\"");
        let err = toml::from_str::<Deployment>(&bad).unwrap_err().to_string();
        assert!(err.contains("unknown hardware preset"), "{err}");
    }
}
