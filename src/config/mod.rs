pub mod hardware;
pub mod model;
pub mod parallel;
pub mod scheduler;
pub mod speculative;
pub mod topology;
pub mod workload;

pub use hardware::{HardwareConfig, KVTier, Precision};
pub use model::{DeepseekV4Model, DenseModel, ModelConfig, ModelCosts, SlidingWindowModel};
pub use parallel::{CommsConfig, ParallelConfig};
pub use scheduler::SchedulerConfig;
pub use speculative::{
    AcceptanceModel, DrafterCost, GammaPolicy, MeasuredCostConfig, SpeculativeConfig,
    SwitchConstraints, TraceBank, TraceRound,
};
pub use topology::{ClusterSpec, DisaggTopology};
pub use workload::{ArrivalPattern, LengthDistribution, RateSchedule, WorkloadConfig};

#[cfg(test)]
use crate::scheduler::SchedulingPolicy;
use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Top-level configuration that aggregates all sub-configs
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub hardware: HardwareConfig,
    #[serde(default)]
    pub parallel: ParallelConfig,
    pub model: ModelConfig,
    pub scheduler: SchedulerConfig,
    pub workload: WorkloadConfig,
    /// Optional speculative decoding. When set, decode steps verify `gamma + 1`
    /// tokens and advance by `accepted + 1` per the acceptance model.
    #[serde(default)]
    pub speculative: Option<SpeculativeConfig>,
}

impl Config {
    /// Load configuration from a TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        let mut config: Config = toml::from_str(&contents)?;
        config.finalize();
        Ok(config)
    }

    /// Fill in derived fields after deserialization. Public so wasm.rs can
    /// call it after `serde_json::from_str`.
    pub fn finalize(&mut self) {
        self.scheduler
            .set_default_prefill_threshold(self.model.max_seq_len());
    }

    /// The single worker pool this config describes: its hardware and
    /// parallel layout as one `ClusterSpec` (no collective-comms model, one
    /// worker).
    pub fn cluster(&self) -> ClusterSpec {
        ClusterSpec {
            hardware: self.hardware.clone(),
            parallel: self.parallel.clone(),
            comms: None,
            num_workers: 1,
        }
    }

    /// Get a default configuration for testing
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
            kv_cache_capacity: 60_000_000_000,
            gpu_memory_utilization: 0.9,
            kv_tiers: Vec::new(),
        };
        let parallel = ParallelConfig::default();

        let model = ModelConfig::Dense(DenseModel {
            name: "Test Model".to_string(),
            num_parameters: 7_000_000_000,
            num_active_parameters: None,
            num_layers: 32,
            hidden_dim: 4096,
            num_heads: 32,
            num_kv_heads: None,
            head_dim: None,
            max_seq_len: 2048,
            precision: crate::config::Precision::Bf16,
        });

        let mut scheduler = SchedulerConfig {
            max_num_batched_tokens: 2048,
            max_num_seqs: 128,
            policy: SchedulingPolicy::FCFS,
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
            max_num_partial_prefills: 1,
            block_size: 16,
            enable_preemption_free: false,
            enable_cascade_attention: false,
        };
        scheduler.set_default_prefill_threshold(model.max_seq_len());

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
            workload,
            speculative: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_kv_storage_matches_formula() {
        let model = ModelConfig::Dense(DenseModel {
            name: "Test".to_string(),
            num_parameters: 7_000_000_000,
            num_active_parameters: None,
            num_layers: 32,
            hidden_dim: 4096,
            num_heads: 32,
            num_kv_heads: None,
            head_dim: None,
            max_seq_len: 2048,
            precision: crate::config::Precision::Bf16,
        });

        // 2 (K+V) * 4096 (hidden) * 2 (bytes) * 32 (layers) = 524,288 per token.
        // For seq_len=100: 52,428,800.
        assert_eq!(model.kv_storage_bytes(100), 52_428_800);
        assert_eq!(model.kv_bytes_read_per_decode_step(100), 52_428_800);
    }

    /// Every TOML shipped under configs/ and examples/ must parse under the
    /// current schema (unknown fields are rejected, so this catches drift).
    #[test]
    fn shipped_configs_parse() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut paths = Vec::new();
        for dir in ["configs", "configs/estate", "examples"] {
            let Ok(entries) = fs::read_dir(root.join(dir)) else {
                continue;
            };
            for e in entries.flatten() {
                let p = e.path();
                if p.extension().and_then(|x| x.to_str()) == Some("toml") {
                    paths.push(p);
                }
            }
        }
        assert!(!paths.is_empty(), "no configs found");
        let mut failures = Vec::new();
        for p in &paths {
            if let Err(e) = Config::from_file(p) {
                failures.push(format!("{}: {e}", p.display()));
            }
        }
        assert!(
            failures.is_empty(),
            "configs failed to parse:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    fn test_config_creation() {
        let config = Config::test_default();
        assert!(config.model.kv_storage_bytes(1) > 0);
    }

    #[test]
    fn test_sliding_window_kv_cache_caps_at_window() {
        let model = ModelConfig::Sliding(SlidingWindowModel {
            name: "Sliding".to_string(),
            num_parameters: 7_000_000_000,
            num_active_parameters: None,
            num_layers: 4,
            hidden_dim: 16,
            num_heads: 4,
            num_kv_heads: Some(2),
            max_seq_len: 2048,
            sliding_window: 8,
            num_sliding_layers: 2,
            precision: crate::config::Precision::Bf16,
        });

        // per layer: 2 * 2 * 4 * 2 = 32 bytes/token.
        // 2 full layers @ seq_len=10: 32 * 2 * 10 = 640
        // 2 sliding layers @ min(10,8)=8: 32 * 2 * 8 = 512
        // total 1,152
        assert_eq!(model.kv_storage_bytes(10), 1_152);
    }
}
