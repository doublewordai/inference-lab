pub mod hardware;
pub mod model;
pub mod scheduler;
pub mod simulation;
pub mod workload;

pub use hardware::HardwareConfig;
pub use model::ModelConfig;
pub use scheduler::SchedulerConfig;
pub use simulation::SimulationConfig;
pub use workload::{LengthDistribution, WorkloadConfig};

use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Top-level configuration that aggregates all sub-configs
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub hardware: HardwareConfig,
    pub model: ModelConfig,
    pub scheduler: SchedulerConfig,
    pub workload: WorkloadConfig,
    #[serde(default)]
    pub simulation: SimulationConfig,
}

impl Config {
    /// Load configuration from a TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        let mut config: Config = toml::from_str(&contents)?;

        // Compute derived fields
        config.hardware.compute_threshold();
        config
            .model
            .compute_kv_cache_size(config.hardware.bytes_per_param);

        // Compute KV cache capacity if not explicitly set
        let model_size_bytes = config.model.num_parameters * config.hardware.bytes_per_param as u64;
        config.hardware.compute_kv_cache_capacity(model_size_bytes);

        config
            .scheduler
            .set_default_prefill_threshold(config.model.max_seq_len);

        Ok(config)
    }

    /// Get a default configuration for testing
    #[cfg(test)]
    pub fn test_default() -> Self {
        let mut hardware = HardwareConfig {
            name: "Test GPU".to_string(),
            compute_flops: 1e15,
            memory_bandwidth: 1e12,
            memory_capacity: 80_000_000_000,
            kv_cache_capacity: 60_000_000_000,
            gpu_memory_utilization: 0.9,
            bytes_per_param: 2,
            compute_bound_threshold: 0,
        };
        hardware.compute_threshold();

        let mut model = ModelConfig {
            name: "Test Model".to_string(),
            num_parameters: 7_000_000_000,
            num_active_parameters: None,
            num_layers: 32,
            hidden_dim: 4096,
            num_heads: 32,
            num_kv_heads: None,
            max_seq_len: 2048,
            sliding_window: None,
            num_sliding_layers: None,
            kv_cache_bytes_per_token: 0,
        };
        model.compute_kv_cache_size(hardware.bytes_per_param);

        let mut scheduler = SchedulerConfig {
            max_num_batched_tokens: 2048,
            max_num_seqs: 128,
            policy: "fcfs".to_string(),
            enable_chunked_prefill: true,
            long_prefill_token_threshold: 0,
            max_num_partial_prefills: 1,
            block_size: 16,
        };
        scheduler.set_default_prefill_threshold(model.max_seq_len);

        let workload = WorkloadConfig {
            arrival_pattern: "poisson".to_string(),
            arrival_rate: 1.0,
            num_concurrent_users: None,
            input_len_dist: LengthDistribution::Fixed { value: 100 },
            output_len_dist: LengthDistribution::Fixed { value: 50 },
            num_requests: Some(10),
            duration_secs: None,
            seed: 42,
        };

        let simulation = SimulationConfig::default();

        Config {
            hardware,
            model,
            scheduler,
            workload,
            simulation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_compute_bound_threshold() {
        let mut hw = HardwareConfig {
            name: "Test".to_string(),
            compute_flops: 1.513e15,
            memory_bandwidth: 3.35e12,
            memory_capacity: 80_000_000_000,
            kv_cache_capacity: 60_000_000_000,
            gpu_memory_utilization: 0.9,
            bytes_per_param: 2,
            compute_bound_threshold: 0,
        };
        hw.compute_threshold();

        // Should be approximately 903 for H100 bf16
        assert!(hw.compute_bound_threshold > 900);
        assert!(hw.compute_bound_threshold < 910);
    }

    #[test]
    fn test_model_kv_cache_calculation() {
        let mut model = ModelConfig {
            name: "Test".to_string(),
            num_parameters: 7_000_000_000,
            num_active_parameters: None,
            num_layers: 32,
            hidden_dim: 4096,
            num_heads: 32,
            num_kv_heads: None,
            max_seq_len: 2048,
            sliding_window: None,
            num_sliding_layers: None,
            kv_cache_bytes_per_token: 0,
        };
        model.compute_kv_cache_size(2); // bf16

        // 2 (K+V) * 4096 (hidden) * 2 (bytes) * 32 (layers) = 524,288 bytes per token
        assert_eq!(model.kv_cache_bytes_per_token, 524_288);

        // For a 100-token sequence
        let size = model.kv_cache_size_for_sequence(100);
        assert_eq!(size, 52_428_800); // 524,288 * 100
    }

    #[test]
    fn test_config_creation() {
        let config = Config::test_default();
        assert!(config.hardware.compute_bound_threshold > 0);
        assert!(config.model.kv_cache_bytes_per_token > 0);
    }
}
