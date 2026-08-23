//! Worker pools and disaggregated topologies.
//!
//! A `ClusterSpec` is one pool of identically-shaped workers (hardware +
//! parallel layout). A `DisaggTopology` is a prefill pool and a decode pool
//! joined by the KV hand-off link between them.

use serde::Deserialize;

use super::{HardwareConfig, MemoryConfig, ModelSpec, ParallelConfig, SchedulerConfig};
use crate::compute::ComputeEngine;

/// A worker pool: one or more identically-shaped workers running the same
/// hardware spec and parallelism layout.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterSpec {
    /// Per-GPU hardware spec: a catalog name (`hardware = "b200"`) or an
    /// inline table.
    #[serde(deserialize_with = "super::hardware_ref")]
    pub hardware: HardwareConfig,
    /// TP / EP layout across the cluster.
    #[serde(default)]
    pub parallel: ParallelConfig,
    /// Number of identical workers in this pool. Defaults to 1.
    #[serde(default = "default_num_workers")]
    pub num_workers: u32,
    /// KV tiers beyond HBM, chosen from the hardware's `[memory]` stores.
    #[serde(default)]
    pub memory: MemoryConfig,
}

fn default_num_workers() -> u32 {
    1
}

impl ClusterSpec {
    /// Aggregate memory bandwidth across the TP group, in bytes/sec.
    pub fn aggregate_memory_bandwidth(&self) -> f64 {
        self.hardware.memory_bandwidth * self.parallel.tp as f64
    }

    /// Aggregate memory capacity across the TP group, in bytes.
    pub fn aggregate_memory_capacity(&self) -> u64 {
        self.hardware
            .memory_capacity
            .saturating_mul(self.parallel.tp as u64)
    }

    /// The roofline cost model for `model` on this cluster: its hardware
    /// (including the collective fabric) and parallel layout.
    pub fn compute_engine(&self, model: ModelSpec) -> ComputeEngine {
        let engine = ComputeEngine::new(self.hardware.clone(), self.parallel.clone(), model);
        match self.active_kv_memory_per_gpu() {
            Some((_, bandwidth)) => engine.with_kv_memory_bandwidth(bandwidth),
            None => engine,
        }
    }

    /// Capacity and direct GPU-read bandwidth of an active-KV tier, per GPU.
    /// Validation restricts this to a capacity-bearing per-GPU store behind
    /// a direct link, so no topology sharing factor is needed here.
    pub fn active_kv_memory_per_gpu(&self) -> Option<(u64, f64)> {
        let name = self.memory.active_tier.as_deref()?;
        let template = self.hardware.memory.as_ref()?;
        let store = template.normal_store(name)?;
        let link = template.gpu_link_to(name)?;
        let capacity = self
            .memory
            .capacity
            .get(name)
            .copied()
            .unwrap_or(store.capacity);
        Some((
            capacity as u64,
            self.memory.active_bandwidth.unwrap_or(link.bandwidth),
        ))
    }

    /// Weights resident on the replica: every stream once, plus the
    /// attention projections replicated on each rank under DP-attention.
    pub fn resident_weight_bytes(&self, model: &ModelSpec) -> u64 {
        let mut bytes = model.weight_residency_bytes();
        if self.parallel.dp_attention && self.parallel.tp > 1 {
            bytes += (self.parallel.tp as u64 - 1) * model.attention_weight_bytes();
        }
        bytes
    }

    /// `ep` shards the experts of the `tp`-wide replica, so it must divide
    /// `tp`; a parallel group wider than one GPU needs a fabric to price its
    /// collectives, and one wider than a node needs `scale_out`.
    pub fn validate(&self) -> Result<(), String> {
        let (tp, ep) = (self.parallel.tp, self.parallel.ep);
        if tp == 0 || ep == 0 || tp % ep != 0 {
            return Err(format!("ep = {ep} must divide tp = {tp}"));
        }
        for (name, g) in [("tp", tp), ("ep", ep)] {
            if g <= 1 {
                continue;
            }
            match &self.hardware.fabric {
                None => {
                    return Err(format!(
                        "{name} = {g} on {} needs a [fabric] block to price its collectives",
                        self.hardware.name
                    ))
                }
                Some(f) if !f.supports_group(g) => {
                    return Err(format!(
                    "{name} = {g} spans nodes of {} GPUs on {} but its [fabric] has no scale_out",
                    f.gpus_per_node, self.hardware.name
                ))
                }
                Some(_) => {}
            }
        }
        self.memory.validate(self.hardware.memory.as_ref())?;
        Ok(())
    }

    /// KV cache bytes available to one worker of this cluster:
    /// `scheduler.kv_cache_capacity` when set, else what
    /// `aggregate_memory_capacity × gpu_memory_utilization` leaves after
    /// `model_size_bytes` of weights (vLLM's rule).
    /// How many KV domains a replica has: `tp` under DP-attention (each
    /// rank schedules its own sequences and holds their KV on its GPU),
    /// else one.
    pub fn kv_ranks(&self) -> u32 {
        if self.parallel.dp_attention {
            self.parallel.tp.max(1)
        } else {
            1
        }
    }

    /// Workers as the memory graph sees them: `(count, gpus per worker)`.
    /// A DP-attention replica contributes `tp` one-GPU workers.
    pub fn graph_workers(&self) -> (usize, u32) {
        let n = self.num_workers.max(1) as usize;
        let ranks = self.kv_ranks();
        if ranks > 1 {
            (n * ranks as usize, 1)
        } else {
            (n, self.parallel.tp.max(1))
        }
    }

    pub fn kv_cache_capacity(&self, scheduler: &SchedulerConfig, model_size_bytes: u64) -> u64 {
        if scheduler.kv_cache_capacity > 0 {
            return scheduler.kv_cache_capacity;
        }
        if let Some((capacity, _)) = self.active_kv_memory_per_gpu() {
            return capacity.saturating_mul(self.parallel.tp as u64);
        }
        let requested =
            (self.aggregate_memory_capacity() as f64 * scheduler.gpu_memory_utilization) as u64;
        requested.saturating_sub(model_size_bytes)
    }
}

/// A disaggregated topology: prefill and decode pools plus the link they
/// use for KV cache hand-off.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DisaggTopology {
    pub prefill: ClusterSpec,
    pub decode: ClusterSpec,
    /// Capacity of the network core between the prefill and decode pools,
    /// bytes/s, shared by every hand-off in flight. A hand-off's route runs
    /// from the prefill worker's GPU through its hardware's `[memory]`
    /// links to the network, over this core, and into the decode worker;
    /// with no network links on either side the core alone is the route.
    /// Unset: unbounded core (the NICs bound the transfer); an error if the
    /// hardware has no network links either.
    #[serde(default)]
    pub kv_link_bw: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_gpu_groups_need_a_fabric() {
        let mut cluster = crate::config::Config::test_default().cluster();
        assert!(cluster.validate().is_ok());
        cluster.parallel.tp = 2;
        let e = cluster.validate().unwrap_err();
        assert!(e.contains("[fabric]"), "{e}");
        cluster.hardware = crate::catalog::hardware("gh200").unwrap();
        assert!(cluster.validate().is_ok());
        cluster.parallel.tp = 8;
        assert!(cluster.validate().is_ok(), "gh200 declares scale_out");
        cluster.hardware.fabric.as_mut().unwrap().scale_out = None;
        let e = cluster.validate().unwrap_err();
        assert!(e.contains("scale_out"), "{e}");
    }

    #[test]
    fn active_grace_kv_uses_per_gpu_capacity_and_c2c_bandwidth() {
        let config = crate::config::Config::test_default();
        let mut cluster = config.cluster();
        cluster.hardware = crate::catalog::hardware("gh200").unwrap();
        cluster.parallel.tp = 16;
        cluster.memory =
            toml::from_str("active_tier = \"grace_dram\"\nactive_bandwidth = 420e9").unwrap();
        cluster.validate().unwrap();

        assert_eq!(
            cluster.active_kv_memory_per_gpu(),
            Some((120_000_000_000, 4.2e11))
        );
        let mut scheduler = config.scheduler;
        scheduler.kv_cache_capacity = 0;
        assert_eq!(
            cluster.kv_cache_capacity(&scheduler, u64::MAX),
            16 * 120_000_000_000
        );
    }

    #[test]
    fn parses_minimal_disagg_topology() {
        let toml_src = r#"
kv_link_bw = 9.0e11

[prefill.hardware]
name = "B300"
flops_fp4 = 1.5e16
flops_fp8 = 7.5e15
memory_bandwidth = 8.0e12
memory_capacity = 309237645312

[prefill.parallel]
tp = 4
ep = 1

[decode.hardware]
name = "B300"
flops_fp4 = 1.5e16
flops_fp8 = 7.5e15
memory_bandwidth = 8.0e12
memory_capacity = 309237645312

[decode.parallel]
tp = 4
ep = 1
"#;
        let topo: DisaggTopology = toml::from_str(toml_src).unwrap();
        assert_eq!(topo.prefill.parallel.tp, 4);
        assert_eq!(topo.decode.parallel.tp, 4);
        assert_eq!(topo.prefill.parallel.ep, 1);
        assert_eq!(topo.prefill.num_workers, 1);
        assert_eq!(topo.decode.num_workers, 1);
    }

    #[test]
    fn rejects_unknown_fields() {
        let toml_src = r#"
kv_link_bw = 9.0e11
inter_node_link_bw = 1.0
[prefill.hardware]
name = "x"
memory_bandwidth = 1.0
memory_capacity = 1
[decode.hardware]
name = "x"
memory_bandwidth = 1.0
memory_capacity = 1
"#;
        assert!(toml::from_str::<DisaggTopology>(toml_src).is_err());
    }
}
