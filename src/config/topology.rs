//! Disaggregated cluster topology.
//!
//! A disaggregated deployment has separate prefill and decode worker pools,
//! each made up of one or more `ClusterSpec`s. Workers live on `Node`s
//! (uniform-bandwidth domains — e.g. one GB300 NVL72 rack is one `Node`),
//! with explicit per-link bandwidth for KV cache hand-off between prefill
//! and decode.
//!
//! These types are config-only for now; the disaggregated simulator that
//! consumes them lands in step 1d.

use serde::Deserialize;

use super::{CommsConfig, HardwareConfig, ParallelConfig};

/// A uniform-bandwidth domain. All GPUs in a `Node` see the same all-to-all
/// bandwidth via `intra_node_link_bw`. NVL72 = one Node with 72 GPUs and an
/// NVSwitch fabric. Cross-rack scenarios use multiple Nodes connected by an
/// `inter_node_link_bw` field on the enclosing topology.
#[derive(Debug, Clone, Deserialize)]
pub struct Node {
    pub name: String,
    pub num_gpus: u32,
    /// Per-GPU all-to-all bandwidth within the node, in bytes/sec. For NVL72
    /// this is NVSwitch-mediated NVLink5 (~900 GB/s per GPU per direction).
    pub intra_node_link_bw: f64,
}

/// A worker pool: one or more identically-shaped workers running the same
/// hardware spec and parallelism layout.
#[derive(Debug, Clone, Deserialize)]
pub struct ClusterSpec {
    /// Per-GPU hardware spec.
    pub hardware: HardwareConfig,
    /// TP / EP layout across the cluster.
    #[serde(default)]
    pub parallel: ParallelConfig,
    /// Optional collective-comms cost model for this cluster's TP / EP
    /// fabric. None = no collective term in `ComputeEngine`.
    #[serde(default)]
    pub comms: Option<CommsConfig>,
    /// Number of identical workers in this pool. Defaults to 1.
    #[serde(default = "default_num_workers")]
    pub num_workers: u32,
    /// Index of the `Node` (in `DisaggTopology::nodes`) this cluster's
    /// workers live on. For a single-rack NVL72 topology, every cluster has
    /// `node = 0`.
    #[serde(default)]
    pub node: usize,
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
        self.hardware.memory_capacity.saturating_mul(self.parallel.tp as u64)
    }

    /// Fill in `hardware.kv_cache_capacity` from `aggregate_memory_capacity *
    /// gpu_memory_utilization - model_size_bytes` if it's still at the
    /// sentinel 0. Matches vLLM's behaviour: requested_memory minus
    /// non-KV-cache memory (approximated as just the model weights).
    pub fn compute_kv_cache_capacity(&mut self, model_size_bytes: u64) {
        if self.hardware.kv_cache_capacity == 0 {
            let requested = (self.aggregate_memory_capacity() as f64
                * self.hardware.gpu_memory_utilization) as u64;
            self.hardware.kv_cache_capacity = requested.saturating_sub(model_size_bytes);
        }
    }
}

/// A disaggregated topology: prefill and decode pools plus the link they
/// use for KV cache hand-off.
#[derive(Debug, Clone, Deserialize)]
pub struct DisaggTopology {
    /// Uniform-bandwidth nodes backing the worker pools.
    pub nodes: Vec<Node>,
    /// Per-GPU bandwidth between nodes, in bytes/sec. `None` for single-node
    /// topologies (NVL72-internal). When set, this is the cross-rack
    /// network link (typical Quantum/Spectrum InfiniBand).
    #[serde(default)]
    pub inter_node_link_bw: Option<f64>,
    pub prefill: ClusterSpec,
    pub decode: ClusterSpec,
    /// Bandwidth used for KV cache hand-off from a prefill worker to a
    /// decode worker, in bytes/sec. Modelled as a single shared link in
    /// step 1d; later we may split per (prefill_worker, decode_worker)
    /// pair.
    pub kv_link_bw: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_disagg_topology() {
        let toml_src = r#"
kv_link_bw = 9.0e11

[[nodes]]
name = "nvl72-rack-0"
num_gpus = 72
intra_node_link_bw = 9.0e11

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
        assert_eq!(topo.nodes.len(), 1);
        assert_eq!(topo.nodes[0].num_gpus, 72);
        assert_eq!(topo.prefill.parallel.tp, 4);
        assert_eq!(topo.decode.parallel.tp, 4);
        assert_eq!(topo.prefill.parallel.ep, 1);
        assert_eq!(topo.prefill.num_workers, 1);
        assert_eq!(topo.decode.num_workers, 1);
        assert!(topo.inter_node_link_bw.is_none());
    }
}
