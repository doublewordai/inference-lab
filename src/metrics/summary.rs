use serde::Serialize;

/// Min / mean / quantiles of one latency measure, in milliseconds.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct LatencyStats {
    pub min: f64,
    pub mean: f64,
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
}

/// End-of-run (or snapshot) summary of a simulation. Serialises to the
/// `--output` JSON layout.
#[derive(Debug, Clone, Serialize)]
pub struct MetricsSummary {
    pub latency_metrics: LatencyMetrics,
    pub throughput_metrics: ThroughputMetrics,
    pub utilization: Utilization,
    pub preemptions: Preemptions,
    pub requests: RequestCounts,
    pub prefix_cache: PrefixCacheMetrics,
    /// Routing into the pool arrivals enter.
    pub router: RouterMetrics,
    /// Disaggregated topologies: routing of hand-offs into the decode pool.
    pub decode_router: Option<RouterMetrics>,
    /// Disaggregated topologies: hand-off transfer totals.
    pub handoff: Option<HandoffMetrics>,
    /// KV memory beyond HBM: stores and links, when the deployment has
    /// tiers.
    pub memory: Option<MemoryMetrics>,
}

/// One store name of the memory graph, totalled over its instances.
#[derive(Debug, Clone, Default, Serialize)]
pub struct StoreMetrics {
    pub name: String,
    pub instances: u64,
    pub capacity_blocks: u64,
    /// Blocks held (resident + arriving) at the end of the run.
    pub held_blocks: u64,
    /// Bytes whose write landed in the store.
    pub bytes_written: u64,
    /// Bytes promoted out of the store.
    pub bytes_read: u64,
    /// Bytes evicted or expired without ever being promoted.
    pub dead_bytes: u64,
    pub evictions: u64,
    pub expired: u64,
}

/// One link name of the memory graph, totalled over its instances (both
/// directions).
#[derive(Debug, Clone, Default, Serialize)]
pub struct EdgeMetrics {
    pub name: String,
    pub instances: u64,
    /// Capacity of one instance, one direction, bytes/s.
    pub capacity: f64,
    pub bytes_moved: f64,
    /// `bytes_moved / (instances × 2 × capacity × elapsed)`: mean use of
    /// the link's total capacity over the run.
    pub utilisation: f64,
}

/// KV memory beyond HBM over the run.
#[derive(Debug, Clone, Default, Serialize)]
pub struct MemoryMetrics {
    pub write_policy: String,
    pub eviction_policy: String,
    pub stores: Vec<StoreMetrics>,
    pub links: Vec<EdgeMetrics>,
    /// Bytes submitted as writes (GPU → store and cascades).
    pub bytes_written: f64,
    /// Bytes submitted as promotions.
    pub bytes_promoted: f64,
    /// Promotions that had to wait for a write still arriving.
    pub write_race_waits: u64,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct LatencyMetrics {
    /// Time to first token, over completed requests.
    pub ttft_ms: LatencyStats,
    /// Arrival to completion, over completed requests.
    pub e2e_ms: LatencyStats,
    /// Per-request mean time per output token (decode span over output
    /// tokens after the first), over completed requests with more than one
    /// output token.
    pub per_token_ms: LatencyStats,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ThroughputMetrics {
    pub input_tokens_per_sec: f64,
    pub output_tokens_per_sec: f64,
    pub requests_per_sec: f64,
}

/// Means over iterations of the per-iteration utilisation figures.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Utilization {
    pub avg_kv_cache_util: f64,
    pub avg_flops_util: f64,
    pub avg_bandwidth_util: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Preemptions {
    /// Preemptions suffered by completed requests.
    pub total: u64,
    pub per_request_mean: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct RequestCounts {
    pub completed: u64,
    pub total: u64,
    /// Refused at submission: the request's whole context needs more KV
    /// than a worker has, so it could never complete.
    #[serde(default)]
    pub rejected: u64,
}

/// What the entry pool's router did (see `crate::router::RouterStats`).
#[derive(Debug, Clone, Default, Serialize)]
pub struct RouterMetrics {
    pub policy: String,
    /// Requests routed to each replica.
    pub per_replica: Vec<u64>,
    /// Requests for which some replica held a nonzero cached prefix at
    /// routing time. Only known to routers that read the prefix signal.
    pub prefix_available: u64,
    /// Of those, requests routed to a replica holding a nonzero prefix.
    pub prefix_routed: u64,
    /// Of those, requests routed away from the longest-prefix holder.
    pub prefix_forgone: u64,
}

impl RouterMetrics {
    pub fn from_stats(policy: &str, stats: &crate::router::RouterStats) -> Self {
        Self {
            policy: policy.to_string(),
            per_replica: stats.per_worker.clone(),
            prefix_available: stats.prefix_available,
            prefix_routed: stats.prefix_routed,
            prefix_forgone: stats.prefix_forgone,
        }
    }
}

/// Hand-off transfers on a disaggregated topology.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct HandoffMetrics {
    pub transfers: u64,
    /// Bytes put on the hand-off link.
    pub bytes: u64,
    /// Bytes not transferred because the chosen decoder already held that
    /// prompt prefix in HBM.
    pub bytes_skipped: u64,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct PrefixCacheMetrics {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    /// Mean cached prefix length per lookup, in tokens.
    pub mean_hit_size: f64,
    /// Lookups whose tier-held prefix was recomputed rather than fetched
    /// (`[memory] source = min_time`), and the tokens recomputed.
    pub recomputed: u64,
    pub recomputed_tokens: u64,
    /// Prefetches started ahead of announced re-entries (`[memory]
    /// prefetch = outlook`), and the tokens they pulled up.
    pub prefetches: u64,
    pub prefetch_tokens: u64,
}
