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
    pub router: RouterMetrics,
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

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct PrefixCacheMetrics {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    /// Mean cached prefix length per lookup, in tokens.
    pub mean_hit_size: f64,
}
