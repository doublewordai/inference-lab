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
    /// Absolute work totals. Unlike throughput, these retain the distinction
    /// between logical prompt length and positions the prefill workers
    /// actually computed.
    pub work: WorkMetrics,
    /// Completion/token gate for the entire input corpus, including requests
    /// before `measurement_start_secs`.
    pub corpus: CorpusMetrics,
    pub utilization: Utilization,
    pub preemptions: Preemptions,
    pub requests: RequestCounts,
    /// Session lifecycle and turns-per-session statistics. `None` for
    /// non-session workloads.
    pub sessions: Option<SessionMetrics>,
    /// Simulated-time boundary, deadline, backlog, and drain accounting.
    pub simulation: SimulationMetrics,
    pub prefix_cache: PrefixCacheMetrics,
    /// Routing into the pool arrivals enter.
    pub router: RouterMetrics,
    /// Disaggregated topologies: routing of hand-offs into the decode pool.
    pub decode_router: Option<RouterMetrics>,
    /// Disaggregated topologies: hand-off transfer totals.
    pub handoff: Option<HandoffMetrics>,
    /// Joint prefill/decode residency of reusable session KV. `None` for an
    /// aggregated topology.
    pub reusable_kv: Option<ReusableKvMetrics>,
    /// Per-worker HBM capacity, resident prefix data, and eviction totals.
    pub hbm: HbmMetrics,
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
    /// Most transfers (promotions, writes, hand-offs) in flight at once.
    #[serde(default)]
    pub peak_transfers_in_flight: u64,
    /// Bytes promoted directly from a sibling worker's HBM.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub peer_hbm_bytes_promoted: u64,
    /// Allocation/store-eviction attempts waiting on a pinned source.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub pin_stalls: u64,
    /// Completed promotions whose unpinned source lost a suffix in flight.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub partial_landings: u64,
}

fn is_zero_u64(value: &u64) -> bool {
    *value == 0
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
    /// Arrival until the prefill pool finishes the prompt. Includes prefill
    /// queueing and execution. Equal to TTFT on an aggregated topology.
    pub prefill_ms: LatencyStats,
    /// Prefill completion until the KV hand-off lands on the decoder.
    pub handoff_ms: LatencyStats,
    /// Hand-off completion until the decoder produces the first token.
    pub decode_ttft_ms: LatencyStats,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ThroughputMetrics {
    pub input_tokens_per_sec: f64,
    pub output_tokens_per_sec: f64,
    pub requests_per_sec: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct WorkMetrics {
    /// Full prompt and output lengths of completed requests.
    pub completed_prompt_tokens: u64,
    pub completed_output_tokens: u64,
    /// Positions actually computed by prefill iterations and output tokens
    /// actually generated by all iterations, including in-flight requests in
    /// a progress snapshot.
    pub prefill_tokens_computed: u64,
    pub output_tokens_generated: u64,
    pub prefill_tokens_per_output_token: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct CorpusMetrics {
    pub requests_admitted: u64,
    pub requests_completed: u64,
    pub requests_rejected: u64,
    pub completed_prompt_tokens: u64,
    pub completed_output_tokens: u64,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct CountStats {
    pub min: u64,
    pub mean: f64,
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
    pub max: u64,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct SessionMetrics {
    pub started: u64,
    /// Sessions whose final trace turn completed.
    pub completed: u64,
    /// Sessions retired because their next turn would arrive after the
    /// configured arrival deadline.
    pub deadline_censored: u64,
    /// Started sessions neither completed nor deadline-censored yet.
    pub unfinished: u64,
    pub turns_completed: u64,
    pub turns_per_started_session: CountStats,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct DeadlineMetrics {
    pub at_secs: f64,
    pub requests_admitted: u64,
    pub requests_completed: u64,
    pub output_tokens_generated: u64,
    pub output_tokens_per_sec: f64,
    pub running: u64,
    pub waiting: u64,
    pub handoffs_in_flight: u64,
    pub prefilling: u64,
    pub decoding: u64,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct SimulationMetrics {
    pub end_time_secs: f64,
    /// Start of the measured interval. Earlier replay work warms engine and
    /// cache state but is excluded from request and token metrics.
    pub measurement_start_secs: f64,
    /// End of the recorded measurement interval, when declared by a replay.
    pub measurement_end_secs: Option<f64>,
    pub arrival_deadline_secs: Option<f64>,
    /// Time after the arrival deadline until the engine became idle.
    pub drain_time_secs: f64,
    /// End-of-deadline state, before any event strictly after the deadline.
    pub at_deadline: Option<DeadlineMetrics>,
}

/// Tokens and their model-specific KV footprint for one residency cell.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct KvAmount {
    pub tokens: u64,
    pub bytes: u64,
}

/// Joint cache state of the reusable prefix of session re-entry requests.
/// First turns and novel suffixes are excluded from every cell.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ReusableKvResidency {
    pub requests: u64,
    pub reusable: KvAmount,
    pub decoder_hit_prefill_hit: KvAmount,
    pub decoder_hit_prefill_miss: KvAmount,
    pub decoder_miss_prefill_hit: KvAmount,
    pub decoder_miss_prefill_miss: KvAmount,
    /// Reusable KV absent on the prefiller, split by where the parent turn
    /// originally materialized it. Their sum, plus unattributed inherited
    /// context, is the prefiller-miss/recompute amount; the decode-origin
    /// cell is the no-write-back contribution.
    pub prefiller_miss_parent_prefill: KvAmount,
    pub prefiller_miss_parent_decode: KvAmount,
    pub prefiller_miss_unattributed: KvAmount,
    pub decoder_hit_byte_fraction: f64,
    pub prefill_hit_byte_fraction: f64,
    pub prefill_hit_given_decoder_miss_byte_fraction: f64,
    pub recompute_byte_fraction: f64,
    /// The two provenance cells above, each divided by all reusable KV bytes.
    pub parent_prefill_recompute_byte_fraction: f64,
    pub parent_decode_recompute_byte_fraction: f64,
    pub unattributed_recompute_byte_fraction: f64,
    pub transfer_byte_fraction: f64,
}

impl ReusableKvResidency {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_cells(
        requests: u64,
        reusable: KvAmount,
        both: KvAmount,
        decoder_only: KvAmount,
        prefill_only: KvAmount,
        neither: KvAmount,
        prefiller_miss_parent_prefill: KvAmount,
        prefiller_miss_parent_decode: KvAmount,
        prefiller_miss_unattributed: KvAmount,
    ) -> Self {
        let fraction = |n: u64, d: u64| if d == 0 { 0.0 } else { n as f64 / d as f64 };
        let decoder_hit = both.bytes + decoder_only.bytes;
        let prefill_hit = both.bytes + prefill_only.bytes;
        let decoder_miss = prefill_only.bytes + neither.bytes;
        debug_assert_eq!(
            decoder_only.bytes + neither.bytes,
            prefiller_miss_parent_prefill.bytes
                + prefiller_miss_parent_decode.bytes
                + prefiller_miss_unattributed.bytes
        );
        Self {
            requests,
            reusable,
            decoder_hit_prefill_hit: both,
            decoder_hit_prefill_miss: decoder_only,
            decoder_miss_prefill_hit: prefill_only,
            decoder_miss_prefill_miss: neither,
            prefiller_miss_parent_prefill,
            prefiller_miss_parent_decode,
            prefiller_miss_unattributed,
            decoder_hit_byte_fraction: fraction(decoder_hit, reusable.bytes),
            prefill_hit_byte_fraction: fraction(prefill_hit, reusable.bytes),
            prefill_hit_given_decoder_miss_byte_fraction: fraction(
                prefill_only.bytes,
                decoder_miss,
            ),
            recompute_byte_fraction: fraction(decoder_only.bytes + neither.bytes, reusable.bytes),
            parent_prefill_recompute_byte_fraction: fraction(
                prefiller_miss_parent_prefill.bytes,
                reusable.bytes,
            ),
            parent_decode_recompute_byte_fraction: fraction(
                prefiller_miss_parent_decode.bytes,
                reusable.bytes,
            ),
            unattributed_recompute_byte_fraction: fraction(
                prefiller_miss_unattributed.bytes,
                reusable.bytes,
            ),
            transfer_byte_fraction: fraction(prefill_only.bytes + neither.bytes, reusable.bytes),
        }
    }
}

/// Unique prompt positions computed on the prefiller, classified using the
/// cache state of the request's final admission. Additional work redone after
/// preemption is the difference between this total and `WorkMetrics`.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct SessionPrefillWork {
    pub new_prompt: KvAmount,
    pub parent_prefill_recompute: KvAmount,
    pub parent_decode_recompute: KvAmount,
    pub unattributed_recompute: KvAmount,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct RankReusableKvMetrics {
    /// Memory-graph worker id and DPA rank within its replica.
    pub worker: u32,
    pub rank: u32,
    pub residency: ReusableKvResidency,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ReusableKvMetrics {
    pub prefill_work: SessionPrefillWork,
    pub total: ReusableKvResidency,
    pub per_decode_rank: Vec<RankReusableKvMetrics>,
    pub recomputed_reusable_tokens_per_output_token: f64,
    pub transferred_reusable_bytes_per_output_token: f64,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct HbmWorkerMetrics {
    pub worker: u32,
    pub rank: u32,
    /// Requests currently executing on this worker.
    pub running: u64,
    /// Requests queued or parked on a KV transfer for this worker.
    pub waiting: u64,
    pub capacity_bytes: u64,
    /// Prefix-cache content resident at the observation point, including
    /// unreferenced but still reusable data.
    pub resident_prefix_bytes: u64,
    /// Referenced or reserved blocks at the observation point.
    pub active_or_reserved_bytes: u64,
    /// Eviction batches and bytes recycled over the run.
    pub eviction_events: u64,
    pub evicted_bytes: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct HbmPoolMetrics {
    pub role: String,
    pub capacity_bytes: u64,
    pub resident_prefix_bytes: u64,
    pub active_or_reserved_bytes: u64,
    pub eviction_events: u64,
    pub evicted_bytes: u64,
    pub workers: Vec<HbmWorkerMetrics>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct HbmMetrics {
    pub pools: Vec<HbmPoolMetrics>,
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
