use super::distribution::{Distribution, RunningMean};
use super::summary::{
    CorpusMetrics, CountStats, HandoffMetrics, HbmMetrics, LatencyMetrics, LatencyStats,
    MemoryMetrics, MetricsSummary, Preemptions, PrefixCacheMetrics, RequestCounts,
    ReusableKvMetrics, RouterMetrics, SessionMetrics, SimulationMetrics, ThroughputMetrics,
    Utilization, WorkMetrics,
};
use crate::kv_cache::PrefixCacheStats;
use crate::request::SessionLifecycle;
use crate::simulation::RequestTiming;
use std::collections::HashMap;

/// A latency measure: the samples plus, aligned with them, the simulated
/// time each was observed (the completion of the request that produced it).
#[derive(Debug, Default)]
struct LatencySeries {
    dist: Distribution,
    timestamps: Vec<f64>,
}

impl LatencySeries {
    fn push(&mut self, value: f64, observed_at: f64) {
        self.dist.push(value);
        self.timestamps.push(observed_at);
    }

    fn stats_ms(&mut self) -> LatencyStats {
        LatencyStats {
            min: self.dist.min() * 1000.0,
            mean: self.dist.mean() * 1000.0,
            p50: self.dist.quantile(0.50) * 1000.0,
            p90: self.dist.quantile(0.90) * 1000.0,
            p99: self.dist.quantile(0.99) * 1000.0,
        }
    }
}

/// A view of one latency series (seconds) with aligned observation times.
#[derive(Debug, Clone, Copy)]
pub struct SeriesRef<'a> {
    pub values: &'a [f64],
    pub timestamps: &'a [f64],
}

/// Views of the latency series.
#[derive(Debug, Clone, Copy)]
pub struct LatencySamples<'a> {
    pub ttft: SeriesRef<'a>,
    pub e2e: SeriesRef<'a>,
    pub tpot: SeriesRef<'a>,
    pub prefill: SeriesRef<'a>,
    pub handoff: SeriesRef<'a>,
    pub decode_ttft: SeriesRef<'a>,
}

/// Position up to which a consumer has already seen each series; use with
/// [`MetricsCollector::samples_since`] to stream only new samples.
#[derive(Debug, Clone, Copy, Default)]
pub struct SampleCursor {
    ttft: usize,
    e2e: usize,
    tpot: usize,
    prefill: usize,
    handoff: usize,
    decode_ttft: usize,
    input_lengths: usize,
    output_lengths: usize,
}

/// One completed request, for the `--request-csv` dump. Times in seconds.
#[derive(Debug, Clone, Copy)]
pub struct RequestRow {
    pub arrival: f64,
    pub recorded_arrival: Option<f64>,
    pub completion: f64,
    pub ttft: f64,
    pub e2e: f64,
    /// Mean time per output token after the first; NaN for one-token requests.
    pub mean_tpot: f64,
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    /// Prompt tokens served from the prefix cache.
    pub cached_tokens: u32,
    /// Prompt-prefix tokens resident on the selected decoder at hand-off.
    pub decode_cached_tokens: Option<u32>,
    pub num_preemptions: u32,
    /// Session workloads: (session ordinal, step), the gap since the parent's
    /// completion, and the KV bytes written into the caches between the
    /// parent's completion and this arrival (the reuse distance).
    pub session: Option<(u32, u32)>,
    /// Memory-graph id of the worker that served the request.
    pub worker: Option<u32>,
    pub gap: Option<f64>,
    /// Prompt tokens shared with the parent's context: the most the prefix
    /// cache could have served.
    pub shared_tokens: Option<u32>,
    pub reuse_distance_bytes: Option<u64>,
    /// `reuse_distance_bytes` plus the free blocks that hits pulled back into
    /// use in between: with it, a bracket on the LRU stack distance.
    pub reuse_touched_bytes: Option<u64>,
    /// The prefix lookup that fixed `cached_tokens`: when, and what it found
    /// in HBM vs the tiers; when the promotion started and landed.
    pub lookup: Option<crate::request::LookupRecord>,
}

/// Accumulates per-request timings and per-iteration utilisation into a
/// [`MetricsSummary`].
pub struct MetricsCollector {
    ttft: LatencySeries,
    e2e: LatencySeries,
    /// Per-request mean TPOT (requests with more than one output token).
    tpot: LatencySeries,
    prefill: LatencySeries,
    handoff: LatencySeries,
    decode_ttft: LatencySeries,

    total_input_tokens: u64,
    total_output_tokens: u64,
    start_time: f64,

    kv_cache_util: RunningMean,
    flops_util: RunningMean,
    bandwidth_util: RunningMean,

    total_preemptions: u64,

    pub completed_requests: u64,
    pub total_requests: u64,
    /// Refused at submission: context larger than a worker's KV cache.
    pub rejected_requests: u64,

    input_lengths: Vec<u32>,
    output_lengths: Vec<u32>,

    // Accumulators for the time-series interval means.
    interval_ttft: RunningMean,
    interval_tpot: RunningMean,

    pub request_rows: Vec<RequestRow>,
    session_turns_completed: HashMap<u32, u64>,
}

#[derive(Debug, Clone, Default)]
pub struct SummaryExtras {
    pub work: WorkMetrics,
    pub corpus: CorpusMetrics,
    pub sessions: Option<SessionMetrics>,
    pub simulation: SimulationMetrics,
    pub reusable_kv: Option<ReusableKvMetrics>,
    pub hbm: HbmMetrics,
}

impl MetricsCollector {
    pub fn new(start_time: f64) -> Self {
        Self {
            ttft: LatencySeries::default(),
            e2e: LatencySeries::default(),
            tpot: LatencySeries::default(),
            prefill: LatencySeries::default(),
            handoff: LatencySeries::default(),
            decode_ttft: LatencySeries::default(),
            total_input_tokens: 0,
            total_output_tokens: 0,
            start_time,
            kv_cache_util: RunningMean::default(),
            flops_util: RunningMean::default(),
            bandwidth_util: RunningMean::default(),
            total_preemptions: 0,
            completed_requests: 0,
            total_requests: 0,
            rejected_requests: 0,
            input_lengths: Vec::new(),
            output_lengths: Vec::new(),
            interval_ttft: RunningMean::default(),
            interval_tpot: RunningMean::default(),
            request_rows: Vec::new(),
            session_turns_completed: HashMap::new(),
        }
    }

    /// Record a completed request.
    pub fn record_request_completion(&mut self, timing: &RequestTiming) {
        if timing.rejected {
            self.rejected_requests += 1;
            return;
        }
        let observed_at = timing.completion_time;
        let ttft = timing.ttft();
        let e2e = timing.e2e();
        self.ttft.push(ttft, observed_at);
        self.e2e.push(e2e, observed_at);
        self.prefill.push(
            (timing.prefill_done_time - timing.arrival_time).max(0.0),
            observed_at,
        );
        self.handoff.push(
            (timing.handoff_done_time - timing.prefill_done_time).max(0.0),
            observed_at,
        );
        self.decode_ttft.push(
            (timing.first_token_time - timing.handoff_done_time).max(0.0),
            observed_at,
        );
        self.interval_ttft.add(ttft);
        let mean_tpot = match timing.tpot() {
            Some(t) => {
                self.tpot.push(t, observed_at);
                self.interval_tpot.add(t);
                t
            }
            None => f64::NAN,
        };

        self.total_input_tokens += timing.num_prompt_tokens as u64;
        self.total_output_tokens += timing.num_output_tokens as u64;
        self.total_preemptions += timing.num_preemptions as u64;

        self.request_rows.push(RequestRow {
            arrival: timing.arrival_time,
            recorded_arrival: timing.recorded_arrival_time,
            completion: timing.completion_time,
            ttft,
            e2e,
            mean_tpot,
            prompt_tokens: timing.num_prompt_tokens,
            output_tokens: timing.num_output_tokens,
            cached_tokens: timing.num_cached_tokens,
            decode_cached_tokens: timing.decode_cached_tokens,
            num_preemptions: timing.num_preemptions,
            session: timing.session.as_ref().map(|s| (s.session, s.step)),
            worker: timing.worker,
            gap: timing.session.as_ref().map(|s| s.gap),
            shared_tokens: timing.session.as_ref().map(|s| s.shared_tokens),
            reuse_distance_bytes: timing.session.as_ref().and_then(|s| s.reuse_distance_bytes),
            reuse_touched_bytes: timing.session.as_ref().and_then(|s| s.reuse_touched_bytes),
            lookup: timing.lookup,
        });

        self.input_lengths.push(timing.num_prompt_tokens);
        self.output_lengths.push(timing.num_output_tokens);
        self.completed_requests += 1;
        if let Some(step) = &timing.session {
            *self
                .session_turns_completed
                .entry(step.session)
                .or_default() += 1;
        }
    }

    pub fn input_lengths(&self) -> &[u32] {
        &self.input_lengths
    }

    pub fn output_lengths(&self) -> &[u32] {
        &self.output_lengths
    }

    pub fn completed_prompt_tokens(&self) -> u64 {
        self.total_input_tokens
    }

    pub fn completed_output_tokens(&self) -> u64 {
        self.total_output_tokens
    }

    pub fn work_metrics(
        &self,
        prefill_tokens_computed: u64,
        output_tokens_generated: u64,
    ) -> WorkMetrics {
        WorkMetrics {
            completed_prompt_tokens: self.total_input_tokens,
            completed_output_tokens: self.total_output_tokens,
            prefill_tokens_computed,
            output_tokens_generated,
            prefill_tokens_per_output_token: if output_tokens_generated == 0 {
                0.0
            } else {
                prefill_tokens_computed as f64 / output_tokens_generated as f64
            },
        }
    }

    pub fn session_metrics(&self, lifecycle: SessionLifecycle) -> SessionMetrics {
        let mut distribution = Distribution::default();
        let mut min = u64::MAX;
        let mut max = 0;
        let mut total = 0;
        for session in 0..lifecycle.started {
            let turns = self
                .session_turns_completed
                .get(&session)
                .copied()
                .unwrap_or(0);
            distribution.push(turns as f64);
            min = min.min(turns);
            max = max.max(turns);
            total += turns;
        }
        let count = CountStats {
            min: if lifecycle.started == 0 { 0 } else { min },
            mean: if lifecycle.started == 0 {
                0.0
            } else {
                total as f64 / lifecycle.started as f64
            },
            p50: distribution.quantile(0.50),
            p90: distribution.quantile(0.90),
            p99: distribution.quantile(0.99),
            max,
        };
        let classified = lifecycle.completed + lifecycle.deadline_censored;
        SessionMetrics {
            started: lifecycle.started as u64,
            completed: lifecycle.completed as u64,
            deadline_censored: lifecycle.deadline_censored as u64,
            unfinished: lifecycle.started.saturating_sub(classified) as u64,
            turns_completed: total,
            turns_per_started_session: count,
        }
    }

    /// All latency samples so far.
    pub fn latency_samples(&self) -> LatencySamples<'_> {
        LatencySamples {
            ttft: SeriesRef {
                values: self.ttft.dist.values(),
                timestamps: &self.ttft.timestamps,
            },
            e2e: SeriesRef {
                values: self.e2e.dist.values(),
                timestamps: &self.e2e.timestamps,
            },
            tpot: SeriesRef {
                values: self.tpot.dist.values(),
                timestamps: &self.tpot.timestamps,
            },
            prefill: SeriesRef {
                values: self.prefill.dist.values(),
                timestamps: &self.prefill.timestamps,
            },
            handoff: SeriesRef {
                values: self.handoff.dist.values(),
                timestamps: &self.handoff.timestamps,
            },
            decode_ttft: SeriesRef {
                values: self.decode_ttft.dist.values(),
                timestamps: &self.decode_ttft.timestamps,
            },
        }
    }

    /// Latency samples and length samples added since `cursor` was last
    /// advanced; advances the cursor.
    pub fn samples_since(
        &self,
        cursor: &mut SampleCursor,
    ) -> (LatencySamples<'_>, (&[u32], &[u32])) {
        let all = self.latency_samples();
        fn slice<'a>(s: SeriesRef<'a>, from: usize) -> SeriesRef<'a> {
            SeriesRef {
                values: &s.values[from..],
                timestamps: &s.timestamps[from..],
            }
        }
        let out = LatencySamples {
            ttft: slice(all.ttft, cursor.ttft),
            e2e: slice(all.e2e, cursor.e2e),
            tpot: slice(all.tpot, cursor.tpot),
            prefill: slice(all.prefill, cursor.prefill),
            handoff: slice(all.handoff, cursor.handoff),
            decode_ttft: slice(all.decode_ttft, cursor.decode_ttft),
        };
        let lengths = (
            &self.input_lengths[cursor.input_lengths..],
            &self.output_lengths[cursor.output_lengths..],
        );
        cursor.ttft = all.ttft.values.len();
        cursor.e2e = all.e2e.values.len();
        cursor.tpot = all.tpot.values.len();
        cursor.prefill = all.prefill.values.len();
        cursor.handoff = all.handoff.values.len();
        cursor.decode_ttft = all.decode_ttft.values.len();
        cursor.input_lengths = self.input_lengths.len();
        cursor.output_lengths = self.output_lengths.len();
        (out, lengths)
    }

    /// Mean TTFT and mean per-request TPOT (both in ms) over the completions
    /// since the previous call, then reset. NaN when there were none, so a
    /// chart can skip the interval.
    pub fn take_interval_latencies(&mut self) -> (f64, f64) {
        let ttft = if self.interval_ttft.count() > 0 {
            self.interval_ttft.mean() * 1000.0
        } else {
            f64::NAN
        };
        let tpot = if self.interval_tpot.count() > 0 {
            self.interval_tpot.mean() * 1000.0
        } else {
            f64::NAN
        };
        self.interval_ttft = RunningMean::default();
        self.interval_tpot = RunningMean::default();
        (ttft, tpot)
    }

    /// Record one iteration's utilisation figures.
    pub fn record_iteration_metrics(
        &mut self,
        kv_cache_util: f64,
        flops_util: f64,
        bandwidth_util: f64,
    ) {
        self.kv_cache_util.add(kv_cache_util);
        self.flops_util.add(flops_util);
        self.bandwidth_util.add(bandwidth_util);
    }

    /// Summary as of `current_time`.
    pub fn compute_summary(
        &mut self,
        current_time: f64,
        prefix_cache: PrefixCacheStats,
        router: RouterMetrics,
        decode_router: Option<RouterMetrics>,
        handoff: Option<HandoffMetrics>,
        memory: Option<MemoryMetrics>,
    ) -> MetricsSummary {
        self.compute_summary_with(
            current_time,
            prefix_cache,
            router,
            decode_router,
            handoff,
            memory,
            SummaryExtras::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute_summary_with(
        &mut self,
        current_time: f64,
        prefix_cache: PrefixCacheStats,
        router: RouterMetrics,
        decode_router: Option<RouterMetrics>,
        handoff: Option<HandoffMetrics>,
        memory: Option<MemoryMetrics>,
        extras: SummaryExtras,
    ) -> MetricsSummary {
        let elapsed = current_time - self.start_time;
        let per_sec = |n: f64| if elapsed > 0.0 { n / elapsed } else { 0.0 };
        MetricsSummary {
            latency_metrics: LatencyMetrics {
                ttft_ms: self.ttft.stats_ms(),
                e2e_ms: self.e2e.stats_ms(),
                per_token_ms: self.tpot.stats_ms(),
                prefill_ms: self.prefill.stats_ms(),
                handoff_ms: self.handoff.stats_ms(),
                decode_ttft_ms: self.decode_ttft.stats_ms(),
            },
            throughput_metrics: ThroughputMetrics {
                input_tokens_per_sec: per_sec(self.total_input_tokens as f64),
                output_tokens_per_sec: per_sec(self.total_output_tokens as f64),
                requests_per_sec: per_sec(self.completed_requests as f64),
            },
            work: extras.work,
            corpus: extras.corpus,
            utilization: Utilization {
                avg_kv_cache_util: self.kv_cache_util.mean(),
                avg_flops_util: self.flops_util.mean(),
                avg_bandwidth_util: self.bandwidth_util.mean(),
            },
            preemptions: Preemptions {
                total: self.total_preemptions,
                per_request_mean: if self.completed_requests > 0 {
                    self.total_preemptions as f64 / self.completed_requests as f64
                } else {
                    0.0
                },
            },
            requests: RequestCounts {
                completed: self.completed_requests,
                total: self.total_requests,
                rejected: self.rejected_requests,
            },
            sessions: extras.sessions,
            simulation: extras.simulation,
            prefix_cache: PrefixCacheMetrics {
                hits: prefix_cache.hits,
                misses: prefix_cache.misses,
                hit_rate: prefix_cache.hit_rate(),
                mean_hit_size: prefix_cache.mean_hit_size(),
                recomputed: prefix_cache.recomputed,
                recomputed_tokens: prefix_cache.recomputed_tokens,
                prefetches: prefix_cache.prefetches,
                prefetch_tokens: prefix_cache.prefetch_tokens,
            },
            router,
            decode_router,
            handoff,
            reusable_kv: extras.reusable_kv,
            hbm: extras.hbm,
            memory,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timing(
        id: &str,
        arrival: f64,
        first_token: f64,
        completion: f64,
        prompt: u32,
        output: u32,
        preemptions: u32,
    ) -> RequestTiming {
        RequestTiming {
            request_id: id.to_string(),
            arrival_time: arrival,
            recorded_arrival_time: None,
            record_metrics: true,
            prefill_done_time: first_token,
            handoff_done_time: first_token,
            first_token_time: first_token,
            completion_time: completion,
            num_prompt_tokens: prompt,
            num_output_tokens: output,
            num_cached_tokens: 0,
            decode_cached_tokens: None,
            session: None,
            worker: None,
            num_preemptions: preemptions,
            rejected: false,
            lookup: None,
        }
    }

    #[test]
    fn latencies_from_timing() {
        let mut m = MetricsCollector::new(0.0);
        // TTFT 1.0s, e2e 4.0s, 3 output tokens over a 3.0s decode span:
        // mean TPOT 1.5s.
        m.record_request_completion(&timing("a", 1.0, 2.0, 5.0, 100, 3, 0));
        let s = m.compute_summary(
            10.0,
            PrefixCacheStats::default(),
            RouterMetrics::default(),
            None,
            None,
            None,
        );
        assert!((s.latency_metrics.ttft_ms.mean - 1000.0).abs() < 1e-9);
        assert!((s.latency_metrics.e2e_ms.mean - 4000.0).abs() < 1e-9);
        assert!((s.latency_metrics.per_token_ms.mean - 1500.0).abs() < 1e-9);
        assert!((s.latency_metrics.prefill_ms.mean - 1000.0).abs() < 1e-9);
        assert_eq!(s.latency_metrics.handoff_ms.mean, 0.0);
        assert_eq!(s.latency_metrics.decode_ttft_ms.mean, 0.0);
        assert_eq!(s.requests.completed, 1);
        assert_eq!(m.request_rows.len(), 1);
        assert!((m.request_rows[0].mean_tpot - 1.5).abs() < 1e-12);
    }

    #[test]
    fn disaggregated_ttft_is_split_into_prefill_handoff_and_decode_wait() {
        let mut m = MetricsCollector::new(0.0);
        let mut t = timing("a", 1.0, 4.0, 5.0, 100, 3, 0);
        t.prefill_done_time = 2.0;
        t.handoff_done_time = 3.0;
        m.record_request_completion(&t);
        let s = m.compute_summary(
            5.0,
            PrefixCacheStats::default(),
            RouterMetrics::default(),
            None,
            None,
            None,
        );
        assert_eq!(s.latency_metrics.prefill_ms.mean, 1000.0);
        assert_eq!(s.latency_metrics.handoff_ms.mean, 1000.0);
        assert_eq!(s.latency_metrics.decode_ttft_ms.mean, 1000.0);
        assert_eq!(s.latency_metrics.ttft_ms.mean, 3000.0);
    }

    #[test]
    fn session_metrics_include_zero_turn_and_censored_sessions() {
        let mut m = MetricsCollector::new(0.0);
        for (id, session) in [("a", 0), ("b", 0), ("c", 1)] {
            let mut t = timing(id, 0.0, 1.0, 2.0, 10, 2, 0);
            t.session = Some(Box::new(crate::request::SessionStep {
                session,
                step: 0,
                gap: 0.0,
                shared_tokens: 0,
                shared_prefill_tokens: 0,
                shared_decode_tokens: 0,
                kind: None,
                parent_bytes_written: None,
                reuse_distance_bytes: None,
                parent_bytes_touched: None,
                reuse_touched_bytes: None,
                next_gap: None,
                next_shared_tokens: 0,
            }));
            m.record_request_completion(&t);
        }
        let sessions = m.session_metrics(SessionLifecycle {
            started: 3,
            completed: 1,
            deadline_censored: 1,
            active: 1,
        });
        assert_eq!(sessions.turns_completed, 3);
        assert_eq!(sessions.unfinished, 1);
        assert_eq!(sessions.turns_per_started_session.min, 0);
        assert_eq!(sessions.turns_per_started_session.mean, 1.0);
        assert_eq!(sessions.turns_per_started_session.p50, 1.0);
        assert_eq!(sessions.turns_per_started_session.max, 2);
    }

    #[test]
    fn one_token_requests_have_no_tpot_sample() {
        let mut m = MetricsCollector::new(0.0);
        m.record_request_completion(&timing("a", 0.0, 1.0, 1.0, 10, 1, 0));
        assert_eq!(m.latency_samples().tpot.values.len(), 0);
        assert_eq!(m.latency_samples().ttft.values.len(), 1);
        assert!(m.request_rows[0].mean_tpot.is_nan());
    }

    #[test]
    fn preemptions_and_throughput_are_aggregated() {
        let mut m = MetricsCollector::new(0.0);
        m.record_request_completion(&timing("a", 0.0, 1.0, 5.0, 100, 50, 2));
        m.record_request_completion(&timing("b", 0.0, 1.0, 5.0, 300, 150, 0));
        let s = m.compute_summary(
            10.0,
            PrefixCacheStats::default(),
            RouterMetrics::default(),
            None,
            None,
            None,
        );
        assert_eq!(s.preemptions.total, 2);
        assert!((s.preemptions.per_request_mean - 1.0).abs() < 1e-12);
        assert!((s.throughput_metrics.input_tokens_per_sec - 40.0).abs() < 1e-9);
        assert!((s.throughput_metrics.output_tokens_per_sec - 20.0).abs() < 1e-9);
        assert!((s.throughput_metrics.requests_per_sec - 0.2).abs() < 1e-12);
    }

    #[test]
    fn samples_since_streams_only_new_samples() {
        let mut m = MetricsCollector::new(0.0);
        let mut cursor = SampleCursor::default();
        m.record_request_completion(&timing("a", 0.0, 1.0, 5.0, 100, 50, 0));
        let (s, (inp, out)) = m.samples_since(&mut cursor);
        assert_eq!(s.ttft.values.len(), 1);
        assert_eq!(inp.len(), 1);
        assert_eq!(out.len(), 1);
        m.record_request_completion(&timing("b", 0.0, 1.0, 5.0, 100, 1, 0));
        m.record_request_completion(&timing("c", 0.0, 1.0, 5.0, 100, 5, 0));
        let (s, (inp, _)) = m.samples_since(&mut cursor);
        assert_eq!(s.ttft.values.len(), 2);
        assert_eq!(s.tpot.values.len(), 1); // "b" is a one-token request
        assert_eq!(inp.len(), 2);
        let (s, _) = m.samples_since(&mut cursor);
        assert_eq!(s.ttft.values.len(), 0);
    }

    #[test]
    fn interval_latencies_reset_after_take() {
        let mut m = MetricsCollector::new(0.0);
        m.record_request_completion(&timing("a", 0.0, 1.0, 3.0, 10, 3, 0));
        let (ttft, tpot) = m.take_interval_latencies();
        assert!((ttft - 1000.0).abs() < 1e-9);
        assert!((tpot - 1000.0).abs() < 1e-9);
        let (ttft, tpot) = m.take_interval_latencies();
        assert!(ttft.is_nan() && tpot.is_nan());
    }

    #[test]
    fn utilization_means_over_iterations() {
        let mut m = MetricsCollector::new(0.0);
        m.record_iteration_metrics(0.5, 0.2, 0.8);
        m.record_iteration_metrics(1.0, 0.4, 0.6);
        let s = m.compute_summary(
            1.0,
            PrefixCacheStats::default(),
            RouterMetrics::default(),
            None,
            None,
            None,
        );
        assert!((s.utilization.avg_kv_cache_util - 0.75).abs() < 1e-12);
        assert!((s.utilization.avg_flops_util - 0.3).abs() < 1e-12);
        assert!((s.utilization.avg_bandwidth_util - 0.7).abs() < 1e-12);
    }
}
