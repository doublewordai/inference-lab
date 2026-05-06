//! Batch-sim driver. Wraps the unified [`Engine`] with a synthetic
//! [`RequestGenerator`] and a [`MetricsCollector`], pumping events to
//! completion. This is what `Commands::Sim` (CLI) and the WASM entry points
//! drive.
//!
//! For real-time HTTP serving, see [`crate::serve::engine`] — same `Engine`,
//! different driver.

use std::collections::HashSet;

use super::engine::{Engine, IterationInfo, RequestTiming, StepKind, Topology};
use crate::config::Config;
use crate::dataset::{BatchTokenizerFn, DatasetLoader};
use crate::metrics::{LatencySampleTriplet, MetricsCollector};
use crate::request::{Request, RequestGenerator};

#[derive(Debug, Clone)]
pub struct TimeSeriesPoint {
    pub time: f64,
    pub arrivals: u64,
    pub running: usize,
    pub waiting: usize,
    pub kv_cache_util: f64,
    pub num_prefilling: usize,
    pub num_decoding: usize,
    pub prefill_tokens: u32,
    pub decode_tokens: u32,
    pub input_throughput: f64,
    pub output_throughput: f64,
    pub ttft_p50: f64,
    pub tpot_p50: f64,
}

pub struct ProgressInfo<'a> {
    pub current_time: f64,
    pub completed_requests: u64,
    pub total_requests: u64,
    pub running: usize,
    pub waiting: usize,
    pub kv_cache_util: f64,
    pub time_series: Option<&'a [TimeSeriesPoint]>,
    pub metrics: Option<crate::metrics::MetricsSummary>,
    pub latency_samples: Option<LatencySampleTriplet<'a>>,
    pub distribution_samples: Option<(&'a [u32], &'a [u32])>,
}

pub struct Simulator {
    engine: Engine,
    request_generator: RequestGenerator,
    metrics: MetricsCollector,
    time_series: Vec<TimeSeriesPoint>,

    sample_interval: f64,
    next_sample_time: f64,

    // Counters for accumulating tokens within a sample window.
    window_prefill_tokens: u32,
    window_decode_tokens: u32,

    last_sent_ttft_count: usize,
    last_sent_e2e_count: usize,
    last_sent_tpot_count: usize,
    last_sent_input_count: usize,
    last_sent_output_count: usize,
}

impl Simulator {
    /// Build a `Simulator` from a `Config`, with an optional batch tokenizer
    /// for dataset-mode workloads. Returns the simulator and the (possibly
    /// updated) config — `num_requests` is filled in from a streamed dataset
    /// count when the source path was provided but the count wasn't.
    pub fn new(
        mut config: Config,
        tokenizer: Option<BatchTokenizerFn>,
    ) -> Result<(Self, Config), String> {
        // The single-cluster `Config` describes one pool of workers. For
        // disagg topologies, callers build a [`Topology`] directly via
        // `Topology::from_disagg` and drive an `Engine` themselves.
        let cluster = crate::config::ClusterSpec {
            hardware: config.hardware.clone(),
            parallel: config.parallel.clone(),
            comms: None,
            num_workers: 1,
            node: 0,
        };
        let topology = Topology::aggregated(
            cluster,
            config.model.clone(),
            config.scheduler.clone(),
        )?;

        let request_generator = if let Some(dataset_path) = &config.workload.dataset_path {
            let tokenizer = tokenizer.ok_or_else(|| {
                format!(
                    "Dataset path '{}' provided but no tokenizer function supplied",
                    dataset_path
                )
            })?;

            if config.workload.num_requests.is_none() {
                let total_entries = DatasetLoader::count_entries(dataset_path)
                    .map_err(|e| format!("Failed to count entries in '{}': {}", dataset_path, e))?;
                config.workload.num_requests = Some(total_entries);
            }

            let dataset_iterator = DatasetLoader::from_file(dataset_path)
                .map_err(|e| format!("Failed to load dataset from '{}': {}", dataset_path, e))?;

            RequestGenerator::from_dataset(
                config.workload.clone(),
                dataset_iterator,
                None,
                tokenizer,
            )
        } else {
            RequestGenerator::new(config.workload.clone())
        };

        let simulator = Self {
            engine: Engine::new(topology),
            request_generator,
            metrics: MetricsCollector::new(0.0),
            time_series: Vec::new(),
            sample_interval: 0.1,
            next_sample_time: 0.0,
            window_prefill_tokens: 0,
            window_decode_tokens: 0,
            last_sent_ttft_count: 0,
            last_sent_e2e_count: 0,
            last_sent_tpot_count: 0,
            last_sent_input_count: 0,
            last_sent_output_count: 0,
        };

        Ok((simulator, config))
    }

    /// Pull all currently-available arrivals from the generator into the
    /// engine. Returns the number of requests submitted.
    fn drain_arrivals(&mut self) -> usize {
        let mut n = 0;
        // For closed-loop, peek_next_arrival_time returns a stale construction
        // value (always 0.0) and is never updated as completions queue
        // replenishment. Use current_time as a floor so closed-loop entries
        // are visible once we reach their arrival time.
        let now = self.engine.current_time();
        let bound = self
            .request_generator
            .peek_next_arrival_time()
            .max(now)
            + 1e-9;
        while let Some(req) = self.request_generator.next_if_before(bound) {
            self.engine.submit(req);
            self.metrics.total_requests += 1;
            n += 1;
        }
        n
    }

    /// Fast-forward sim time to the next arrival or pending event when the
    /// engine has nothing to do (Poisson idle gaps, dataset stragglers).
    fn maybe_skip_idle(&mut self) {
        if !self.engine.is_idle() {
            return;
        }
        let next_gen = self.request_generator.peek_next_arrival_time();
        if next_gen.is_finite() && next_gen > self.engine.current_time() {
            self.engine.advance_to(next_gen);
        }
    }

    pub fn run_with_callback<F>(&mut self, mut callback: F) -> Result<(), String>
    where
        F: FnMut(ProgressInfo),
    {
        let mut last_callback_time = 0.0;
        let callback_interval = 1.0;

        loop {
            self.drain_arrivals();
            self.maybe_skip_idle();

            if self.engine.next_event_time().is_none() {
                if self.should_terminate() {
                    self.emit_progress(&mut callback, true);
                    break;
                }
                // Nothing to do but no termination yet — bump sim a bit and
                // re-poll the generator. This guards against weird states
                // where a closed-loop generator has nothing pending and the
                // engine is empty but `is_finished` is false.
                self.engine.advance_to(self.engine.current_time() + 1e-3);
                continue;
            }

            let outcome = self.engine.step()?;
            if let Some(iter) = &outcome.iteration {
                self.handle_iteration(iter);
            }
            for completion in &outcome.completions {
                self.handle_completion(completion);
            }

            // Sample the time series at fixed sim-time intervals.
            while self.engine.current_time() >= self.next_sample_time {
                let prefilling = self.engine.aggregate_prefilling();
                let decoding = self.engine.aggregate_running() - prefilling;

                let prefill_tokens = self.window_prefill_tokens;
                let decode_tokens = self.window_decode_tokens;
                let input_throughput = prefill_tokens as f64 / self.sample_interval;
                let output_throughput = decode_tokens as f64 / self.sample_interval;
                let (ttft_mean, tpot_mean) = self.metrics.get_interval_latencies();

                self.time_series.push(TimeSeriesPoint {
                    time: self.engine.current_time(),
                    arrivals: self.metrics.total_requests,
                    running: self.engine.aggregate_running(),
                    waiting: self.engine.aggregate_waiting(),
                    kv_cache_util: self.engine.kv_cache_util(),
                    num_prefilling: prefilling,
                    num_decoding: decoding,
                    prefill_tokens,
                    decode_tokens,
                    input_throughput,
                    output_throughput,
                    ttft_p50: ttft_mean,
                    tpot_p50: tpot_mean,
                });
                self.window_prefill_tokens = 0;
                self.window_decode_tokens = 0;
                self.next_sample_time += self.sample_interval;
            }

            // Progress callback every callback_interval of sim time.
            if matches!(outcome.kind, StepKind::Iteration)
                && self.engine.current_time() - last_callback_time >= callback_interval
            {
                self.emit_progress(&mut callback, false);
                last_callback_time = self.engine.current_time();
            }

            if self.should_terminate() {
                self.emit_progress(&mut callback, true);
                break;
            }
        }

        Ok(())
    }

    fn handle_iteration(&mut self, iter: &IterationInfo) {
        // Token-by-phase counters and bandwidth/flops trackers.
        let mut prefill_ids: HashSet<&str> = HashSet::new();
        for prog in &iter.progress {
            if prog.was_prefill {
                prefill_ids.insert(&prog.request_id);
                self.window_prefill_tokens += prog.num_tokens;
            } else {
                self.window_decode_tokens += prog.num_tokens;
            }
        }
        self.metrics.record_iteration_metrics(
            self.engine.kv_cache_util(),
            iter.flops_util,
            iter.bandwidth_util,
        );
    }

    fn handle_completion(&mut self, timing: &RequestTiming) {
        // Build a Request stand-in from the timing for the metrics collector.
        // MetricsCollector wants the canonical Request (it pulls TTFT,
        // token_generation_times, num_preemptions, etc.). The engine doesn't
        // hand the original Request back, so we synthesise the fields the
        // collector reads.
        let mut req = Request::new(
            timing.request_id.clone(),
            0,
            timing.arrival_time,
            timing.num_prompt_tokens,
            timing.num_output_tokens,
        );
        req.first_token_time = Some(timing.first_token_time);
        req.completion_time = Some(timing.completion_time);
        req.num_output_tokens = timing.num_output_tokens;
        // Per-token timestamps: we lost individual times, so synthesise an
        // even decode cadence between first_token and completion. This
        // preserves the per-token-latency mean and percentiles to within
        // sample_interval but loses jitter detail. Acceptable trade-off
        // given the engine doesn't carry per-token samples.
        if timing.num_output_tokens > 0 {
            let n = timing.num_output_tokens as usize;
            let span = (timing.completion_time - timing.first_token_time).max(0.0);
            let dt = if n > 1 { span / (n - 1) as f64 } else { 0.0 };
            req.token_generation_times = (0..n)
                .map(|i| timing.first_token_time + i as f64 * dt)
                .collect();
        }

        self.metrics.record_request_completion(&req);
        self.request_generator
            .on_request_complete(timing.completion_time);
    }

    fn emit_progress<F: FnMut(ProgressInfo)>(&mut self, callback: &mut F, _final_step: bool) {
        let summary = self.compute_summary_inner();
        let latency_samples = self.metrics.get_latency_samples();
        let input_lengths = self.metrics.get_input_lengths();
        let output_lengths = self.metrics.get_output_lengths();

        let ttft_delta = &latency_samples.0 .0[self.last_sent_ttft_count..];
        let ttft_ts_delta = &latency_samples.0 .1[self.last_sent_ttft_count..];
        let e2e_delta = &latency_samples.1 .0[self.last_sent_e2e_count..];
        let e2e_ts_delta = &latency_samples.1 .1[self.last_sent_e2e_count..];
        let tpot_delta = &latency_samples.2 .0[self.last_sent_tpot_count..];
        let tpot_ts_delta = &latency_samples.2 .1[self.last_sent_tpot_count..];
        let input_delta = &input_lengths[self.last_sent_input_count..];
        let output_delta = &output_lengths[self.last_sent_output_count..];

        let progress = ProgressInfo {
            current_time: self.engine.current_time(),
            completed_requests: self.metrics.completed_requests,
            total_requests: self.metrics.total_requests,
            running: self.engine.aggregate_running(),
            waiting: self.engine.aggregate_waiting(),
            kv_cache_util: self.engine.kv_cache_util(),
            time_series: Some(&self.time_series),
            metrics: Some(summary),
            latency_samples: Some((
                (ttft_delta, ttft_ts_delta),
                (e2e_delta, e2e_ts_delta),
                (tpot_delta, tpot_ts_delta),
            )),
            distribution_samples: Some((input_delta, output_delta)),
        };
        callback(progress);

        self.last_sent_ttft_count = latency_samples.0 .0.len();
        self.last_sent_e2e_count = latency_samples.1 .0.len();
        self.last_sent_tpot_count = latency_samples.2 .0.len();
        self.last_sent_input_count = input_lengths.len();
        self.last_sent_output_count = output_lengths.len();
    }

    fn compute_summary_inner(&mut self) -> crate::metrics::MetricsSummary {
        let (hits, misses, hit_size_sum, hit_size_count) = self.engine.aggregate_prefix_cache();
        self.metrics.compute_summary(
            self.engine.current_time(),
            hits,
            misses,
            hit_size_sum,
            hit_size_count,
        )
    }

    pub fn get_metrics_summary(&mut self) -> crate::metrics::MetricsSummary {
        self.compute_summary_inner()
    }

    pub fn get_time_series_data(&self) -> &[TimeSeriesPoint] {
        &self.time_series
    }

    pub fn get_input_lengths(&self) -> &[u32] {
        self.metrics.get_input_lengths()
    }

    pub fn get_output_lengths(&self) -> &[u32] {
        self.metrics.get_output_lengths()
    }

    pub fn get_current_time(&self) -> f64 {
        self.engine.current_time()
    }

    pub fn get_latency_samples(&self) -> LatencySampleTriplet<'_> {
        self.metrics.get_latency_samples()
    }

    fn should_terminate(&self) -> bool {
        self.request_generator.is_finished() && self.engine.is_idle()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_minimal_test_config() -> Config {
        let mut config = Config::test_default();
        config.workload.num_requests = Some(10);
        config.workload.arrival_rate = 10.0;
        config
    }

    #[test]
    fn test_simulation_completes_all_requests() {
        let config = create_minimal_test_config();
        let mut simulator = Simulator::new(config, None).unwrap().0;
        simulator.run_with_callback(|_| {}).unwrap();
        let summary = simulator.get_metrics_summary();
        assert_eq!(summary.completed_requests, summary.total_requests);
        assert_eq!(summary.completed_requests, 10);
    }

    #[test]
    fn test_simulation_time_progresses() {
        let config = create_minimal_test_config();
        let mut simulator = Simulator::new(config, None).unwrap().0;
        let start = simulator.get_current_time();
        simulator.run_with_callback(|_| {}).unwrap();
        assert!(simulator.get_current_time() > start);
    }

    #[test]
    fn test_simulation_metrics_reasonable() {
        let config = create_minimal_test_config();
        let mut simulator = Simulator::new(config, None).unwrap().0;
        simulator.run_with_callback(|_| {}).unwrap();
        let s = simulator.get_metrics_summary();
        assert!(s.ttft_mean > 0.0 && s.ttft_mean.is_finite());
        assert!(s.e2e_mean > 0.0 && s.e2e_mean.is_finite());
        assert!(s.per_token_mean > 0.0 && s.per_token_mean.is_finite());
        assert!(s.ttft_min <= s.ttft_p50);
        assert!(s.ttft_p50 <= s.ttft_p90);
        assert!(s.ttft_p90 <= s.ttft_p99);
        assert!(s.input_tokens_per_sec > 0.0);
        assert!(s.output_tokens_per_sec > 0.0);
        assert!(s.requests_per_sec > 0.0);
    }

    #[test]
    fn test_simulation_with_fcfs_policy() {
        let mut config = create_minimal_test_config();
        config.scheduler.policy = "fcfs".to_string();
        let mut simulator = Simulator::new(config, None).unwrap().0;
        simulator.run_with_callback(|_| {}).unwrap();
        assert_eq!(simulator.get_metrics_summary().completed_requests, 10);
    }

    #[test]
    fn test_simulation_with_chunked_prefill() {
        let mut config = create_minimal_test_config();
        config.scheduler.enable_chunked_prefill = true;
        config.scheduler.long_prefill_token_threshold = 512;
        let mut simulator = Simulator::new(config, None).unwrap().0;
        simulator.run_with_callback(|_| {}).unwrap();
        assert_eq!(simulator.get_metrics_summary().completed_requests, 10);
    }

    #[test]
    fn test_simulation_time_series_collected() {
        let config = create_minimal_test_config();
        let mut simulator = Simulator::new(config, None).unwrap().0;
        simulator.run_with_callback(|_| {}).unwrap();
        let ts = simulator.get_time_series_data();
        assert!(!ts.is_empty());
        for i in 1..ts.len() {
            assert!(ts[i].time >= ts[i - 1].time);
        }
    }

    #[test]
    fn test_simulation_latency_samples_collected() {
        let config = create_minimal_test_config();
        let mut simulator = Simulator::new(config, None).unwrap().0;
        simulator.run_with_callback(|_| {}).unwrap();
        let ((ttft, ttft_ts), (e2e, e2e_ts), (tpot, tpot_ts)) = simulator.get_latency_samples();
        assert!(!ttft.is_empty());
        assert!(!e2e.is_empty());
        assert_eq!(ttft.len(), ttft_ts.len());
        assert_eq!(e2e.len(), e2e_ts.len());
        assert_eq!(tpot.len(), tpot_ts.len());
    }
}
