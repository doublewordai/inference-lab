//! Batch-sim driver. Wraps the unified [`Engine`] with a [`RequestGenerator`]
//! and a [`MetricsCollector`], pumping events to completion. This is what
//! `Commands::Sim` (CLI) and the WASM entry points drive.
//!
//! For real-time HTTP serving, see `crate::serve::engine` — same `Engine`,
//! different driver.

use super::engine::{Engine, IterationInfo, RequestTiming, StepKind, Topology};
use super::spec::DepthSample;
use crate::config::Config;
use crate::dataset::{BatchTokenizerFn, DatasetLoader};
use crate::metrics::{LatencySamples, MetricsCollector, MetricsSummary, RequestRow, SampleCursor};
use crate::request::RequestGenerator;

/// One sample of the fixed-interval time series.
#[derive(Debug, Clone)]
pub struct TimeSeriesPoint {
    pub time: f64,
    /// Requests submitted so far.
    pub arrivals: u64,
    pub running: usize,
    pub waiting: usize,
    pub kv_cache_util: f64,
    pub num_prefilling: usize,
    pub num_decoding: usize,
    /// Prompt positions computed in the interval.
    pub prefill_tokens: u32,
    /// Output tokens generated in the interval.
    pub decode_tokens: u32,
    pub input_throughput: f64,
    pub output_throughput: f64,
    /// Mean TTFT (ms) over requests completed in the interval; NaN if none.
    pub ttft_interval_mean_ms: f64,
    /// Mean per-request TPOT (ms) over requests completed in the interval;
    /// NaN if none.
    pub tpot_interval_mean_ms: f64,
}

/// Snapshot handed to the progress callback.
pub struct ProgressInfo<'a> {
    pub current_time: f64,
    pub completed_requests: u64,
    pub total_requests: u64,
    pub running: usize,
    pub waiting: usize,
    pub kv_cache_util: f64,
    pub time_series: &'a [TimeSeriesPoint],
    pub metrics: MetricsSummary,
    /// Latency samples added since the previous callback.
    pub latency_samples: LatencySamples<'a>,
    /// Prompt / output lengths of requests completed since the previous
    /// callback.
    pub input_lengths: &'a [u32],
    pub output_lengths: &'a [u32],
}

pub struct Simulator {
    engine: Engine,
    request_generator: RequestGenerator,
    metrics: MetricsCollector,
    config: Config,
    time_series: Vec<TimeSeriesPoint>,

    sample_interval: f64,
    next_sample_time: f64,

    // Counters for accumulating tokens within a sample window.
    window_prefill_tokens: u32,
    window_decode_tokens: u32,

    /// Samples already delivered to the progress callback.
    sent: SampleCursor,
}

impl Simulator {
    /// Build a `Simulator` from a `Config`, with an optional batch tokenizer
    /// for dataset-mode workloads. When the workload names a dataset but no
    /// `num_requests`, the dataset is counted and the config's `num_requests`
    /// filled in (see [`Simulator::config`]).
    pub fn new(mut config: Config, tokenizer: Option<BatchTokenizerFn>) -> Result<Self, String> {
        // The single-cluster `Config` describes one pool of workers. For
        // disagg topologies, callers build a [`Topology`] directly via
        // `Topology::from_disagg` and drive an `Engine` themselves.
        let topology = Topology::aggregated(
            config.cluster(),
            config.model.clone(),
            config.scheduler.clone(),
        )?;

        let request_generator = if let Some(dataset_path) = &config.workload.dataset_path {
            let tokenizer = tokenizer.ok_or_else(|| {
                format!("Dataset path '{dataset_path}' provided but no tokenizer function supplied")
            })?;

            if config.workload.num_requests.is_none() {
                let total_entries = DatasetLoader::count_entries(dataset_path)
                    .map_err(|e| format!("Failed to count entries in '{dataset_path}': {e}"))?;
                config.workload.num_requests = Some(total_entries);
            }

            let dataset_iterator = DatasetLoader::from_file(dataset_path)
                .map_err(|e| format!("Failed to load dataset from '{dataset_path}': {e}"))?;

            RequestGenerator::from_dataset(
                config.workload.clone(),
                config.scheduler.block_size,
                dataset_iterator,
                tokenizer,
            )
        } else {
            RequestGenerator::new(config.workload.clone())
        };

        let mut engine = Engine::new(topology);
        if let Some(spec) = &config.speculative {
            engine.enable_speculative(spec.clone(), config.workload.seed)?;
        }

        Ok(Self {
            engine,
            request_generator,
            metrics: MetricsCollector::new(0.0),
            config,
            time_series: Vec::new(),
            sample_interval: 0.1,
            next_sample_time: 0.0,
            window_prefill_tokens: 0,
            window_decode_tokens: 0,
            sent: SampleCursor::default(),
        })
    }

    /// The configuration in force, including any fields filled in at
    /// construction (`workload.num_requests` from a counted dataset).
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Pull all currently-available arrivals from the generator into the
    /// engine. Returns the number of requests submitted.
    fn drain_arrivals(&mut self) -> usize {
        let mut n = 0;
        // Closed-loop replenishment is keyed on completions, not on the
        // generator's own clock; use current_time as a floor so those
        // entries are visible once we reach their arrival time.
        let now = self.engine.current_time();
        let bound = self.request_generator.peek_next_arrival_time().max(now) + 1e-9;
        while let Some(req) = self.request_generator.next_if_before(bound) {
            self.engine.submit(req);
            self.metrics.total_requests += 1;
            n += 1;
        }
        n
    }

    /// Fast-forward sim time to the next arrival when the engine has nothing
    /// to do (Poisson idle gaps, dataset stragglers).
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
                    self.emit_progress(&mut callback);
                    break;
                }
                // No pending event, no arrival to jump to, and work still in
                // the system: nothing can ever make progress (a request that
                // can never be scheduled, e.g. a prompt longer than the KV
                // cache, or a closed loop with no users).
                return Err(format!(
                    "simulation stalled at t={:.3}: {} running, {} waiting, no pending events",
                    self.engine.current_time(),
                    self.engine.aggregate_running(),
                    self.engine.aggregate_waiting()
                ));
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
                let (ttft_mean, tpot_mean) = self.metrics.take_interval_latencies();

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
                    input_throughput: prefill_tokens as f64 / self.sample_interval,
                    output_throughput: decode_tokens as f64 / self.sample_interval,
                    ttft_interval_mean_ms: ttft_mean,
                    tpot_interval_mean_ms: tpot_mean,
                });
                self.window_prefill_tokens = 0;
                self.window_decode_tokens = 0;
                self.next_sample_time += self.sample_interval;
            }

            // Progress callback every callback_interval of sim time.
            if matches!(outcome.kind, StepKind::Iteration)
                && self.engine.current_time() - last_callback_time >= callback_interval
            {
                self.emit_progress(&mut callback);
                last_callback_time = self.engine.current_time();
            }

            if self.should_terminate() {
                self.emit_progress(&mut callback);
                break;
            }
        }

        Ok(())
    }

    fn handle_iteration(&mut self, iter: &IterationInfo) {
        for prog in &iter.progress {
            if prog.was_prefill {
                self.window_prefill_tokens += prog.num_tokens;
            }
            self.window_decode_tokens += prog.num_output;
        }
        self.metrics.record_iteration_metrics(
            self.engine.kv_cache_util(),
            iter.flops_util,
            iter.bandwidth_util,
        );
    }

    fn handle_completion(&mut self, timing: &RequestTiming) {
        self.metrics.record_request_completion(timing);
        self.request_generator
            .on_request_complete(timing.completion_time);
    }

    fn emit_progress<F: FnMut(ProgressInfo)>(&mut self, callback: &mut F) {
        let metrics = self.summary();
        let (latency_samples, (input_lengths, output_lengths)) =
            self.metrics.samples_since(&mut self.sent);
        callback(ProgressInfo {
            current_time: self.engine.current_time(),
            completed_requests: self.metrics.completed_requests,
            total_requests: self.metrics.total_requests,
            running: self.engine.aggregate_running(),
            waiting: self.engine.aggregate_waiting(),
            kv_cache_util: self.engine.kv_cache_util(),
            time_series: &self.time_series,
            metrics,
            latency_samples,
            input_lengths,
            output_lengths,
        });
    }

    /// Metrics summary as of the current simulated time.
    pub fn summary(&mut self) -> MetricsSummary {
        self.metrics.compute_summary(
            self.engine.current_time(),
            self.engine.aggregate_prefix_cache(),
        )
    }

    /// Per-request rows for the `--request-csv` dump.
    pub fn request_rows(&self) -> &[RequestRow] {
        &self.metrics.request_rows
    }

    /// Per-second speculative draft-depth series from the engine.
    pub fn spec_depth_series(&self) -> Vec<DepthSample> {
        self.engine.spec_depth_series()
    }

    pub fn time_series(&self) -> &[TimeSeriesPoint] {
        &self.time_series
    }

    pub fn input_lengths(&self) -> &[u32] {
        self.metrics.input_lengths()
    }

    pub fn output_lengths(&self) -> &[u32] {
        self.metrics.output_lengths()
    }

    pub fn current_time(&self) -> f64 {
        self.engine.current_time()
    }

    /// All latency samples so far.
    pub fn latency_samples(&self) -> LatencySamples<'_> {
        self.metrics.latency_samples()
    }

    fn should_terminate(&self) -> bool {
        self.request_generator.is_finished() && self.engine.is_idle()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LengthDistribution;

    #[test]
    fn kv_cache_smaller_than_a_block_is_a_config_error() {
        let mut config = create_minimal_test_config();
        config.scheduler.kv_cache_capacity = 1;
        let err = Simulator::new(config, None).err().unwrap();
        assert!(err.contains("less than one"), "{err}");
    }

    fn create_minimal_test_config() -> Config {
        let mut config = Config::test_default();
        config.workload.num_requests = Some(10);
        config.workload.arrival_rate = 10.0;
        config
    }

    #[test]
    fn test_simulation_completes_all_requests() {
        let config = create_minimal_test_config();
        let mut simulator = Simulator::new(config, None).unwrap();
        simulator.run_with_callback(|_| {}).unwrap();
        let summary = simulator.summary();
        assert_eq!(summary.requests.completed, summary.requests.total);
        assert_eq!(summary.requests.completed, 10);
    }

    #[test]
    fn test_simulation_time_progresses() {
        let config = create_minimal_test_config();
        let mut simulator = Simulator::new(config, None).unwrap();
        let start = simulator.current_time();
        simulator.run_with_callback(|_| {}).unwrap();
        assert!(simulator.current_time() > start);
    }

    #[test]
    fn test_simulation_metrics_reasonable() {
        let config = create_minimal_test_config();
        let mut simulator = Simulator::new(config, None).unwrap();
        simulator.run_with_callback(|_| {}).unwrap();
        let s = simulator.summary();
        let l = &s.latency_metrics;
        assert!(l.ttft_ms.mean > 0.0 && l.ttft_ms.mean.is_finite());
        assert!(l.e2e_ms.mean > 0.0 && l.e2e_ms.mean.is_finite());
        assert!(l.per_token_ms.mean > 0.0 && l.per_token_ms.mean.is_finite());
        assert!(l.ttft_ms.min <= l.ttft_ms.p50);
        assert!(l.ttft_ms.p50 <= l.ttft_ms.p90);
        assert!(l.ttft_ms.p90 <= l.ttft_ms.p99);
        assert!(s.throughput_metrics.input_tokens_per_sec > 0.0);
        assert!(s.throughput_metrics.output_tokens_per_sec > 0.0);
        assert!(s.throughput_metrics.requests_per_sec > 0.0);
    }

    #[test]
    fn test_simulation_with_chunked_prefill() {
        let mut config = create_minimal_test_config();
        config.scheduler.enable_chunked_prefill = true;
        config.scheduler.long_prefill_token_threshold = 512;
        let mut simulator = Simulator::new(config, None).unwrap();
        simulator.run_with_callback(|_| {}).unwrap();
        assert_eq!(simulator.summary().requests.completed, 10);
    }

    #[test]
    fn test_simulation_time_series_collected() {
        let config = create_minimal_test_config();
        let mut simulator = Simulator::new(config, None).unwrap();
        simulator.run_with_callback(|_| {}).unwrap();
        let ts = simulator.time_series();
        assert!(!ts.is_empty());
        for i in 1..ts.len() {
            assert!(ts[i].time >= ts[i - 1].time);
        }
        // Every output token is counted exactly once across the windows.
        let decode_total: u64 = ts.iter().map(|p| p.decode_tokens as u64).sum();
        let output_total: u64 = simulator.output_lengths().iter().map(|&x| x as u64).sum();
        assert_eq!(decode_total, output_total);
    }

    #[test]
    fn unschedulable_request_is_an_error_not_a_hang() {
        let mut config = create_minimal_test_config();
        // A prompt longer than the whole KV cache (64 blocks of 16 tokens)
        // can never be admitted.
        config.scheduler.kv_cache_capacity = 64 * config.model.kv_storage_bytes(16);
        config.workload.input_len_dist = LengthDistribution::Fixed { value: 4096 };
        config.workload.num_requests = Some(1);
        let mut simulator = Simulator::new(config, None).unwrap();
        let err = simulator.run_with_callback(|_| {}).unwrap_err();
        assert!(err.contains("stalled"), "{err}");
    }

    #[test]
    fn closed_loop_with_jitter_starts_at_the_first_staggered_arrival() {
        let mut config = create_minimal_test_config();
        config.workload.arrival_pattern = crate::config::ArrivalPattern::ClosedLoop;
        config.workload.num_concurrent_users = Some(4);
        config.workload.closed_loop_jitter_secs = Some(0.05);
        config.workload.num_requests = Some(12);
        let mut simulator = Simulator::new(config, None).unwrap();
        // Nothing is due at t=0; the simulator must jump to the earliest
        // jittered arrival instead of reporting a stall.
        simulator.run_with_callback(|_| {}).unwrap();
        let summary = simulator.summary();
        assert_eq!(summary.requests.completed, 12);
    }

    #[test]
    fn test_progress_callback_streams_each_sample_once() {
        let config = create_minimal_test_config();
        let mut simulator = Simulator::new(config, None).unwrap();
        let mut streamed = 0usize;
        simulator
            .run_with_callback(|p| streamed += p.latency_samples.ttft.values.len())
            .unwrap();
        assert_eq!(streamed, simulator.latency_samples().ttft.values.len());
        assert_eq!(streamed, 10);
    }
}
