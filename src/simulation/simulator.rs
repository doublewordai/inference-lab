use crate::compute::ComputeEngine;
use crate::config::Config;
use crate::kv_cache::KVCacheManager;
use crate::metrics::MetricsCollector;
use crate::request::RequestGenerator;
use crate::scheduler::Scheduler;

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
    pub input_throughput: f64,  // Input tokens per second (windowed)
    pub output_throughput: f64, // Output tokens per second (windowed)
    pub ttft_p50: f64,          // TTFT p50 in recent window (ms)
    pub tpot_p50: f64,          // TPOT p50 in recent window (ms)
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
    pub latency_samples: Option<(
        (&'a [f64], &'a [f64]),
        (&'a [f64], &'a [f64]),
        (&'a [f64], &'a [f64]),
    )>,
    pub distribution_samples: Option<(&'a [u32], &'a [u32])>, // (input_lengths, output_lengths)
}

pub struct Simulator {
    scheduler: Scheduler,
    compute_engine: ComputeEngine,
    request_generator: RequestGenerator,
    metrics: MetricsCollector,
    time_series_data: Vec<TimeSeriesPoint>,
    sample_interval: f64,
    next_sample_time: f64,
    prev_sample_time: f64,

    current_time: f64,
    iteration: u64,

    // Track last sent sample counts for streaming deltas (one per metric type)
    last_sent_ttft_count: usize,
    last_sent_e2e_count: usize,
    last_sent_tpot_count: usize,
    last_sent_input_count: usize,
    last_sent_output_count: usize,
}

impl Simulator {
    pub fn new(config: Config) -> Result<Self, String> {
        let kv_cache_manager = KVCacheManager::new(
            config.hardware.kv_cache_capacity,
            config.scheduler.block_size,
            config.model.kv_cache_bytes_per_token,
            true, // enable_prefix_caching
        );

        let scheduler = Scheduler::new(
            config.scheduler.clone(),
            config.hardware.clone(),
            config.model.clone(),
            kv_cache_manager,
        )?;

        let compute_engine = ComputeEngine::new(config.hardware, config.model);
        let request_generator = RequestGenerator::new(config.workload);
        let metrics = MetricsCollector::new(0.0);

        Ok(Self {
            scheduler,
            compute_engine,
            request_generator,
            metrics,
            time_series_data: Vec::new(),
            sample_interval: 0.1,
            next_sample_time: 0.0,
            prev_sample_time: 0.0,
            current_time: 0.0,
            iteration: 0,
            last_sent_ttft_count: 0,
            last_sent_e2e_count: 0,
            last_sent_tpot_count: 0,
            last_sent_input_count: 0,
            last_sent_output_count: 0,
        })
    }

    /// Run the simulation with progress callbacks
    pub fn run_with_callback<F>(&mut self, mut callback: F) -> Result<(), String>
    where
        F: FnMut(ProgressInfo),
    {
        let mut last_callback_time = 0.0;
        let callback_interval = 1.0; // Call callback every 1.0 seconds

        loop {
            self.iteration += 1;

            // 1. Generate new arrivals up to current_time
            while let Some(request) = self.request_generator.next_if_before(self.current_time) {
                self.scheduler.add_request(request);
                self.metrics.total_requests += 1;
            }

            // 2. Run scheduler
            let decision = self.scheduler.schedule(self.current_time);

            // 3. Calculate iteration time and utilization (build batch once, reuse for both)
            let (iteration_time, bandwidth_util, flops_util) = if decision.num_scheduled() > 0 {
                // Build batch of scheduled requests and tokens in a single pass
                let running = self.scheduler.running_mut();
                let mut batch_requests = Vec::new();
                let mut tokens_per_req = Vec::new();

                for (i, &idx) in decision.scheduled_new.iter().enumerate() {
                    if let Some(req) = running.get(idx) {
                        batch_requests.push(req);
                        tokens_per_req.push(decision.tokens_for_new[i]);
                    }
                }

                for (i, &idx) in decision.scheduled_running.iter().enumerate() {
                    if let Some(req) = running.get(idx) {
                        batch_requests.push(req);
                        tokens_per_req.push(decision.tokens_for_running[i]);
                    }
                }

                let iteration_time = self
                    .compute_engine
                    .calculate_iteration_time(&batch_requests, &tokens_per_req);

                let bytes_transferred = self
                    .compute_engine
                    .calculate_bytes_transferred(&batch_requests, &tokens_per_req);
                let bandwidth_util = self
                    .compute_engine
                    .calculate_bandwidth_utilization(bytes_transferred, iteration_time);

                let flops_util = self.compute_engine.calculate_flops_utilization(
                    &batch_requests,
                    &tokens_per_req,
                    iteration_time,
                );

                (iteration_time, bandwidth_util, flops_util)
            } else {
                (0.001, 0.0, 0.0) // Small time step when idle
            };

            // 4. Advance time
            self.current_time += iteration_time;

            // 5. Determine which requests were prefilling vs decoding BEFORE updating state
            let mut prefilling_reqs = std::collections::HashSet::new();
            for &idx in decision.scheduled_new.iter().chain(decision.scheduled_running.iter()) {
                if let Some(request) = self.scheduler.running().get(idx) {
                    if request.is_prefill() {
                        prefilling_reqs.insert(idx);
                    }
                }
            }

            // 6. Update request states
            for (i, &idx) in decision.scheduled_new.iter().enumerate() {
                if let Some(request) = self.scheduler.running_mut().get_mut(idx) {
                    request.record_generated_tokens(decision.tokens_for_new[i], self.current_time);
                }
            }
            for (i, &idx) in decision.scheduled_running.iter().enumerate() {
                if let Some(request) = self.scheduler.running_mut().get_mut(idx) {
                    request.record_generated_tokens(decision.tokens_for_running[i], self.current_time);
                }
            }

            // 7. Record iteration metrics (before moving completed requests)
            let kv_util = self.scheduler.kv_cache_manager().utilization();

            self.metrics
                .record_iteration_metrics(kv_util, flops_util, bandwidth_util);

            // 8. Record time-series data (BEFORE handling completed requests)
            if self.current_time >= self.next_sample_time {
                // Calculate prefill vs decode breakdown
                let running = self.scheduler.running();
                let mut num_prefilling = 0;
                let mut num_decoding = 0;
                let mut prefill_tokens = 0;
                let mut decode_tokens = 0;

                for req in running {
                    if req.is_prefill() {
                        num_prefilling += 1;
                    } else {
                        num_decoding += 1;
                    }
                }

                // Count tokens scheduled in this iteration
                for (i, &idx) in decision.scheduled_new.iter().enumerate() {
                    let tokens = decision.tokens_for_new[i];
                    if prefilling_reqs.contains(&idx) {
                        prefill_tokens += tokens;
                    } else {
                        decode_tokens += tokens;
                    }
                }
                for (i, &idx) in decision.scheduled_running.iter().enumerate() {
                    let tokens = decision.tokens_for_running[i];
                    if prefilling_reqs.contains(&idx) {
                        prefill_tokens += tokens;
                    } else {
                        decode_tokens += tokens;
                    }
                }

                // Calculate windowed throughput (tokens per second)
                let input_throughput = prefill_tokens as f64 / self.sample_interval;
                let output_throughput = decode_tokens as f64 / self.sample_interval;

                // Get latency mean for events since last sample
                let (ttft_mean, tpot_mean) = self.metrics.get_interval_latencies();

                self.time_series_data.push(TimeSeriesPoint {
                    time: self.current_time,
                    arrivals: self.metrics.total_requests,
                    running: self.scheduler.num_running(),
                    waiting: self.scheduler.num_waiting(),
                    kv_cache_util: kv_util,
                    num_prefilling,
                    num_decoding,
                    prefill_tokens,
                    decode_tokens,
                    input_throughput,
                    output_throughput,
                    ttft_p50: ttft_mean,
                    tpot_p50: tpot_mean,
                });
                self.prev_sample_time = self.current_time;
                self.next_sample_time = self.current_time + self.sample_interval;
            }

            // 9. Handle completed requests
            for request in decision.completed {
                // Free KV cache blocks
                self.scheduler
                    .kv_cache_manager_mut()
                    .free_blocks(&request.kv_blocks);

                self.metrics.record_request_completion(&request);

                // For closed-loop workloads, generate a new request when one completes
                self.request_generator
                    .on_request_complete(self.current_time);
            }

            // 10. Send progress update if enough time has passed
            if self.current_time - last_callback_time >= callback_interval {
                // Compute summary first (requires &mut self)
                let summary = self.metrics.compute_summary(self.current_time);

                // Then get immutable references
                let latency_samples = self.metrics.get_latency_samples();
                let input_lengths = self.metrics.get_input_lengths();
                let output_lengths = self.metrics.get_output_lengths();

                // Only send new samples since last callback (delta) - track each metric separately
                let ttft_delta = &latency_samples.0 .0[self.last_sent_ttft_count..];
                let ttft_timestamps_delta = &latency_samples.0 .1[self.last_sent_ttft_count..];
                let e2e_delta = &latency_samples.1 .0[self.last_sent_e2e_count..];
                let e2e_timestamps_delta = &latency_samples.1 .1[self.last_sent_e2e_count..];
                let tpot_delta = &latency_samples.2 .0[self.last_sent_tpot_count..];
                let tpot_timestamps_delta = &latency_samples.2 .1[self.last_sent_tpot_count..];
                let input_delta = &input_lengths[self.last_sent_input_count..];
                let output_delta = &output_lengths[self.last_sent_output_count..];

                let progress = ProgressInfo {
                    current_time: self.current_time,
                    completed_requests: self.metrics.completed_requests,
                    total_requests: self.metrics.total_requests,
                    running: self.scheduler.num_running(),
                    waiting: self.scheduler.num_waiting(),
                    kv_cache_util: kv_util,
                    time_series: Some(&self.time_series_data),
                    metrics: Some(summary),
                    latency_samples: Some((
                        (ttft_delta, ttft_timestamps_delta),
                        (e2e_delta, e2e_timestamps_delta),
                        (tpot_delta, tpot_timestamps_delta),
                    )),
                    distribution_samples: Some((input_delta, output_delta)),
                };
                callback(progress);

                // Update last sent sample counts for each metric
                self.last_sent_ttft_count = latency_samples.0 .0.len();
                self.last_sent_e2e_count = latency_samples.1 .0.len();
                self.last_sent_tpot_count = latency_samples.2 .0.len();
                self.last_sent_input_count = input_lengths.len();
                self.last_sent_output_count = output_lengths.len();
                last_callback_time = self.current_time;
            }

            // Note: Periodic logging removed for callback mode - callback replaces it

            // 11. Check termination conditions
            if self.should_terminate() {
                // Send final progress update with any remaining samples
                // Compute summary first (requires &mut self)
                let summary = self.metrics.compute_summary(self.current_time);

                // Then get immutable references
                let latency_samples = self.metrics.get_latency_samples();
                let input_lengths = self.metrics.get_input_lengths();
                let output_lengths = self.metrics.get_output_lengths();

                // Only send new samples since last callback (delta) - track each metric separately
                let ttft_delta = &latency_samples.0 .0[self.last_sent_ttft_count..];
                let ttft_timestamps_delta = &latency_samples.0 .1[self.last_sent_ttft_count..];
                let e2e_delta = &latency_samples.1 .0[self.last_sent_e2e_count..];
                let e2e_timestamps_delta = &latency_samples.1 .1[self.last_sent_e2e_count..];
                let tpot_delta = &latency_samples.2 .0[self.last_sent_tpot_count..];
                let tpot_timestamps_delta = &latency_samples.2 .1[self.last_sent_tpot_count..];
                let input_delta = &input_lengths[self.last_sent_input_count..];
                let output_delta = &output_lengths[self.last_sent_output_count..];

                let progress = ProgressInfo {
                    current_time: self.current_time,
                    completed_requests: self.metrics.completed_requests,
                    total_requests: self.metrics.total_requests,
                    running: self.scheduler.num_running(),
                    waiting: self.scheduler.num_waiting(),
                    kv_cache_util: kv_util,
                    time_series: Some(&self.time_series_data),
                    metrics: Some(summary),
                    latency_samples: Some((
                        (ttft_delta, ttft_timestamps_delta),
                        (e2e_delta, e2e_timestamps_delta),
                        (tpot_delta, tpot_timestamps_delta),
                    )),
                    distribution_samples: Some((input_delta, output_delta)),
                };
                callback(progress);
                break;
            }
        }

        Ok(())
    }

    pub fn get_metrics_summary(&mut self) -> crate::metrics::MetricsSummary {
        self.metrics.compute_summary(self.current_time)
    }

    pub fn get_time_series_data(&self) -> &[TimeSeriesPoint] {
        &self.time_series_data
    }

    pub fn get_input_lengths(&self) -> &[u32] {
        self.metrics.get_input_lengths()
    }

    pub fn get_output_lengths(&self) -> &[u32] {
        self.metrics.get_output_lengths()
    }

    pub fn get_current_time(&self) -> f64 {
        self.current_time
    }

    pub fn get_latency_samples(
        &self,
    ) -> (
        (&[f64], &[f64]), // (ttft_samples, ttft_timestamps)
        (&[f64], &[f64]), // (e2e_samples, e2e_timestamps)
        (&[f64], &[f64]), // (tpot_samples, tpot_timestamps)
    ) {
        self.metrics.get_latency_samples()
    }

    fn should_terminate(&self) -> bool {
        // Check if we've generated all requests and completed them all
        self.request_generator.is_finished()
            && self.scheduler.num_running() == 0
            && self.scheduler.num_waiting() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_minimal_test_config() -> Config {
        let mut config = Config::test_default();
        config.workload.num_requests = Some(10); // Small number for fast tests
        config.workload.arrival_rate = 10.0; // Fast arrival
        config
    }

    #[test]
    fn test_simulation_completes_all_requests() {
        let config = create_minimal_test_config();
        let mut simulator = Simulator::new(config).unwrap();

        simulator.run_with_callback(|_| {}).unwrap();

        let summary = simulator.get_metrics_summary();

        // All requests should complete
        assert_eq!(summary.completed_requests, summary.total_requests);
        assert_eq!(summary.completed_requests, 10);
    }

    #[test]
    fn test_simulation_time_progresses() {
        let config = create_minimal_test_config();
        let mut simulator = Simulator::new(config).unwrap();

        let start_time = simulator.get_current_time();
        simulator.run_with_callback(|_| {}).unwrap();
        let end_time = simulator.get_current_time();

        // Time should advance
        assert!(end_time > start_time);
    }

    #[test]
    fn test_simulation_metrics_reasonable() {
        let config = create_minimal_test_config();
        let mut simulator = Simulator::new(config).unwrap();

        simulator.run_with_callback(|_| {}).unwrap();

        let summary = simulator.get_metrics_summary();

        // Latencies should be positive and finite
        assert!(summary.ttft_mean > 0.0 && summary.ttft_mean.is_finite());
        assert!(summary.e2e_mean > 0.0 && summary.e2e_mean.is_finite());
        assert!(summary.per_token_mean > 0.0 && summary.per_token_mean.is_finite());

        // Percentiles should be ordered
        assert!(summary.ttft_min <= summary.ttft_p50);
        assert!(summary.ttft_p50 <= summary.ttft_p90);
        assert!(summary.ttft_p90 <= summary.ttft_p99);

        assert!(summary.e2e_min <= summary.e2e_p50);
        assert!(summary.e2e_p50 <= summary.e2e_p90);
        assert!(summary.e2e_p90 <= summary.e2e_p99);

        // Utilization should be between 0 and 1
        assert!(summary.avg_kv_cache_util >= 0.0 && summary.avg_kv_cache_util <= 1.0);
        assert!(summary.avg_flops_util >= 0.0 && summary.avg_flops_util <= 1.0);
        assert!(summary.avg_bandwidth_util >= 0.0 && summary.avg_bandwidth_util <= 1.0);

        // Throughput should be positive
        assert!(summary.input_tokens_per_sec > 0.0);
        assert!(summary.output_tokens_per_sec > 0.0);
        assert!(summary.requests_per_sec > 0.0);
    }

    #[test]
    fn test_simulation_no_infinite_loop() {
        let config = create_minimal_test_config();
        let mut simulator = Simulator::new(config).unwrap();

        // Run with a callback that counts iterations
        let mut iteration_count = 0;
        simulator
            .run_with_callback(|_| {
                iteration_count += 1;
            })
            .unwrap();

        // Should terminate in reasonable number of iterations
        // With 10 requests and fast arrival, should be < 1000 iterations
        assert!(iteration_count < 1000);
        assert!(iteration_count > 0);
    }

    #[test]
    fn test_simulation_with_fcfs_policy() {
        let mut config = create_minimal_test_config();
        config.scheduler.policy = "fcfs".to_string();

        let mut simulator = Simulator::new(config).unwrap();
        simulator.run_with_callback(|_| {}).unwrap();

        let summary = simulator.get_metrics_summary();
        assert_eq!(summary.completed_requests, 10);
    }

    #[test]
    fn test_simulation_with_sjf_policy() {
        let mut config = create_minimal_test_config();
        config.scheduler.policy = "sjf".to_string();

        let mut simulator = Simulator::new(config).unwrap();
        simulator.run_with_callback(|_| {}).unwrap();

        let summary = simulator.get_metrics_summary();
        assert_eq!(summary.completed_requests, 10);
    }

    #[test]
    fn test_simulation_with_priority_policy() {
        let mut config = create_minimal_test_config();
        config.scheduler.policy = "priority".to_string();

        let mut simulator = Simulator::new(config).unwrap();
        simulator.run_with_callback(|_| {}).unwrap();

        let summary = simulator.get_metrics_summary();
        assert_eq!(summary.completed_requests, 10);
    }

    #[test]
    fn test_simulation_different_policies_produce_results() {
        let mut config_fcfs = create_minimal_test_config();
        config_fcfs.scheduler.policy = "fcfs".to_string();
        config_fcfs.workload.seed = 42;

        let mut config_sjf = create_minimal_test_config();
        config_sjf.scheduler.policy = "sjf".to_string();
        config_sjf.workload.seed = 42; // Same seed for comparison

        let mut sim_fcfs = Simulator::new(config_fcfs).unwrap();
        let mut sim_sjf = Simulator::new(config_sjf).unwrap();

        sim_fcfs.run_with_callback(|_| {}).unwrap();
        sim_sjf.run_with_callback(|_| {}).unwrap();

        let summary_fcfs = sim_fcfs.get_metrics_summary();
        let summary_sjf = sim_sjf.get_metrics_summary();

        // Both should complete all requests
        assert_eq!(summary_fcfs.completed_requests, 10);
        assert_eq!(summary_sjf.completed_requests, 10);

        // Metrics should be reasonable for both (not testing exact equality,
        // as policies may produce different latencies)
        assert!(summary_fcfs.e2e_mean > 0.0);
        assert!(summary_sjf.e2e_mean > 0.0);
    }

    #[test]
    fn test_simulation_with_chunked_prefill() {
        let mut config = create_minimal_test_config();
        config.scheduler.enable_chunked_prefill = true;
        config.scheduler.long_prefill_token_threshold = 512;

        let mut simulator = Simulator::new(config).unwrap();
        simulator.run_with_callback(|_| {}).unwrap();

        let summary = simulator.get_metrics_summary();
        assert_eq!(summary.completed_requests, 10);
    }

    #[test]
    fn test_simulation_preemption_metrics() {
        let config = create_minimal_test_config();
        let mut simulator = Simulator::new(config).unwrap();

        simulator.run_with_callback(|_| {}).unwrap();

        let summary = simulator.get_metrics_summary();

        // Preemption metrics should be non-negative
        assert!(summary.total_preemptions >= 0);
        assert!(summary.preemptions_per_request_mean >= 0.0);
    }

    #[test]
    fn test_simulation_time_series_collected() {
        let config = create_minimal_test_config();
        let mut simulator = Simulator::new(config).unwrap();

        simulator.run_with_callback(|_| {}).unwrap();

        let time_series = simulator.get_time_series_data();

        // Should have collected some time series data
        assert!(!time_series.is_empty());

        // Time should be monotonically increasing
        for i in 1..time_series.len() {
            assert!(time_series[i].time >= time_series[i - 1].time);
        }
    }

    #[test]
    fn test_simulation_latency_samples_collected() {
        let config = create_minimal_test_config();
        let mut simulator = Simulator::new(config).unwrap();

        simulator.run_with_callback(|_| {}).unwrap();

        let ((ttft, ttft_ts), (e2e, e2e_ts), (tpot, tpot_ts)) = simulator.get_latency_samples();

        // Should have latency samples
        assert!(!ttft.is_empty());
        assert!(!e2e.is_empty());

        // Timestamps should match samples
        assert_eq!(ttft.len(), ttft_ts.len());
        assert_eq!(e2e.len(), e2e_ts.len());
        assert_eq!(tpot.len(), tpot_ts.len());
    }
}
