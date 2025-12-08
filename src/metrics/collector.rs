use super::quantile::StreamingQuantiles;
use super::summary::MetricsSummary;
use crate::request::Request;

pub struct MetricsCollector {
    // Latency metrics (in seconds) - keep recent samples with timestamps for windowing
    ttft_samples: Vec<f64>,
    ttft_timestamps: Vec<f64>,
    e2e_latency_samples: Vec<f64>,
    e2e_timestamps: Vec<f64>,
    per_token_latency_samples: Vec<f64>,
    per_token_timestamps: Vec<f64>,

    // Streaming quantiles for efficient percentile computation
    ttft_quantiles: StreamingQuantiles,
    e2e_quantiles: StreamingQuantiles,
    per_token_quantiles: StreamingQuantiles,

    // Throughput metrics
    total_input_tokens: u64,
    total_output_tokens: u64,
    start_time: f64,

    // Throughput tracking
    // (computed from totals at the end)

    // Resource utilization (sampled periodically)
    kv_cache_utilization_samples: Vec<f64>,
    flops_utilization_samples: Vec<f64>,
    bandwidth_utilization_samples: Vec<f64>,

    // Preemption metrics
    total_preemptions: u64,
    preemptions_per_request: Vec<u32>,

    // Request tracking
    pub completed_requests: u64,
    pub total_requests: u64,

    // Length distributions
    input_lengths: Vec<u32>,
    output_lengths: Vec<u32>,

    // Interval accumulators for O(1) mean computation
    current_interval_ttft_sum: f64,
    current_interval_ttft_count: u32,
    current_interval_tpot_sum: f64,
    current_interval_tpot_count: u32,
}

impl MetricsCollector {
    pub fn new(start_time: f64) -> Self {
        Self {
            ttft_samples: Vec::new(),
            ttft_timestamps: Vec::new(),
            e2e_latency_samples: Vec::new(),
            e2e_timestamps: Vec::new(),
            per_token_latency_samples: Vec::new(),
            per_token_timestamps: Vec::new(),
            ttft_quantiles: StreamingQuantiles::new(),
            e2e_quantiles: StreamingQuantiles::new(),
            per_token_quantiles: StreamingQuantiles::new(),
            total_input_tokens: 0,
            total_output_tokens: 0,
            start_time,
            kv_cache_utilization_samples: Vec::new(),
            flops_utilization_samples: Vec::new(),
            bandwidth_utilization_samples: Vec::new(),
            total_preemptions: 0,
            preemptions_per_request: Vec::new(),
            completed_requests: 0,
            total_requests: 0,
            input_lengths: Vec::new(),
            output_lengths: Vec::new(),
            current_interval_ttft_sum: 0.0,
            current_interval_ttft_count: 0,
            current_interval_tpot_sum: 0.0,
            current_interval_tpot_count: 0,
        }
    }

    /// Record completion of a request
    pub fn record_request_completion(&mut self, request: &Request) {
        let completion_time = request.completion_time.unwrap_or(0.0);

        // TTFT (Time To First Token) - use completion_time as timestamp since that's when we observe it
        if let Some(ttft_time) = request.first_token_time {
            let ttft = ttft_time - request.arrival_time;
            self.ttft_samples.push(ttft);
            self.ttft_timestamps.push(completion_time); // Use completion time, not first token time
            self.ttft_quantiles.add(ttft);
            // Accumulate for interval mean
            self.current_interval_ttft_sum += ttft;
            self.current_interval_ttft_count += 1;
        }

        // E2E latency (excluding time spent preempted)
        if let Some(completion_time) = request.completion_time {
            let e2e = completion_time - request.arrival_time - request.preempted_time;
            self.e2e_latency_samples.push(e2e);
            self.e2e_timestamps.push(completion_time);
            self.e2e_quantiles.add(e2e);
        }

        // Per-token latency (for decode phase)
        for i in 1..request.token_generation_times.len() {
            let prev_time = request.token_generation_times[i - 1];
            let curr_time = request.token_generation_times[i];
            let tpot = curr_time - prev_time;
            self.per_token_latency_samples.push(tpot);
            self.per_token_timestamps.push(curr_time);
            self.per_token_quantiles.add(tpot);
            // Accumulate for interval mean
            self.current_interval_tpot_sum += tpot;
            self.current_interval_tpot_count += 1;
        }

        // Throughput counters
        self.total_input_tokens += request.num_prompt_tokens as u64;
        self.total_output_tokens += request.num_output_tokens as u64;

        // Preemption tracking
        self.preemptions_per_request.push(request.num_preemptions);
        self.total_preemptions += request.num_preemptions as u64;

        // Length distributions
        self.input_lengths.push(request.num_prompt_tokens);
        self.output_lengths.push(request.num_output_tokens);

        self.completed_requests += 1;
    }

    /// Get input length distribution
    pub fn get_input_lengths(&self) -> &[u32] {
        &self.input_lengths
    }

    /// Get output length distribution
    pub fn get_output_lengths(&self) -> &[u32] {
        &self.output_lengths
    }

    /// Get latency samples with timestamps
    pub fn get_latency_samples(
        &self,
    ) -> (
        (&[f64], &[f64]), // (ttft_samples, ttft_timestamps)
        (&[f64], &[f64]), // (e2e_samples, e2e_timestamps)
        (&[f64], &[f64]), // (tpot_samples, tpot_timestamps)
    ) {
        (
            (&self.ttft_samples, &self.ttft_timestamps),
            (&self.e2e_latency_samples, &self.e2e_timestamps),
            (&self.per_token_latency_samples, &self.per_token_timestamps),
        )
    }

    /// Get latency mean for the current interval and reset accumulators
    /// Returns (ttft_mean_ms, tpot_mean_ms) for samples since last call
    /// Returns f64::NAN if no samples in interval (allows chart to skip intervals and draw lines)
    pub fn get_interval_latencies(&mut self) -> (f64, f64) {
        // Calculate mean from accumulators
        let ttft_mean = if self.current_interval_ttft_count > 0 {
            (self.current_interval_ttft_sum / self.current_interval_ttft_count as f64) * 1000.0
        } else {
            f64::NAN // Use NaN so chart skips this point
        };

        let tpot_mean = if self.current_interval_tpot_count > 0 {
            (self.current_interval_tpot_sum / self.current_interval_tpot_count as f64) * 1000.0
        } else {
            f64::NAN // Use NaN so chart skips this point
        };

        // Reset accumulators for next interval
        self.current_interval_ttft_sum = 0.0;
        self.current_interval_ttft_count = 0;
        self.current_interval_tpot_sum = 0.0;
        self.current_interval_tpot_count = 0;

        (ttft_mean, tpot_mean)
    }

    /// Record iteration metrics (utilization)
    pub fn record_iteration_metrics(
        &mut self,
        kv_cache_util: f64,
        flops_util: f64,
        bandwidth_util: f64,
    ) {
        self.kv_cache_utilization_samples.push(kv_cache_util);
        self.flops_utilization_samples.push(flops_util);
        self.bandwidth_utilization_samples.push(bandwidth_util);
    }

    /// Record throughput metrics (no longer needed - computed from totals)
    pub fn record_throughput_sample(&mut self, _current_time: f64) {
        // Throughput percentiles removed - they were measuring cumulative throughput over time
        // which is not meaningful. We now only report mean throughput.
    }

    /// Compute final summary statistics
    pub fn compute_summary(&self, current_time: f64) -> MetricsSummary {
        let elapsed = current_time - self.start_time;

        MetricsSummary {
            // Latency (convert to milliseconds) - use streaming quantiles for O(1) lookups
            ttft_min: self.ttft_quantiles.min() * 1000.0,
            ttft_mean: self.ttft_quantiles.mean() * 1000.0,
            ttft_p50: self.ttft_quantiles.p50() * 1000.0,
            ttft_p90: self.ttft_quantiles.p90() * 1000.0,
            ttft_p99: self.ttft_quantiles.p99() * 1000.0,

            e2e_min: self.e2e_quantiles.min() * 1000.0,
            e2e_mean: self.e2e_quantiles.mean() * 1000.0,
            e2e_p50: self.e2e_quantiles.p50() * 1000.0,
            e2e_p90: self.e2e_quantiles.p90() * 1000.0,
            e2e_p99: self.e2e_quantiles.p99() * 1000.0,

            per_token_min: self.per_token_quantiles.min() * 1000.0,
            per_token_mean: self.per_token_quantiles.mean() * 1000.0,
            per_token_p50: self.per_token_quantiles.p50() * 1000.0,
            per_token_p90: self.per_token_quantiles.p90() * 1000.0,
            per_token_p99: self.per_token_quantiles.p99() * 1000.0,

            // Throughput
            input_tokens_per_sec: self.total_input_tokens as f64 / elapsed,
            output_tokens_per_sec: self.total_output_tokens as f64 / elapsed,
            requests_per_sec: self.completed_requests as f64 / elapsed,

            // Utilization (average over all samples)
            avg_kv_cache_util: mean(&self.kv_cache_utilization_samples),
            avg_flops_util: mean(&self.flops_utilization_samples),
            avg_bandwidth_util: mean(&self.bandwidth_utilization_samples),

            // Preemption
            total_preemptions: self.total_preemptions,
            preemptions_per_request_mean: mean_u32(&self.preemptions_per_request),

            // Counts
            completed_requests: self.completed_requests,
            total_requests: self.total_requests,
        }
    }
}

/// Calculate mean of samples
fn mean(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let valid_samples: Vec<f64> = samples
        .iter()
        .filter(|x| !x.is_nan() && x.is_finite())
        .copied()
        .collect();
    if valid_samples.is_empty() {
        return 0.0;
    }
    valid_samples.iter().sum::<f64>() / valid_samples.len() as f64
}

/// Calculate mean of u32 samples
fn mean_u32(samples: &[u32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().map(|&x| x as f64).sum::<f64>() / samples.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean(&samples), 3.0);

        let empty: Vec<f64> = vec![];
        assert_eq!(mean(&empty), 0.0);
    }

    #[test]
    fn test_mean_u32() {
        let samples = vec![1, 2, 3, 4, 5];
        assert_eq!(mean_u32(&samples), 3.0);
    }

    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new(0.0);

        collector.total_requests = 10;

        // Create a test request
        let mut req = Request::new("req-1".to_string(), 0, 1.0, 100, 50);
        req.first_token_time = Some(2.0);
        req.completion_time = Some(5.0);
        req.num_output_tokens = 50;
        req.num_preemptions = 2;
        req.token_generation_times = vec![2.0, 2.1, 2.2];

        collector.record_request_completion(&req);

        assert_eq!(collector.completed_requests, 1);
        assert_eq!(collector.ttft_samples.len(), 1);
        assert_eq!(collector.e2e_latency_samples.len(), 1);
        assert_eq!(collector.per_token_latency_samples.len(), 2);

        // TTFT should be 2.0 - 1.0 = 1.0
        assert_eq!(collector.ttft_samples[0], 1.0);

        // E2E should be 5.0 - 1.0 = 4.0
        assert_eq!(collector.e2e_latency_samples[0], 4.0);

        let summary = collector.compute_summary(10.0);
        assert_eq!(summary.completed_requests, 1);
        assert_eq!(summary.total_requests, 10);
    }

    #[test]
    fn test_ttft_calculation() {
        let mut collector = MetricsCollector::new(0.0);

        // Request: arrival=1.0, first_token=3.5
        let mut req = Request::new("req-1".to_string(), 0, 1.0, 100, 20);
        req.first_token_time = Some(3.5);
        req.completion_time = Some(5.0);

        collector.record_request_completion(&req);

        // TTFT = first_token_time - arrival_time = 3.5 - 1.0 = 2.5
        assert_eq!(collector.ttft_samples[0], 2.5);
        assert_eq!(collector.ttft_timestamps[0], 5.0); // timestamp is completion_time
    }

    #[test]
    fn test_e2e_without_preemption() {
        let mut collector = MetricsCollector::new(0.0);

        // Request with no preemption: arrival=1.0, completion=5.0, preempted_time=0.0
        let mut req = Request::new("req-1".to_string(), 0, 1.0, 100, 20);
        req.completion_time = Some(5.0);
        req.preempted_time = 0.0;

        collector.record_request_completion(&req);

        // E2E = completion - arrival - preempted_time = 5.0 - 1.0 - 0.0 = 4.0
        assert_eq!(collector.e2e_latency_samples[0], 4.0);
    }

    #[test]
    fn test_e2e_with_preemption() {
        let mut collector = MetricsCollector::new(0.0);

        // Request with preemption: arrival=1.0, completion=10.0, preempted_time=3.0
        let mut req = Request::new("req-1".to_string(), 0, 1.0, 100, 20);
        req.completion_time = Some(10.0);
        req.preempted_time = 3.0; // spent 3 seconds preempted
        req.num_preemptions = 2;

        collector.record_request_completion(&req);

        // E2E = completion - arrival - preempted_time = 10.0 - 1.0 - 3.0 = 6.0
        assert_eq!(collector.e2e_latency_samples[0], 6.0);
        assert_eq!(collector.total_preemptions, 2);
    }

    #[test]
    fn test_e2e_with_multiple_preemptions() {
        let mut collector = MetricsCollector::new(0.0);

        // Request: arrival=0.0, completion=20.0, preempted_time=7.5
        let mut req = Request::new("req-1".to_string(), 0, 0.0, 100, 20);
        req.completion_time = Some(20.0);
        req.preempted_time = 7.5;
        req.num_preemptions = 5;

        collector.record_request_completion(&req);

        // E2E = 20.0 - 0.0 - 7.5 = 12.5 (actual processing time)
        assert_eq!(collector.e2e_latency_samples[0], 12.5);
        assert_eq!(collector.total_preemptions, 5);
        assert_eq!(collector.preemptions_per_request[0], 5);
    }

    #[test]
    fn test_tpot_calculation() {
        let mut collector = MetricsCollector::new(0.0);

        // Request with token generation times
        let mut req = Request::new("req-1".to_string(), 0, 1.0, 100, 20);
        req.completion_time = Some(5.0);
        // Times: 2.0, 2.1, 2.25, 2.5
        req.token_generation_times = vec![2.0, 2.1, 2.25, 2.5];

        collector.record_request_completion(&req);

        // TPOT samples: [2.1-2.0, 2.25-2.1, 2.5-2.25] = [0.1, 0.15, 0.25]
        assert_eq!(collector.per_token_latency_samples.len(), 3);
        assert!((collector.per_token_latency_samples[0] - 0.1).abs() < 1e-10);
        assert!((collector.per_token_latency_samples[1] - 0.15).abs() < 1e-10);
        assert_eq!(collector.per_token_latency_samples[2], 0.25);

        // Timestamps should be the current token generation time
        assert_eq!(collector.per_token_timestamps[0], 2.1);
        assert_eq!(collector.per_token_timestamps[1], 2.25);
        assert_eq!(collector.per_token_timestamps[2], 2.5);
    }

    #[test]
    fn test_ttft_not_affected_by_preemption() {
        let mut collector = MetricsCollector::new(0.0);

        // TTFT should NOT subtract preemption time
        // arrival=1.0, first_token=5.0, preempted_time=2.0
        let mut req = Request::new("req-1".to_string(), 0, 1.0, 100, 20);
        req.first_token_time = Some(5.0);
        req.completion_time = Some(10.0);
        req.preempted_time = 2.0;

        collector.record_request_completion(&req);

        // TTFT = first_token_time - arrival_time = 5.0 - 1.0 = 4.0
        // (preemption time NOT subtracted)
        assert_eq!(collector.ttft_samples[0], 4.0);
    }

    #[test]
    fn test_multiple_requests_aggregation() {
        let mut collector = MetricsCollector::new(0.0);

        // Request 1: TTFT=1.0, E2E=3.0
        let mut req1 = Request::new("req-1".to_string(), 0, 1.0, 100, 10);
        req1.first_token_time = Some(2.0);
        req1.completion_time = Some(4.0);
        req1.preempted_time = 0.0;

        // Request 2: TTFT=2.0, E2E=5.0 (with preemption)
        let mut req2 = Request::new("req-2".to_string(), 0, 2.0, 100, 10);
        req2.first_token_time = Some(4.0);
        req2.completion_time = Some(10.0);
        req2.preempted_time = 1.0; // 8.0 - 1.0 = 7.0 actual, then - 2.0 arrival = 5.0

        collector.record_request_completion(&req1);
        collector.record_request_completion(&req2);

        assert_eq!(collector.ttft_samples.len(), 2);
        assert_eq!(collector.e2e_latency_samples.len(), 2);

        // Check TTFT values
        assert_eq!(collector.ttft_samples[0], 1.0);
        assert_eq!(collector.ttft_samples[1], 2.0);

        // Check E2E values
        assert_eq!(collector.e2e_latency_samples[0], 3.0); // 4.0 - 1.0 - 0.0
        assert_eq!(collector.e2e_latency_samples[1], 7.0); // 10.0 - 2.0 - 1.0
    }

    #[test]
    fn test_throughput_calculation() {
        let mut collector = MetricsCollector::new(0.0);

        // Request 1: 100 input, 50 output tokens
        let mut req1 = Request::new("req-1".to_string(), 0, 1.0, 100, 50);
        req1.completion_time = Some(5.0);
        req1.num_output_tokens = 50;

        // Request 2: 200 input, 30 output tokens
        let mut req2 = Request::new("req-2".to_string(), 0, 2.0, 200, 30);
        req2.completion_time = Some(6.0);
        req2.num_output_tokens = 30;

        collector.record_request_completion(&req1);
        collector.record_request_completion(&req2);

        // Total: 300 input, 80 output, 2 requests over 10 seconds
        let summary = collector.compute_summary(10.0);

        assert_eq!(summary.input_tokens_per_sec, 30.0); // 300 / 10
        assert_eq!(summary.output_tokens_per_sec, 8.0); // 80 / 10
        assert_eq!(summary.requests_per_sec, 0.2); // 2 / 10
    }

    #[test]
    fn test_metrics_summary_conversion_to_milliseconds() {
        let mut collector = MetricsCollector::new(0.0);

        // Request with known latencies in seconds
        let mut req = Request::new("req-1".to_string(), 0, 1.0, 100, 10);
        req.first_token_time = Some(1.5); // TTFT = 0.5 seconds
        req.completion_time = Some(3.0); // E2E = 2.0 seconds
        req.preempted_time = 0.0;
        req.token_generation_times = vec![1.5, 1.6]; // TPOT = 0.1 seconds

        collector.record_request_completion(&req);

        let summary = collector.compute_summary(10.0);

        // All metrics should be converted to milliseconds (use approximate comparison)
        assert!((summary.ttft_mean - 500.0).abs() < 0.001); // 0.5 * 1000
        assert!((summary.e2e_mean - 2000.0).abs() < 0.001); // 2.0 * 1000
        assert!((summary.per_token_mean - 100.0).abs() < 0.001); // 0.1 * 1000
    }
}
