use super::summary::MetricsSummary;
use super::quantile::StreamingQuantiles;
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

    // Throughput samples (sampled periodically)
    input_tokens_per_sec_samples: Vec<f64>,
    output_tokens_per_sec_samples: Vec<f64>,
    requests_per_sec_samples: Vec<f64>,

    // Streaming quantiles for throughput
    input_tokens_per_sec_quantiles: StreamingQuantiles,
    output_tokens_per_sec_quantiles: StreamingQuantiles,
    requests_per_sec_quantiles: StreamingQuantiles,

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
            input_tokens_per_sec_samples: Vec::new(),
            output_tokens_per_sec_samples: Vec::new(),
            requests_per_sec_samples: Vec::new(),
            input_tokens_per_sec_quantiles: StreamingQuantiles::new(),
            output_tokens_per_sec_quantiles: StreamingQuantiles::new(),
            requests_per_sec_quantiles: StreamingQuantiles::new(),
            kv_cache_utilization_samples: Vec::new(),
            flops_utilization_samples: Vec::new(),
            bandwidth_utilization_samples: Vec::new(),
            total_preemptions: 0,
            preemptions_per_request: Vec::new(),
            completed_requests: 0,
            total_requests: 0,
            input_lengths: Vec::new(),
            output_lengths: Vec::new(),
        }
    }

    /// Record completion of a request
    pub fn record_request_completion(&mut self, request: &Request) {
        let completion_time = request.completion_time.unwrap_or(0.0);

        // TTFT (Time To First Token) - use completion_time as timestamp since that's when we observe it
        if let Some(ttft_time) = request.first_token_time {
            let ttft = ttft_time - request.arrival_time;
            self.ttft_samples.push(ttft);
            self.ttft_timestamps.push(completion_time);  // Use completion time, not first token time
            self.ttft_quantiles.add(ttft);
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
    pub fn get_latency_samples(&self) -> (
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

    /// Get latency mean for events in a time range
    /// Returns (ttft_mean_ms, tpot_mean_ms) for samples in the given time range
    /// Returns f64::NAN if no samples in interval (allows chart to skip intervals and draw lines)
    pub fn get_interval_latencies(&self, start_time: f64, end_time: f64) -> (f64, f64) {
        // Collect TTFT samples in interval
        let ttft_interval: Vec<f64> = self.ttft_samples.iter()
            .zip(&self.ttft_timestamps)
            .filter(|(_, &timestamp)| timestamp > start_time && timestamp <= end_time)
            .map(|(&sample, _)| sample * 1000.0) // Convert to ms
            .collect();

        // Collect TPOT samples in interval
        let tpot_interval: Vec<f64> = self.per_token_latency_samples.iter()
            .zip(&self.per_token_timestamps)
            .filter(|(_, &timestamp)| timestamp > start_time && timestamp <= end_time)
            .map(|(&sample, _)| sample * 1000.0) // Convert to ms
            .collect();

        // Calculate mean for each
        let ttft_mean = if ttft_interval.is_empty() {
            f64::NAN  // Use NaN so chart skips this point
        } else {
            ttft_interval.iter().sum::<f64>() / ttft_interval.len() as f64
        };

        let tpot_mean = if tpot_interval.is_empty() {
            f64::NAN  // Use NaN so chart skips this point
        } else {
            tpot_interval.iter().sum::<f64>() / tpot_interval.len() as f64
        };

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

    /// Record throughput metrics (sampled periodically)
    pub fn record_throughput_sample(&mut self, current_time: f64) {
        let elapsed = current_time - self.start_time;
        if elapsed > 0.0 {
            let input_tps = self.total_input_tokens as f64 / elapsed;
            let output_tps = self.total_output_tokens as f64 / elapsed;
            let req_ps = self.completed_requests as f64 / elapsed;

            self.input_tokens_per_sec_samples.push(input_tps);
            self.output_tokens_per_sec_samples.push(output_tps);
            self.requests_per_sec_samples.push(req_ps);

            self.input_tokens_per_sec_quantiles.add(input_tps);
            self.output_tokens_per_sec_quantiles.add(output_tps);
            self.requests_per_sec_quantiles.add(req_ps);
        }
    }

    /// Compute final summary statistics
    pub fn compute_summary(&self, current_time: f64) -> MetricsSummary {
        let elapsed = current_time - self.start_time;

        MetricsSummary {
            // Latency (convert to milliseconds) - use streaming quantiles for O(1) lookups
            ttft_mean: self.ttft_quantiles.mean() * 1000.0,
            ttft_p50: self.ttft_quantiles.p50() * 1000.0,
            ttft_p90: self.ttft_quantiles.p90() * 1000.0,
            ttft_p99: self.ttft_quantiles.p99() * 1000.0,

            e2e_mean: self.e2e_quantiles.mean() * 1000.0,
            e2e_p50: self.e2e_quantiles.p50() * 1000.0,
            e2e_p90: self.e2e_quantiles.p90() * 1000.0,
            e2e_p99: self.e2e_quantiles.p99() * 1000.0,

            per_token_mean: self.per_token_quantiles.mean() * 1000.0,
            per_token_p50: self.per_token_quantiles.p50() * 1000.0,
            per_token_p90: self.per_token_quantiles.p90() * 1000.0,
            per_token_p99: self.per_token_quantiles.p99() * 1000.0,

            // Throughput - use streaming quantiles for O(1) lookups
            input_tokens_per_sec: self.total_input_tokens as f64 / elapsed,
            input_tokens_per_sec_p50: self.input_tokens_per_sec_quantiles.p50(),
            input_tokens_per_sec_p90: self.input_tokens_per_sec_quantiles.p90(),
            input_tokens_per_sec_p99: self.input_tokens_per_sec_quantiles.p99(),

            output_tokens_per_sec: self.total_output_tokens as f64 / elapsed,
            output_tokens_per_sec_p50: self.output_tokens_per_sec_quantiles.p50(),
            output_tokens_per_sec_p90: self.output_tokens_per_sec_quantiles.p90(),
            output_tokens_per_sec_p99: self.output_tokens_per_sec_quantiles.p99(),

            requests_per_sec: self.completed_requests as f64 / elapsed,
            requests_per_sec_p50: self.requests_per_sec_quantiles.p50(),
            requests_per_sec_p90: self.requests_per_sec_quantiles.p90(),
            requests_per_sec_p99: self.requests_per_sec_quantiles.p99(),

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

/// Calculate percentile of a sorted array
fn percentile(samples: &[f64], p: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = samples.iter().filter(|x| !x.is_nan()).copied().collect();
    if sorted.is_empty() {
        return 0.0;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len() as f64 - 1.0) * p) as usize;
    sorted[idx]
}

/// Calculate mean of samples
fn mean(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let valid_samples: Vec<f64> = samples.iter().filter(|x| !x.is_nan() && x.is_finite()).copied().collect();
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
    fn test_percentile() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(percentile(&samples, 0.0), 1.0);
        assert_eq!(percentile(&samples, 0.5), 3.0);
        assert_eq!(percentile(&samples, 1.0), 5.0);
    }

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
}
