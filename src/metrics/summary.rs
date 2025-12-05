/// Summary of all metrics from the simulation
#[derive(Debug)]
pub struct MetricsSummary {
    // Latency metrics (in milliseconds)
    pub ttft_min: f64,
    pub ttft_mean: f64,
    pub ttft_p50: f64,
    pub ttft_p90: f64,
    pub ttft_p99: f64,

    pub e2e_min: f64,
    pub e2e_mean: f64,
    pub e2e_p50: f64,
    pub e2e_p90: f64,
    pub e2e_p99: f64,

    pub per_token_min: f64,
    pub per_token_mean: f64,
    pub per_token_p50: f64,
    pub per_token_p90: f64,
    pub per_token_p99: f64,

    // Throughput metrics
    pub input_tokens_per_sec: f64,
    pub output_tokens_per_sec: f64,
    pub requests_per_sec: f64,

    // Resource utilization (average over all samples)
    pub avg_kv_cache_util: f64,
    pub avg_flops_util: f64,
    pub avg_bandwidth_util: f64,

    // Preemption metrics
    pub total_preemptions: u64,
    pub preemptions_per_request_mean: f64,

    // Request counts
    pub completed_requests: u64,
    pub total_requests: u64,
}

impl MetricsSummary {
    pub fn print(&self) {
        println!("\n=== Final Metrics ===\n");

        println!("Latency Metrics (ms):");
        println!(
            "  TTFT: min={:.2}, mean={:.2}, p50={:.2}, p90={:.2}, p99={:.2}",
            self.ttft_min, self.ttft_mean, self.ttft_p50, self.ttft_p90, self.ttft_p99
        );
        println!(
            "  E2E:  min={:.2}, mean={:.2}, p50={:.2}, p90={:.2}, p99={:.2}",
            self.e2e_min, self.e2e_mean, self.e2e_p50, self.e2e_p90, self.e2e_p99
        );
        println!(
            "  Per-token: min={:.2}, mean={:.2}, p50={:.2}, p90={:.2}, p99={:.2}",
            self.per_token_min,
            self.per_token_mean,
            self.per_token_p50,
            self.per_token_p90,
            self.per_token_p99
        );

        println!("\nThroughput Metrics:");
        println!("  Input tokens/sec:  {:.2}", self.input_tokens_per_sec);
        println!("  Output tokens/sec: {:.2}", self.output_tokens_per_sec);
        println!("  Requests/sec:      {:.2}", self.requests_per_sec);

        println!("\nResource Utilization:");
        println!(
            "  Avg KV cache:      {:.1}%",
            self.avg_kv_cache_util * 100.0
        );
        println!("  Avg FLOPS:         {:.1}%", self.avg_flops_util * 100.0);
        println!(
            "  Avg bandwidth:     {:.1}%",
            self.avg_bandwidth_util * 100.0
        );

        println!("\nPreemption Metrics:");
        println!("  Total preemptions: {}", self.total_preemptions);
        println!(
            "  Avg per request:   {:.2}",
            self.preemptions_per_request_mean
        );

        println!("\nRequest Statistics:");
        println!(
            "  Completed: {}/{}",
            self.completed_requests, self.total_requests
        );
    }
}
