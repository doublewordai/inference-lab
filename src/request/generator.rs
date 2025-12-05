use super::Request;
use crate::config::WorkloadConfig;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Exp};

/// Generates requests based on workload configuration
pub struct RequestGenerator {
    workload: WorkloadConfig,
    rng: StdRng,
    next_arrival_time: f64,
    requests_generated: usize,
    next_request_id: u64,
    /// For closed-loop: track pending requests to generate when completions occur
    pending_closed_loop_requests: Vec<f64>,
}

impl RequestGenerator {
    pub fn new(workload: WorkloadConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(workload.seed);
        let is_closed_loop = workload.arrival_pattern.to_lowercase() == "closed_loop";

        // For closed-loop, initialize with N requests at time 0
        let mut pending_closed_loop_requests = Vec::new();
        if is_closed_loop {
            if let Some(num_users) = workload.num_concurrent_users {
                // Generate initial requests for all concurrent users
                for _ in 0..num_users {
                    pending_closed_loop_requests.push(0.0);
                }
            }
        }

        let next_arrival_time = if is_closed_loop && !pending_closed_loop_requests.is_empty() {
            0.0 // Start immediately with the first batch
        } else {
            Self::sample_next_arrival(
                0.0,
                &workload.arrival_pattern,
                workload.arrival_rate,
                &mut rng,
            )
        };

        Self {
            workload,
            rng,
            next_arrival_time,
            requests_generated: 0,
            next_request_id: 0,
            pending_closed_loop_requests,
        }
    }

    /// Get the next request if its arrival time is before the given time
    /// Returns None if no request is ready or all requests have been generated
    pub fn next_if_before(&mut self, current_time: f64) -> Option<Request> {
        let is_closed_loop = self.workload.arrival_pattern.to_lowercase() == "closed_loop";

        // For closed-loop, check pending requests
        if is_closed_loop {
            // Check if we've generated all requests
            if let Some(max_requests) = self.workload.num_requests {
                if self.requests_generated >= max_requests {
                    // Clear any remaining pending requests that won't be used
                    self.pending_closed_loop_requests.clear();
                    return None;
                }
            }

            // Find the earliest pending request that has arrived
            if let Some(pos) = self.pending_closed_loop_requests.iter().position(|&t| t <= current_time) {
                let arrival_time = self.pending_closed_loop_requests.remove(pos);

                // Generate request
                let request_id = format!("req-{}", self.next_request_id);
                self.next_request_id += 1;

                let num_prompt_tokens = self.workload.input_len_dist.sample(&mut self.rng);
                let max_output_tokens = self.workload.output_len_dist.sample(&mut self.rng);

                let request = Request::new(
                    request_id,
                    0, // Default priority
                    arrival_time,
                    num_prompt_tokens,
                    max_output_tokens,
                );

                self.requests_generated += 1;
                return Some(request);
            }
            return None;
        }

        // Original logic for non-closed-loop patterns
        // Check if we've generated all requests
        if let Some(max_requests) = self.workload.num_requests {
            if self.requests_generated >= max_requests {
                return None;
            }
        }

        // Check if next request has arrived
        if self.next_arrival_time > current_time {
            return None;
        }

        // Generate request
        let request_id = format!("req-{}", self.next_request_id);
        self.next_request_id += 1;

        let num_prompt_tokens = self.workload.input_len_dist.sample(&mut self.rng);
        let max_output_tokens = self.workload.output_len_dist.sample(&mut self.rng);

        let request = Request::new(
            request_id,
            0, // Default priority
            self.next_arrival_time,
            num_prompt_tokens,
            max_output_tokens,
        );

        self.requests_generated += 1;

        // Sample next arrival time
        self.next_arrival_time = Self::sample_next_arrival(
            self.next_arrival_time,
            &self.workload.arrival_pattern,
            self.workload.arrival_rate,
            &mut self.rng,
        );

        Some(request)
    }

    /// Sample the next arrival time based on the arrival pattern
    fn sample_next_arrival(
        current_time: f64,
        pattern: &str,
        rate: f64,
        rng: &mut StdRng,
    ) -> f64 {
        match pattern.to_lowercase().as_str() {
            "poisson" => {
                // Poisson process: inter-arrival times are exponentially distributed
                let exp = Exp::new(rate).unwrap();
                let inter_arrival = exp.sample(rng);
                current_time + inter_arrival
            }
            "uniform" => {
                // Uniform: constant inter-arrival time
                let inter_arrival = 1.0 / rate;
                current_time + inter_arrival
            }
            "burst" => {
                // Burst: requests arrive in bursts with gaps
                // Simple implementation: alternate between fast and slow
                if rng.gen_bool(0.2) {
                    // 20% chance of burst
                    current_time + rng.gen_range(0.001..0.01)
                } else {
                    current_time + rng.gen_range(0.5..2.0)
                }
            }
            "fixed_rate" => {
                // Fixed rate: exact inter-arrival time
                current_time + 1.0 / rate
            }
            "batched" => {
                // Batched: all requests arrive at time 0
                0.0
            }
            _ => {
                // Default to Poisson
                let exp = Exp::new(rate).unwrap();
                current_time + exp.sample(rng)
            }
        }
    }

    /// Check if all requests have been generated
    pub fn is_finished(&self) -> bool {
        if let Some(max_requests) = self.workload.num_requests {
            let is_closed_loop = self.workload.arrival_pattern.to_lowercase() == "closed_loop";
            if is_closed_loop {
                // For closed-loop, we're finished when we've generated max_requests
                // AND have no pending requests
                self.requests_generated >= max_requests && self.pending_closed_loop_requests.is_empty()
            } else {
                self.requests_generated >= max_requests
            }
        } else {
            false
        }
    }

    /// Called when a request completes (for closed-loop pattern)
    /// Generates a new request for that "user slot" at the completion time
    pub fn on_request_complete(&mut self, completion_time: f64) {
        let is_closed_loop = self.workload.arrival_pattern.to_lowercase() == "closed_loop";
        if !is_closed_loop {
            return; // Only applicable to closed-loop
        }

        // Check if we should generate more requests
        if let Some(max_requests) = self.workload.num_requests {
            if self.requests_generated >= max_requests {
                return; // Already generated all requested requests
            }
        }

        // Add a new pending request at the completion time
        self.pending_closed_loop_requests.push(completion_time);
    }

    /// Get number of requests generated so far
    pub fn num_generated(&self) -> usize {
        self.requests_generated
    }

    /// Peek at the next arrival time without generating the request
    pub fn peek_next_arrival(&self) -> f64 {
        self.next_arrival_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LengthDistribution;

    fn create_test_workload(pattern: &str, rate: f64, num_requests: usize) -> WorkloadConfig {
        WorkloadConfig {
            arrival_pattern: pattern.to_string(),
            arrival_rate: rate,
            num_concurrent_users: None,
            input_len_dist: LengthDistribution::Fixed { value: 100 },
            output_len_dist: LengthDistribution::Fixed { value: 50 },
            num_requests: Some(num_requests),
            duration_secs: None,
            seed: 42,
        }
    }

    #[test]
    fn test_generator_creation() {
        let workload = create_test_workload("poisson", 1.0, 10);
        let generator = RequestGenerator::new(workload);

        assert_eq!(generator.num_generated(), 0);
        assert!(!generator.is_finished());
    }

    #[test]
    fn test_generate_requests() {
        let workload = create_test_workload("poisson", 10.0, 5);
        let mut generator = RequestGenerator::new(workload);

        let mut requests = Vec::new();
        let mut current_time = 0.0;

        while !generator.is_finished() {
            // Advance time significantly to ensure all requests arrive
            current_time += 10.0;

            while let Some(req) = generator.next_if_before(current_time) {
                requests.push(req);
            }
        }

        assert_eq!(requests.len(), 5);
        assert!(generator.is_finished());
    }

    #[test]
    fn test_arrival_ordering() {
        let workload = create_test_workload("poisson", 5.0, 10);
        let mut generator = RequestGenerator::new(workload);

        let mut requests = Vec::new();
        let mut current_time = 0.0;

        while !generator.is_finished() {
            current_time += 10.0;
            while let Some(req) = generator.next_if_before(current_time) {
                requests.push(req);
            }
        }

        // Check that arrival times are monotonically increasing
        for i in 1..requests.len() {
            assert!(requests[i].arrival_time >= requests[i - 1].arrival_time);
        }
    }

    #[test]
    fn test_fixed_rate_arrival() {
        let workload = create_test_workload("fixed_rate", 2.0, 4);
        let mut generator = RequestGenerator::new(workload);

        let mut requests = Vec::new();
        let mut current_time = 0.0;

        while !generator.is_finished() {
            current_time += 10.0;
            while let Some(req) = generator.next_if_before(current_time) {
                requests.push(req);
            }
        }

        assert_eq!(requests.len(), 4);

        // Check that inter-arrival times are approximately 1/rate = 0.5 seconds
        for i in 1..requests.len() {
            let inter_arrival = requests[i].arrival_time - requests[i - 1].arrival_time;
            assert!((inter_arrival - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_request_properties() {
        let workload = create_test_workload("poisson", 1.0, 1);
        let mut generator = RequestGenerator::new(workload);

        let req = generator.next_if_before(10.0).unwrap();

        assert_eq!(req.num_prompt_tokens, 100);
        assert_eq!(req.max_output_tokens, 50);
        assert_eq!(req.priority, 0);
        assert!(req.request_id.starts_with("req-"));
    }

    #[test]
    fn test_peek_next_arrival() {
        let workload = create_test_workload("poisson", 1.0, 10);
        let mut generator = RequestGenerator::new(workload);

        let next_arrival = generator.peek_next_arrival();
        assert!(next_arrival > 0.0);

        // Generate the request
        let req = generator.next_if_before(next_arrival + 1.0).unwrap();

        // Check that arrival time matches what we peeked
        assert_eq!(req.arrival_time, next_arrival);
    }
}
