use super::Request;
use crate::config::WorkloadConfig;
use crate::dataset::{BatchTokenizerFn, DatasetEntry, UnparsedEntry};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Exp};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread;

/// Generates requests based on workload configuration
pub struct RequestGenerator {
    workload: WorkloadConfig,
    rng: StdRng,
    next_arrival_time: f64,
    requests_generated: usize,
    next_request_id: u64,
    /// For closed-loop: track pending requests to generate when completions occur
    pending_closed_loop_requests: Vec<f64>,
    /// Dataset receiver (if using dataset mode) - receives pre-loaded entries from background thread
    dataset_receiver: Option<Receiver<Option<DatasetEntry>>>,
    /// Track if dataset has been exhausted (received None from channel)
    dataset_exhausted: bool,
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
                pending_closed_loop_requests = vec![0.0; num_users]
            }
        };

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
            dataset_receiver: None,
            dataset_exhausted: false,
        }
    }

    /// Create a new generator from a dataset iterator
    /// Spawns a background thread to read, parse, and batch-tokenize entries in parallel
    /// Buffer size controls memory usage (entries buffered ahead of simulation)
    pub fn from_dataset<I>(
        workload: WorkloadConfig,
        dataset_iterator: I,
        _total_entries: Option<usize>,
        tokenizer: BatchTokenizerFn,
    ) -> Self
    where
        I: Iterator<Item = Result<Option<UnparsedEntry>, Box<dyn std::error::Error>>>
            + Send
            + 'static,
    {
        let rng = StdRng::seed_from_u64(workload.seed);
        let is_closed_loop = workload.arrival_pattern.to_lowercase() == "closed_loop";

        // For closed-loop, initialize with N requests at time 0
        let mut pending_closed_loop_requests = Vec::new();
        if is_closed_loop {
            if let Some(num_users) = workload.num_concurrent_users {
                pending_closed_loop_requests = vec![0.0; num_users]
            }
        };

        let next_arrival_time = if is_closed_loop && !pending_closed_loop_requests.is_empty() {
            0.0
        } else {
            0.0 // First request arrives at t=0
        };

        // Spawn background thread to load and batch-tokenize entries
        // Buffer size: 5000 entries (~10-50MB depending on token counts)
        let (sender, receiver) = sync_channel::<Option<DatasetEntry>>(5000);

        thread::spawn(move || {
            let batch_size: usize = std::env::var("TOKENIZER_BATCH_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(32); // Default: 32 (optimal for latency/throughput balance)
            let mut batch = Vec::with_capacity(batch_size);

            for result in dataset_iterator {
                match result {
                    Ok(Some(unparsed)) => {
                        batch.push(unparsed);

                        // Process batch when full
                        if batch.len() >= batch_size {
                            if let Err(_) =
                                Self::tokenize_and_send_batch(&mut batch, &tokenizer, &sender)
                            {
                                // Receiver dropped, simulation ended early
                                break;
                            }
                        }
                    }
                    Ok(None) => {
                        // End of dataset - flush remaining batch and send completion signal
                        if !batch.is_empty() {
                            let _ = Self::tokenize_and_send_batch(&mut batch, &tokenizer, &sender);
                        }
                        let _ = sender.send(None);
                        break;
                    }
                    Err(e) => {
                        eprintln!("Error loading dataset entry: {}", e);
                        break;
                    }
                }
            }
        });

        Self {
            workload,
            rng,
            next_arrival_time,
            requests_generated: 0,
            next_request_id: 0,
            pending_closed_loop_requests,
            dataset_receiver: Some(receiver),
            dataset_exhausted: false,
        }
    }

    /// Check if using dataset mode
    pub fn is_dataset_mode(&self) -> bool {
        self.dataset_receiver.is_some()
    }

    /// Batch tokenize and send entries to the channel
    /// Returns Err if the receiver dropped (simulation ended)
    fn tokenize_and_send_batch(
        batch: &mut Vec<UnparsedEntry>,
        tokenizer: &BatchTokenizerFn,
        sender: &std::sync::mpsc::SyncSender<Option<DatasetEntry>>,
    ) -> Result<(), ()> {
        if batch.is_empty() {
            return Ok(());
        }

        // Collect all message arrays to tokenize together
        let message_arrays: Vec<&[_]> = batch.iter().map(|e| e.messages.as_slice()).collect();

        // Batch tokenize all entries at once (much faster!)
        let all_tokens = match tokenizer(&message_arrays) {
            Ok(tokens) => tokens,
            Err(e) => {
                eprintln!("Batch tokenization failed: {}", e);
                return Err(());
            }
        };

        // Send tokenized entries
        for (unparsed, prompt_tokens) in batch.drain(..).zip(all_tokens.into_iter()) {
            let entry = DatasetEntry {
                request_id: unparsed.request_id,
                prompt_tokens,
                max_output_tokens: unparsed.max_output_tokens,
            };

            // Send to channel - returns Err if receiver dropped
            if sender.send(Some(entry)).is_err() {
                return Err(());
            }
        }
        Ok(())
    }

    /// Get the next scheduled arrival time
    pub fn peek_next_arrival_time(&self) -> f64 {
        self.next_arrival_time
    }

    /// Compute block hashes from token IDs (for real prefix caching)
    /// Uses incremental hashing: hash of block i includes all tokens up to block i
    fn compute_block_hashes(tokens: &[u32], block_size: usize) -> Vec<u64> {
        let num_blocks = tokens.len().div_ceil(block_size);
        let mut hashes = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let end = ((block_idx + 1) * block_size).min(tokens.len());
            let block_tokens = &tokens[..end]; // All tokens up to this block

            // Hash all tokens cumulatively
            let mut hasher = DefaultHasher::new();
            block_tokens.hash(&mut hasher);
            hashes.push(hasher.finish());
        }

        hashes
    }

    /// Get the next request if its arrival time is before the given time
    /// Returns None if no request is ready or all requests have been generated
    pub fn next_if_before(&mut self, current_time: f64) -> Option<Request> {
        // Dataset mode
        if self.is_dataset_mode() {
            return self.next_from_dataset(current_time);
        }

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
            if let Some(pos) = self
                .pending_closed_loop_requests
                .iter()
                .position(|&t| t <= current_time)
            {
                let arrival_time = self.pending_closed_loop_requests.remove(pos);

                // Generate request
                let request_id = format!("req-{}", self.next_request_id);
                self.next_request_id += 1;

                let num_prompt_tokens = self.workload.input_len_dist.sample(&mut self.rng);
                let max_output_tokens = self.workload.output_len_dist.sample(&mut self.rng);

                let mut request = Request::new(
                    request_id,
                    0, // Default priority
                    arrival_time,
                    num_prompt_tokens,
                    max_output_tokens,
                );

                // Generate block hashes for the prompt
                // For now, sample from a small range to get realistic cache hit rates
                // First few blocks more likely to be shared (simulating common system prompts)
                let num_blocks = num_prompt_tokens.div_ceil(16) as usize; // Assume 16-token blocks
                request.prompt_block_hashes = (0..num_blocks)
                    .map(|_| self.rng.gen_range(0..u64::MAX))
                    .collect();

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

        let mut request = Request::new(
            request_id,
            0, // Default priority
            self.next_arrival_time,
            num_prompt_tokens,
            max_output_tokens,
        );

        // Generate block hashes for the prompt
        // For now, sample from a small range to get realistic cache hit rates
        // First few blocks more likely to be shared (simulating common system prompts)
        let num_blocks = num_prompt_tokens.div_ceil(16) as usize; // Assume 16-token blocks
        request.prompt_block_hashes = (0..num_blocks)
            .map(|_| self.rng.gen_range(0..u64::MAX))
            .collect();

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
    fn sample_next_arrival(current_time: f64, pattern: &str, rate: f64, rng: &mut StdRng) -> f64 {
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

    /// Get next request from dataset (receives from background thread)
    fn next_from_dataset(&mut self, current_time: f64) -> Option<Request> {
        // If already exhausted, no more requests
        if self.dataset_exhausted {
            return None;
        }

        let is_closed_loop = self.workload.arrival_pattern.to_lowercase() == "closed_loop";

        // For closed-loop, handle pending requests
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
            if let Some(pos) = self
                .pending_closed_loop_requests
                .iter()
                .position(|&t| t <= current_time)
            {
                let arrival_time = self.pending_closed_loop_requests.remove(pos);

                // Receive from channel
                let entry = match self.dataset_receiver.as_ref()?.recv() {
                    Ok(Some(e)) => e,
                    Ok(None) => {
                        // End of dataset signaled
                        self.dataset_exhausted = true;
                        return None;
                    }
                    Err(_) => {
                        // Channel error (sender dropped)
                        self.dataset_exhausted = true;
                        return None;
                    }
                };

                // Sample actual output length from distribution
                let sampled_output_len = self.workload.output_len_dist.sample(&mut self.rng);

                // If max_output_tokens is specified in the dataset, cap at that; otherwise use sampled value
                let max_output_tokens = entry.max_output_tokens.unwrap_or(16384);
                let target_output_tokens = sampled_output_len.min(max_output_tokens);

                let mut request = Request::new_with_target(
                    entry.request_id.clone(),
                    0,
                    arrival_time,
                    entry.num_prompt_tokens(),
                    max_output_tokens,
                    target_output_tokens,
                );

                request.prompt_block_hashes = Self::compute_block_hashes(&entry.prompt_tokens, 16);
                self.requests_generated += 1;

                return Some(request);
            }
            return None;
        }

        // Non-closed-loop: original logic
        // Check if it's time for next arrival
        if self.next_arrival_time > current_time {
            return None;
        }

        // Receive from channel
        let entry = match self.dataset_receiver.as_ref()?.recv() {
            Ok(Some(e)) => e,
            Ok(None) => {
                // End of dataset signaled
                self.dataset_exhausted = true;
                return None;
            }
            Err(_) => {
                // Channel error (sender dropped)
                self.dataset_exhausted = true;
                return None;
            }
        };

        let arrival_time = self.next_arrival_time;

        // Sample actual output length from distribution
        let sampled_output_len = self.workload.output_len_dist.sample(&mut self.rng);

        // If max_output_tokens is specified in the dataset, cap at that; otherwise use sampled value
        let max_output_tokens = entry.max_output_tokens.unwrap_or(16384);
        let target_output_tokens = sampled_output_len.min(max_output_tokens);

        let mut request = Request::new_with_target(
            entry.request_id.clone(),
            0,
            arrival_time,
            entry.num_prompt_tokens(),
            max_output_tokens,
            target_output_tokens,
        );

        request.prompt_block_hashes = Self::compute_block_hashes(&entry.prompt_tokens, 16);
        self.requests_generated += 1;

        // Sample next arrival time AFTER creating the request
        // but ONLY if we haven't hit the request limit
        // This prevents sampling a bogus future time for a request that won't exist
        let should_sample_next = if let Some(max_requests) = self.workload.num_requests {
            self.requests_generated < max_requests
        } else {
            // No limit set, keep sampling (will stop when channel sends None)
            true
        };

        if should_sample_next {
            self.next_arrival_time = Self::sample_next_arrival(
                self.next_arrival_time,
                &self.workload.arrival_pattern,
                self.workload.arrival_rate,
                &mut self.rng,
            );
        }

        Some(request)
    }

    /// Check if all requests have been generated
    pub fn is_finished(&self) -> bool {
        let is_closed_loop = self.workload.arrival_pattern.to_lowercase() == "closed_loop";

        // Dataset mode: check if we've hit limit or dataset is exhausted
        if self.is_dataset_mode() {
            // If num_requests is set, check against that limit
            if let Some(max_requests) = self.workload.num_requests {
                if is_closed_loop {
                    // For closed-loop with dataset, we're finished when we've generated max_requests
                    // AND have no pending requests (or dataset is exhausted)
                    return (self.requests_generated >= max_requests
                        && self.pending_closed_loop_requests.is_empty())
                        || self.dataset_exhausted;
                } else {
                    return self.requests_generated >= max_requests;
                }
            }
            // Otherwise, check if dataset has been fully consumed
            return self.dataset_exhausted;
        }

        // Synthetic workload mode
        if let Some(max_requests) = self.workload.num_requests {
            if is_closed_loop {
                // For closed-loop, we're finished when we've generated max_requests
                // AND have no pending requests
                self.requests_generated >= max_requests
                    && self.pending_closed_loop_requests.is_empty()
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
            dataset_path: None,
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
