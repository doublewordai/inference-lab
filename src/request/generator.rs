//! Turns a [`WorkloadConfig`] into a stream of [`Request`]s, either synthetic
//! (lengths sampled from the configured distributions) or from a dataset
//! (real prompts, tokenised on a background thread).

use super::Request;
use crate::config::{ArrivalPattern, RateSchedule, WorkloadConfig};
use crate::dataset::{BatchTokenizerFn, DatasetEntry, UnparsedEntry};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Exp};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread;

/// Output-length cap applied to dataset entries that don't specify one.
const DEFAULT_DATASET_MAX_OUTPUT: u32 = 16384;

/// Where request contents come from.
enum Source {
    /// Prompt / output lengths sampled from the workload distributions. No
    /// prompt content, so no prefix identity (no prefix-cache sharing).
    Synthetic,
    /// Entries pre-tokenised by a background thread. Prompt block hashes are
    /// computed at the scheduler's `block_size` so the KV cache can share
    /// blocks between prompts with a common prefix.
    Dataset {
        receiver: Receiver<Option<DatasetEntry>>,
        block_size: u32,
        exhausted: bool,
    },
}

/// Generates requests based on workload configuration.
pub struct RequestGenerator {
    workload: WorkloadConfig,
    rng: StdRng,
    /// Next open-loop arrival time (unused for closed loop).
    next_arrival_time: f64,
    requests_generated: usize,
    next_request_id: u64,
    /// Closed loop: arrival times of requests waiting to be issued (one per
    /// user slot that has completed and not yet been replenished).
    pending_closed_loop: Vec<f64>,
    source: Source,
}

impl RequestGenerator {
    /// Synthetic workload.
    pub fn new(workload: WorkloadConfig) -> Self {
        Self::build(workload, Source::Synthetic)
    }

    /// Dataset workload. Spawns a background thread that reads, parses and
    /// batch-tokenises entries ahead of the simulation. `block_size` is the
    /// scheduler's KV block size, used to hash prompt blocks.
    pub fn from_dataset<I>(
        workload: WorkloadConfig,
        block_size: u32,
        dataset_iterator: I,
        tokenizer: BatchTokenizerFn,
    ) -> Self
    where
        I: Iterator<Item = Result<UnparsedEntry, Box<dyn std::error::Error>>> + Send + 'static,
    {
        // Buffer size: 5000 entries (~10-50MB depending on token counts).
        // `None` on the channel marks the end of the dataset.
        let (sender, receiver) = sync_channel::<Option<DatasetEntry>>(5000);
        thread::spawn(move || {
            let batch_size: usize = std::env::var("TOKENIZER_BATCH_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(32);
            let mut batch = Vec::with_capacity(batch_size);
            for result in dataset_iterator {
                match result {
                    Ok(unparsed) => {
                        batch.push(unparsed);
                        if batch.len() >= batch_size
                            && Self::tokenize_and_send_batch(&mut batch, &tokenizer, &sender)
                                .is_err()
                        {
                            return; // receiver dropped: simulation ended
                        }
                    }
                    Err(e) => {
                        log::error!("Error loading dataset entry: {e}");
                        break;
                    }
                }
            }
            // End of dataset (or a read error): flush the partial batch, then
            // signal the end.
            if Self::tokenize_and_send_batch(&mut batch, &tokenizer, &sender).is_ok() {
                let _ = sender.send(None);
            }
        });

        Self::build(
            workload,
            Source::Dataset {
                receiver,
                block_size,
                exhausted: false,
            },
        )
    }

    fn build(workload: WorkloadConfig, source: Source) -> Self {
        let mut rng = StdRng::seed_from_u64(workload.seed);
        let mut pending_closed_loop = Vec::new();
        let next_arrival_time;
        if workload.arrival_pattern.is_closed_loop() {
            // Seed the N initial arrivals. With `closed_loop_jitter_secs > 0`,
            // stagger them uniformly across [0, jitter) to break
            // synchronization; the stagger persists since each user
            // replenishes immediately at completion.
            let jitter = workload.closed_loop_jitter_secs.unwrap_or(0.0);
            pending_closed_loop = (0..workload.num_concurrent_users.unwrap_or(0))
                .map(|_| {
                    if jitter > 0.0 {
                        rng.gen_range(0.0..jitter)
                    } else {
                        0.0
                    }
                })
                .collect();
            next_arrival_time = 0.0;
        } else {
            next_arrival_time = Self::sample_next_arrival(&workload, 0.0, &mut rng);
        }
        Self {
            workload,
            rng,
            next_arrival_time,
            requests_generated: 0,
            next_request_id: 0,
            pending_closed_loop,
            source,
        }
    }

    /// Batch tokenize and send entries to the channel. Returns Err if the
    /// receiver dropped (simulation ended) or tokenization failed.
    fn tokenize_and_send_batch(
        batch: &mut Vec<UnparsedEntry>,
        tokenizer: &BatchTokenizerFn,
        sender: &SyncSender<Option<DatasetEntry>>,
    ) -> Result<(), ()> {
        if batch.is_empty() {
            return Ok(());
        }
        let prompt_inputs: Vec<_> = batch.iter().map(|e| e.prompt_input.clone()).collect();
        let all_tokens = match tokenizer(&prompt_inputs) {
            Ok(tokens) => tokens,
            Err(e) => {
                log::error!("Batch tokenization failed: {e}");
                return Err(());
            }
        };
        for (unparsed, prompt_tokens) in batch.drain(..).zip(all_tokens) {
            let entry = DatasetEntry {
                request_id: unparsed.request_id,
                prompt_tokens,
                max_output_tokens: unparsed.max_output_tokens,
            };
            if sender.send(Some(entry)).is_err() {
                return Err(());
            }
        }
        Ok(())
    }

    /// Next open-loop arrival time (0 for closed loop, where arrivals follow
    /// completions).
    pub fn peek_next_arrival_time(&self) -> f64 {
        if self.workload.arrival_pattern.is_closed_loop() {
            // Closed loop: the earliest pending user slot (jittered at init,
            // then each completion time); infinite when no slot is pending.
            self.pending_closed_loop
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min)
        } else {
            self.next_arrival_time
        }
    }

    /// Incremental block hashes: hash `i` covers all tokens up to the end of
    /// block `i`, so two prompts share hashes exactly over their common
    /// block-aligned prefix.
    fn compute_block_hashes(tokens: &[u32], block_size: u32) -> Vec<u64> {
        let block_size = block_size.max(1) as usize;
        let num_blocks = tokens.len().div_ceil(block_size);
        let mut hashes = Vec::with_capacity(num_blocks);
        let mut hasher = DefaultHasher::new();
        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = ((block_idx + 1) * block_size).min(tokens.len());
            // Feeding blocks incrementally into one hasher hashes the prefix.
            tokens[start..end].hash(&mut hasher);
            hashes.push(hasher.clone().finish());
        }
        hashes
    }

    fn reached_request_limit(&self) -> bool {
        self.workload
            .num_requests
            .is_some_and(|max| self.requests_generated >= max)
    }

    /// The next request if it has arrived by `current_time`; `None` if no
    /// request is due or the workload is exhausted.
    pub fn next_if_before(&mut self, current_time: f64) -> Option<Request> {
        if self.reached_request_limit() {
            self.pending_closed_loop.clear();
            return None;
        }
        let arrival_time = if self.workload.arrival_pattern.is_closed_loop() {
            let pos = self
                .pending_closed_loop
                .iter()
                .position(|&t| t <= current_time)?;
            self.pending_closed_loop.remove(pos)
        } else {
            if self.next_arrival_time > current_time {
                return None;
            }
            self.next_arrival_time
        };

        let request = match &mut self.source {
            Source::Synthetic => {
                let request_id = format!("req-{}", self.next_request_id);
                self.next_request_id += 1;
                let prompt = self.workload.input_len_dist.sample(&mut self.rng);
                let output = self.workload.output_len_dist.sample(&mut self.rng);
                Request::new(request_id, 0, arrival_time, prompt, output)
            }
            Source::Dataset {
                receiver,
                block_size,
                exhausted,
            } => {
                if *exhausted {
                    return None;
                }
                let entry = match receiver.recv() {
                    Ok(Some(e)) => e,
                    // End of dataset, or the loader thread died.
                    Ok(None) | Err(_) => {
                        *exhausted = true;
                        return None;
                    }
                };
                // Output length: sampled, capped by the entry's own limit.
                let max_output = entry
                    .max_output_tokens
                    .unwrap_or(DEFAULT_DATASET_MAX_OUTPUT);
                let target = self
                    .workload
                    .output_len_dist
                    .sample(&mut self.rng)
                    .min(max_output);
                let mut req = Request::new_with_target(
                    entry.request_id,
                    0,
                    arrival_time,
                    entry.prompt_tokens.len() as u32,
                    max_output,
                    target,
                );
                req.prompt_block_hashes =
                    Self::compute_block_hashes(&entry.prompt_tokens, *block_size);
                req
            }
        };
        self.requests_generated += 1;

        // Sample the next open-loop arrival only if more requests will exist.
        if !self.workload.arrival_pattern.is_closed_loop() && !self.reached_request_limit() {
            self.next_arrival_time =
                Self::sample_next_arrival(&self.workload, self.next_arrival_time, &mut self.rng);
        }
        Some(request)
    }

    /// Sample the next arrival time after `current_time`. When a
    /// `rate_schedule` is set it supplies the instantaneous rate
    /// λ(current_time) (piecewise-constant non-homogeneous Poisson: λ is
    /// frozen at the arrival instant, accurate when the schedule period far
    /// exceeds the gap scale); otherwise the constant `arrival_rate` is used.
    fn sample_next_arrival(workload: &WorkloadConfig, current_time: f64, rng: &mut StdRng) -> f64 {
        let rate = workload
            .rate_schedule
            .as_ref()
            .map(|s: &RateSchedule| s.rate_at(current_time))
            .unwrap_or(workload.arrival_rate)
            .max(1e-9);
        match workload.arrival_pattern {
            ArrivalPattern::Poisson => current_time + Exp::new(rate).unwrap().sample(rng),
            ArrivalPattern::Uniform => current_time + 1.0 / rate,
            ArrivalPattern::Burst => {
                if rng.gen_bool(0.2) {
                    current_time + rng.gen_range(0.001..0.01)
                } else {
                    current_time + rng.gen_range(0.5..2.0)
                }
            }
            ArrivalPattern::Batched => 0.0,
            ArrivalPattern::ClosedLoop => 0.0,
        }
    }

    /// Whether every request has been generated (or the dataset ran out).
    pub fn is_finished(&self) -> bool {
        if let Source::Dataset {
            exhausted: true, ..
        } = self.source
        {
            return true;
        }
        match self.workload.num_requests {
            Some(_) => {
                self.reached_request_limit()
                    && (!self.workload.arrival_pattern.is_closed_loop()
                        || self.pending_closed_loop.is_empty())
            }
            None => false,
        }
    }

    /// Closed loop: a request completed at `completion_time`; its user slot
    /// issues a new request there. No-op for other patterns.
    pub fn on_request_complete(&mut self, completion_time: f64) {
        if !self.workload.arrival_pattern.is_closed_loop() || self.reached_request_limit() {
            return;
        }
        // Replenish immediately at completion. The stagger established at
        // init time persists, since fixed ISL/OSL means each user has the
        // same cycle time — once unsynchronized, they stay that way.
        self.pending_closed_loop.push(completion_time);
    }

    /// Requests generated so far.
    pub fn num_generated(&self) -> usize {
        self.requests_generated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LengthDistribution;

    fn create_test_workload(
        pattern: ArrivalPattern,
        rate: f64,
        num_requests: usize,
    ) -> WorkloadConfig {
        WorkloadConfig {
            dataset_path: None,
            arrival_pattern: pattern,
            arrival_rate: rate,
            rate_schedule: None,
            num_concurrent_users: None,
            input_len_dist: LengthDistribution::Fixed { value: 100 },
            output_len_dist: LengthDistribution::Fixed { value: 50 },
            num_requests: Some(num_requests),
            duration_secs: None,
            seed: 42,
            closed_loop_jitter_secs: None,
        }
    }

    fn drain(generator: &mut RequestGenerator) -> Vec<Request> {
        let mut requests = Vec::new();
        let mut current_time = 0.0;
        while !generator.is_finished() {
            current_time += 10.0;
            while let Some(req) = generator.next_if_before(current_time) {
                requests.push(req);
            }
        }
        requests
    }

    #[test]
    fn test_generator_creation() {
        let generator =
            RequestGenerator::new(create_test_workload(ArrivalPattern::Poisson, 1.0, 10));
        assert_eq!(generator.num_generated(), 0);
        assert!(!generator.is_finished());
    }

    #[test]
    fn test_generate_requests() {
        let mut generator =
            RequestGenerator::new(create_test_workload(ArrivalPattern::Poisson, 10.0, 5));
        let requests = drain(&mut generator);
        assert_eq!(requests.len(), 5);
        assert!(generator.is_finished());
    }

    #[test]
    fn test_arrival_ordering() {
        let mut generator =
            RequestGenerator::new(create_test_workload(ArrivalPattern::Poisson, 5.0, 10));
        let requests = drain(&mut generator);
        for i in 1..requests.len() {
            assert!(requests[i].arrival_time >= requests[i - 1].arrival_time);
        }
    }

    #[test]
    fn test_uniform_arrival() {
        let mut generator =
            RequestGenerator::new(create_test_workload(ArrivalPattern::Uniform, 2.0, 4));
        let requests = drain(&mut generator);
        assert_eq!(requests.len(), 4);
        for i in 1..requests.len() {
            let inter_arrival = requests[i].arrival_time - requests[i - 1].arrival_time;
            assert!((inter_arrival - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_request_properties() {
        let mut generator =
            RequestGenerator::new(create_test_workload(ArrivalPattern::Poisson, 1.0, 1));
        let req = generator.next_if_before(10.0).unwrap();
        assert_eq!(req.num_prompt_tokens, 100);
        assert_eq!(req.max_output_tokens, 50);
        assert_eq!(req.priority, 0);
        assert!(req.request_id.starts_with("req-"));
        // Synthetic prompts have no content identity.
        assert!(req.prompt_block_hashes.is_empty());
    }

    #[test]
    fn test_peek_next_arrival() {
        let mut generator =
            RequestGenerator::new(create_test_workload(ArrivalPattern::Poisson, 1.0, 10));
        let next_arrival = generator.peek_next_arrival_time();
        assert!(next_arrival > 0.0);
        let req = generator.next_if_before(next_arrival + 1.0).unwrap();
        assert_eq!(req.arrival_time, next_arrival);
    }

    #[test]
    fn test_closed_loop_replenishes_on_completion() {
        let mut workload = create_test_workload(ArrivalPattern::ClosedLoop, 0.0, 5);
        workload.num_concurrent_users = Some(2);
        let mut generator = RequestGenerator::new(workload);
        // Two users start at t=0.
        assert!(generator.next_if_before(0.0).is_some());
        assert!(generator.next_if_before(0.0).is_some());
        assert!(generator.next_if_before(0.0).is_none());
        // A completion at t=3 issues the next request at t=3.
        generator.on_request_complete(3.0);
        assert!(generator.next_if_before(2.9).is_none());
        let req = generator.next_if_before(3.0).unwrap();
        assert_eq!(req.arrival_time, 3.0);
        assert!(!generator.is_finished());
        // Completions beyond the request limit issue nothing.
        for t in [4.0, 5.0, 6.0] {
            generator.on_request_complete(t);
        }
        assert!(generator.next_if_before(10.0).is_some());
        assert!(generator.next_if_before(10.0).is_some());
        assert!(generator.next_if_before(10.0).is_none());
        assert!(generator.is_finished());
    }

    #[test]
    fn test_closed_loop_peek_tracks_pending_slots() {
        let mut workload = create_test_workload(ArrivalPattern::ClosedLoop, 0.0, 4);
        workload.num_concurrent_users = Some(2);
        workload.closed_loop_jitter_secs = Some(0.5);
        let mut generator = RequestGenerator::new(workload);
        // Jittered starts: nothing is due at t=0, and peek reports the
        // earliest pending slot so the simulator can jump to it.
        assert!(generator.next_if_before(0.0).is_none());
        let first = generator.peek_next_arrival_time();
        assert!(first > 0.0 && first < 0.5, "{first}");
        let req = generator.next_if_before(first).unwrap();
        assert_eq!(req.arrival_time, first);
        let second = generator.peek_next_arrival_time();
        assert!(second >= first && second < 0.5, "{second}");
        assert!(generator.next_if_before(0.5).is_some());
        // No pending slot: peek is infinite until a completion refills one.
        assert!(generator.peek_next_arrival_time().is_infinite());
        generator.on_request_complete(3.0);
        assert_eq!(generator.peek_next_arrival_time(), 3.0);
    }

    #[test]
    fn test_block_hashes_are_prefix_incremental_at_block_size() {
        let a: Vec<u32> = (0..100).collect();
        let mut b = a.clone();
        b[70] = 999; // differs inside block 4 (tokens 64..80) at block_size 16
        let ha = RequestGenerator::compute_block_hashes(&a, 16);
        let hb = RequestGenerator::compute_block_hashes(&b, 16);
        assert_eq!(ha.len(), 7); // ceil(100 / 16)
        assert_eq!(&ha[..4], &hb[..4]);
        assert_ne!(ha[4], hb[4]);
        assert_ne!(ha[5], hb[5]); // incremental: later hashes differ too
                                  // Block size drives the hash count.
        assert_eq!(RequestGenerator::compute_block_hashes(&a, 64).len(), 2);
    }
}
