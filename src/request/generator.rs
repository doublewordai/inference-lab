//! Turns a [`WorkloadConfig`] into a stream of [`Request`]s: synthetic
//! (lengths sampled from the configured distributions), from a dataset (real
//! prompts, tokenised on a background thread), or sessions (chains of
//! re-entering requests, see [`super::session`]).

use super::session::{SessionSource, SessionSpec};
use super::Request;
use crate::config::{ArrivalPattern, RateSchedule, WorkloadConfig};
use crate::dataset::{BatchTokenizerFn, DatasetEntry, UnparsedEntry};
use crate::simulation::RequestTiming;
use rand::{rngs::StdRng, RngExt, SeedableRng};
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
    /// Sessions: the arrival pattern starts sessions; each further step is
    /// queued inside the source at its parent's completion plus gap.
    Sessions(SessionSource),
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

    /// Session workload. `block_size` is the scheduler's KV block size, used
    /// to build each step's block hashes.
    pub fn from_sessions(
        workload: WorkloadConfig,
        block_size: u32,
        sessions: Vec<SessionSpec>,
    ) -> Self {
        Self::build(
            workload,
            Source::Sessions(SessionSource::new(sessions, block_size)),
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
                        rng.random_range(0.0..jitter)
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

    /// Next arrival time: the next clock arrival (open loop) or the earliest
    /// pending user slot (closed loop; infinite when none is pending), and
    /// for sessions the earliest queued step if that comes first.
    pub fn peek_next_arrival_time(&self) -> f64 {
        let next_start = if self.reached_start_limit() {
            f64::INFINITY
        } else if self.workload.arrival_pattern.is_closed_loop() {
            // Closed loop: the earliest pending user slot (jittered at init,
            // then each completion time); infinite when no slot is pending.
            self.pending_closed_loop
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min)
        } else {
            self.next_arrival_time
        };
        let next = match &self.source {
            Source::Sessions(src) => src.peek_pending().unwrap_or(f64::INFINITY).min(next_start),
            _ => next_start,
        };
        if Self::within_duration(self.workload.duration_secs, next) {
            next
        } else {
            f64::INFINITY
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

    /// Whether an arrival belongs to the configured simulation window.
    /// Requests admitted by the deadline are allowed to finish afterwards.
    fn within_duration(duration_secs: Option<f64>, arrival_time: f64) -> bool {
        duration_secs.is_none_or(|deadline| arrival_time <= deadline)
    }

    /// Sessions: no more sessions may start (`num_sessions` reached). Always
    /// false for other sources.
    fn reached_start_limit(&self) -> bool {
        match (&self.source, self.workload.num_sessions) {
            (Source::Sessions(src), Some(max)) => src.num_started() as usize >= max,
            _ => false,
        }
    }

    /// The next request if it has arrived by `current_time`; `None` if no
    /// request is due or the workload is exhausted.
    pub fn next_if_before(&mut self, current_time: f64) -> Option<Request> {
        if self.reached_request_limit() {
            self.pending_closed_loop.clear();
            return None;
        }
        let duration_secs = self.workload.duration_secs;
        // Sessions: a queued step that is due comes before any new session.
        if let Source::Sessions(src) = &mut self.source {
            if src
                .peek_pending()
                .is_some_and(|t| Self::within_duration(duration_secs, t))
            {
                if let Some(req) = src.next_due(current_time) {
                    self.requests_generated += 1;
                    return Some(req);
                }
            }
            if self.reached_start_limit() {
                self.pending_closed_loop.clear();
                return None;
            }
        }
        let arrival_time = if self.workload.arrival_pattern.is_closed_loop() {
            let pos = self
                .pending_closed_loop
                .iter()
                .position(|&t| t <= current_time && Self::within_duration(duration_secs, t))?;
            self.pending_closed_loop.remove(pos)
        } else {
            if self.next_arrival_time > current_time
                || !Self::within_duration(duration_secs, self.next_arrival_time)
            {
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
            Source::Sessions(src) => src.start_session(arrival_time),
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
                if rng.random_bool(0.2) {
                    current_time + rng.random_range(0.001..0.01)
                } else {
                    current_time + rng.random_range(0.5..2.0)
                }
            }
            ArrivalPattern::Batched => 0.0,
            ArrivalPattern::ClosedLoop => 0.0,
        }
    }

    /// Whether every request has been generated: the request limit is
    /// reached, the dataset ran out, or (sessions) no more sessions may start
    /// and every started session has issued its last step.
    pub fn is_finished(&self) -> bool {
        match &self.source {
            Source::Dataset {
                exhausted: true, ..
            } => return true,
            Source::Sessions(src)
                if self.reached_start_limit()
                    && src.num_active() == 0
                    && src.peek_pending().is_none() =>
            {
                return true;
            }
            _ => {}
        }
        if self.workload.duration_secs.is_some() && self.peek_next_arrival_time().is_infinite() {
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

    /// A request completed. Sessions: queue the session's next step at
    /// `completion + gap`. Closed loop: the user slot issues a new request
    /// (a new session, in session mode) at the completion time. Returns
    /// `true` if the completed request has a successor step in its session.
    pub fn on_request_complete(&mut self, timing: &RequestTiming) -> bool {
        let completion_time = timing.completion_time;
        let mut has_successor = false;
        let mut slot_freed = true;
        if let (Source::Sessions(src), Some(step)) = (&mut self.source, &timing.session) {
            has_successor =
                src.on_step_complete_before(step, completion_time, self.workload.duration_secs);
            // The user slot stays busy until the session's last step is done.
            slot_freed = !has_successor;
        }
        if slot_freed
            && self.workload.arrival_pattern.is_closed_loop()
            && !self.reached_request_limit()
            && !self.reached_start_limit()
            && Self::within_duration(self.workload.duration_secs, completion_time)
        {
            // Replenish immediately at completion. The stagger established at
            // init time persists, since fixed ISL/OSL means each user has the
            // same cycle time — once unsynchronized, they stay that way.
            self.pending_closed_loop.push(completion_time);
        }
        has_successor
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
            sessions_path: None,
            num_sessions: None,
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

    /// A completion timing for a request nobody tracks (non-session).
    fn done_at(t: f64) -> RequestTiming {
        RequestTiming {
            request_id: "x".into(),
            arrival_time: 0.0,
            prefill_done_time: t,
            handoff_done_time: t,
            first_token_time: t,
            completion_time: t,
            num_prompt_tokens: 1,
            num_output_tokens: 1,
            num_cached_tokens: 0,
            session: None,
            num_preemptions: 0,
            rejected: false,
        }
    }

    /// Completion timing for a session request.
    fn completed(req: &Request, t: f64) -> RequestTiming {
        RequestTiming {
            request_id: req.request_id.clone(),
            arrival_time: req.arrival_time,
            prefill_done_time: t,
            handoff_done_time: t,
            first_token_time: t,
            completion_time: t,
            num_prompt_tokens: req.num_prompt_tokens,
            num_output_tokens: req.target_output_tokens,
            num_cached_tokens: 0,
            session: req.session.clone(),
            num_preemptions: 0,
            rejected: false,
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
    fn duration_stops_open_loop_arrivals_at_the_deadline() {
        let mut workload = create_test_workload(ArrivalPattern::Uniform, 1.0, 100);
        workload.duration_secs = Some(2.0);
        let mut generator = RequestGenerator::new(workload);

        let requests = drain(&mut generator);

        assert_eq!(
            requests.iter().map(|r| r.arrival_time).collect::<Vec<_>>(),
            vec![1.0, 2.0]
        );
        assert_eq!(generator.num_generated(), 2);
        assert!(generator.is_finished());
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
        generator.on_request_complete(&done_at(3.0));
        assert!(generator.next_if_before(2.9).is_none());
        let req = generator.next_if_before(3.0).unwrap();
        assert_eq!(req.arrival_time, 3.0);
        assert!(!generator.is_finished());
        // Completions beyond the request limit issue nothing.
        for t in [4.0, 5.0, 6.0] {
            generator.on_request_complete(&done_at(t));
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
        generator.on_request_complete(&done_at(3.0));
        assert_eq!(generator.peek_next_arrival_time(), 3.0);
    }

    fn session_specs() -> Vec<crate::request::SessionSpec> {
        use crate::request::{SessionSpec, StepSpec};
        let step = |input, new, output, gap| StepSpec {
            input,
            new,
            output,
            gap,
            kind: None,
        };
        vec![
            SessionSpec {
                id: "a".into(),
                steps: vec![step(64, 64, 16, 0.0), step(96, 16, 16, 2.0)],
            },
            SessionSpec {
                id: "b".into(),
                steps: vec![step(32, 32, 8, 0.0)],
            },
        ]
    }

    #[test]
    fn sessions_steps_follow_completion_plus_gap_and_come_before_new_starts() {
        // Uniform session starts every 10 s; a 2-step session's second step
        // is due at completion + 2 s, ahead of the next session start.
        let mut workload = create_test_workload(ArrivalPattern::Uniform, 0.1, 100);
        workload.num_sessions = Some(2);
        let mut g = RequestGenerator::from_sessions(workload, 16, session_specs());
        assert_eq!(g.peek_next_arrival_time(), 10.0);
        let a0 = g.next_if_before(10.0).unwrap();
        assert_eq!(a0.request_id, "s0/0");
        assert!(g.next_if_before(15.0).is_none());
        // a0 completes at 12: step 1 due at 14, before the 20 s start.
        assert!(g.on_request_complete(&completed(&a0, 12.0)));
        assert_eq!(g.peek_next_arrival_time(), 14.0);
        assert!(g.next_if_before(13.9).is_none());
        let a1 = g.next_if_before(14.0).unwrap();
        assert_eq!(a1.request_id, "s0/1");
        assert_eq!(a1.arrival_time, 14.0);
        assert_eq!(a1.session.as_ref().unwrap().shared_tokens, 80);
        assert!(!g.on_request_complete(&completed(&a1, 15.0)));
        // Second session starts at 20; the file cycles but num_sessions caps.
        let b0 = g.next_if_before(20.0).unwrap();
        assert_eq!(b0.request_id, "s1/0");
        assert!(!g.is_finished());
        assert!(g.next_if_before(1000.0).is_none());
        assert!(g.peek_next_arrival_time().is_infinite());
        assert!(!g.on_request_complete(&completed(&b0, 21.0)));
        assert!(g.is_finished());
        assert_eq!(g.num_generated(), 3);
    }

    #[test]
    fn duration_stops_future_session_steps_and_starts() {
        let mut workload = create_test_workload(ArrivalPattern::Uniform, 1.0, 100);
        workload.duration_secs = Some(1.0);
        let mut generator = RequestGenerator::from_sessions(workload, 16, session_specs());

        let first = generator.next_if_before(1.0).unwrap();
        assert_eq!(first.request_id, "s0/0");
        // Step 1 would arrive at completion (2 s) + its 2 s gap, and the
        // next session would start at 2 s. Both fall after the deadline.
        assert!(!generator.on_request_complete(&completed(&first, 2.0)));
        assert!(generator.next_if_before(100.0).is_none());
        assert!(generator.peek_next_arrival_time().is_infinite());
        assert!(generator.is_finished());
    }

    #[test]
    fn sessions_closed_loop_slot_is_held_for_the_whole_session() {
        let mut workload = create_test_workload(ArrivalPattern::ClosedLoop, 0.0, 100);
        workload.num_concurrent_users = Some(1);
        workload.num_sessions = Some(2);
        let mut g = RequestGenerator::from_sessions(workload, 16, session_specs());
        let a0 = g.next_if_before(0.0).unwrap();
        assert!(g.next_if_before(0.0).is_none());
        // Completing step 0 queues step 1 and does not free the slot.
        g.on_request_complete(&completed(&a0, 1.0));
        assert!(g.next_if_before(2.9).is_none());
        let a1 = g.next_if_before(3.0).unwrap();
        assert_eq!(a1.request_id, "s0/1");
        // The session's last step frees the slot: a new session starts.
        g.on_request_complete(&completed(&a1, 4.0));
        let b0 = g.next_if_before(4.0).unwrap();
        assert_eq!(b0.request_id, "s1/0");
        assert!(g.next_if_before(100.0).is_none());
        g.on_request_complete(&completed(&b0, 5.0));
        assert!(g.next_if_before(100.0).is_none());
        assert!(g.is_finished());
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
