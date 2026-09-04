//! Batch-sim driver. Wraps the unified [`Engine`] with a [`RequestGenerator`]
//! and a [`MetricsCollector`], pumping events to completion. This is what
//! `Commands::Sim` (CLI) and the WASM entry points drive.
//!
//! For real-time HTTP serving, see `crate::serve::engine` — same `Engine`,
//! different driver.

use super::engine::{
    Engine, IterationInfo, KvAmountStats, RequestTiming, ReusableKvStats, StepKind, Topology,
};
use super::spec::DepthSample;
use crate::config::Config;
use crate::dataset::{BatchTokenizerFn, DatasetLoader};
use crate::metrics::{
    CorpusMetrics, DeadlineMetrics, HandoffMetrics, HbmMetrics, HbmPoolMetrics, HbmWorkerMetrics,
    KvAmount, LatencySamples, MetricsCollector, MetricsSummary, RankReusableKvMetrics, RequestRow,
    ReusableKvMetrics, ReusableKvResidency, RouterMetrics, SampleCursor, SessionPrefillWork,
    SimulationMetrics, SummaryExtras,
};
use crate::request::{ReplayManifest, RequestGenerator};
use std::collections::HashMap;
use std::ops::ControlFlow;

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
    pub prefill_tokens: u64,
    /// Output tokens generated in the interval.
    pub decode_tokens: u64,
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

    /// Sim-time spacing of `time_series` samples; `None` collects nothing.
    /// Off by default: a sample every 0.1 s over a months-long session
    /// replay is gigabytes of points, and only interactive front-ends read
    /// them.
    sample_interval: Option<f64>,
    next_sample_time: f64,

    // Counters for accumulating tokens within a sample window.
    window_prefill_tokens: u64,
    window_decode_tokens: u64,
    total_prefill_tokens: u64,
    total_decode_tokens: u64,
    deadline_snapshot: Option<DeadlineMetrics>,
    measurement_start_secs: f64,
    measurement_end_secs: Option<f64>,
    corpus: CorpusMetrics,

    /// Intended arrival time of every request admitted to the engine. This
    /// includes requests still running or waiting at a hard simulation
    /// deadline, so finite-horizon analyses can report right-censoring rather
    /// than silently dropping them.
    admitted_arrival_times: Vec<f64>,

    /// Samples already delivered to the progress callback.
    sent: SampleCursor,

    /// Session workloads: engine KV bytes written as of each completed step
    /// that has a successor, keyed by (session, step). Stamped on the
    /// successor when it is generated; the engine turns it into a reuse
    /// distance at the successor's arrival.
    bytes_at_completion: HashMap<(u32, u32), (u64, u64)>,
}

impl Simulator {
    /// Build a `Simulator` from a `Config`, with an optional batch tokenizer
    /// for dataset-mode workloads. When the workload names a dataset but no
    /// `num_requests`, the dataset is counted and the config's `num_requests`
    /// filled in (see [`Simulator::config`]).
    pub fn new(mut config: Config, tokenizer: Option<BatchTokenizerFn>) -> Result<Self, String> {
        let measurement_start_secs = match config.workload.measurement_start_secs {
            Some(seconds) if seconds.is_finite() && seconds >= 0.0 => seconds,
            Some(_) => return Err("measurement_start_secs must be non-negative and finite".into()),
            None => 0.0,
        };
        let measurement_end_secs = match config.workload.measurement_duration_secs {
            Some(seconds) if seconds.is_finite() && seconds > 0.0 => {
                Some(measurement_start_secs + seconds)
            }
            Some(_) => return Err("measurement_duration_secs must be positive and finite".into()),
            None => None,
        };

        // One pool of workers, or — with a `[prefill]` block — a prefill
        // pool in front of the config's own pool, which then decodes.
        let topology = match config.disagg() {
            Some(disagg) => {
                Topology::from_disagg(&disagg, config.model.clone(), config.scheduler.clone())?
            }

            None => Topology::aggregated(
                config.cluster(),
                config.model.clone(),
                config.scheduler.clone(),
            )?,
        }
        .with_routers(&config.router, config.decode_router());

        if config.workload.dataset_path.is_some() && config.workload.replay_manifest_path.is_some()
        {
            return Err("workload sets both dataset_path and replay_manifest_path".into());
        }
        let request_generator = if let Some(replay_path) = &config.workload.replay_manifest_path {
            let manifests = ReplayManifest::load(replay_path)?;
            RequestGenerator::from_replay_manifest(
                config.workload.clone(),
                config.scheduler.block_size,
                manifests,
            )
        } else if let Some(dataset_path) = &config.workload.dataset_path {
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
        if let Some(tc) = &config.time_correction {
            engine.set_time_correction(tc.alpha, tc.beta);
        }
        if let Some(spec) = &config.speculative {
            engine.enable_speculative(spec.clone(), config.workload.seed)?;
        }

        Ok(Self {
            engine,
            request_generator,
            metrics: MetricsCollector::new(measurement_start_secs),
            config,
            time_series: Vec::new(),
            sample_interval: None,
            next_sample_time: measurement_start_secs,
            window_prefill_tokens: 0,
            window_decode_tokens: 0,
            total_prefill_tokens: 0,
            total_decode_tokens: 0,
            deadline_snapshot: None,
            measurement_start_secs,
            measurement_end_secs,
            corpus: CorpusMetrics::default(),
            admitted_arrival_times: Vec::new(),
            sent: SampleCursor::default(),
            bytes_at_completion: HashMap::new(),
        })
    }

    /// Collect a [`TimeSeriesPoint`] every `interval` seconds of sim time,
    /// readable through [`Simulator::time_series`] and the progress callback.
    pub fn with_time_series(mut self, interval: f64) -> Self {
        assert!(interval > 0.0, "sample interval must be positive");
        self.sample_interval = Some(interval);
        self
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
        // Admit only work whose simulated arrival time has actually been
        // reached. When idle, `maybe_skip_idle` advances to the next arrival
        // and the caller drains again. Submitting one future arrival on every
        // engine iteration would eventually pre-create an entire long-horizon
        // workload and make admission/session snapshots include future work.
        let now = self.engine.current_time();
        let bound = now + 1e-9;
        while let Some(mut req) = self.request_generator.next_if_before(bound) {
            self.corpus.requests_admitted += 1;
            let measurement_time = req.recorded_arrival_time.unwrap_or(req.arrival_time);
            req.record_metrics = measurement_time >= self.measurement_start_secs
                && self
                    .measurement_end_secs
                    .is_none_or(|end| measurement_time < end);
            if req.record_metrics {
                self.admitted_arrival_times.push(req.arrival_time);
                self.metrics.total_requests += 1;
            }
            if let Some(step) = &mut req.session {
                if let Some((written, touched)) = self
                    .bytes_at_completion
                    .remove(&(step.session, step.step.wrapping_sub(1)))
                {
                    step.parent_bytes_written = Some(written);
                    step.parent_bytes_touched = Some(touched);
                }
            }
            self.engine.submit(req);
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

    /// Capture the state after every event at or before the inclusive arrival
    /// deadline, but before processing the first event strictly after it.
    fn maybe_capture_deadline(&mut self) {
        let Some(deadline) = self.arrival_deadline_secs() else {
            return;
        };
        if self.deadline_snapshot.is_some() || self.engine.current_time() > deadline {
            return;
        }
        let no_engine_event_before_end = self
            .engine
            .next_event_time()
            .is_none_or(|time| time > deadline);
        let no_arrival_before_end = self.request_generator.is_finished()
            || self.request_generator.peek_next_arrival_time() > deadline;
        if !no_engine_event_before_end || !no_arrival_before_end {
            return;
        }
        let running = self.engine.aggregate_running();
        let prefilling = self.engine.aggregate_prefilling();
        self.deadline_snapshot = Some(DeadlineMetrics {
            at_secs: deadline,
            requests_admitted: self.metrics.total_requests,
            requests_completed: self.metrics.completed_requests,
            output_tokens_generated: self.total_decode_tokens,
            output_tokens_per_sec: if deadline > self.measurement_start_secs {
                self.total_decode_tokens as f64 / (deadline - self.measurement_start_secs)
            } else {
                0.0
            },
            running: running as u64,
            waiting: self.engine.aggregate_waiting() as u64,
            handoffs_in_flight: self.engine.handoffs_in_flight() as u64,
            prefilling: prefilling as u64,
            decoding: running.saturating_sub(prefilling) as u64,
        })
    }

    pub fn run_with_callback<F>(&mut self, mut callback: F) -> Result<(), String>
    where
        F: FnMut(ProgressInfo),
    {
        self.run_with_callback_mode(
            &mut |info| {
                callback(info);
                ControlFlow::Continue(())
            },
            false,
            1.0,
        )
    }

    /// Run until the configured workload deadline without draining work after
    /// it. The final callback is an exact deadline snapshot: no engine event
    /// or arrival remains at or before the deadline, and later work stays
    /// right-censored.
    pub fn run_until_deadline_with_callback<F>(&mut self, mut callback: F) -> Result<(), String>
    where
        F: FnMut(ProgressInfo),
    {
        if self.config.workload.duration_secs.is_none() {
            return Err("run_until_deadline_with_callback requires workload.duration_secs".into());
        }
        self.run_with_callback_mode(
            &mut |info| {
                callback(info);
                ControlFlow::Continue(())
            },
            true,
            1.0,
        )
    }

    /// Run toward the configured deadline, stopping immediately without a
    /// drain when the callback returns [`ControlFlow::Break`].
    pub fn run_until_deadline_with_control<F>(&mut self, mut callback: F) -> Result<(), String>
    where
        F: FnMut(ProgressInfo) -> ControlFlow<()>,
    {
        if self.config.workload.duration_secs.is_none() {
            return Err("run_until_deadline_with_control requires workload.duration_secs".into());
        }
        self.run_with_callback_mode(&mut callback, true, 1.0)
    }

    /// Run toward the configured deadline with a caller-selected progress
    /// callback interval. A coarser interval avoids repeatedly constructing a
    /// full metrics summary in long simulations while preserving the exact
    /// endpoint snapshot.
    pub fn run_until_deadline_with_control_interval<F>(
        &mut self,
        callback_interval_secs: f64,
        mut callback: F,
    ) -> Result<(), String>
    where
        F: FnMut(ProgressInfo) -> ControlFlow<()>,
    {
        if self.config.workload.duration_secs.is_none() {
            return Err(
                "run_until_deadline_with_control_interval requires workload.duration_secs".into(),
            );
        }
        if !callback_interval_secs.is_finite() || callback_interval_secs <= 0.0 {
            return Err("callback interval must be positive and finite".into());
        }
        self.run_with_callback_mode(&mut callback, true, callback_interval_secs)
    }

    fn run_with_callback_mode<F>(
        &mut self,
        callback: &mut F,
        stop_at_deadline: bool,
        callback_interval: f64,
    ) -> Result<(), String>
    where
        F: FnMut(ProgressInfo) -> ControlFlow<()>,
    {
        let mut last_callback_time = 0.0;
        loop {
            self.drain_arrivals();
            self.maybe_skip_idle();
            self.drain_arrivals();
            self.maybe_capture_deadline();

            if stop_at_deadline && self.deadline_snapshot.is_some() {
                self.engine
                    .advance_to(self.arrival_deadline_secs().unwrap());
                let _ = self.emit_progress(callback);
                break;
            }

            if self.engine.next_event_time().is_none() {
                if self.should_terminate() {
                    let _ = self.emit_progress(callback);
                    break;
                }
                // No pending event, no arrival to jump to, and work still in
                // the system: nothing can ever make progress (a request that
                // can never be scheduled, e.g. a prompt longer than the KV
                // cache, or a closed loop with no users).
                return Err(format!(
                    "simulation stalled at t={:.3}: {} running, {} waiting, no pending events{}",
                    self.engine.current_time(),
                    self.engine.aggregate_running(),
                    self.engine.aggregate_waiting(),
                    self.engine.describe_stuck_workers()
                ));
            }

            if let Some(at) = std::env::var("INFERENCE_LAB_DUMP_AT")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
            {
                if self.engine.current_time() >= at {
                    eprintln!(
                        "dump at t={:.3}: {} running, {} waiting, next event {:?}{}",
                        self.engine.current_time(),
                        self.engine.aggregate_running(),
                        self.engine.aggregate_waiting(),
                        self.engine.next_event_time(),
                        self.engine.describe_stuck_workers()
                    );
                    if let Some(m) = self.engine.memory_metrics() {
                        eprintln!("{}", serde_json::to_string_pretty(&m).unwrap_or_default());
                    }
                    std::process::exit(3);
                }
            }
            let outcome = self.engine.step()?;
            if let Some(iter) = &outcome.iteration {
                self.handle_iteration(iter);
            }
            for completion in &outcome.completions {
                self.handle_completion(completion);
            }

            // Sample the time series at fixed sim-time intervals.
            while self
                .sample_interval
                .is_some_and(|_| self.engine.current_time() >= self.next_sample_time)
            {
                let interval = self.sample_interval.unwrap();
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
                    input_throughput: prefill_tokens as f64 / interval,
                    output_throughput: decode_tokens as f64 / interval,
                    ttft_interval_mean_ms: ttft_mean,
                    tpot_interval_mean_ms: tpot_mean,
                });
                self.window_prefill_tokens = 0;
                self.window_decode_tokens = 0;
                self.next_sample_time += interval;
            }

            // Progress callback every callback_interval of sim time.
            if matches!(outcome.kind, StepKind::Iteration)
                && self.engine.current_time() - last_callback_time >= callback_interval
            {
                if self.emit_progress(callback).is_break() {
                    return Ok(());
                }
                last_callback_time = self.engine.current_time();
            }

            if self.should_terminate() {
                // A finite request cap can drain the simulation before the
                // arrival deadline. The state at the later deadline is then
                // the same idle state, so materialize that snapshot before
                // leaving the event loop.
                self.maybe_capture_deadline();
                let _ = self.emit_progress(callback);
                break;
            }
        }

        Ok(())
    }

    fn handle_iteration(&mut self, iter: &IterationInfo) {
        for prog in &iter.progress {
            if !prog.record_metrics {
                continue;
            }
            if prog.was_prefill {
                self.window_prefill_tokens += prog.num_tokens as u64;
                self.total_prefill_tokens += prog.num_tokens as u64;
            }
            self.window_decode_tokens += prog.num_output as u64;
            self.total_decode_tokens += prog.num_output as u64;
        }
        if iter.end_time >= self.measurement_start_secs {
            self.metrics.record_iteration_metrics(
                self.engine.kv_cache_util(),
                iter.flops_util,
                iter.bandwidth_util,
            );
        }
    }

    fn handle_completion(&mut self, timing: &RequestTiming) {
        if timing.rejected {
            self.corpus.requests_rejected += 1;
        } else {
            self.corpus.requests_completed += 1;
            self.corpus.completed_prompt_tokens += u64::from(timing.num_prompt_tokens);
            self.corpus.completed_output_tokens += u64::from(timing.num_output_tokens);
        }
        if timing.record_metrics {
            self.metrics.record_request_completion(timing);
        }
        let has_successor = self.request_generator.on_request_complete(timing);
        if has_successor {
            if let Some(step) = &timing.session {
                self.bytes_at_completion.insert(
                    (step.session, step.step),
                    (
                        self.engine.kv_bytes_written(),
                        self.engine.kv_bytes_touched(),
                    ),
                );
            }
        }
    }

    fn emit_progress<F>(&mut self, callback: &mut F) -> ControlFlow<()>
    where
        F: FnMut(ProgressInfo) -> ControlFlow<()>,
    {
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
        })
    }

    /// Metrics summary as of the current simulated time.
    pub fn summary(&mut self) -> MetricsSummary {
        let router =
            RouterMetrics::from_stats(self.config.router.name(), self.engine.router_stats());
        let decode_router = self
            .engine
            .decode_router_stats()
            .map(|rs| RouterMetrics::from_stats(self.config.decode_router().name(), rs));
        let handoff = decode_router.as_ref().map(|_| {
            let h = self.engine.handoff_stats();
            HandoffMetrics {
                transfers: h.transfers,
                bytes: h.bytes,
                bytes_skipped: h.bytes_skipped,
            }
        });
        let memory = self.engine.memory_metrics();
        let to_amount = |amount: KvAmountStats| KvAmount {
            tokens: amount.tokens,
            bytes: amount.bytes,
        };
        let to_residency = |stats: ReusableKvStats| {
            ReusableKvResidency::from_cells(
                stats.requests,
                to_amount(stats.reusable),
                to_amount(stats.both),
                to_amount(stats.decoder_only),
                to_amount(stats.prefill_only),
                to_amount(stats.neither),
                to_amount(stats.prefiller_miss_parent_prefill),
                to_amount(stats.prefiller_miss_parent_decode),
                to_amount(stats.prefiller_miss_unattributed),
            )
        };
        let reusable_kv = self
            .engine
            .reusable_kv_stats()
            .map(|(total, ranks, prefill_work)| {
                let total = to_residency(total);
                let output = self.total_decode_tokens;
                ReusableKvMetrics {
                    prefill_work: SessionPrefillWork {
                        new_prompt: to_amount(prefill_work.new_prompt),
                        parent_prefill_recompute: to_amount(prefill_work.parent_prefill_recompute),
                        parent_decode_recompute: to_amount(prefill_work.parent_decode_recompute),
                        unattributed_recompute: to_amount(prefill_work.unattributed_recompute),
                    },
                    recomputed_reusable_tokens_per_output_token: if output == 0 {
                        0.0
                    } else {
                        (total.decoder_hit_prefill_miss.tokens
                            + total.decoder_miss_prefill_miss.tokens) as f64
                            / output as f64
                    },
                    transferred_reusable_bytes_per_output_token: if output == 0 {
                        0.0
                    } else {
                        (total.decoder_miss_prefill_hit.bytes
                            + total.decoder_miss_prefill_miss.bytes) as f64
                            / output as f64
                    },
                    total,
                    per_decode_rank: ranks
                        .into_iter()
                        .map(|rank| RankReusableKvMetrics {
                            worker: rank.worker,
                            rank: rank.rank,
                            residency: to_residency(rank.stats),
                        })
                        .collect(),
                }
            });
        let hbm = HbmMetrics {
            pools: self
                .engine
                .hbm_pool_stats()
                .into_iter()
                .map(|pool| {
                    let workers: Vec<_> = pool
                        .workers
                        .into_iter()
                        .map(|worker| HbmWorkerMetrics {
                            worker: worker.worker,
                            rank: worker.rank,
                            running: worker.running,
                            waiting: worker.waiting,
                            capacity_bytes: worker.capacity_bytes,
                            resident_prefix_bytes: worker.resident_prefix_bytes,
                            active_or_reserved_bytes: worker.active_or_reserved_bytes,
                            eviction_events: worker.eviction_events,
                            evicted_bytes: worker.evicted_bytes,
                        })
                        .collect();
                    HbmPoolMetrics {
                        role: pool.role,
                        capacity_bytes: workers.iter().map(|w| w.capacity_bytes).sum(),
                        resident_prefix_bytes: workers
                            .iter()
                            .map(|w| w.resident_prefix_bytes)
                            .sum(),
                        active_or_reserved_bytes: workers
                            .iter()
                            .map(|w| w.active_or_reserved_bytes)
                            .sum(),
                        eviction_events: workers.iter().map(|w| w.eviction_events).sum(),
                        evicted_bytes: workers.iter().map(|w| w.evicted_bytes).sum(),
                        workers,
                    }
                })
                .collect(),
        };
        let sessions = self
            .request_generator
            .session_lifecycle()
            .map(|lifecycle| self.metrics.session_metrics(lifecycle));
        let end_time = self.engine.current_time();
        let arrival_deadline_secs = self.arrival_deadline_secs();
        let simulation = SimulationMetrics {
            end_time_secs: end_time,
            measurement_start_secs: self.measurement_start_secs,
            measurement_end_secs: self.measurement_end_secs,
            arrival_deadline_secs,
            drain_time_secs: arrival_deadline_secs
                .map_or(0.0, |deadline| (end_time - deadline).max(0.0)),
            at_deadline: self.deadline_snapshot,
        };
        let extras = SummaryExtras {
            work: self
                .metrics
                .work_metrics(self.total_prefill_tokens, self.total_decode_tokens),
            corpus: self.corpus,
            sessions,
            simulation,
            reusable_kv,
            hbm,
        };
        let metrics_end_time = if self.should_terminate() {
            self.measurement_end_secs.unwrap_or(end_time)
        } else {
            self.measurement_end_secs
                .map_or(end_time, |measurement_end| end_time.min(measurement_end))
        };
        self.metrics.compute_summary_with(
            metrics_end_time,
            self.engine.aggregate_prefix_cache(),
            router,
            decode_router,
            handoff,
            memory,
            extras,
        )
    }

    /// Per-request rows for the `--request-csv` dump.
    pub fn request_rows(&self) -> &[RequestRow] {
        &self.metrics.request_rows
    }

    /// Intended arrival times for all admitted requests, including work still
    /// incomplete at a hard simulation deadline.
    pub fn admitted_arrival_times(&self) -> &[f64] {
        &self.admitted_arrival_times
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

    fn arrival_deadline_secs(&self) -> Option<f64> {
        self.config.workload.duration_secs
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
    fn prefill_block_runs_a_disaggregated_topology() {
        let mut config = create_minimal_test_config();
        config.hardware = crate::catalog::hardware("b200").unwrap();
        config.parallel.tp = 1;
        config.replicas = 2;
        config.prefill = Some(crate::config::PrefillSpec {
            replicas: 1,
            ..Default::default()
        });
        let mut simulator = Simulator::new(config, None).unwrap();
        simulator.run_with_callback(|_| {}).unwrap();
        let summary = simulator.summary();
        assert_eq!(summary.requests.completed, 10);
        let handoff = summary.handoff.expect("disagg runs report hand-offs");
        assert_eq!(handoff.transfers, 10);
        assert!(summary.decode_router.is_some());
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
    fn duration_stops_arrivals_then_drains_in_flight_requests() {
        let mut config = create_minimal_test_config();
        config.workload.arrival_pattern = crate::config::ArrivalPattern::Uniform;
        config.workload.arrival_rate = 1.0;
        config.workload.num_requests = Some(100);
        config.workload.duration_secs = Some(1.0);
        let mut simulator = Simulator::new(config, None).unwrap();

        simulator.run_with_callback(|_| {}).unwrap();

        let summary = simulator.summary();
        assert_eq!(summary.requests.total, 1);
        assert_eq!(summary.requests.completed, 1);
        assert!(simulator.current_time() > 1.0);
        let deadline = summary.simulation.at_deadline.unwrap();
        assert_eq!(deadline.at_secs, 1.0);
        assert_eq!(deadline.requests_admitted, 1);
        assert_eq!(deadline.requests_completed, 0);
        assert_eq!(deadline.running + deadline.waiting, 1);
        assert!(summary.simulation.drain_time_secs > 0.0);
    }

    #[test]
    fn deadline_run_stops_without_draining_and_retains_admission_times() {
        let mut config = create_minimal_test_config();
        config.workload.arrival_pattern = crate::config::ArrivalPattern::Uniform;
        config.workload.arrival_rate = 1.0;
        config.workload.num_requests = Some(100);
        config.workload.duration_secs = Some(1.0);
        let mut simulator = Simulator::new(config, None).unwrap();

        simulator.run_until_deadline_with_callback(|_| {}).unwrap();

        let summary = simulator.summary();
        assert_eq!(simulator.current_time(), 1.0);
        assert_eq!(summary.simulation.drain_time_secs, 0.0);
        assert_eq!(simulator.admitted_arrival_times(), &[1.0]);
        assert!(simulator.request_rows().is_empty());
        let deadline = summary.simulation.at_deadline.unwrap();
        assert_eq!(deadline.requests_admitted, 1);
        assert_eq!(deadline.requests_completed, 0);
        assert_eq!(deadline.running + deadline.waiting, 1);
    }

    #[test]
    fn future_arrivals_are_admitted_only_after_the_clock_reaches_them() {
        let mut config = create_minimal_test_config();
        config.workload.arrival_pattern = crate::config::ArrivalPattern::Uniform;
        config.workload.arrival_rate = 1.0;
        config.workload.num_requests = Some(100);
        config.workload.duration_secs = Some(100.0);
        let mut simulator = Simulator::new(config, None).unwrap();

        assert_eq!(simulator.current_time(), 0.0);
        assert_eq!(simulator.drain_arrivals(), 0);
        assert!(simulator.admitted_arrival_times().is_empty());

        simulator.maybe_skip_idle();
        assert_eq!(simulator.current_time(), 1.0);
        assert_eq!(simulator.drain_arrivals(), 1);
        assert_eq!(simulator.admitted_arrival_times(), &[1.0]);
    }

    #[test]
    fn controlled_deadline_run_can_stop_early_without_draining() {
        let mut config = create_minimal_test_config();
        config.workload.arrival_pattern = crate::config::ArrivalPattern::Uniform;
        config.workload.arrival_rate = 1.0;
        config.workload.num_requests = Some(100);
        config.workload.duration_secs = Some(100.0);
        let mut simulator = Simulator::new(config, None).unwrap();

        simulator
            .run_until_deadline_with_control(|_| ControlFlow::Break(()))
            .unwrap();

        assert!(simulator.current_time() < 100.0);
        let summary = simulator.summary();
        assert_eq!(summary.simulation.drain_time_secs, 0.0);
        assert!(summary.simulation.at_deadline.is_none());
    }

    #[test]
    fn deadline_snapshot_is_reported_when_the_workload_drains_early() {
        let mut config = create_minimal_test_config();
        config.workload.arrival_pattern = crate::config::ArrivalPattern::Batched;
        config.workload.num_requests = Some(1);
        config.workload.duration_secs = Some(100.0);
        let mut simulator = Simulator::new(config, None).unwrap();

        simulator.run_with_callback(|_| {}).unwrap();

        let summary = simulator.summary();
        assert!(summary.simulation.end_time_secs < 100.0);
        assert_eq!(summary.simulation.drain_time_secs, 0.0);
        let deadline = summary.simulation.at_deadline.unwrap();
        assert_eq!(deadline.at_secs, 100.0);
        assert_eq!(deadline.requests_admitted, 1);
        assert_eq!(deadline.requests_completed, 1);
        assert_eq!(
            deadline.running + deadline.waiting + deadline.handoffs_in_flight,
            0
        );
        assert_eq!(
            deadline.output_tokens_per_sec,
            deadline.output_tokens_generated as f64 / 100.0
        );
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
        let mut simulator = Simulator::new(config, None).unwrap().with_time_series(0.1);
        simulator.run_with_callback(|_| {}).unwrap();
        let ts = simulator.time_series();
        assert!(!ts.is_empty());
        for i in 1..ts.len() {
            assert!(ts[i].time >= ts[i - 1].time);
        }
        // Every output token is counted exactly once across the windows.
        let decode_total: u64 = ts.iter().map(|p| p.decode_tokens).sum();
        let output_total: u64 = simulator.output_lengths().iter().map(|&x| x as u64).sum();
        assert_eq!(decode_total, output_total);
    }

    #[test]
    fn unschedulable_request_is_rejected_not_a_hang() {
        let mut config = create_minimal_test_config();
        // A prompt longer than the whole KV cache (64 blocks of 16 tokens)
        // can never be admitted: refused at submission, run completes.
        config.scheduler.kv_cache_capacity = 64 * config.model.kv_storage_bytes(16);
        config.workload.input_len_dist = LengthDistribution::Fixed { value: 4096 };
        config.workload.num_requests = Some(1);
        let mut simulator = Simulator::new(config, None).unwrap();
        simulator.run_with_callback(|_| {}).unwrap();
        let summary = simulator.summary();
        assert_eq!(summary.requests.rejected, 1);
        assert_eq!(summary.requests.completed, 0);
    }

    #[test]
    fn batchbench_manifest_runs_with_recorded_start_and_turn_delay() {
        let dir = std::env::temp_dir().join(format!("il-batchbench-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("plans.jsonl");
        std::fs::write(
            &path,
            concat!(
                r#"{"schema_version":2,"trajectory_id":"prod-1","start_after_ms":1250,"requests":["#,
                r#"{"prompt_tokens":64,"output_tokens":16,"overhead_tokens":16,"delay_after_ms":200,"blocks":["#,
                r#"{"seed":"system","tokens":32,"role":"system"},{"seed":"user","tokens":16,"role":"user"}]},"#,
                r#"{"prompt_tokens":96,"output_tokens":8,"overhead_tokens":16,"blocks":["#,
                r#"{"seed":"system","tokens":32,"role":"system"},{"seed":"user","tokens":16,"role":"user"},"#,
                r#"{"seed":"reply","tokens":16,"role":"assistant","live":true},{"seed":"tool","tokens":16,"role":"tool"}]}]}"#,
                "\n"
            ),
        )
        .unwrap();
        let mut config = create_minimal_test_config();
        config.workload.replay_manifest_path = Some(path.to_string_lossy().into_owned());
        config.workload.num_requests = None;
        config.workload.arrival_pattern = crate::config::ArrivalPattern::Poisson;
        config.workload.arrival_rate = 999.0; // ignored for manifest replay

        let mut simulator = Simulator::new(config, None).unwrap();
        simulator.run_with_callback(|_| {}).unwrap();

        let rows = simulator.request_rows();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].arrival, 1.25);
        assert!((rows[1].arrival - (rows[0].completion + 0.2)).abs() < 1e-9);
        assert_eq!(rows[1].shared_tokens, Some(80));
        let sessions = simulator.summary().sessions.unwrap();
        assert_eq!((sessions.started, sessions.completed), (1, 1));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn batchbench_block_seeds_share_prefix_cache_across_trajectories() {
        let dir = std::env::temp_dir().join(format!("il-batchbench-prefix-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("plans.jsonl");
        std::fs::write(
            &path,
            concat!(
                r#"{"schema_version":2,"trajectory_id":"a","start_after_ms":0,"requests":["#,
                r#"{"prompt_tokens":64,"output_tokens":8,"overhead_tokens":16,"blocks":["#,
                r#"{"seed":"shared-system","tokens":32,"role":"system"},{"seed":"user-a","tokens":16,"role":"user"}]}]}"#,
                "\n",
                r#"{"schema_version":2,"trajectory_id":"b","start_after_ms":1000,"requests":["#,
                r#"{"prompt_tokens":64,"output_tokens":8,"overhead_tokens":16,"blocks":["#,
                r#"{"seed":"shared-system","tokens":32,"role":"system"},{"seed":"user-b","tokens":16,"role":"user"}]}]}"#,
                "\n"
            ),
        )
        .unwrap();
        let mut config = create_minimal_test_config();
        config.replicas = 1;
        config.workload.replay_manifest_path = Some(path.to_string_lossy().into_owned());
        config.workload.num_requests = None;

        let mut simulator = Simulator::new(config, None).unwrap();
        simulator.run_with_callback(|_| {}).unwrap();

        let rows = simulator.request_rows();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].cached_tokens, 0);
        assert_eq!(rows[1].cached_tokens, 32);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn measurement_start_warms_cache_but_excludes_earlier_requests() {
        let dir = std::env::temp_dir().join(format!("il-batchbench-window-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("plans.jsonl");
        let plan = |trajectory: &str,
                    user: &str,
                    start_after_ms: u64,
                    recorded_start_after_ms: u64| {
            format!(
                concat!(
                    r#"{{"schema_version":2,"trajectory_id":"{}","start_after_ms":{},"requests":["#,
                    r#"{{"prompt_tokens":64,"output_tokens":8,"recorded_start_after_ms":{},"overhead_tokens":16,"blocks":["#,
                    r#"{{"seed":"shared-system","tokens":32,"role":"system"}},"#,
                    r#"{{"seed":"{}","tokens":16,"role":"user"}}]}}]}}"#,
                    "\n"
                ),
                trajectory, start_after_ms, recorded_start_after_ms, user
            )
        };
        std::fs::write(
            &path,
            plan("warm", "warm-user", 0, 0) + &plan("measure", "measure-user", 5_000, 10_000),
        )
        .unwrap();

        let mut config = create_minimal_test_config();
        config.replicas = 1;
        config.workload.replay_manifest_path = Some(path.to_string_lossy().into_owned());
        config.workload.measurement_start_secs = Some(10.0);
        config.workload.measurement_duration_secs = Some(20.0);
        config.workload.num_requests = None;

        let mut simulator = Simulator::new(config, None).unwrap();
        simulator.run_with_callback(|_| {}).unwrap();

        let rows = simulator.request_rows();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].arrival, 5.0);
        assert_eq!(rows[0].recorded_arrival, Some(10.0));
        assert_eq!(rows[0].cached_tokens, 32);
        let summary = simulator.summary();
        assert_eq!(summary.requests.total, 1);
        assert_eq!(summary.requests.completed, 1);
        assert_eq!(summary.corpus.requests_admitted, 2);
        assert_eq!(summary.corpus.requests_completed, 2);
        assert_eq!(summary.corpus.completed_prompt_tokens, 128);
        assert_eq!(summary.corpus.completed_output_tokens, 16);
        assert_eq!(summary.work.completed_prompt_tokens, 64);
        assert_eq!(summary.work.completed_output_tokens, 8);
        assert_eq!(summary.throughput_metrics.output_tokens_per_sec, 0.4);
        assert_eq!(summary.simulation.measurement_start_secs, 10.0);
        assert_eq!(summary.simulation.measurement_end_secs, Some(30.0));
        std::fs::remove_dir_all(&dir).ok();
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
    fn request_rows_report_the_serving_worker() {
        let mut config = create_minimal_test_config();
        config.replicas = 2;
        config.workload.arrival_pattern = crate::config::ArrivalPattern::Batched;
        config.workload.num_requests = Some(2);
        let mut simulator = Simulator::new(config, None).unwrap();

        simulator.run_with_callback(|_| {}).unwrap();

        let mut workers: Vec<_> = simulator
            .request_rows()
            .iter()
            .map(|row| row.worker.expect("worker is stamped at delivery"))
            .collect();
        workers.sort_unstable();
        assert_eq!(workers, [0, 1]);
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
