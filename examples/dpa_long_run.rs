//! Long, finite-horizon DPA run for observing equilibration or divergence.
//!
//! Arrivals remain enabled through the configured endpoint and the simulator
//! stops there without draining. Cohorts therefore report both completed and
//! right-censored requests. Periodic stderr progress includes wall-clock speed
//! and ETA so multi-day simulated runs can be timed before a broader sweep.
//!
//! Run:
//!   cargo run --release --example dpa_long_run -- --lambda 0.02 --json

use inference_lab::catalog;
use inference_lab::config::{
    ArrivalPattern, Config, LengthDistribution, MemoryConfig, ParallelConfig, PrefillSpec,
    RouterConfig, SchedulerConfig, WorkloadConfig,
};
use inference_lab::metrics::{Distribution, MetricsSummary, RequestRow};
use inference_lab::scheduler::SchedulingPolicy;
use inference_lab::simulation::Simulator;
use serde::Serialize;
use std::ops::ControlFlow;
use std::time::Instant;

const DPA_RANKS: u32 = 8;
const PREFILL_REPLICAS: u32 = 1;
const SEED: u64 = 42;
const DEFAULT_LAMBDA_RANK: f64 = 0.02;
const DEFAULT_DURATION_SECS: f64 = 10.0 * 24.0 * 60.0 * 60.0;
const DEFAULT_LOG_INTERVAL_SECS: f64 = 60.0 * 60.0;
const DEFAULT_QUEUE_SAMPLE_INTERVAL_SECS: f64 = 60.0;
const DEFAULT_MAX_WAITING: usize = 1_000;
const DEFAULT_GROWTH_EPOCHS: usize = 6;
const ANALYSIS_COHORT_SECS: f64 = 120.0;
const SLO_WINDOW_SECS: f64 = 600.0;
const TTFT_SLO_SECS: f64 = 30.0;
const DEFAULT_TTFT_QUANTILE: f64 = 0.95;

#[derive(Debug)]
struct Args {
    json: bool,
    lambda_rank: f64,
    duration_secs: f64,
    log_interval_secs: f64,
    queue_sample_interval_secs: f64,
    max_waiting: usize,
    growth_epochs: usize,
    stop_on_ttft_breach: bool,
    ttft_quantile: f64,
    prefill_hbm_factor: f64,
}

fn parse_args() -> Args {
    let mut args = Args {
        json: false,
        lambda_rank: DEFAULT_LAMBDA_RANK,
        duration_secs: DEFAULT_DURATION_SECS,
        log_interval_secs: DEFAULT_LOG_INTERVAL_SECS,
        queue_sample_interval_secs: DEFAULT_QUEUE_SAMPLE_INTERVAL_SECS,
        max_waiting: DEFAULT_MAX_WAITING,
        growth_epochs: DEFAULT_GROWTH_EPOCHS,
        stop_on_ttft_breach: false,
        ttft_quantile: DEFAULT_TTFT_QUANTILE,
        prefill_hbm_factor: 1.0,
    };
    let mut raw = std::env::args().skip(1);
    while let Some(arg) = raw.next() {
        let value = |raw: &mut std::iter::Skip<std::env::Args>, flag: &str| {
            raw.next()
                .unwrap_or_else(|| panic!("{flag} requires a value"))
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("{flag} must be a number"))
        };
        match arg.as_str() {
            "--json" => args.json = true,
            "--lambda" => args.lambda_rank = value(&mut raw, "--lambda"),
            "--duration" => args.duration_secs = value(&mut raw, "--duration"),
            "--log-every" => args.log_interval_secs = value(&mut raw, "--log-every"),
            "--sample-every" => args.queue_sample_interval_secs = value(&mut raw, "--sample-every"),
            "--max-waiting" => {
                args.max_waiting = raw
                    .next()
                    .expect("--max-waiting requires a value")
                    .parse::<usize>()
                    .expect("--max-waiting must be an integer")
            }
            "--growth-epochs" => {
                args.growth_epochs = raw
                    .next()
                    .expect("--growth-epochs requires a value")
                    .parse::<usize>()
                    .expect("--growth-epochs must be an integer")
            }
            "--stop-on-ttft-breach" => args.stop_on_ttft_breach = true,
            "--ttft-quantile" => args.ttft_quantile = value(&mut raw, "--ttft-quantile"),
            "--prefill-hbm-factor" => {
                args.prefill_hbm_factor = value(&mut raw, "--prefill-hbm-factor")
            }
            _ => panic!("unknown argument {arg:?}"),
        }
    }
    assert!(args.lambda_rank > 0.0, "--lambda must be positive");
    assert!(args.duration_secs > 0.0, "--duration must be positive");
    assert!(args.log_interval_secs > 0.0, "--log-every must be positive");
    assert!(
        args.queue_sample_interval_secs > 0.0,
        "--sample-every must be positive"
    );
    assert!(args.max_waiting > 0, "--max-waiting must be positive");
    assert!(args.growth_epochs > 0, "--growth-epochs must be positive");
    assert!(
        (0.0..=1.0).contains(&args.ttft_quantile),
        "--ttft-quantile must be between 0 and 1"
    );
    assert!(
        args.prefill_hbm_factor.is_finite() && args.prefill_hbm_factor > 0.0,
        "--prefill-hbm-factor must be positive and finite"
    );
    assert!(
        ((args.duration_secs / SLO_WINDOW_SECS).round() - args.duration_secs / SLO_WINDOW_SECS)
            .abs()
            < 1e-9,
        "--duration must be a multiple of 600 seconds"
    );
    args
}

fn scheduler() -> SchedulerConfig {
    SchedulerConfig {
        max_num_batched_tokens: 8192,
        max_num_seqs: 32768,
        enable_chunked_prefill: true,
        long_prefill_token_threshold: 0,
        max_num_partial_prefills: 1,
        block_size: 64,
        gpu_memory_utilization: 0.9,
        kv_cache_capacity: 0,
        max_model_len: None,
        policy: SchedulingPolicy::FCFS,
        enable_preemption_free: false,
        enable_cascade_attention: false,
        balance_set: None,
        max_waiting: 0,
    }
}

fn parallel() -> ParallelConfig {
    ParallelConfig {
        tp: DPA_RANKS,
        ep: DPA_RANKS,
        dp_attention: true,
    }
}

fn config(lambda_rank: f64, duration_secs: f64, prefill_hbm_factor: f64) -> Config {
    let parallel = parallel();
    let hardware = catalog::hardware("b200").expect("b200 in catalog");
    let prefill_hardware = if prefill_hbm_factor == 1.0 {
        None
    } else {
        let mut hardware = hardware.clone();
        hardware.name = format!("{}-prefill-hbm-{prefill_hbm_factor:.3}x", hardware.name);
        hardware.memory_capacity =
            (hardware.memory_capacity as f64 * prefill_hbm_factor).round() as u64;
        Some(hardware)
    };
    let mut config = Config {
        hardware,
        parallel: parallel.clone(),
        model: catalog::model("glm-5.2-fp8").expect("glm-5.2-fp8 in catalog"),
        scheduler: scheduler(),
        replicas: 1,
        router: RouterConfig::SessionAffinity {},
        decode_router: Some(RouterConfig::SessionAffinity {}),
        memory: MemoryConfig::default(),
        prefill: Some(PrefillSpec {
            hardware: prefill_hardware,
            parallel: Some(parallel),
            replicas: PREFILL_REPLICAS,
            memory: MemoryConfig::default(),
            kv_link_bw: None,
        }),
        time_correction: None,
        workload: WorkloadConfig {
            dataset_path: None,
            replay_manifest_path: Some("plans.jsonl".into()),
            measurement_start_secs: None,
            measurement_duration_secs: None,
            num_trajectories: None,
            arrival_pattern: ArrivalPattern::Poisson,
            arrival_rate: lambda_rank * DPA_RANKS as f64,
            rate_schedule: None,
            num_concurrent_users: None,
            closed_loop_jitter_secs: None,
            input_len_dist: LengthDistribution::Fixed { value: 1 },
            output_len_dist: LengthDistribution::Fixed { value: 1 },
            num_requests: None,
            duration_secs: Some(duration_secs),
            seed: SEED,
        },
        speculative: None,
        fault: None,
    };
    config.finalize();
    config
}

#[derive(Debug, Clone, Serialize)]
struct ProgressSample {
    simulated_secs: f64,
    simulated_days: f64,
    percent: f64,
    wall_elapsed_secs: f64,
    recent_sim_secs_per_wall_sec: f64,
    estimated_wall_secs_remaining: f64,
    requests_admitted: u64,
    requests_completed: u64,
    running: usize,
    waiting: usize,
    prefill_queue: QueueDistribution,
    decode_queue: QueueDistribution,
    epoch_min_waiting: usize,
    epoch_max_waiting: usize,
    active_sessions: Option<u64>,
    interval_output_tokens_per_sec: f64,
    interval_completed_ttft_samples: usize,
    interval_completed_ttft_quantile: f64,
    interval_completed_ttft_quantile_ms: Option<f64>,
    interval_cause: IntervalCause,
}

#[derive(Debug, Clone, Serialize)]
struct QueueDistribution {
    samples: usize,
    current_waiting: usize,
    mean_waiting: f64,
    p50_waiting: f64,
    p99_waiting: f64,
    max_waiting: usize,
    zero_sample_fraction: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
enum StopReason {
    MaxWaiting {
        limit: usize,
        observed: usize,
    },
    ConsecutiveQueueGrowth {
        epochs: usize,
        start_waiting: usize,
        end_waiting: usize,
    },
    CompletedTtftSloBreach {
        interval_secs: f64,
        completed_samples: usize,
        quantile: f64,
        observed_ms: f64,
        limit_ms: f64,
    },
}

#[derive(Debug, Clone, Serialize)]
struct TtftCohort {
    index: usize,
    start_secs: f64,
    end_secs: f64,
    requests_admitted: usize,
    requests_completed_by_endpoint: usize,
    requests_censored_at_endpoint: usize,
    exact: bool,
    completed_p50_ms: Option<f64>,
    completed_p90_ms: Option<f64>,
    completed_p95_ms: Option<f64>,
    completed_p99_ms: Option<f64>,
    completed_max_ms: Option<f64>,
}

#[derive(Debug, Clone, Copy, Default)]
struct CauseTotals {
    prefill_tokens_computed: u64,
    new_prompt_tokens: u64,
    parent_prefill_recompute_tokens: u64,
    parent_decode_recompute_tokens: u64,
    unattributed_recompute_tokens: u64,
    reusable_tokens: u64,
    prefiller_miss_reusable_tokens: u64,
    prefill_eviction_events: u64,
    prefill_evicted_bytes: u64,
    decode_eviction_events: u64,
    decode_evicted_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
struct IntervalCause {
    interval_secs: f64,
    prefill_tokens_computed: u64,
    new_prompt_tokens: u64,
    parent_prefill_recompute_tokens: u64,
    parent_decode_recompute_tokens: u64,
    unattributed_recompute_tokens: u64,
    computed_minus_classified_tokens: i128,
    reusable_tokens: u64,
    prefiller_miss_reusable_tokens: u64,
    prefiller_miss_reusable_token_fraction: f64,
    prefill_eviction_events: u64,
    prefill_evicted_bytes: u64,
    decode_eviction_events: u64,
    decode_evicted_bytes: u64,
    prefill_capacity_bytes: u64,
    prefill_resident_prefix_bytes: u64,
    prefill_active_or_reserved_bytes: u64,
    decode_capacity_bytes: u64,
    decode_resident_prefix_bytes: u64,
    decode_active_or_reserved_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
struct LargestReusableMiss {
    lookup_secs: f64,
    arrival_secs: f64,
    ttft_ms: f64,
    session: u32,
    step: u32,
    prompt_tokens: u32,
    shared_tokens: u32,
    cached_tokens: u32,
    reusable_miss_tokens: u32,
}

#[derive(Debug, Clone, Serialize)]
struct RequestCauseCohort {
    index: usize,
    start_secs: f64,
    end_secs: f64,
    completed_session_requests: usize,
    reusable_miss_requests: usize,
    reusable_miss_tokens: u64,
    reusable_miss_p99_tokens: Option<f64>,
    largest_reusable_miss: Option<LargestReusableMiss>,
}

#[derive(Debug, Serialize)]
struct LongRunResult {
    description: String,
    lambda_rank_sessions_s: f64,
    lambda_node_sessions_s: f64,
    seed: u64,
    prefill_hbm_factor: f64,
    endpoint_secs: f64,
    endpoint_days: f64,
    actual_end_secs: f64,
    actual_end_days: f64,
    completed_endpoint: bool,
    stop_reason: Option<StopReason>,
    log_interval_secs: f64,
    queue_sample_interval_secs: f64,
    max_waiting_limit: usize,
    growth_epochs_limit: usize,
    stop_on_ttft_breach: bool,
    ttft_quantile: f64,
    ttft_slo_ms: f64,
    completed_ttft_quantile_ms: Option<f64>,
    analysis_cohort_secs: f64,
    slo_window_secs: f64,
    wall_elapsed_secs: f64,
    prefill_queue: QueueDistribution,
    decode_queue: QueueDistribution,
    endpoint_metrics: MetricsSummary,
    progress: Vec<ProgressSample>,
    request_cause_cohorts: Vec<RequestCauseCohort>,
    ttft_analysis_cohorts: Vec<TtftCohort>,
    ttft_slo_windows: Vec<TtftCohort>,
}

fn pool_waiting(metrics: &MetricsSummary, role: &str) -> usize {
    metrics
        .hbm
        .pools
        .iter()
        .filter(|pool| pool.role == role)
        .flat_map(|pool| &pool.workers)
        .map(|worker| worker.waiting as usize)
        .sum()
}

fn pool_cache_state(metrics: &MetricsSummary, role: &str) -> (u64, u64, u64, u64, u64) {
    metrics
        .hbm
        .pools
        .iter()
        .filter(|pool| pool.role == role)
        .fold((0, 0, 0, 0, 0), |totals, pool| {
            (
                totals.0 + pool.capacity_bytes,
                totals.1 + pool.resident_prefix_bytes,
                totals.2 + pool.active_or_reserved_bytes,
                totals.3 + pool.eviction_events,
                totals.4 + pool.evicted_bytes,
            )
        })
}

fn cause_totals(metrics: &MetricsSummary) -> CauseTotals {
    let mut totals = CauseTotals {
        prefill_tokens_computed: metrics.work.prefill_tokens_computed,
        ..CauseTotals::default()
    };
    if let Some(reusable) = &metrics.reusable_kv {
        totals.new_prompt_tokens = reusable.prefill_work.new_prompt.tokens;
        totals.parent_prefill_recompute_tokens =
            reusable.prefill_work.parent_prefill_recompute.tokens;
        totals.parent_decode_recompute_tokens =
            reusable.prefill_work.parent_decode_recompute.tokens;
        totals.unattributed_recompute_tokens = reusable.prefill_work.unattributed_recompute.tokens;
        totals.reusable_tokens = reusable.total.reusable.tokens;
        totals.prefiller_miss_reusable_tokens = reusable.total.decoder_hit_prefill_miss.tokens
            + reusable.total.decoder_miss_prefill_miss.tokens;
    }
    let (_, _, _, prefill_events, prefill_bytes) = pool_cache_state(metrics, "prefill");
    let (_, _, _, decode_events, decode_bytes) = pool_cache_state(metrics, "decode");
    totals.prefill_eviction_events = prefill_events;
    totals.prefill_evicted_bytes = prefill_bytes;
    totals.decode_eviction_events = decode_events;
    totals.decode_evicted_bytes = decode_bytes;
    totals
}

fn interval_cause(
    metrics: &MetricsSummary,
    previous: CauseTotals,
    interval_secs: f64,
) -> (CauseTotals, IntervalCause) {
    let current = cause_totals(metrics);
    let delta = |now: u64, before: u64| now.saturating_sub(before);
    let prefill_tokens_computed = delta(
        current.prefill_tokens_computed,
        previous.prefill_tokens_computed,
    );
    let new_prompt_tokens = delta(current.new_prompt_tokens, previous.new_prompt_tokens);
    let parent_prefill_recompute_tokens = delta(
        current.parent_prefill_recompute_tokens,
        previous.parent_prefill_recompute_tokens,
    );
    let parent_decode_recompute_tokens = delta(
        current.parent_decode_recompute_tokens,
        previous.parent_decode_recompute_tokens,
    );
    let unattributed_recompute_tokens = delta(
        current.unattributed_recompute_tokens,
        previous.unattributed_recompute_tokens,
    );
    let classified = new_prompt_tokens
        + parent_prefill_recompute_tokens
        + parent_decode_recompute_tokens
        + unattributed_recompute_tokens;
    let reusable_tokens = delta(current.reusable_tokens, previous.reusable_tokens);
    let prefiller_miss_reusable_tokens = delta(
        current.prefiller_miss_reusable_tokens,
        previous.prefiller_miss_reusable_tokens,
    );
    let (prefill_capacity, prefill_resident, prefill_active, _, _) =
        pool_cache_state(metrics, "prefill");
    let (decode_capacity, decode_resident, decode_active, _, _) =
        pool_cache_state(metrics, "decode");
    (
        current,
        IntervalCause {
            interval_secs,
            prefill_tokens_computed,
            new_prompt_tokens,
            parent_prefill_recompute_tokens,
            parent_decode_recompute_tokens,
            unattributed_recompute_tokens,
            computed_minus_classified_tokens: prefill_tokens_computed as i128 - classified as i128,
            reusable_tokens,
            prefiller_miss_reusable_tokens,
            prefiller_miss_reusable_token_fraction: if reusable_tokens == 0 {
                0.0
            } else {
                prefiller_miss_reusable_tokens as f64 / reusable_tokens as f64
            },
            prefill_eviction_events: delta(
                current.prefill_eviction_events,
                previous.prefill_eviction_events,
            ),
            prefill_evicted_bytes: delta(
                current.prefill_evicted_bytes,
                previous.prefill_evicted_bytes,
            ),
            decode_eviction_events: delta(
                current.decode_eviction_events,
                previous.decode_eviction_events,
            ),
            decode_evicted_bytes: delta(
                current.decode_evicted_bytes,
                previous.decode_evicted_bytes,
            ),
            prefill_capacity_bytes: prefill_capacity,
            prefill_resident_prefix_bytes: prefill_resident,
            prefill_active_or_reserved_bytes: prefill_active,
            decode_capacity_bytes: decode_capacity,
            decode_resident_prefix_bytes: decode_resident,
            decode_active_or_reserved_bytes: decode_active,
        },
    )
}

fn summarize_queue(distribution: &mut Distribution, current_waiting: usize) -> QueueDistribution {
    let samples = distribution.len();
    let zero_samples = distribution
        .values()
        .iter()
        .filter(|&&waiting| waiting == 0.0)
        .count();
    QueueDistribution {
        samples,
        current_waiting,
        mean_waiting: distribution.mean(),
        p50_waiting: distribution.quantile(0.50),
        p99_waiting: distribution.quantile(0.99),
        max_waiting: distribution.quantile(1.0) as usize,
        zero_sample_fraction: if samples == 0 {
            0.0
        } else {
            zero_samples as f64 / samples as f64
        },
    }
}

fn format_wall(seconds: f64) -> String {
    if !seconds.is_finite() {
        return "unknown".into();
    }
    if seconds >= 3600.0 {
        format!("{:.1}h", seconds / 3600.0)
    } else if seconds >= 60.0 {
        format!("{:.1}m", seconds / 60.0)
    } else {
        format!("{seconds:.1}s")
    }
}

fn build_cohorts(
    arrivals: &[f64],
    completed: &[RequestRow],
    endpoint_secs: f64,
    cohort_secs: f64,
) -> Vec<TtftCohort> {
    let count = (endpoint_secs / cohort_secs).round() as usize;
    let mut admitted = vec![0usize; count];
    let mut distributions: Vec<Distribution> =
        (0..count).map(|_| Distribution::default()).collect();

    for &arrival in arrivals {
        if (0.0..endpoint_secs).contains(&arrival) {
            admitted[(arrival / cohort_secs).floor() as usize] += 1;
        }
    }
    for row in completed {
        if (0.0..endpoint_secs).contains(&row.arrival) && row.completion <= endpoint_secs {
            distributions[(row.arrival / cohort_secs).floor() as usize].push(row.ttft);
        }
    }

    distributions
        .into_iter()
        .enumerate()
        .map(|(index, mut distribution)| {
            let completed = distribution.len();
            let value = |distribution: &mut Distribution, quantile| {
                (!distribution.is_empty()).then(|| distribution.quantile(quantile) * 1000.0)
            };
            TtftCohort {
                index,
                start_secs: index as f64 * cohort_secs,
                end_secs: (index + 1) as f64 * cohort_secs,
                requests_admitted: admitted[index],
                requests_completed_by_endpoint: completed,
                requests_censored_at_endpoint: admitted[index].saturating_sub(completed),
                exact: admitted[index] == completed,
                completed_p50_ms: value(&mut distribution, 0.50),
                completed_p90_ms: value(&mut distribution, 0.90),
                completed_p95_ms: value(&mut distribution, 0.95),
                completed_p99_ms: value(&mut distribution, 0.99),
                completed_max_ms: value(&mut distribution, 1.0),
            }
        })
        .collect()
}

fn build_request_cause_cohorts(
    completed: &[RequestRow],
    endpoint_secs: f64,
    cohort_secs: f64,
) -> Vec<RequestCauseCohort> {
    struct BuildCohort {
        completed_session_requests: usize,
        reusable_miss_requests: usize,
        reusable_miss_tokens: u64,
        reusable_misses: Distribution,
        largest_reusable_miss: Option<LargestReusableMiss>,
    }

    let count = (endpoint_secs / cohort_secs).ceil() as usize;
    let mut cohorts: Vec<_> = (0..count)
        .map(|_| BuildCohort {
            completed_session_requests: 0,
            reusable_miss_requests: 0,
            reusable_miss_tokens: 0,
            reusable_misses: Distribution::default(),
            largest_reusable_miss: None,
        })
        .collect();

    for row in completed {
        let (Some((session, step)), Some(shared_tokens)) = (row.session, row.shared_tokens) else {
            continue;
        };
        let lookup_secs = row.lookup.map_or(row.arrival, |lookup| lookup.at);
        if !(0.0..endpoint_secs).contains(&lookup_secs) {
            continue;
        }
        let cohort = &mut cohorts[(lookup_secs / cohort_secs).floor() as usize];
        cohort.completed_session_requests += 1;
        let cached_tokens = row.cached_tokens.min(shared_tokens);
        let reusable_miss_tokens = shared_tokens.saturating_sub(cached_tokens);
        if reusable_miss_tokens == 0 {
            continue;
        }
        cohort.reusable_miss_requests += 1;
        cohort.reusable_miss_tokens += reusable_miss_tokens as u64;
        cohort.reusable_misses.push(reusable_miss_tokens as f64);
        let largest = LargestReusableMiss {
            lookup_secs,
            arrival_secs: row.arrival,
            ttft_ms: row.ttft * 1000.0,
            session,
            step,
            prompt_tokens: row.prompt_tokens,
            shared_tokens,
            cached_tokens,
            reusable_miss_tokens,
        };
        if cohort
            .largest_reusable_miss
            .as_ref()
            .is_none_or(|previous| reusable_miss_tokens > previous.reusable_miss_tokens)
        {
            cohort.largest_reusable_miss = Some(largest);
        }
    }

    cohorts
        .into_iter()
        .enumerate()
        .map(|(index, mut cohort)| RequestCauseCohort {
            index,
            start_secs: index as f64 * cohort_secs,
            end_secs: ((index + 1) as f64 * cohort_secs).min(endpoint_secs),
            completed_session_requests: cohort.completed_session_requests,
            reusable_miss_requests: cohort.reusable_miss_requests,
            reusable_miss_tokens: cohort.reusable_miss_tokens,
            reusable_miss_p99_tokens: (!cohort.reusable_misses.is_empty())
                .then(|| cohort.reusable_misses.quantile(0.99)),
            largest_reusable_miss: cohort.largest_reusable_miss,
        })
        .collect()
}

fn main() -> Result<(), String> {
    let args = parse_args();
    let started = Instant::now();
    let mut simulator = Simulator::new(
        config(
            args.lambda_rank,
            args.duration_secs,
            args.prefill_hbm_factor,
        ),
        None,
    )?;
    let mut progress = Vec::new();
    let mut next_log = args.log_interval_secs;
    let mut last_log_sim = 0.0;
    let mut last_log_wall = Instant::now();
    let mut last_output_tokens = 0u64;
    let mut interval_ttft = Distribution::default();
    let mut interval_prefill_queue = Distribution::default();
    let mut interval_decode_queue = Distribution::default();
    let mut overall_prefill_queue = Distribution::default();
    let mut overall_decode_queue = Distribution::default();
    let mut epoch_min_waiting = usize::MAX;
    let mut epoch_max_waiting = 0usize;
    let mut last_cause_totals = CauseTotals::default();
    let mut stop_reason = None;

    eprintln!(
        "starting {:.3}-day run: lambda/rank={:.6}, lambda/node={:.6}, prefill HBM={:.3}x, sample/log every {:.0}/{:.0} simulated seconds, stop at waiting {}, {} consecutive growing epochs{}",
        args.duration_secs / 86_400.0,
        args.lambda_rank,
        args.lambda_rank * DPA_RANKS as f64,
        args.prefill_hbm_factor,
        args.queue_sample_interval_secs,
        args.log_interval_secs,
        args.max_waiting,
        args.growth_epochs,
        if args.stop_on_ttft_breach {
            ", or a completed-request interval TTFT quantile breach"
        } else {
            ""
        },
    );

    simulator.run_until_deadline_with_control_interval(args.queue_sample_interval_secs, |info| {
        for &sample in info.latency_samples.ttft.values {
            interval_ttft.push(sample);
        }
        let prefill_waiting = pool_waiting(&info.metrics, "prefill");
        let decode_waiting = pool_waiting(&info.metrics, "decode");
        interval_prefill_queue.push(prefill_waiting as f64);
        interval_decode_queue.push(decode_waiting as f64);
        overall_prefill_queue.push(prefill_waiting as f64);
        overall_decode_queue.push(decode_waiting as f64);
        epoch_min_waiting = epoch_min_waiting.min(info.waiting);
        epoch_max_waiting = epoch_max_waiting.max(info.waiting);
        let waiting_limit_hit = epoch_max_waiting >= args.max_waiting;
        let regular_log =
            info.current_time + 1e-9 >= next_log || info.current_time >= args.duration_secs;
        if !regular_log && !waiting_limit_hit {
            return ControlFlow::Continue(());
        }

        let now = Instant::now();
        let wall_elapsed = started.elapsed().as_secs_f64();
        let recent_wall = now.duration_since(last_log_wall).as_secs_f64();
        let recent_sim = info.current_time - last_log_sim;
        let recent_speed = if recent_wall > 0.0 {
            recent_sim / recent_wall
        } else {
            f64::INFINITY
        };
        let eta = if recent_speed > 0.0 {
            (args.duration_secs - info.current_time).max(0.0) / recent_speed
        } else {
            f64::INFINITY
        };
        let output_tokens = info.metrics.work.output_tokens_generated;
        let output_rate = if recent_sim > 0.0 {
            (output_tokens - last_output_tokens) as f64 / recent_sim
        } else {
            0.0
        };
        let ttft_samples = interval_ttft.len();
        let ttft_quantile_ms = (!interval_ttft.is_empty())
            .then(|| interval_ttft.quantile(args.ttft_quantile) * 1000.0);
        let active_sessions = info.metrics.sessions.map(|sessions| sessions.unfinished);
        let prefill_queue = summarize_queue(&mut interval_prefill_queue, prefill_waiting);
        let decode_queue = summarize_queue(&mut interval_decode_queue, decode_waiting);
        let (cause_totals, interval_cause) =
            interval_cause(&info.metrics, last_cause_totals, recent_sim);
        let sample = ProgressSample {
            simulated_secs: info.current_time,
            simulated_days: info.current_time / 86_400.0,
            percent: info.current_time / args.duration_secs * 100.0,
            wall_elapsed_secs: wall_elapsed,
            recent_sim_secs_per_wall_sec: recent_speed,
            estimated_wall_secs_remaining: eta,
            requests_admitted: info.total_requests,
            requests_completed: info.completed_requests,
            running: info.running,
            waiting: info.waiting,
            prefill_queue,
            decode_queue,
            epoch_min_waiting,
            epoch_max_waiting,
            active_sessions,
            interval_output_tokens_per_sec: output_rate,
            interval_completed_ttft_samples: ttft_samples,
            interval_completed_ttft_quantile: args.ttft_quantile,
            interval_completed_ttft_quantile_ms: ttft_quantile_ms,
            interval_cause,
        };
        eprintln!(
            "  sim={:>6.2}d {:>6.2}% wall={:>7} speed={:>8.0}x eta={:>7} admitted={} completed={} running={} waiting={} waiting_min/max={}/{} queue_p99[prefill={:.1},decode={:.1}] queue_now[prefill={},decode={}] active_sessions={} output={:.1} tok/s interval_ttft_q{:.3}={} prefill_Mtok[computed={:.1},new={:.1},parent_prefill={:.1},parent_decode={:.1},other={:.1}] reusable_miss={:.1}% evicted_GB[prefill={:.1},decode={:.1}]",
            sample.simulated_days,
            sample.percent,
            format_wall(sample.wall_elapsed_secs),
            sample.recent_sim_secs_per_wall_sec,
            format_wall(sample.estimated_wall_secs_remaining),
            sample.requests_admitted,
            sample.requests_completed,
            sample.running,
            sample.waiting,
            sample.epoch_min_waiting,
            sample.epoch_max_waiting,
            sample.prefill_queue.p99_waiting,
            sample.decode_queue.p99_waiting,
            sample.prefill_queue.current_waiting,
            sample.decode_queue.current_waiting,
            sample
                .active_sessions
                .map_or_else(|| "n/a".into(), |value| value.to_string()),
            sample.interval_output_tokens_per_sec,
            sample.interval_completed_ttft_quantile,
            sample.interval_completed_ttft_quantile_ms.map_or_else(
                || "n/a".into(),
                |value| format!("{:.3}s", value / 1000.0)
            ),
            sample.interval_cause.prefill_tokens_computed as f64 / 1e6,
            sample.interval_cause.new_prompt_tokens as f64 / 1e6,
            sample.interval_cause.parent_prefill_recompute_tokens as f64 / 1e6,
            sample.interval_cause.parent_decode_recompute_tokens as f64 / 1e6,
            sample.interval_cause.unattributed_recompute_tokens as f64 / 1e6,
            sample.interval_cause.prefiller_miss_reusable_token_fraction * 100.0,
            sample.interval_cause.prefill_evicted_bytes as f64 / 1e9,
            sample.interval_cause.decode_evicted_bytes as f64 / 1e9,
        );
        let ttft_breach_reason = (args.stop_on_ttft_breach
            && sample
                .interval_completed_ttft_quantile_ms
                .is_some_and(|observed_ms| observed_ms > TTFT_SLO_SECS * 1000.0))
        .then(|| StopReason::CompletedTtftSloBreach {
            interval_secs: recent_sim,
            completed_samples: sample.interval_completed_ttft_samples,
            quantile: args.ttft_quantile,
            observed_ms: sample.interval_completed_ttft_quantile_ms.unwrap(),
            limit_ms: TTFT_SLO_SECS * 1000.0,
        });
        progress.push(sample);

        let growth_reason = if progress.len() > args.growth_epochs {
            let start = progress.len() - args.growth_epochs - 1;
            let endpoints = &progress[start..];
            let increasing = endpoints
                .windows(2)
                .all(|pair| pair[0].waiting < pair[1].waiting);
            let stayed_nonzero = endpoints[1..]
                .iter()
                .all(|sample| sample.epoch_min_waiting > 0);
            (increasing && stayed_nonzero).then(|| StopReason::ConsecutiveQueueGrowth {
                epochs: args.growth_epochs,
                start_waiting: endpoints[0].waiting,
                end_waiting: endpoints.last().unwrap().waiting,
            })
        } else {
            None
        };
        let reason = waiting_limit_hit
            .then_some(StopReason::MaxWaiting {
                limit: args.max_waiting,
                observed: epoch_max_waiting,
            })
            .or(growth_reason)
            .or(ttft_breach_reason);
        if let Some(reason) = reason {
            eprintln!("  EARLY STOP: {reason:?}");
            stop_reason = Some(reason);
            return ControlFlow::Break(());
        }

        interval_ttft = Distribution::default();
        interval_prefill_queue = Distribution::default();
        interval_decode_queue = Distribution::default();
        epoch_min_waiting = info.waiting;
        epoch_max_waiting = info.waiting;
        last_log_sim = info.current_time;
        last_log_wall = now;
        last_output_tokens = output_tokens;
        last_cause_totals = cause_totals;
        while next_log <= info.current_time {
            next_log += args.log_interval_secs;
        }
        ControlFlow::Continue(())
    })?;

    let wall_elapsed_secs = started.elapsed().as_secs_f64();
    let actual_end_secs = simulator.current_time();
    let analysis_end_secs = (actual_end_secs / SLO_WINDOW_SECS).floor() * SLO_WINDOW_SECS;
    let ttft_analysis_cohorts = build_cohorts(
        simulator.admitted_arrival_times(),
        simulator.request_rows(),
        analysis_end_secs,
        ANALYSIS_COHORT_SECS,
    );
    let ttft_slo_windows = build_cohorts(
        simulator.admitted_arrival_times(),
        simulator.request_rows(),
        analysis_end_secs,
        SLO_WINDOW_SECS,
    );
    let request_cause_cohorts = build_request_cause_cohorts(
        simulator.request_rows(),
        actual_end_secs,
        args.log_interval_secs,
    );
    let endpoint_metrics = simulator.summary();
    let prefill_waiting = pool_waiting(&endpoint_metrics, "prefill");
    let decode_waiting = pool_waiting(&endpoint_metrics, "decode");
    let prefill_queue = summarize_queue(&mut overall_prefill_queue, prefill_waiting);
    let decode_queue = summarize_queue(&mut overall_decode_queue, decode_waiting);
    let completed_endpoint = actual_end_secs + 1e-9 >= args.duration_secs;
    let mut completed_ttft = Distribution::default();
    for row in simulator.request_rows() {
        completed_ttft.push(row.ttft);
    }
    let completed_ttft_quantile_ms =
        (!completed_ttft.is_empty()).then(|| completed_ttft.quantile(args.ttft_quantile) * 1000.0);
    let result = LongRunResult {
        description:
            "Empty-start run with arrivals enabled until the endpoint or an overload sanity gate, with no drain"
                .into(),
        lambda_rank_sessions_s: args.lambda_rank,
        lambda_node_sessions_s: args.lambda_rank * DPA_RANKS as f64,
        seed: SEED,
        prefill_hbm_factor: args.prefill_hbm_factor,
        endpoint_secs: args.duration_secs,
        endpoint_days: args.duration_secs / 86_400.0,
        actual_end_secs,
        actual_end_days: actual_end_secs / 86_400.0,
        completed_endpoint,
        stop_reason,
        log_interval_secs: args.log_interval_secs,
        queue_sample_interval_secs: args.queue_sample_interval_secs,
        max_waiting_limit: args.max_waiting,
        growth_epochs_limit: args.growth_epochs,
        stop_on_ttft_breach: args.stop_on_ttft_breach,
        ttft_quantile: args.ttft_quantile,
        ttft_slo_ms: TTFT_SLO_SECS * 1000.0,
        completed_ttft_quantile_ms,
        analysis_cohort_secs: ANALYSIS_COHORT_SECS,
        slo_window_secs: SLO_WINDOW_SECS,
        wall_elapsed_secs,
        prefill_queue,
        decode_queue,
        endpoint_metrics,
        progress,
        request_cause_cohorts,
        ttft_analysis_cohorts,
        ttft_slo_windows,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
    } else {
        println!(
            "stopped at {:.3}/{:.3} simulated days in {}: admitted {}, completed {}, waiting {}, queue p99 prefill/decode {:.1}/{:.1}, output {:.1} tok/s overall, reason {:?}",
            result.actual_end_days,
            result.endpoint_days,
            format_wall(result.wall_elapsed_secs),
            result.endpoint_metrics.requests.total,
            result.endpoint_metrics.requests.completed,
            result.progress.last().map_or(0, |sample| sample.waiting),
            result.prefill_queue.p99_waiting,
            result.decode_queue.p99_waiting,
            result.endpoint_metrics.work.output_tokens_generated as f64 / result.actual_end_secs,
            result.stop_reason,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(arrival: f64, completion: f64, ttft: f64) -> RequestRow {
        RequestRow {
            arrival,
            completion,
            ttft,
            e2e: completion - arrival,
            mean_tpot: f64::NAN,
            prompt_tokens: 1,
            output_tokens: 1,
            cached_tokens: 0,
            decode_cached_tokens: None,
            num_preemptions: 0,
            session: None,
            worker: None,
            gap: None,
            shared_tokens: None,
            reuse_distance_bytes: None,
            reuse_touched_bytes: None,
            lookup: None,
        }
    }

    #[test]
    fn endpoint_cohorts_report_right_censoring() {
        let arrivals = [1.0, 2.0, 121.0];
        let completed = [row(1.0, 5.0, 4.0), row(121.0, 125.0, 4.0)];
        let cohorts = build_cohorts(&arrivals, &completed, 240.0, 120.0);
        assert_eq!(cohorts.len(), 2);
        assert_eq!(cohorts[0].requests_admitted, 2);
        assert_eq!(cohorts[0].requests_completed_by_endpoint, 1);
        assert_eq!(cohorts[0].requests_censored_at_endpoint, 1);
        assert!(!cohorts[0].exact);
        assert!(cohorts[1].exact);
    }

    #[test]
    fn queue_summary_reports_interpolated_p99_and_zero_fraction() {
        let mut distribution = Distribution::default();
        for waiting in [0.0, 0.0, 2.0, 10.0] {
            distribution.push(waiting);
        }
        let summary = summarize_queue(&mut distribution, 10);
        assert_eq!(summary.samples, 4);
        assert_eq!(summary.current_waiting, 10);
        assert_eq!(summary.mean_waiting, 3.0);
        assert!((summary.p99_waiting - 9.76).abs() < 1e-12);
        assert_eq!(summary.max_waiting, 10);
        assert_eq!(summary.zero_sample_fraction, 0.5);
    }

    #[test]
    fn request_cause_cohorts_find_largest_reusable_miss_at_lookup_time() {
        let mut request = row(3_650.0, 3_800.0, 150.0);
        request.prompt_tokens = 100;
        request.cached_tokens = 20;
        request.session = Some((7, 3));
        request.shared_tokens = Some(80);
        request.lookup = Some(inference_lab::request::LookupRecord {
            at: 3_700.0,
            hbm_tokens: 20,
            ..Default::default()
        });
        let cohorts = build_request_cause_cohorts(&[request], 7_200.0, 3_600.0);
        assert_eq!(cohorts.len(), 2);
        assert_eq!(cohorts[0].completed_session_requests, 0);
        assert_eq!(cohorts[1].completed_session_requests, 1);
        assert_eq!(cohorts[1].reusable_miss_requests, 1);
        assert_eq!(cohorts[1].reusable_miss_tokens, 60);
        let largest = cohorts[1].largest_reusable_miss.as_ref().unwrap();
        assert_eq!(largest.reusable_miss_tokens, 60);
        assert_eq!((largest.session, largest.step), (7, 3));
    }

    #[test]
    fn prefill_hbm_factor_changes_only_prefill_hardware_capacity() {
        let base = config(0.0035, 600.0, 1.0);
        let increased = config(0.0035, 600.0, 2.0);
        assert!(base.prefill.as_ref().unwrap().hardware.is_none());
        assert_eq!(
            increased.hardware.memory_capacity,
            base.hardware.memory_capacity
        );
        assert_eq!(
            increased
                .prefill
                .as_ref()
                .unwrap()
                .hardware
                .as_ref()
                .unwrap()
                .memory_capacity,
            base.hardware.memory_capacity * 2
        );
    }
}
