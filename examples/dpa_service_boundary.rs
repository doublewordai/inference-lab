//! Sustainable-service boundary for the HBM-only DPA KV-cache experiment.
//!
//! A run discards the empty-cache transient by waiting until every prefill
//! and decode rank has cumulatively evicted one local HBM capacity. It then
//! observes consecutive 600-second arrival windows with arrivals still
//! enabled. Every window must meet the configured p99 TTFT SLO. Shorter
//! 120-second cohorts are also retained for diagnosing movement within each
//! SLO window. The simulator continues and drains, so the latency of every
//! request admitted to a cohort is observed rather than right-censored at the
//! cohort boundary.
//!
//! Run:
//!   cargo run --release --example dpa_service_boundary -- --json
//!   cargo run --release --example dpa_service_boundary -- --lambda 0.15
//!   cargo run --release --example dpa_service_boundary -- --sweep --json

use inference_lab::catalog;
use inference_lab::config::{
    ArrivalPattern, Config, LengthDistribution, MemoryConfig, ParallelConfig, PrefillSpec,
    RouterConfig, SchedulerConfig, WorkloadConfig,
};
use inference_lab::metrics::{Distribution, HbmWorkerMetrics, MetricsSummary};
use inference_lab::scheduler::SchedulingPolicy;
use inference_lab::simulation::Simulator;
use serde::Serialize;

const DPA_RANKS: u32 = 8;
const PREFILL_REPLICAS: u32 = 1;
const SEED: u64 = 42;
const DEFAULT_LOW: f64 = 0.02;
const DEFAULT_HIGH: f64 = 0.025;
const DEFAULT_RESOLUTION: f64 = 0.005;
const DEFAULT_MEASUREMENT_SECS: f64 = 7200.0;
const DEFAULT_MAX_ARRIVAL_SECS: f64 = 12_000.0;
const DEFAULT_TTFT_SLO_MS: f64 = 30_000.0;
const ANALYSIS_COHORT_SECS: f64 = 120.0;
const SLO_WINDOW_SECS: f64 = 600.0;
const LAMBDA_RANK_SWEEP: &[f64] = &[
    0.015, 0.0175, 0.020, 0.02125, 0.0225, 0.02375, 0.025, 0.0275, 0.030, 0.040,
];

#[derive(Debug)]
struct Args {
    json: bool,
    sweep: bool,
    lambda_rank: Option<f64>,
    low: f64,
    high: f64,
    resolution: f64,
    measurement_secs: f64,
    max_arrival_secs: f64,
    ttft_slo_ms: f64,
}

fn parse_args() -> Args {
    let mut args = Args {
        json: false,
        sweep: false,
        lambda_rank: None,
        low: DEFAULT_LOW,
        high: DEFAULT_HIGH,
        resolution: DEFAULT_RESOLUTION,
        measurement_secs: DEFAULT_MEASUREMENT_SECS,
        max_arrival_secs: DEFAULT_MAX_ARRIVAL_SECS,
        ttft_slo_ms: DEFAULT_TTFT_SLO_MS,
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
            "--sweep" => args.sweep = true,
            "--lambda" => args.lambda_rank = Some(value(&mut raw, "--lambda")),
            "--low" => args.low = value(&mut raw, "--low"),
            "--high" => args.high = value(&mut raw, "--high"),
            "--resolution" => args.resolution = value(&mut raw, "--resolution"),
            "--measurement" => args.measurement_secs = value(&mut raw, "--measurement"),
            "--max-arrival" => args.max_arrival_secs = value(&mut raw, "--max-arrival"),
            "--ttft-slo-ms" => args.ttft_slo_ms = value(&mut raw, "--ttft-slo-ms"),
            _ => panic!("unknown argument {arg:?}"),
        }
    }
    assert!(args.low > 0.0, "--low must be positive");
    assert!(args.high > args.low, "--high must exceed --low");
    assert!(args.resolution > 0.0, "--resolution must be positive");
    assert!(
        args.measurement_secs > 0.0,
        "--measurement must be positive"
    );
    assert!(
        ((args.measurement_secs / ANALYSIS_COHORT_SECS).round()
            - args.measurement_secs / ANALYSIS_COHORT_SECS)
            .abs()
            < 1e-9,
        "--measurement must be a multiple of 120 seconds"
    );
    assert!(
        ((args.measurement_secs / SLO_WINDOW_SECS).round()
            - args.measurement_secs / SLO_WINDOW_SECS)
            .abs()
            < 1e-9,
        "--measurement must be a multiple of 600 seconds"
    );
    assert!(
        args.max_arrival_secs > args.measurement_secs,
        "--max-arrival must exceed --measurement"
    );
    assert!(args.ttft_slo_ms > 0.0, "--ttft-slo-ms must be positive");
    if let Some(lambda) = args.lambda_rank {
        assert!(lambda > 0.0, "--lambda must be positive");
    }
    assert!(
        !(args.sweep && args.lambda_rank.is_some()),
        "--sweep and --lambda are mutually exclusive"
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
    }
}

fn parallel() -> ParallelConfig {
    ParallelConfig {
        tp: DPA_RANKS,
        ep: DPA_RANKS,
        dp_attention: true,
    }
}

fn config(lambda_rank: f64, max_arrival_secs: f64) -> Config {
    let parallel = parallel();
    let mut config = Config {
        hardware: catalog::hardware("b200").expect("b200 in catalog"),
        parallel: parallel.clone(),
        model: catalog::model("glm-5.2-fp8").expect("glm-5.2-fp8 in catalog"),
        scheduler: scheduler(),
        replicas: 1,
        router: RouterConfig::SessionAffinity {},
        decode_router: Some(RouterConfig::SessionAffinity {}),
        memory: MemoryConfig::default(),
        prefill: Some(PrefillSpec {
            hardware: None,
            parallel: Some(parallel),
            replicas: PREFILL_REPLICAS,
            memory: MemoryConfig::default(),
            kv_link_bw: None,
        }),
        time_correction: None,
        workload: WorkloadConfig {
            dataset_path: None,
            sessions_path: Some("data/sessions/tracelab.jsonl".into()),
            num_sessions: None,
            stationary_start_sessions: None,
            resample_sessions: true,
            arrival_pattern: ArrivalPattern::Poisson,
            arrival_rate: lambda_rank * DPA_RANKS as f64,
            rate_schedule: None,
            num_concurrent_users: None,
            closed_loop_jitter_secs: None,
            input_len_dist: LengthDistribution::Fixed { value: 1 },
            output_len_dist: LengthDistribution::Fixed { value: 1 },
            num_requests: None,
            duration_secs: Some(max_arrival_secs),
            seed: SEED,
        },
        speculative: None,
        fault: None,
    };
    config.finalize();
    config
}

#[derive(Debug, Clone, Serialize)]
struct WorkerLabel {
    role: String,
    worker: u32,
    rank: u32,
    capacity_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
struct QueueSample {
    elapsed_secs: f64,
    output_tokens: u64,
    waiting: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct WorkerResult {
    role: String,
    worker: u32,
    rank: u32,
    start_waiting: u64,
    end_waiting: u64,
    max_waiting: u64,
    zero_sample_fraction: f64,
}

#[derive(Debug, Clone, Serialize)]
struct TtftCohort {
    index: usize,
    start_secs: f64,
    end_secs: f64,
    requests: usize,
    p50_ms: f64,
    p90_ms: f64,
    p99_ms: f64,
    max_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct BoundaryRun {
    lambda_rank_sessions_s: f64,
    lambda_node_sessions_s: f64,
    seed: u64,
    turnover_secs: f64,
    measurement_secs: f64,
    output_tokens: f64,
    output_tokens_per_sec: f64,
    ttft_slo_ms: f64,
    ttft_samples: usize,
    ttft_p99_ms: f64,
    ttft_analysis_cohorts: Vec<TtftCohort>,
    ttft_slo_windows: Vec<TtftCohort>,
    tpot_samples: usize,
    tpot_p50_ms: f64,
    tpot_p99_ms: f64,
    slo_met: bool,
    workers: Vec<WorkerResult>,
    worker_order: Vec<WorkerLabel>,
    queue_samples: Vec<QueueSample>,
}

#[derive(Debug, Serialize)]
struct SearchResult {
    criterion: String,
    ttft_slo_ms: f64,
    analysis_cohort_secs: f64,
    analysis_cohorts: usize,
    slo_window_secs: f64,
    slo_windows: usize,
    resolution_sessions_s_per_rank: f64,
    last_slo_met_lambda_rank: f64,
    first_slo_breach_lambda_rank: f64,
    last_slo_met_output_tokens_per_sec: f64,
    runs: Vec<BoundaryRun>,
}

#[derive(Debug, Serialize)]
struct SweepResult {
    criterion: String,
    ttft_slo_ms: f64,
    analysis_cohort_secs: f64,
    analysis_cohorts: usize,
    slo_window_secs: f64,
    slo_windows: usize,
    lambda_rank_sessions_s: &'static [f64],
    runs: Vec<BoundaryRun>,
}

fn flatten_workers(summary: &MetricsSummary) -> Vec<(String, HbmWorkerMetrics)> {
    summary
        .hbm
        .pools
        .iter()
        .flat_map(|pool| {
            pool.workers
                .iter()
                .copied()
                .map(|worker| (pool.role.clone(), worker))
        })
        .collect()
}

fn interpolated_output(samples: &[QueueSample], at: f64) -> f64 {
    let before = samples
        .iter()
        .rev()
        .find(|sample| sample.elapsed_secs <= at)
        .expect("measurement has a sample before its endpoint");
    let Some(after) = samples.iter().find(|sample| sample.elapsed_secs >= at) else {
        return before.output_tokens as f64;
    };
    if after.elapsed_secs == before.elapsed_secs {
        return before.output_tokens as f64;
    }
    let fraction = (at - before.elapsed_secs) / (after.elapsed_secs - before.elapsed_secs);
    before.output_tokens as f64
        + fraction * (after.output_tokens as f64 - before.output_tokens as f64)
}

fn summarize_ttft_cohorts(
    samples: &[(f64, f64)],
    measurement_secs: f64,
    cohort_secs: f64,
) -> Result<Vec<TtftCohort>, String> {
    let num_cohorts = (measurement_secs / cohort_secs).round() as usize;
    let mut distributions: Vec<Distribution> =
        (0..num_cohorts).map(|_| Distribution::default()).collect();
    for &(elapsed_arrival, ttft_secs) in samples {
        let cohort = ((elapsed_arrival / cohort_secs).floor() as usize).min(num_cohorts - 1);
        distributions[cohort].push(ttft_secs);
    }
    if let Some(index) = distributions.iter().position(Distribution::is_empty) {
        return Err(format!(
            "empty {cohort_secs:.0}-second TTFT cohort at index {index}"
        ));
    }
    Ok(distributions
        .into_iter()
        .enumerate()
        .map(|(index, mut distribution)| TtftCohort {
            index,
            start_secs: index as f64 * cohort_secs,
            end_secs: (index + 1) as f64 * cohort_secs,
            requests: distribution.len(),
            p50_ms: distribution.quantile(0.50) * 1000.0,
            p90_ms: distribution.quantile(0.90) * 1000.0,
            p99_ms: distribution.quantile(0.99) * 1000.0,
            max_ms: distribution.quantile(1.0) * 1000.0,
        })
        .collect())
}

fn run_one(
    lambda_rank: f64,
    measurement_secs: f64,
    max_arrival_secs: f64,
    ttft_slo_ms: f64,
) -> Result<BoundaryRun, String> {
    eprintln!(
        "lambda_rank={lambda_rank:.6}, lambda_node={:.6}",
        lambda_rank * DPA_RANKS as f64
    );
    let mut simulator = Simulator::new(config(lambda_rank, max_arrival_secs), None)?;
    let mut turnover_time = None;
    let mut worker_order: Vec<WorkerLabel> = Vec::new();
    let mut queue_samples = Vec::new();

    simulator.run_with_callback(|info| {
        let workers = flatten_workers(&info.metrics);
        if turnover_time.is_none()
            && !workers.is_empty()
            && workers
                .iter()
                .all(|(_, worker)| worker.evicted_bytes >= worker.capacity_bytes)
        {
            turnover_time = Some(info.current_time);
            worker_order = workers
                .iter()
                .map(|(role, worker)| WorkerLabel {
                    role: role.clone(),
                    worker: worker.worker,
                    rank: worker.rank,
                    capacity_bytes: worker.capacity_bytes,
                })
                .collect();
            eprintln!("  turnover at {:.3}s", info.current_time);
        }
        let Some(turnover) = turnover_time else {
            return;
        };
        let elapsed = info.current_time - turnover;
        if elapsed > measurement_secs + 2.0 {
            return;
        }
        assert_eq!(workers.len(), worker_order.len());
        for ((role, worker), expected) in workers.iter().zip(&worker_order) {
            assert_eq!(
                (role.as_str(), worker.worker, worker.rank),
                (expected.role.as_str(), expected.worker, expected.rank)
            );
        }
        queue_samples.push(QueueSample {
            elapsed_secs: elapsed,
            output_tokens: info.metrics.work.output_tokens_generated,
            waiting: workers.iter().map(|(_, worker)| worker.waiting).collect(),
        });
    })?;

    let turnover_secs = turnover_time.ok_or_else(|| {
        format!("lambda {lambda_rank:.6} did not turn over every HBM by {max_arrival_secs:.0}s")
    })?;
    if turnover_secs + measurement_secs > max_arrival_secs {
        return Err(format!(
            "lambda {lambda_rank:.6} turns over at {turnover_secs:.1}s, so arrivals stop before the {measurement_secs:.0}s measurement ends; increase --max-arrival"
        ));
    }
    let last_elapsed = queue_samples
        .last()
        .map(|sample| sample.elapsed_secs)
        .unwrap_or(0.0);
    if last_elapsed < measurement_secs {
        return Err(format!(
            "lambda {lambda_rank:.6} has only {last_elapsed:.1}s after turnover; increase --max-arrival"
        ));
    }

    let start_output = queue_samples[0].output_tokens as f64;
    let end_output = interpolated_output(&queue_samples, measurement_secs);
    let output_tokens = end_output - start_output;
    let workers = worker_order
        .iter()
        .enumerate()
        .map(|(worker_idx, label)| {
            let within_window = queue_samples
                .iter()
                .filter(|sample| sample.elapsed_secs <= measurement_secs)
                .collect::<Vec<_>>();
            let start_waiting = queue_samples[0].waiting[worker_idx];
            let end_waiting = queue_samples
                .iter()
                .rev()
                .find(|sample| sample.elapsed_secs <= measurement_secs)
                .expect("measurement has an endpoint queue sample")
                .waiting[worker_idx];
            let max_waiting = within_window
                .iter()
                .map(|sample| sample.waiting[worker_idx])
                .max()
                .unwrap_or(0);
            let zero_samples = within_window
                .iter()
                .filter(|sample| sample.waiting[worker_idx] == 0)
                .count();
            WorkerResult {
                role: label.role.clone(),
                worker: label.worker,
                rank: label.rank,
                start_waiting,
                end_waiting,
                max_waiting,
                zero_sample_fraction: zero_samples as f64 / within_window.len() as f64,
            }
        })
        .collect::<Vec<_>>();

    let end_time = turnover_secs + measurement_secs;
    let mut ttft = Distribution::default();
    let mut tpot = Distribution::default();
    let mut ttft_samples = Vec::new();
    for row in simulator.request_rows() {
        if row.arrival >= turnover_secs && row.arrival < end_time {
            ttft.push(row.ttft);
            ttft_samples.push((row.arrival - turnover_secs, row.ttft));
            if row.mean_tpot.is_finite() {
                tpot.push(row.mean_tpot);
            }
        }
    }
    let ttft_analysis_cohorts =
        summarize_ttft_cohorts(&ttft_samples, measurement_secs, ANALYSIS_COHORT_SECS)
            .map_err(|error| format!("lambda {lambda_rank:.6} has an {error}"))?;
    let ttft_slo_windows = summarize_ttft_cohorts(&ttft_samples, measurement_secs, SLO_WINDOW_SECS)
        .map_err(|error| format!("lambda {lambda_rank:.6} has an {error}"))?;
    let slo_met = ttft_slo_windows
        .iter()
        .all(|window| window.p99_ms <= ttft_slo_ms);

    let result = BoundaryRun {
        lambda_rank_sessions_s: lambda_rank,
        lambda_node_sessions_s: lambda_rank * DPA_RANKS as f64,
        seed: SEED,
        turnover_secs,
        measurement_secs,
        output_tokens,
        output_tokens_per_sec: output_tokens / measurement_secs,
        ttft_slo_ms,
        ttft_samples: ttft.len(),
        ttft_p99_ms: ttft.quantile(0.99) * 1000.0,
        ttft_analysis_cohorts,
        ttft_slo_windows,
        tpot_samples: tpot.len(),
        tpot_p50_ms: tpot.quantile(0.50) * 1000.0,
        tpot_p99_ms: tpot.quantile(0.99) * 1000.0,
        slo_met,
        workers,
        worker_order,
        queue_samples,
    };
    eprintln!(
        "  {}: {:.1} output tok/s, 600s-window p99 TTFTs {:?}ms",
        if result.slo_met {
            "SLO MET"
        } else {
            "SLO BREACH"
        },
        result.output_tokens_per_sec,
        result
            .ttft_slo_windows
            .iter()
            .map(|window| window.p99_ms.round())
            .collect::<Vec<_>>()
    );
    Ok(result)
}

fn print_run(run: &BoundaryRun) {
    println!(
        "lambda/rank {:.6}: {:<10} {:>8.1} output tok/s, turnover {:>7.1}s, 600s-window p99 TTFTs {:?}ms, p50/p99 TPOT {:>6.2}/{:>6.2}ms",
        run.lambda_rank_sessions_s,
        if run.slo_met { "SLO MET" } else { "SLO BREACH" },
        run.output_tokens_per_sec,
        run.turnover_secs,
        run.ttft_slo_windows
            .iter()
            .map(|window| window.p99_ms.round())
            .collect::<Vec<_>>(),
        run.tpot_p50_ms,
        run.tpot_p99_ms,
    );
    for worker in &run.workers {
        println!(
            "  {:>7} rank {}: waiting {} -> {} (max {}), empty {:.1}% of samples",
            worker.role,
            worker.rank,
            worker.start_waiting,
            worker.end_waiting,
            worker.max_waiting,
            worker.zero_sample_fraction * 100.0,
        );
    }
}

fn main() -> Result<(), String> {
    let args = parse_args();
    if let Some(lambda) = args.lambda_rank {
        let run = run_one(
            lambda,
            args.measurement_secs,
            args.max_arrival_secs,
            args.ttft_slo_ms,
        )?;
        if args.json {
            println!("{}", serde_json::to_string_pretty(&run).unwrap());
        } else {
            print_run(&run);
        }
        return Ok(());
    }

    if args.sweep {
        let runs = LAMBDA_RANK_SWEEP
            .iter()
            .map(|&lambda| {
                run_one(
                    lambda,
                    args.measurement_secs,
                    args.max_arrival_secs,
                    args.ttft_slo_ms,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let result = SweepResult {
            criterion: format!(
                "p99 TTFT <= {:.0} ms in each of {} consecutive {:.0}-second post-turnover arrival windows",
                args.ttft_slo_ms,
                (args.measurement_secs / SLO_WINDOW_SECS).round() as usize,
                SLO_WINDOW_SECS,
            ),
            ttft_slo_ms: args.ttft_slo_ms,
            analysis_cohort_secs: ANALYSIS_COHORT_SECS,
            analysis_cohorts: (args.measurement_secs / ANALYSIS_COHORT_SECS).round() as usize,
            slo_window_secs: SLO_WINDOW_SECS,
            slo_windows: (args.measurement_secs / SLO_WINDOW_SECS).round() as usize,
            lambda_rank_sessions_s: LAMBDA_RANK_SWEEP,
            runs,
        };
        if args.json {
            println!("{}", serde_json::to_string_pretty(&result).unwrap());
        } else {
            for run in &result.runs {
                print_run(run);
            }
        }
        return Ok(());
    }

    let mut runs = Vec::new();
    let low = run_one(
        args.low,
        args.measurement_secs,
        args.max_arrival_secs,
        args.ttft_slo_ms,
    )?;
    if !low.slo_met {
        return Err(format!(
            "lower bracket {:.6} breaches the TTFT SLO; choose a lower --low",
            args.low
        ));
    }
    runs.push(low);
    let high = run_one(
        args.high,
        args.measurement_secs,
        args.max_arrival_secs,
        args.ttft_slo_ms,
    )?;
    if high.slo_met {
        return Err(format!(
            "upper bracket {:.6} meets the TTFT SLO; choose a higher --high",
            args.high
        ));
    }
    runs.push(high);

    let (mut slo_met, mut slo_breach) = (args.low, args.high);
    while slo_breach - slo_met > args.resolution {
        let candidate = (slo_met + slo_breach) / 2.0;
        let run = run_one(
            candidate,
            args.measurement_secs,
            args.max_arrival_secs,
            args.ttft_slo_ms,
        )?;
        if run.slo_met {
            slo_met = candidate;
        } else {
            slo_breach = candidate;
        }
        runs.push(run);
    }
    let slo_run = runs
        .iter()
        .find(|run| run.lambda_rank_sessions_s == slo_met)
        .expect("SLO-met endpoint was measured");
    let result = SearchResult {
        criterion: format!(
            "p99 TTFT <= {:.0} ms in each of {} consecutive {:.0}-second post-turnover arrival cohorts",
            args.ttft_slo_ms,
            (args.measurement_secs / SLO_WINDOW_SECS).round() as usize,
            SLO_WINDOW_SECS,
        ),
        ttft_slo_ms: args.ttft_slo_ms,
        analysis_cohort_secs: ANALYSIS_COHORT_SECS,
        analysis_cohorts: (args.measurement_secs / ANALYSIS_COHORT_SECS).round() as usize,
        slo_window_secs: SLO_WINDOW_SECS,
        slo_windows: (args.measurement_secs / SLO_WINDOW_SECS).round() as usize,
        resolution_sessions_s_per_rank: args.resolution,
        last_slo_met_lambda_rank: slo_met,
        first_slo_breach_lambda_rank: slo_breach,
        last_slo_met_output_tokens_per_sec: slo_run.output_tokens_per_sec,
        runs,
    };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&result).unwrap());
    } else {
        for run in &result.runs {
            print_run(run);
        }
        println!(
            "boundary: TTFT SLO met through {:.6}, breached by {:.6} sessions/s/rank; {:.1} output tok/s at the last SLO-met point",
            result.last_slo_met_lambda_rank,
            result.first_slo_breach_lambda_rank,
            result.last_slo_met_output_tokens_per_sec,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(elapsed_secs: f64, output_tokens: u64, waiting: u64) -> QueueSample {
        QueueSample {
            elapsed_secs,
            output_tokens,
            waiting: vec![waiting],
        }
    }

    #[test]
    fn output_counter_is_interpolated_at_the_exact_window_end() {
        let samples = [sample(0.0, 100, 0), sample(1.2, 220, 0)];
        assert!((interpolated_output(&samples, 1.0) - 200.0).abs() < 1e-9);
    }

    #[test]
    fn ttft_samples_are_grouped_by_arrival_time() {
        let samples = [(0.0, 1.0), (119.999, 2.0), (120.0, 3.0), (239.0, 4.0)];
        let cohorts = summarize_ttft_cohorts(&samples, 240.0, 120.0).unwrap();
        assert_eq!(cohorts.len(), 2);
        assert_eq!(cohorts[0].requests, 2);
        assert_eq!(cohorts[1].requests, 2);
        assert_eq!(cohorts[0].start_secs, 0.0);
        assert_eq!(cohorts[1].start_secs, 120.0);
        assert_eq!(cohorts[1].max_ms, 4000.0);
    }
}
