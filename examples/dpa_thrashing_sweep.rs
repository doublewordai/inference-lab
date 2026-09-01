//! HBM-only load sweep for the KV-tier blog post.
//!
//! One disaggregated prefill replica feeds one decode replica. Both are
//! GLM-5.2-FP8 on 8xB200 with EP8 and data-parallel attention: eight
//! rank-local attention/KV domains sharing expert computation across the
//! replica. Sessions are statically assigned by ordinal, so each re-entry
//! returns to the same DPA rank. `lambda_rank` is the session-start rate per
//! rank; the node receives `8 * lambda_rank`.
//! The default tables report selected metrics; `--json` retains every run
//! and the complete core metrics summary.
//!
//! Run:
//!   cargo run --release --example dpa_thrashing_sweep
//!   cargo run --release --example dpa_thrashing_sweep -- --json
//!   cargo run --release --example dpa_thrashing_sweep -- --lambda 0.03
//!   cargo run --release --example dpa_thrashing_sweep -- --quick

use inference_lab::catalog;
use inference_lab::config::{
    ArrivalPattern, Config, LengthDistribution, MemoryConfig, ParallelConfig, PrefillSpec,
    RouterConfig, SchedulerConfig, WorkloadConfig,
};
use inference_lab::metrics::MetricsSummary;
use inference_lab::scheduler::SchedulingPolicy;
use inference_lab::simulation::Simulator;

const DPA_RANKS: u32 = 8;
const PREFILL_REPLICAS: u32 = 1;
const DEFAULT_DURATION_SECS: f64 = 600.0;
const SEEDS: &[u64] = &[42];
const LAMBDA_RANK_SWEEP: &[f64] = &[
    0.05, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
];

#[derive(Debug)]
struct Args {
    json: bool,
    quick: bool,
    lambda_rank: Option<f64>,
    duration_secs: Option<f64>,
}

fn parse_args() -> Args {
    let mut json = false;
    let mut quick = false;
    let mut lambda_rank = None;
    let mut duration_secs = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--json" => json = true,
            "--quick" => quick = true,
            "--lambda" => {
                let value = args.next().expect("--lambda requires a value");
                let parsed: f64 = value.parse().expect("--lambda must be a number");
                assert!(parsed > 0.0, "--lambda must be positive");
                lambda_rank = Some(parsed);
            }
            "--duration" => {
                let value = args.next().expect("--duration requires seconds");
                let parsed: f64 = value.parse().expect("--duration must be a number");
                assert!(parsed > 0.0, "--duration must be positive");
                duration_secs = Some(parsed);
            }
            _ => panic!("unknown argument {arg:?}"),
        }
    }
    Args {
        json,
        quick,
        lambda_rank,
        duration_secs,
    }
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
        // `tp` is currently the simulator's replica world-size field. With
        // DPA enabled, attention is rank-local rather than tensor-parallel.
        tp: DPA_RANKS,
        ep: DPA_RANKS,
        dp_attention: true,
    }
}

fn config(lambda_rank: f64, seed: u64, duration_secs: f64) -> Config {
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
            // The per-rank B200 NIC paths bound hand-offs.
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
            duration_secs: Some(duration_secs),
            seed,
        },
        speculative: None,
        fault: None,
    };
    config.finalize();
    config
}

#[derive(Debug, serde::Serialize)]
struct RunResult {
    lambda_rank_sessions_s: f64,
    lambda_node_sessions_s: f64,
    seed: u64,
    duration_secs: f64,
    metrics: MetricsSummary,
}

fn run_one(lambda_rank: f64, seed: u64, duration_secs: f64) -> RunResult {
    let mut simulator =
        Simulator::new(config(lambda_rank, seed, duration_secs), None).expect("build simulator");
    simulator.run_with_callback(|_| {}).expect("run simulator");
    RunResult {
        lambda_rank_sessions_s: lambda_rank,
        lambda_node_sessions_s: lambda_rank * DPA_RANKS as f64,
        seed,
        duration_secs,
        metrics: simulator.summary(),
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn percent(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64 * 100.0
    }
}

fn per_second_per_rank(value: u64, elapsed_secs: f64) -> f64 {
    if elapsed_secs == 0.0 {
        0.0
    } else {
        value as f64 / elapsed_secs / DPA_RANKS as f64
    }
}

fn main() {
    let args = parse_args();
    let lambdas: Vec<f64> = match args.lambda_rank {
        Some(lambda) => vec![lambda],
        None if args.quick => vec![0.03],
        None => LAMBDA_RANK_SWEEP.to_vec(),
    };
    let seeds: &[u64] = if args.quick { &SEEDS[..1] } else { SEEDS };
    let duration_secs = args.duration_secs.unwrap_or(if args.quick {
        30.0
    } else {
        DEFAULT_DURATION_SECS
    });

    let mut results = Vec::with_capacity(lambdas.len() * seeds.len());
    for &lambda_rank in &lambdas {
        for &seed in seeds {
            eprintln!(
                "lambda_rank={lambda_rank:.3}, lambda_node={:.3}, seed={seed}",
                lambda_rank * DPA_RANKS as f64,
            );
            results.push(run_one(lambda_rank, seed, duration_secs));
        }
    }

    if args.json {
        println!("{}", serde_json::to_string_pretty(&results).unwrap());
        return;
    }

    println!("Performance");
    println!(
        "{:<23} {:>18} {:>18} {:>18} {:>18} {:>21}",
        "Session starts/s/rank",
        format!("Output tok/s @{duration_secs:.0}s"),
        "p99 TTFT (ms)",
        "p50 TPOT (ms)",
        "p99 TPOT (ms)",
        format!("Unfinished @{duration_secs:.0}s"),
    );
    for &lambda_rank in &lambdas {
        let runs: Vec<_> = results
            .iter()
            .filter(|result| result.lambda_rank_sessions_s == lambda_rank)
            .collect();
        let output_at_deadline = median(
            runs.iter()
                .map(|result| {
                    result
                        .metrics
                        .simulation
                        .at_deadline
                        .expect("duration-limited run reports its deadline")
                        .output_tokens_per_sec
                })
                .collect(),
        );
        let ttft_p99 = median(
            runs.iter()
                .map(|result| result.metrics.latency_metrics.ttft_ms.p99)
                .collect(),
        );
        let tpot_p50 = median(
            runs.iter()
                .map(|result| result.metrics.latency_metrics.per_token_ms.p50)
                .collect(),
        );
        let tpot_p99 = median(
            runs.iter()
                .map(|result| result.metrics.latency_metrics.per_token_ms.p99)
                .collect(),
        );
        let unfinished = median(
            runs.iter()
                .map(|result| {
                    let deadline = result.metrics.simulation.at_deadline.unwrap();
                    (deadline.running + deadline.waiting + deadline.handoffs_in_flight) as f64
                })
                .collect(),
        );
        println!(
            "{:<23.3} {:>18.0} {:>18.1} {:>18.2} {:>18.2} {:>21.0}",
            lambda_rank, output_at_deadline, ttft_p99, tpot_p50, tpot_p99, unfinished,
        );
    }

    println!();
    println!("Prefill computation");
    println!(
        "{:<23} {:>23} {:>18} {:>31} {:>30} {:>26}",
        "Session starts/s/rank",
        "Computed tok/s/rank",
        "New prompt (%)",
        "Previous-output recompute (%)",
        "Evicted-context recompute (%)",
        "Preemption recompute (%)",
    );
    for &lambda_rank in &lambdas {
        let runs: Vec<_> = results
            .iter()
            .filter(|result| result.lambda_rank_sessions_s == lambda_rank)
            .collect();
        let accounting = |result: &&RunResult| {
            result
                .metrics
                .reusable_kv
                .as_ref()
                .expect("disaggregated session run reports reusable KV")
                .prefill_work
        };
        let computed_per_sec = median(
            runs.iter()
                .map(|result| {
                    per_second_per_rank(
                        result.metrics.work.prefill_tokens_computed,
                        result.metrics.simulation.end_time_secs,
                    )
                })
                .collect(),
        );
        let new_prompt = median(
            runs.iter()
                .map(|result| {
                    let r = accounting(result);
                    percent(
                        r.new_prompt.tokens,
                        result.metrics.work.prefill_tokens_computed,
                    )
                })
                .collect(),
        );
        let previous_output = median(
            runs.iter()
                .map(|result| {
                    let r = accounting(result);
                    percent(
                        r.parent_decode_recompute.tokens,
                        result.metrics.work.prefill_tokens_computed,
                    )
                })
                .collect(),
        );
        let evicted_context = median(
            runs.iter()
                .map(|result| {
                    let r = accounting(result);
                    percent(
                        r.parent_prefill_recompute.tokens + r.unattributed_recompute.tokens,
                        result.metrics.work.prefill_tokens_computed,
                    )
                })
                .collect(),
        );
        let preemption_recompute = median(
            runs.iter()
                .map(|result| {
                    let r = accounting(result);
                    let initially_computed = r.new_prompt.tokens
                        + r.parent_prefill_recompute.tokens
                        + r.parent_decode_recompute.tokens
                        + r.unattributed_recompute.tokens;
                    let recomputed = result
                        .metrics
                        .work
                        .prefill_tokens_computed
                        .checked_sub(initially_computed)
                        .expect("prefill work covers its initial token partition");
                    percent(recomputed, result.metrics.work.prefill_tokens_computed)
                })
                .collect(),
        );
        println!(
            "{:<23.3} {:>23.0} {:>17.1}% {:>30.1}% {:>29.1}% {:>25.1}%",
            lambda_rank,
            computed_per_sec,
            new_prompt,
            previous_output,
            evicted_context,
            preemption_recompute,
        );
    }

    println!();
    println!("Decoder cache and KV transfer");
    println!(
        "{:<23} {:>36} {:>28} {:>27} {:>32} {:>27}",
        "Session starts/s/rank",
        "Prior context present on decoder (%)",
        "KV transferred (GB/s/rank)",
        "New prompt KV (% transfer)",
        "From prefiller cache (% transfer)",
        "After recompute (% transfer)",
    );
    for &lambda_rank in &lambdas {
        let runs: Vec<_> = results
            .iter()
            .filter(|result| result.lambda_rank_sessions_s == lambda_rank)
            .collect();
        let accounting = |result: &&RunResult| {
            result
                .metrics
                .reusable_kv
                .as_ref()
                .expect("disaggregated session run reports reusable KV")
                .total
        };
        let decoder_hit = median(
            runs.iter()
                .map(|result| accounting(result).decoder_hit_byte_fraction * 100.0)
                .collect(),
        );
        let transfer_gb_per_sec = median(
            runs.iter()
                .map(|result| {
                    let handoff = result
                        .metrics
                        .handoff
                        .expect("disaggregated run reports hand-offs");
                    per_second_per_rank(handoff.bytes, result.metrics.simulation.end_time_secs)
                        / 1e9
                })
                .collect(),
        );
        let transfer_share = |result: &&RunResult| {
            let r = accounting(result);
            let handoff = result
                .metrics
                .handoff
                .expect("disaggregated run reports hand-offs");
            let from_prefiller = r.decoder_miss_prefill_hit.bytes;
            let after_recompute = r.decoder_miss_prefill_miss.bytes;
            let new_prompt = handoff
                .bytes
                .checked_sub(from_prefiller + after_recompute)
                .expect("hand-off bytes cover inherited transfers");
            (handoff.bytes, new_prompt, from_prefiller, after_recompute)
        };
        let new_prompt = median(
            runs.iter()
                .map(|result| {
                    let (total, new_prompt, _, _) = transfer_share(result);
                    percent(new_prompt, total)
                })
                .collect(),
        );
        let from_prefiller = median(
            runs.iter()
                .map(|result| {
                    let (total, _, from_prefiller, _) = transfer_share(result);
                    percent(from_prefiller, total)
                })
                .collect(),
        );
        let after_recompute = median(
            runs.iter()
                .map(|result| {
                    let (total, _, _, after_recompute) = transfer_share(result);
                    percent(after_recompute, total)
                })
                .collect(),
        );
        println!(
            "{:<23.3} {:>35.1}% {:>28.2} {:>26.1}% {:>31.1}% {:>26.1}%",
            lambda_rank,
            decoder_hit,
            transfer_gb_per_sec,
            new_prompt,
            from_prefiller,
            after_recompute,
        );
    }

    println!();
    println!("Session outcomes");
    println!(
        "{:<23} {:>23} {:>25} {:>18}",
        "Session starts/s/rank",
        "Full traces complete (%)",
        "Turns/started session",
        "Drain time (s)",
    );
    for &lambda_rank in &lambdas {
        let runs: Vec<_> = results
            .iter()
            .filter(|result| result.lambda_rank_sessions_s == lambda_rank)
            .collect();
        let sessions = |result: &&RunResult| {
            *result
                .metrics
                .sessions
                .as_ref()
                .expect("session workload reports session metrics")
        };
        let completion = median(
            runs.iter()
                .map(|result| {
                    let session = sessions(result);
                    if session.started == 0 {
                        0.0
                    } else {
                        session.completed as f64 / session.started as f64 * 100.0
                    }
                })
                .collect(),
        );
        let turns = median(
            runs.iter()
                .map(|result| sessions(result).turns_per_started_session.mean)
                .collect(),
        );
        let drain = median(
            runs.iter()
                .map(|result| result.metrics.simulation.drain_time_secs)
                .collect(),
        );
        println!(
            "{:<23.3} {:>22.1}% {:>25.2} {:>18.1}",
            lambda_rank, completion, turns, drain,
        );
    }
}
