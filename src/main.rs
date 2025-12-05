use clap::Parser;
use sim::{Config, Simulator};
use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "cli")]
use colored::Colorize;
#[cfg(feature = "cli")]
use tabled::{
    settings::Style,
    Table, Tabled,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "LLM Inference Simulator", long_about = None)]
struct Args {
    /// Path to the TOML configuration file
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,

    /// Minimal output (final metrics only)
    #[arg(short, long)]
    quiet: bool,

    /// Show detailed progress during simulation
    #[arg(short, long)]
    verbose: bool,

    /// Very verbose debug output
    #[arg(long)]
    debug: bool,

    /// Disable colored output
    #[arg(long)]
    no_color: bool,

    /// Save metrics to JSON file
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum VerbosityLevel {
    Quiet,
    Normal,
    Verbose,
    Debug,
}

impl Args {
    fn verbosity_level(&self) -> VerbosityLevel {
        if self.debug {
            VerbosityLevel::Debug
        } else if self.verbose {
            VerbosityLevel::Verbose
        } else if self.quiet {
            VerbosityLevel::Quiet
        } else {
            VerbosityLevel::Normal
        }
    }
}

#[cfg(feature = "cli")]
#[derive(Tabled)]
struct LatencyRow {
    #[tabled(rename = "Metric")]
    metric: String,
    #[tabled(rename = "Mean")]
    mean: String,
    #[tabled(rename = "p50")]
    p50: String,
    #[tabled(rename = "p90")]
    p90: String,
    #[tabled(rename = "p99")]
    p99: String,
}

#[cfg(feature = "cli")]
#[derive(Tabled)]
struct ThroughputRow {
    #[tabled(rename = "Metric")]
    metric: String,
    #[tabled(rename = "Mean")]
    mean: String,
    #[tabled(rename = "p50")]
    p50: String,
    #[tabled(rename = "p90")]
    p90: String,
    #[tabled(rename = "p99")]
    p99: String,
}

fn main() {
    env_logger::init();

    let args = Args::parse();
    let verbosity = args.verbosity_level();
    let use_color = !args.no_color;

    // Header
    if verbosity >= VerbosityLevel::Normal {
        if use_color {
            println!("{}", "LLM Inference Simulator".bright_cyan().bold());
        } else {
            println!("LLM Inference Simulator");
        }
        println!("Loading configuration from: {:?}\n", args.config);
    }

    // Load configuration
    let config = match Config::from_file(&args.config) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Error loading configuration: {}", e);
            std::process::exit(1);
        }
    };

    // Print configuration summary
    if verbosity >= VerbosityLevel::Normal {
        if use_color {
            println!("{}", "Configuration:".green().bold());
        } else {
            println!("Configuration:");
        }
        println!("  Hardware: {}", config.hardware.name);
        println!("  Model: {}", config.model.name);
        println!(
            "  Compute-bound threshold: {} tokens",
            config.hardware.compute_bound_threshold
        );
        println!(
            "  Max batched tokens: {}",
            config.scheduler.max_num_batched_tokens
        );
        if config.scheduler.enable_chunked_prefill {
            println!(
                "  Chunked prefill enabled: {} tokens/chunk",
                config.scheduler.long_prefill_token_threshold
            );
        }
        println!("  Arrival rate: {} req/sec", config.workload.arrival_rate);
        println!(
            "  Number of requests: {}",
            config
                .workload
                .num_requests
                .map(|n| n.to_string())
                .unwrap_or_else(|| "unlimited".to_string())
        );
        println!();
    }

    // Create simulator
    let mut simulator = match Simulator::new(config.clone()) {
        Ok(sim) => sim,
        Err(e) => {
            eprintln!("Error creating simulator: {}", e);
            std::process::exit(1);
        }
    };

    let start_time = Instant::now();

    // Run simulation based on verbosity
    match verbosity {
        VerbosityLevel::Quiet => {
            run_quiet(&mut simulator, use_color);
        }
        VerbosityLevel::Normal => {
            run_with_dashboard(&mut simulator, use_color, &config);
        }
        VerbosityLevel::Verbose => {
            run_verbose(&mut simulator, use_color, &config);
        }
        VerbosityLevel::Debug => {
            // Debug mode uses the old run() method which has detailed logging
            simulator.run();
            let elapsed = start_time.elapsed();
            if verbosity >= VerbosityLevel::Normal {
                println!(
                    "\nSimulation completed in {:.2}s (real time)",
                    elapsed.as_secs_f64()
                );
            }
            // Metrics already printed by run()
            return;
        }
    }

    let elapsed = start_time.elapsed();

    // Print final metrics
    let summary = simulator.get_metrics_summary();
    print_final_metrics(&summary, simulator.get_current_time(), elapsed, verbosity, use_color);

    // Save to JSON if requested
    if let Some(output_path) = args.output {
        match save_metrics_json(&summary, &output_path) {
            Ok(_) => {
                if verbosity >= VerbosityLevel::Normal {
                    println!("\nMetrics saved to: {:?}", output_path);
                }
            }
            Err(e) => {
                eprintln!("Error saving metrics to JSON: {}", e);
            }
        }
    }
}

fn run_quiet(simulator: &mut Simulator, _use_color: bool) {
    simulator
        .run_with_callback(|_progress| {
            // No output during simulation
        })
        .unwrap();
}

fn run_with_dashboard(simulator: &mut Simulator, use_color: bool, config: &Config) {
    let total_requests = config.workload.num_requests.unwrap_or(1000) as u64;

    if use_color {
        println!("{}", "━".repeat(60).bright_black());
        println!("{}", "Simulation Progress".bright_cyan().bold());
        println!("{}", "━".repeat(60).bright_black());
    } else {
        println!("{}", "━".repeat(60));
        println!("Simulation Progress");
        println!("{}", "━".repeat(60));
    }

    let mut first_update = true;
    let num_lines = 5; // Number of lines the dashboard uses (including final separator)

    simulator
        .run_with_callback(|progress| {
            let percent = (progress.completed_requests as f64 / total_requests as f64 * 100.0)
                .min(100.0);
            let bar_width = 40;
            let filled = (bar_width as f64 * percent / 100.0) as usize;
            let bar: String = "█".repeat(filled) + &"░".repeat(bar_width - filled);

            // Clear previous dashboard (move cursor up and clear lines)
            if !first_update {
                // ANSI escape: move cursor up N lines and clear from cursor to end of screen
                print!("\x1B[{}A\x1B[J", num_lines);
            }
            first_update = false;

            if use_color {
                println!(
                    "  Progress: [{}] {}/{} ({:.0}%)",
                    bar.cyan(),
                    progress.completed_requests,
                    total_requests,
                    percent
                );
                println!(
                    "  Time:     {:.1}s simulated",
                    progress.current_time.to_string().yellow()
                );
                println!(
                    "  Queue:    {} running, {} waiting",
                    progress.running.to_string().green(),
                    progress.waiting.to_string().blue()
                );
                println!(
                    "  KV Cache: {:.1}% utilized",
                    (progress.kv_cache_util * 100.0).to_string().magenta()
                );
                println!("{}", "━".repeat(60).bright_black());
            } else {
                println!(
                    "  Progress: [{}] {}/{} ({:.0}%)",
                    bar, progress.completed_requests, total_requests, percent
                );
                println!("  Time:     {:.1}s simulated", progress.current_time);
                println!(
                    "  Queue:    {} running, {} waiting",
                    progress.running, progress.waiting
                );
                println!("  KV Cache: {:.1}% utilized", progress.kv_cache_util * 100.0);
                println!("{}", "━".repeat(60));
            }
        })
        .unwrap();
}

fn run_verbose(simulator: &mut Simulator, use_color: bool, config: &Config) {
    let total_requests = config.workload.num_requests.unwrap_or(1000) as u64;

    if use_color {
        println!("{}", "Starting simulation...".green());
    } else {
        println!("Starting simulation...");
    }

    simulator
        .run_with_callback(|progress| {
            println!(
                "[{:.1}s] {}/{} requests | {} running, {} waiting | KV: {:.1}% | FLOPS: {:.1}% | BW: {:.1}%",
                progress.current_time,
                progress.completed_requests,
                total_requests,
                progress.running,
                progress.waiting,
                progress.kv_cache_util * 100.0,
                progress.metrics.as_ref().map(|m| m.avg_flops_util * 100.0).unwrap_or(0.0),
                progress.metrics.as_ref().map(|m| m.avg_bandwidth_util * 100.0).unwrap_or(0.0),
            );
        })
        .unwrap();
}

#[cfg(feature = "cli")]
fn print_final_metrics(
    summary: &sim::metrics::MetricsSummary,
    sim_time: f64,
    real_time: std::time::Duration,
    verbosity: VerbosityLevel,
    use_color: bool,
) {
    if verbosity == VerbosityLevel::Quiet {
        // Minimal output for quiet mode
        println!(
            "Simulating... done ({:.1}s simulated, {:.2}s real)",
            sim_time,
            real_time.as_secs_f64()
        );
        println!(
            "TTFT: {:.2}ms (p50: {:.2}ms, p99: {:.2}ms)",
            summary.ttft_mean, summary.ttft_p50, summary.ttft_p99
        );
        println!(
            "E2E:  {:.2}ms (p50: {:.2}ms, p99: {:.2}ms)",
            summary.e2e_mean, summary.e2e_p50, summary.e2e_p99
        );
        println!("Throughput: {:.0} output tok/s", summary.output_tokens_per_sec);
        return;
    }

    // Header
    if use_color {
        println!(
            "\n{} ({:.1}s simulated, {:.2}s real)",
            "Simulation Complete".bright_green().bold(),
            sim_time,
            real_time.as_secs_f64()
        );
        println!("{}", "━".repeat(80).bright_black());
    } else {
        println!(
            "\nSimulation Complete ({:.1}s simulated, {:.2}s real)",
            sim_time,
            real_time.as_secs_f64()
        );
        println!("{}", "━".repeat(80));
    }

    // Latency Metrics Table
    if use_color {
        println!("\n{}", "LATENCY METRICS".yellow().bold());
    } else {
        println!("\nLATENCY METRICS");
    }

    let latency_rows = vec![
        LatencyRow {
            metric: "TTFT (ms)".to_string(),
            mean: format!("{:.2}", summary.ttft_mean),
            p50: format!("{:.2}", summary.ttft_p50),
            p90: format!("{:.2}", summary.ttft_p90),
            p99: format!("{:.2}", summary.ttft_p99),
        },
        LatencyRow {
            metric: "E2E Latency (ms)".to_string(),
            mean: format!("{:.2}", summary.e2e_mean),
            p50: format!("{:.2}", summary.e2e_p50),
            p90: format!("{:.2}", summary.e2e_p90),
            p99: format!("{:.2}", summary.e2e_p99),
        },
        LatencyRow {
            metric: "Per-Token Latency (ms)".to_string(),
            mean: format!("{:.2}", summary.per_token_mean),
            p50: format!("{:.2}", summary.per_token_p50),
            p90: format!("{:.2}", summary.per_token_p90),
            p99: format!("{:.2}", summary.per_token_p99),
        },
    ];

    let latency_table = Table::new(&latency_rows).with(Style::rounded()).to_string();
    println!("{}", latency_table);

    // Throughput Metrics Table
    if use_color {
        println!("\n{}", "THROUGHPUT METRICS".yellow().bold());
    } else {
        println!("\nTHROUGHPUT METRICS");
    }

    let throughput_rows = vec![
        ThroughputRow {
            metric: "Input Tokens/sec".to_string(),
            mean: format!("{:.2}", summary.input_tokens_per_sec),
            p50: format!("{:.2}", summary.input_tokens_per_sec_p50),
            p90: format!("{:.2}", summary.input_tokens_per_sec_p90),
            p99: format!("{:.2}", summary.input_tokens_per_sec_p99),
        },
        ThroughputRow {
            metric: "Output Tokens/sec".to_string(),
            mean: format!("{:.2}", summary.output_tokens_per_sec),
            p50: format!("{:.2}", summary.output_tokens_per_sec_p50),
            p90: format!("{:.2}", summary.output_tokens_per_sec_p90),
            p99: format!("{:.2}", summary.output_tokens_per_sec_p99),
        },
        ThroughputRow {
            metric: "Requests/sec".to_string(),
            mean: format!("{:.2}", summary.requests_per_sec),
            p50: format!("{:.2}", summary.requests_per_sec_p50),
            p90: format!("{:.2}", summary.requests_per_sec_p90),
            p99: format!("{:.2}", summary.requests_per_sec_p99),
        },
    ];

    let throughput_table = Table::new(&throughput_rows)
        .with(Style::rounded())
        .to_string();
    println!("{}", throughput_table);

    // Utilization Section
    if use_color {
        println!("\n{}", "UTILIZATION".yellow().bold());
    } else {
        println!("\nUTILIZATION");
    }
    println!("  • KV Cache:  {:.1}% avg", summary.avg_kv_cache_util * 100.0);
    println!("  • FLOPS:     {:.1}% avg", summary.avg_flops_util * 100.0);
    println!(
        "  • Bandwidth: {:.1}% avg",
        summary.avg_bandwidth_util * 100.0
    );
    println!(
        "  • Preemptions: {} total ({:.2} per request avg)",
        summary.total_preemptions, summary.preemptions_per_request_mean
    );

    // Summary Section
    if use_color {
        println!("\n{}", "SUMMARY".yellow().bold());
    } else {
        println!("\nSUMMARY");
    }
    println!(
        "  • Total Requests: {} completed",
        summary.completed_requests
    );
    println!("  • Simulation Time: {:.1}s", sim_time);
    println!("  • Real Time: {:.2}s", real_time.as_secs_f64());
}

#[cfg(not(feature = "cli"))]
fn print_final_metrics(
    summary: &sim::metrics::MetricsSummary,
    sim_time: f64,
    real_time: std::time::Duration,
    verbosity: VerbosityLevel,
    _use_color: bool,
) {
    // Fallback for when CLI features are not available
    println!("\nSimulation Complete ({:.1}s)", sim_time);
    println!("TTFT: {:.2}ms (p50: {:.2}ms)", summary.ttft_mean, summary.ttft_p50);
    println!("E2E: {:.2}ms (p50: {:.2}ms)", summary.e2e_mean, summary.e2e_p50);
}

fn save_metrics_json(
    summary: &sim::metrics::MetricsSummary,
    path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    use serde_json::json;

    let json_data = json!({
        "latency_metrics": {
            "ttft_ms": {
                "mean": summary.ttft_mean,
                "p50": summary.ttft_p50,
                "p90": summary.ttft_p90,
                "p99": summary.ttft_p99,
            },
            "e2e_ms": {
                "mean": summary.e2e_mean,
                "p50": summary.e2e_p50,
                "p90": summary.e2e_p90,
                "p99": summary.e2e_p99,
            },
            "per_token_ms": {
                "mean": summary.per_token_mean,
                "p50": summary.per_token_p50,
                "p90": summary.per_token_p90,
                "p99": summary.per_token_p99,
            },
        },
        "throughput_metrics": {
            "input_tokens_per_sec": {
                "mean": summary.input_tokens_per_sec,
                "p50": summary.input_tokens_per_sec_p50,
                "p90": summary.input_tokens_per_sec_p90,
                "p99": summary.input_tokens_per_sec_p99,
            },
            "output_tokens_per_sec": {
                "mean": summary.output_tokens_per_sec,
                "p50": summary.output_tokens_per_sec_p50,
                "p90": summary.output_tokens_per_sec_p90,
                "p99": summary.output_tokens_per_sec_p99,
            },
            "requests_per_sec": {
                "mean": summary.requests_per_sec,
                "p50": summary.requests_per_sec_p50,
                "p90": summary.requests_per_sec_p90,
                "p99": summary.requests_per_sec_p99,
            },
        },
        "utilization": {
            "avg_kv_cache_util": summary.avg_kv_cache_util,
            "avg_flops_util": summary.avg_flops_util,
            "avg_bandwidth_util": summary.avg_bandwidth_util,
        },
        "preemptions": {
            "total": summary.total_preemptions,
            "per_request_mean": summary.preemptions_per_request_mean,
        },
        "requests": {
            "completed": summary.completed_requests,
            "total": summary.total_requests,
        },
    });

    std::fs::write(path, serde_json::to_string_pretty(&json_data)?)?;
    Ok(())
}
