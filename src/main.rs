use clap::Parser;
use inference_lab::{BatchTokenizerFn, Config, Message, Simulator};
use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "cli")]
use tokenizers::Tokenizer;

#[cfg(feature = "cli")]
use minijinja::Environment;

#[cfg(feature = "cli")]
use colored::Colorize;
#[cfg(feature = "cli")]
use tabled::{settings::Style, Table, Tabled};

#[derive(Parser, Debug)]
#[command(author, version, about = "LLM Inference Simulator", long_about = None)]
struct Args {
    /// Path to the TOML configuration file
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,

    /// Path to tokenizer.json file (required if dataset_path is set in config)
    #[arg(short, long)]
    tokenizer: Option<PathBuf>,

    /// Chat template (Jinja2 format) or path to template file. Use "None" for simple concatenation. Required when using datasets.
    #[arg(long)]
    chat_template: Option<String>,

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

    /// Override the random seed (overrides config file)
    #[arg(long)]
    seed: Option<u64>,
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
    #[tabled(rename = "Min")]
    min: String,
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
    #[tabled(rename = "Value")]
    value: String,
}

/// Apply chat template to messages using Jinja2
#[cfg(feature = "cli")]
fn apply_chat_template(template: &str, messages: &[Message]) -> Result<String, String> {
    let env = Environment::new();
    let tmpl = env
        .template_from_str(template)
        .map_err(|e| format!("Invalid template: {}", e))?;

    // Convert messages to the format expected by chat templates
    let messages_json: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| {
            serde_json::json!({
                "role": m.role,
                "content": m.content
            })
        })
        .collect();

    // Render the template
    let context = serde_json::json!({
        "messages": messages_json,
        "add_generation_prompt": true,
        "bos_token": "<s>",
        "eos_token": "</s>",
    });

    tmpl.render(context)
        .map_err(|e| format!("Template rendering failed: {}", e))
}

/// Load a tokenizer from a file and create a BatchTokenizerFn
#[cfg(feature = "cli")]
fn load_tokenizer(
    tokenizer_path: &PathBuf,
    chat_template: Option<String>,
) -> Result<BatchTokenizerFn, String> {
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

    // Load template: check if it's a file path first, otherwise use as template string
    let template = match chat_template {
        Some(ref t) if t == "None" => None,
        Some(t) => {
            // Try to read as file first
            if let Ok(content) = std::fs::read_to_string(&t) {
                Some(content)
            } else {
                // Not a file, use as template string directly
                Some(t)
            }
        }
        None => None,
    };

    Ok(Box::new(move |message_batches: &[&[Message]]| {
        // Apply chat template and collect all texts
        let texts: Result<Vec<String>, String> = message_batches
            .iter()
            .map(|messages| {
                if let Some(ref tmpl) = template {
                    apply_chat_template(tmpl, messages)
                } else {
                    // Simple concatenation fallback
                    Ok(messages
                        .iter()
                        .map(|m| format!("{}: {}", m.role, m.content))
                        .collect::<Vec<_>>()
                        .join("\n"))
                }
            })
            .collect();

        let texts = texts?;

        // Batch encode all texts at once (much faster!)
        let encodings = tokenizer
            .encode_batch(texts, false)
            .map_err(|e| format!("Failed to batch tokenize: {}", e))?;

        // Extract token IDs from all encodings
        Ok(encodings
            .into_iter()
            .map(|enc| enc.get_ids().to_vec())
            .collect())
    }))
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
    let mut config = match Config::from_file(&args.config) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Error loading configuration: {}", e);
            std::process::exit(1);
        }
    };

    // Override seed if provided via CLI
    if let Some(seed) = args.seed {
        config.workload.seed = seed;
        if verbosity >= VerbosityLevel::Normal {
            println!("Overriding seed with CLI value: {}\n", seed);
        }
    }

    // Load tokenizer if needed for dataset mode
    #[cfg(feature = "cli")]
    let tokenizer = if config.workload.dataset_path.is_some() {
        match &args.tokenizer {
            Some(tokenizer_path) => {
                // Check if chat template is provided
                if args.chat_template.is_none() {
                    eprintln!("Error: --chat-template is required when using datasets.");
                    eprintln!("Use --chat-template \"<template>\" or --chat-template None for simple concatenation.");
                    std::process::exit(1);
                }

                if verbosity >= VerbosityLevel::Normal {
                    println!("Loading tokenizer from: {:?}", tokenizer_path);
                    if let Some(ref tmpl) = args.chat_template {
                        if tmpl == "None" {
                            println!("Using simple message concatenation (no chat template)");
                        } else if std::path::Path::new(tmpl).exists() {
                            println!("Loading chat template from: {:?}", tmpl);
                        } else {
                            println!("Using custom chat template (inline)");
                        }
                    }
                }

                match load_tokenizer(tokenizer_path, args.chat_template.clone()) {
                    Ok(tok) => Some(tok),
                    Err(e) => {
                        eprintln!("Error loading tokenizer: {}", e);
                        std::process::exit(1);
                    }
                }
            }
            None => {
                eprintln!("Error: dataset_path is set in config but no tokenizer specified.");
                eprintln!("Please provide a tokenizer using --tokenizer <path-to-tokenizer.json>");
                std::process::exit(1);
            }
        }
    } else {
        None
    };

    #[cfg(not(feature = "cli"))]
    let tokenizer = None;

    // Create simulator (returns updated config with counted dataset entries if applicable)
    let (mut simulator, config) = match Simulator::new_with_tokenizer(config, tokenizer) {
        Ok((sim, cfg)) => (sim, cfg),
        Err(e) => {
            eprintln!("Error creating simulator: {}", e);
            std::process::exit(1);
        }
    };

    // Print configuration summary (after simulator creation to show updated dataset entry count)
    if verbosity >= VerbosityLevel::Normal {
        if use_color {
            println!("{}", "Configuration:".green().bold());
        } else {
            println!("Configuration:");
        }
        println!("  Hardware: {}", config.hardware.name);
        println!("  Model: {}", config.model.name);
        println!(
            "  Max batched tokens: {}",
            config.scheduler.max_num_batched_tokens
        );

        // Print arrival pattern with relevant details
        match config.workload.arrival_pattern.to_lowercase().as_str() {
            "closed_loop" => {
                if let Some(users) = config.workload.num_concurrent_users {
                    println!("  Arrival: closed-loop ({} concurrent users)", users);
                } else {
                    println!("  Arrival: closed-loop");
                }
            }
            "batched" => {
                println!("  Arrival: batched (all requests at t=0)");
            }
            pattern => {
                println!(
                    "  Arrival: {} ({} req/sec)",
                    pattern, config.workload.arrival_rate
                );
            }
        }
        println!(
            "  Number of requests: {}",
            config
                .workload
                .num_requests
                .map(|n| n.to_string())
                .unwrap_or_else(|| "unlimited".to_string())
        );
        println!("  Seed: {}", config.workload.seed);
        println!();
    }

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
            // Debug mode with no progress callbacks
            simulator.run_with_callback(|_| {}).unwrap();
            let elapsed = start_time.elapsed();
            println!("\nSimulation complete!");
            if verbosity >= VerbosityLevel::Normal {
                println!(
                    "Simulation completed in {:.2}s (real time)\n",
                    elapsed.as_secs_f64()
                );
            }
            // Print final metrics for debug mode
            let summary = simulator.get_metrics_summary();
            summary.print();
            return;
        }
    }

    let elapsed = start_time.elapsed();

    // Print final metrics
    let summary = simulator.get_metrics_summary();
    print_final_metrics(
        &summary,
        simulator.get_current_time(),
        elapsed,
        verbosity,
        use_color,
    );

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
            let percent =
                (progress.completed_requests as f64 / total_requests as f64 * 100.0).min(100.0);
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
                    "  Time:     {}s simulated",
                    format!("{:.1}", progress.current_time).yellow()
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
                println!(
                    "  KV Cache: {:.1}% utilized",
                    progress.kv_cache_util * 100.0
                );
                println!("{}", "━".repeat(60));
            }
        })
        .unwrap();
}

fn run_verbose(simulator: &mut Simulator, use_color: bool, config: &Config) {
    if use_color {
        println!("{}", "Starting simulation...".green());
    } else {
        println!("Starting simulation...");
    }

    simulator
        .run_with_callback(|progress| {
            // Use actual total_requests from progress (updated dynamically)
            // For dataset mode, this tracks requests as they arrive
            let total_display = if let Some(num_req) = config.workload.num_requests {
                num_req.to_string()
            } else {
                progress.total_requests.to_string()
            };

            println!(
                "[{:.1}s] {}/{} requests | {} running, {} waiting | KV: {:.1}% | FLOPS: {:.1}% | BW: {:.1}%",
                progress.current_time,
                progress.completed_requests,
                total_display,
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
    summary: &inference_lab::metrics::MetricsSummary,
    sim_time: f64,
    real_time: std::time::Duration,
    verbosity: VerbosityLevel,
    use_color: bool,
) {
    // Quiet mode now uses the same output as normal mode (no special case)

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
            min: format!("{:.2}", summary.ttft_min),
            mean: format!("{:.2}", summary.ttft_mean),
            p50: format!("{:.2}", summary.ttft_p50),
            p90: format!("{:.2}", summary.ttft_p90),
            p99: format!("{:.2}", summary.ttft_p99),
        },
        LatencyRow {
            metric: "E2E Latency (ms)".to_string(),
            min: format!("{:.2}", summary.e2e_min),
            mean: format!("{:.2}", summary.e2e_mean),
            p50: format!("{:.2}", summary.e2e_p50),
            p90: format!("{:.2}", summary.e2e_p90),
            p99: format!("{:.2}", summary.e2e_p99),
        },
        LatencyRow {
            metric: "Per-Token Latency (ms)".to_string(),
            min: format!("{:.2}", summary.per_token_min),
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
            value: format!("{:.2}", summary.input_tokens_per_sec),
        },
        ThroughputRow {
            metric: "Output Tokens/sec".to_string(),
            value: format!("{:.2}", summary.output_tokens_per_sec),
        },
        ThroughputRow {
            metric: "Requests/sec".to_string(),
            value: format!("{:.2}", summary.requests_per_sec),
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
    println!(
        "  • KV Cache:  {:.1}% avg",
        summary.avg_kv_cache_util * 100.0
    );
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

    // Prefix Cache Section
    if summary.prefix_cache_hits + summary.prefix_cache_misses > 0 {
        if use_color {
            println!("\n{}", "PREFIX CACHE".yellow().bold());
        } else {
            println!("\nPREFIX CACHE");
        }
        println!("  • Hits:      {}", summary.prefix_cache_hits);
        println!("  • Misses:    {}", summary.prefix_cache_misses);
        println!(
            "  • Avg hit size: {}/{}={}",
            summary.prefix_cache_hit_size_sum,
            summary.prefix_cache_hit_size_count,
            summary.prefix_cache_hit_size_sum / summary.prefix_cache_hit_size_count
        );
        println!(
            "  • Hit Rate:  {:.1}%",
            summary.prefix_cache_hit_rate * 100.0
        );
    }
}

#[cfg(not(feature = "cli"))]
fn print_final_metrics(
    summary: &inference_lab::metrics::MetricsSummary,
    sim_time: f64,
    real_time: std::time::Duration,
    verbosity: VerbosityLevel,
    _use_color: bool,
) {
    // Fallback for when CLI features are not available
    println!("\nSimulation Complete ({:.1}s)", sim_time);
    println!(
        "TTFT: {:.2}ms (p50: {:.2}ms)",
        summary.ttft_mean, summary.ttft_p50
    );
    println!(
        "E2E: {:.2}ms (p50: {:.2}ms)",
        summary.e2e_mean, summary.e2e_p50
    );
}

fn save_metrics_json(
    summary: &inference_lab::metrics::MetricsSummary,
    path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    use serde_json::json;

    let json_data = json!({
        "latency_metrics": {
            "ttft_ms": {
                "min": summary.ttft_min,
                "mean": summary.ttft_mean,
                "p50": summary.ttft_p50,
                "p90": summary.ttft_p90,
                "p99": summary.ttft_p99,
            },
            "e2e_ms": {
                "min": summary.e2e_min,
                "mean": summary.e2e_mean,
                "p50": summary.e2e_p50,
                "p90": summary.e2e_p90,
                "p99": summary.e2e_p99,
            },
            "per_token_ms": {
                "min": summary.per_token_min,
                "mean": summary.per_token_mean,
                "p50": summary.per_token_p50,
                "p90": summary.per_token_p90,
                "p99": summary.per_token_p99,
            },
        },
        "throughput_metrics": {
            "input_tokens_per_sec": summary.input_tokens_per_sec,
            "output_tokens_per_sec": summary.output_tokens_per_sec,
            "requests_per_sec": summary.requests_per_sec,
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
        "prefix_cache": {
            "hits": summary.prefix_cache_hits,
            "misses": summary.prefix_cache_misses,
            "hit_rate": summary.prefix_cache_hit_rate,
        },
    });

    std::fs::write(path, serde_json::to_string_pretty(&json_data)?)?;
    Ok(())
}
