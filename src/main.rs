use clap::Parser;
use inference_lab::config::{ArrivalPattern, Config, Deployment, ModelConfig, WorkloadConfig};
use inference_lab::dataset::{BatchTokenizerFn, Message, PromptInput};
use inference_lab::simulation::Simulator;
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

// --- CLI argument structures ---

#[cfg(feature = "serve")]
#[derive(Parser, Debug)]
#[command(author, version, about = "LLM Inference Simulator", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[cfg(feature = "serve")]
#[derive(clap::Subcommand, Debug)]
enum Commands {
    /// Run a batch simulation
    Sim(SimArgs),
    /// Start an OpenAI-compatible inference server
    Serve(ServeArgs),
}

#[cfg(feature = "serve")]
#[derive(Parser, Debug)]
struct ServeArgs {
    /// Model config file, or a directory of them (one model each)
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,

    /// Hardware entry to serve each model on ([hardware.<name>] in the model
    /// config). Optional when a config has exactly one entry; with a
    /// directory, models without this entry are skipped.
    #[arg(long)]
    hardware: Option<String>,

    /// Workload file whose output_len_dist samples each response's length
    /// (capped at the request's max_tokens). Without it responses run to
    /// max_tokens.
    #[arg(short, long)]
    workload: Option<PathBuf>,

    /// Port to listen on
    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    /// Host to bind to
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Path to tokenizer.json file (for accurate token counting)
    #[arg(short, long)]
    tokenizer: Option<PathBuf>,

    /// Honor <<respond:{...}>> echo-directives in message text (scripted responses that
    /// bypass the engine — see serve::directive) and the x-inference-lab-fault header
    /// (mid-stream death injection — see serve::fault). Test-harness feature: leave OFF
    /// anywhere untrusted clients can reach this server, or requests can spoof model
    /// output and stall/abort connections at will.
    #[arg(long, default_value_t = false)]
    enable_directives: bool,
}

#[derive(Parser, Debug)]
#[cfg_attr(not(feature = "serve"), command(author, version, about = "LLM Inference Simulator", long_about = None))]
struct SimArgs {
    /// Model config file
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,

    /// Hardware entry to run on ([hardware.<name>] in the model config).
    /// Optional when the config has exactly one entry.
    #[arg(long)]
    hardware: Option<String>,

    /// Workload file
    #[arg(short, long)]
    workload: PathBuf,

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

    /// Save per-request metrics to a CSV file (arrival, completion, ttft,
    /// e2e, mean_tpot, prompt_toks, output_toks, cached_toks, preemptions, and
    /// for session workloads session, step, gap, shared_toks,
    /// reuse_distance_bytes, reuse_touched_bytes); times in seconds
    #[arg(long)]
    request_csv: Option<PathBuf>,

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

impl SimArgs {
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

    Ok(Box::new(move |prompt_inputs: &[PromptInput]| {
        // Apply chat template and collect all texts
        let texts: Result<Vec<String>, String> = prompt_inputs
            .iter()
            .map(|prompt_input| match prompt_input {
                PromptInput::Messages(messages) => {
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
                }
                PromptInput::Prompt(prompt) => Ok(prompt.clone()),
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

// --- Entry points ---

#[cfg(feature = "serve")]
#[tokio::main]
async fn main() {
    env_logger::init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Sim(args) => run_sim(args),
        Commands::Serve(args) => {
            let workload = args.workload.as_ref().map(|p| {
                WorkloadConfig::from_file(p).unwrap_or_else(|e| {
                    eprintln!("Error loading workload: {e}");
                    std::process::exit(1);
                })
            });
            let deployments = if args.config.is_dir() {
                let mut paths: Vec<PathBuf> = match std::fs::read_dir(&args.config) {
                    Ok(entries) => entries
                        .flatten()
                        .map(|e| e.path())
                        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("toml"))
                        .collect(),
                    Err(e) => {
                        eprintln!("Error reading config directory {:?}: {}", args.config, e);
                        std::process::exit(1);
                    }
                };
                paths.sort();
                let mut deployments = Vec::new();
                for path in &paths {
                    let cfg = ModelConfig::from_file(path).unwrap_or_else(|e| {
                        eprintln!("Error loading config: {e}");
                        std::process::exit(1);
                    });
                    match &args.hardware {
                        Some(hw) if !cfg.hardware_names().contains(&hw.as_str()) => {
                            println!("  Skipping {}: no [hardware.{hw}] entry", path.display());
                        }
                        hw => deployments.push(load_deployment(&cfg, path, hw.as_deref())),
                    }
                }
                if deployments.is_empty() {
                    eprintln!("No model configs to serve in {:?}", args.config);
                    std::process::exit(1);
                }
                deployments
            } else {
                let cfg = ModelConfig::from_file(&args.config).unwrap_or_else(|e| {
                    eprintln!("Error loading config: {e}");
                    std::process::exit(1);
                });
                vec![load_deployment(
                    &cfg,
                    &args.config,
                    args.hardware.as_deref(),
                )]
            };

            if let Err(e) = inference_lab::serve::start_server(
                deployments,
                workload,
                args.host,
                args.port,
                args.tokenizer,
                args.enable_directives,
            )
            .await
            {
                eprintln!("Server error: {}", e);
                std::process::exit(1);
            }
        }
    }
}

#[cfg(all(feature = "cli", not(feature = "serve")))]
fn main() {
    env_logger::init();
    let args = SimArgs::parse();
    run_sim(args);
}

/// Resolve one hardware entry of a model config, exiting with the loader's
/// message on failure.
fn load_deployment(
    cfg: &ModelConfig,
    path: &std::path::Path,
    hardware: Option<&str>,
) -> Deployment {
    cfg.deployment(hardware).unwrap_or_else(|e| {
        eprintln!("Error in {}: {e}", path.display());
        if hardware.is_none() {
            eprintln!("Pass --hardware <name>.");
        }
        std::process::exit(1);
    })
}

fn run_sim(args: SimArgs) {
    let verbosity = args.verbosity_level();
    if args.no_color {
        // One switch for every `.color()` call below.
        colored::control::set_override(false);
    }

    // Header
    if verbosity >= VerbosityLevel::Normal {
        println!("{}", "LLM Inference Simulator".bright_cyan().bold());
        println!(
            "Loading configuration from: {:?} (workload {:?})\n",
            args.config, args.workload
        );
    }

    // Load configuration: model config × hardware entry × workload.
    let model_config = ModelConfig::from_file(&args.config).unwrap_or_else(|e| {
        eprintln!("Error loading configuration: {e}");
        std::process::exit(1);
    });
    let deployment = load_deployment(&model_config, &args.config, args.hardware.as_deref());
    let workload = WorkloadConfig::from_file(&args.workload).unwrap_or_else(|e| {
        eprintln!("Error loading workload: {e}");
        std::process::exit(1);
    });
    let mut config = Config::new(deployment, workload);

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

    let mut simulator = match Simulator::new(config, tokenizer) {
        Ok(sim) => sim,
        Err(e) => {
            eprintln!("Error creating simulator: {}", e);
            std::process::exit(1);
        }
    };
    // The simulator may have filled in `num_requests` from a counted dataset.
    let config = simulator.config().clone();

    // Print configuration summary (after simulator creation to show updated dataset entry count)
    if verbosity >= VerbosityLevel::Normal {
        println!("{}", "Configuration:".green().bold());
        println!("  Hardware: {}", config.hardware.name);
        println!("  Model: {}", config.model.name);
        println!(
            "  Max batched tokens: {}",
            config.scheduler.max_num_batched_tokens
        );

        match config.workload.arrival_pattern {
            ArrivalPattern::ClosedLoop => {
                if let Some(users) = config.workload.num_concurrent_users {
                    println!("  Arrival: closed-loop ({} concurrent users)", users);
                } else {
                    println!("  Arrival: closed-loop");
                }
            }
            ArrivalPattern::Batched => {
                println!("  Arrival: batched (all requests at t=0)");
            }
            pattern => {
                println!(
                    "  Arrival: {:?} ({} req/sec)",
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
            run_quiet(&mut simulator);
        }
        VerbosityLevel::Normal => {
            run_with_dashboard(&mut simulator, &config);
        }
        VerbosityLevel::Verbose => {
            run_verbose(&mut simulator, &config);
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
            let summary = simulator.summary();
            println!("{}", serde_json::to_string_pretty(&summary).unwrap());
            return;
        }
    }

    let elapsed = start_time.elapsed();

    // Print final metrics
    let summary = simulator.summary();
    print_final_metrics(&summary, simulator.current_time(), elapsed);

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

    // Save per-request CSV if requested (plus a sibling .depth.csv with the
    // per-second mean chosen draft depth and decode batch, when speculating)
    if let Some(csv_path) = args.request_csv {
        let depth = simulator.spec_depth_series();
        if !depth.is_empty() {
            let mut d = String::from("second,mean_draft,mean_decode_batch\n");
            for p in &depth {
                d.push_str(&format!(
                    "{},{:.4},{:.2}\n",
                    p.second, p.mean_draft, p.mean_decode_batch
                ));
            }
            let dp = csv_path.with_extension("depth.csv");
            if let Err(e) = std::fs::write(&dp, d) {
                eprintln!("Error saving depth CSV: {}", e);
            }
        }
        let mut out = String::from(
            "arrival,completion,ttft,e2e,mean_tpot,prompt_toks,output_toks,cached_toks,preemptions,session,step,gap,shared_toks,reuse_distance_bytes,reuse_touched_bytes\n",
        );
        for r in simulator.request_rows() {
            let opt = |v: Option<String>| v.unwrap_or_default();
            out.push_str(&format!(
                "{:.4},{:.4},{:.5},{:.4},{:.6},{},{},{},{},{},{},{},{},{},{}\n",
                r.arrival,
                r.completion,
                r.ttft,
                r.e2e,
                r.mean_tpot,
                r.prompt_tokens,
                r.output_tokens,
                r.cached_tokens,
                r.num_preemptions,
                opt(r.session.map(|(s, _)| s.to_string())),
                opt(r.session.map(|(_, st)| st.to_string())),
                opt(r.gap.map(|g| format!("{g:.3}"))),
                opt(r.shared_tokens.map(|t| t.to_string())),
                opt(r.reuse_distance_bytes.map(|b| b.to_string())),
                opt(r.reuse_touched_bytes.map(|b| b.to_string())),
            ));
        }
        match std::fs::write(&csv_path, out) {
            Ok(_) => {
                if verbosity >= VerbosityLevel::Normal {
                    println!("Per-request CSV saved to: {:?}", csv_path);
                }
            }
            Err(e) => eprintln!("Error saving request CSV: {}", e),
        }
    }
}

fn run_quiet(simulator: &mut Simulator) {
    simulator
        .run_with_callback(|_progress| {
            // No output during simulation
        })
        .unwrap();
}

fn run_with_dashboard(simulator: &mut Simulator, config: &Config) {
    let total_requests = config.workload.num_requests.unwrap_or(1000) as u64;

    println!("{}", "━".repeat(60).bright_black());
    println!("{}", "Simulation Progress".bright_cyan().bold());
    println!("{}", "━".repeat(60).bright_black());

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
        })
        .unwrap();
}

fn run_verbose(simulator: &mut Simulator, config: &Config) {
    println!("Starting simulation...");

    simulator
        .run_with_callback(|progress| {
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
                progress.metrics.utilization.avg_flops_util * 100.0,
                progress.metrics.utilization.avg_bandwidth_util * 100.0,
            );
        })
        .unwrap();
}

#[cfg(feature = "cli")]
fn print_final_metrics(
    summary: &inference_lab::metrics::MetricsSummary,
    sim_time: f64,
    real_time: std::time::Duration,
) {
    println!(
        "\n{} ({:.1}s simulated, {:.2}s real)",
        "Simulation Complete".bright_green().bold(),
        sim_time,
        real_time.as_secs_f64()
    );
    println!("{}", "━".repeat(80).bright_black());

    // Latency Metrics Table
    println!("\n{}", "LATENCY METRICS".yellow().bold());

    let row = |metric: &str, l: &inference_lab::metrics::LatencyStats| LatencyRow {
        metric: metric.to_string(),
        min: format!("{:.2}", l.min),
        mean: format!("{:.2}", l.mean),
        p50: format!("{:.2}", l.p50),
        p90: format!("{:.2}", l.p90),
        p99: format!("{:.2}", l.p99),
    };
    let lat = &summary.latency_metrics;
    let latency_rows = vec![
        row("TTFT (ms)", &lat.ttft_ms),
        row("E2E Latency (ms)", &lat.e2e_ms),
        row("Per-Token Latency (ms)", &lat.per_token_ms),
    ];

    let latency_table = Table::new(&latency_rows).with(Style::rounded()).to_string();
    println!("{}", latency_table);

    // Throughput Metrics Table
    println!("\n{}", "THROUGHPUT METRICS".yellow().bold());

    let throughput_rows = vec![
        ThroughputRow {
            metric: "Input Tokens/sec".to_string(),
            value: format!("{:.2}", summary.throughput_metrics.input_tokens_per_sec),
        },
        ThroughputRow {
            metric: "Output Tokens/sec".to_string(),
            value: format!("{:.2}", summary.throughput_metrics.output_tokens_per_sec),
        },
        ThroughputRow {
            metric: "Requests/sec".to_string(),
            value: format!("{:.2}", summary.throughput_metrics.requests_per_sec),
        },
    ];

    let throughput_table = Table::new(&throughput_rows)
        .with(Style::rounded())
        .to_string();
    println!("{}", throughput_table);

    // Utilization Section
    println!("\n{}", "UTILIZATION".yellow().bold());
    let util = &summary.utilization;
    println!("  • KV Cache:  {:.1}% avg", util.avg_kv_cache_util * 100.0);
    println!("  • FLOPS:     {:.1}% avg", util.avg_flops_util * 100.0);
    println!("  • Bandwidth: {:.1}% avg", util.avg_bandwidth_util * 100.0);
    println!(
        "  • Preemptions: {} total ({:.2} per request avg)",
        summary.preemptions.total, summary.preemptions.per_request_mean
    );

    // Summary Section
    println!("\n{}", "SUMMARY".yellow().bold());
    println!(
        "  • Total Requests: {} completed",
        summary.requests.completed
    );
    if summary.requests.rejected > 0 {
        println!(
            "  • {} {} rejected: context larger than a worker's KV cache",
            "⚠".yellow(),
            summary.requests.rejected
        );
    }
    println!("  • Simulation Time: {:.1}s", sim_time);
    println!("  • Real Time: {:.2}s", real_time.as_secs_f64());

    // Prefix Cache Section
    let pc = &summary.prefix_cache;
    if pc.hits + pc.misses > 0 {
        println!("\n{}", "PREFIX CACHE".yellow().bold());
        println!("  • Hits:      {}", pc.hits);
        println!("  • Misses:    {}", pc.misses);
        println!("  • Avg hit size: {:.1} tokens", pc.mean_hit_size);
        println!("  • Hit Rate:  {:.1}%", pc.hit_rate * 100.0);
        if pc.recomputed > 0 {
            println!(
                "  • Recomputed instead of fetched: {} lookups, {} tokens",
                pc.recomputed, pc.recomputed_tokens
            );
        }
        if pc.prefetches > 0 {
            println!(
                "  • Prefetched: {} prefixes, {} tokens",
                pc.prefetches, pc.prefetch_tokens
            );
        }
    }

    // Router Section (only interesting with more than one replica)
    let print_router = |title: &str, rt: &inference_lab::metrics::RouterMetrics| {
        println!("\n{}", title.yellow().bold());
        println!("  • Policy:    {}", rt.policy);
        let per: Vec<String> = rt.per_replica.iter().map(|n| n.to_string()).collect();
        println!("  • Per replica: [{}]", per.join(", "));
        if rt.prefix_available > 0 {
            println!(
                "  • Prefix held somewhere: {} requests; routed to a holder: {} ({:.1}%); \
                 routed away from the longest holder: {}",
                rt.prefix_available,
                rt.prefix_routed,
                100.0 * rt.prefix_routed as f64 / rt.prefix_available as f64,
                rt.prefix_forgone
            );
        }
    };
    if summary.router.per_replica.len() > 1 {
        print_router("ROUTER", &summary.router);
    }
    if let Some(dr) = &summary.decode_router {
        if dr.per_replica.len() > 1 {
            print_router("DECODE ROUTER", dr);
        }
    }
    if let Some(m) = &summary.memory {
        println!("\n{}", "MEMORY".yellow().bold());
        println!(
            "  • Write: {}; eviction: {}; written {:.2} GB; promoted {:.2} GB; promotions waiting on a write: {}",
            m.write_policy,
            m.eviction_policy,
            m.bytes_written / 1e9,
            m.bytes_promoted / 1e9,
            m.write_race_waits
        );
        for st in &m.stores {
            println!(
                "  • {}: {} × {} blocks, {} held; written {:.2} GB, read {:.2} GB, dead {:.2} GB; evictions {}, expired {}",
                st.name,
                st.instances,
                st.capacity_blocks,
                st.held_blocks,
                st.bytes_written as f64 / 1e9,
                st.bytes_read as f64 / 1e9,
                st.dead_bytes as f64 / 1e9,
                st.evictions,
                st.expired
            );
        }
        for l in &m.links {
            if l.bytes_moved > 0.0 {
                println!(
                    "  • link {}: {} × {:.1} GB/s; moved {:.2} GB ({:.1}% of capacity)",
                    l.name,
                    l.instances / 2,
                    l.capacity / 1e9,
                    l.bytes_moved / 1e9,
                    100.0 * l.utilisation
                );
            }
        }
    }
    if let Some(h) = &summary.handoff {
        if h.transfers > 0 {
            println!("\n{}", "HAND-OFF".yellow().bold());
            println!("  • Transfers: {}", h.transfers);
            println!(
                "  • Bytes moved: {:.2} GB; skipped (resident on decoder): {:.2} GB ({:.1}%)",
                h.bytes as f64 / 1e9,
                h.bytes_skipped as f64 / 1e9,
                100.0 * h.bytes_skipped as f64 / (h.bytes + h.bytes_skipped).max(1) as f64
            );
        }
    }
}

#[cfg(not(feature = "cli"))]
fn print_final_metrics(
    summary: &inference_lab::metrics::MetricsSummary,
    sim_time: f64,
    _real_time: std::time::Duration,
) {
    // Fallback for when CLI features are not available
    println!("\nSimulation Complete ({:.1}s)", sim_time);
    let l = &summary.latency_metrics;
    println!(
        "TTFT: {:.2}ms (p50: {:.2}ms)",
        l.ttft_ms.mean, l.ttft_ms.p50
    );
    println!("E2E: {:.2}ms (p50: {:.2}ms)", l.e2e_ms.mean, l.e2e_ms.p50);
}

fn save_metrics_json(
    summary: &inference_lab::metrics::MetricsSummary,
    path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::write(path, serde_json::to_string_pretty(summary)?)?;
    Ok(())
}
