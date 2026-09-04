//! Disaggregated tiering experiments for the KV-tier blog post.
//!
//! Four experiments, each sweeping one knob with the others fixed:
//!   1. tier_capacity  — HBM-only → +DRAM → +NVMe, varying capacity
//!   2. tier_bandwidth — fix high capacity, sweep link BW
//!   3. prefill_ratio  — sweep prefill workers × tier config
//!   4. load_sweep     — sweep session count, generous tiers
//!
//! All use disaggregated prefill/decode with GLM-5.2 on B200s and trajectories
//! from a Batchbench `plans.jsonl` manifest.
//!
//! Run: cargo run --release --example disagg_tier_sweep -- <experiment> [--json]

mod common;

use std::collections::{BTreeMap, HashMap};

use inference_lab::catalog;
use inference_lab::config::{
    ArrivalPattern, ClusterSpec, DisaggTopology, MemoryConfig, RouterConfig, SchedulerConfig,
    WorkloadConfig,
};
use inference_lab::metrics::{HandoffMetrics, MetricsCollector, MetricsSummary, RouterMetrics};
use inference_lab::request::{ReplayManifest, RequestGenerator};
use inference_lab::scheduler::SchedulingPolicy;
use inference_lab::simulation::{Engine, Topology};
use rayon::prelude::*;

fn glm52() -> inference_lab::config::ModelSpec {
    catalog::model("glm-5.2-fp8").expect("glm-5.2-fp8 in catalog")
}

fn b200() -> inference_lab::config::HardwareConfig {
    catalog::hardware("b200").expect("b200 in catalog")
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
        enable_preemption_free: true,
        enable_cascade_attention: false,
        balance_set: None,
        max_waiting: 0,
    }
}

fn decode_cluster(num_workers: u32) -> ClusterSpec {
    ClusterSpec {
        hardware: b200(),
        parallel: inference_lab::config::ParallelConfig {
            tp: 8,
            ep: 1,
            dp_attention: false,
        },
        num_workers,
        memory: Default::default(),
    }
}

fn load_manifests() -> Vec<ReplayManifest> {
    ReplayManifest::load("plans.jsonl").expect("load Batchbench replay manifest")
}

#[derive(Clone)]
struct RunConfig {
    label: String,
    prefill_workers: u32,
    decode_workers: u32,
    memory: MemoryConfig,
    num_sessions: usize,
    kv_link_bw: f64,
    hardware_override: Option<inference_lab::config::HardwareConfig>,
}

#[derive(Debug, serde::Serialize)]
struct RunResult {
    label: String,
    num_sessions: usize,
    prefill_workers: u32,
    decode_workers: u32,
    output_tok_s: f64,
    input_tok_s: f64,
    ttft_mean_ms: f64,
    ttft_p50_ms: f64,
    ttft_p99_ms: f64,
    tpot_mean_ms: f64,
    cache_hit_rate: f64,
    recomputed: u64,
    prefetches: u64,
}

impl RunResult {
    fn from_summary(cfg: &RunConfig, s: &MetricsSummary) -> Self {
        Self {
            label: cfg.label.clone(),
            num_sessions: cfg.num_sessions,
            prefill_workers: cfg.prefill_workers,
            decode_workers: cfg.decode_workers,
            output_tok_s: s.throughput_metrics.output_tokens_per_sec,
            input_tok_s: s.throughput_metrics.input_tokens_per_sec,
            ttft_mean_ms: s.latency_metrics.ttft_ms.mean,
            ttft_p50_ms: s.latency_metrics.ttft_ms.p50,
            ttft_p99_ms: s.latency_metrics.ttft_ms.p99,
            tpot_mean_ms: s.latency_metrics.per_token_ms.mean,
            cache_hit_rate: s.prefix_cache.hit_rate,
            recomputed: s.prefix_cache.recomputed,
            prefetches: s.prefix_cache.prefetches,
        }
    }
}

fn run_one(cfg: &RunConfig, manifests: &[ReplayManifest]) -> RunResult {
    let hw = cfg.hardware_override.clone().unwrap_or_else(b200);
    let prefill = ClusterSpec {
        hardware: hw,
        parallel: inference_lab::config::ParallelConfig {
            tp: 8,
            ep: 1,
            dp_attention: false,
        },
        num_workers: cfg.prefill_workers,
        memory: cfg.memory.clone(),
    };
    let topo = DisaggTopology {
        prefill,
        decode: decode_cluster(cfg.decode_workers),
        kv_link_bw: Some(cfg.kv_link_bw),
    };
    let model = glm52();
    let sched = scheduler();

    let topology = Topology::from_disagg(&topo, model.clone(), sched.clone())
        .expect("build topology")
        .with_routers(
            &RouterConfig::PrefixAffinity {
                max_load_ratio: Some(1.5),
            },
            &RouterConfig::KvAwareDecode { load_weight: 64.0 },
        );

    let workload = WorkloadConfig {
        dataset_path: None,
        replay_manifest_path: None,
        measurement_start_secs: None,
        measurement_duration_secs: None,
        num_trajectories: Some(cfg.num_sessions),
        arrival_pattern: ArrivalPattern::Poisson,
        arrival_rate: 1.0,
        rate_schedule: None,
        num_concurrent_users: None,
        closed_loop_jitter_secs: None,
        input_len_dist: inference_lab::config::LengthDistribution::Fixed { value: 1024 },
        output_len_dist: inference_lab::config::LengthDistribution::Fixed { value: 256 },
        num_requests: None,
        duration_secs: Some(600.0),
        seed: 42,
    };

    let mut gen =
        RequestGenerator::from_replay_manifest(workload, sched.block_size, manifests.to_vec());
    let mut engine = Engine::new(topology);
    let mut metrics = MetricsCollector::new(0.0);
    let mut bytes_at_completion: HashMap<(u32, u32), (u64, u64)> = HashMap::new();

    loop {
        // Drain arrivals.
        let now = engine.current_time();
        let bound = gen.peek_next_arrival_time().max(now) + 1e-9;
        while let Some(mut req) = gen.next_if_before(bound) {
            if let Some(step) = &mut req.session {
                if let Some((written, touched)) =
                    bytes_at_completion.remove(&(step.session, step.step.wrapping_sub(1)))
                {
                    step.parent_bytes_written = Some(written);
                    step.parent_bytes_touched = Some(touched);
                }
            }
            engine.submit(req);
            metrics.total_requests += 1;
        }

        // Skip idle gaps.
        if engine.is_idle() {
            let next = gen.peek_next_arrival_time();
            if next.is_finite() && next > engine.current_time() {
                engine.advance_to(next);
            }
        }

        if engine.next_event_time().is_none() {
            if gen.is_finished() && engine.is_idle() {
                break;
            }
            break;
        }

        let outcome = engine.step().expect("engine step");
        for prog in outcome.iteration.iter().flat_map(|i| i.progress.iter()) {
            metrics.record_iteration_metrics(
                engine.kv_cache_util(),
                outcome.iteration.as_ref().map_or(0.0, |i| i.flops_util),
                outcome.iteration.as_ref().map_or(0.0, |i| i.bandwidth_util),
            );
            let _ = prog;
        }
        for completion in &outcome.completions {
            metrics.record_request_completion(completion);
            let has_successor = gen.on_request_complete(completion);
            if has_successor {
                if let Some(step) = &completion.session {
                    bytes_at_completion.insert(
                        (step.session, step.step),
                        (engine.kv_bytes_written(), engine.kv_bytes_touched()),
                    );
                }
            }
        }

        if engine.current_time() > 600.0 {
            break;
        }
    }

    let router = RouterMetrics::from_stats("prefix_affinity", engine.router_stats());
    let decode_router = engine
        .decode_router_stats()
        .map(|rs| RouterMetrics::from_stats("kv_aware_decode", rs));
    let handoff = decode_router.as_ref().map(|_| {
        let h = engine.handoff_stats();
        HandoffMetrics {
            transfers: h.transfers,
            bytes: h.bytes,
            bytes_skipped: h.bytes_skipped,
        }
    });
    let memory = engine.memory_metrics();
    let summary = metrics.compute_summary(
        engine.current_time(),
        engine.aggregate_prefix_cache(),
        router,
        decode_router,
        handoff,
        memory,
    );

    RunResult::from_summary(cfg, &summary)
}

// ── Experiment definitions ──────────────────────────────────────────────

fn mem_hbm_only() -> MemoryConfig {
    Default::default()
}

fn mem_dram(cap: f64) -> MemoryConfig {
    let mut capacity = BTreeMap::new();
    capacity.insert("host_dram".into(), cap);
    MemoryConfig {
        tiers: vec!["host_dram".into()],
        capacity,
        write: Some(inference_lab::config::WritePolicy::WriteThrough {}),
        eviction: Some(inference_lab::config::EvictionPolicy::Lru {}),
        ..Default::default()
    }
}

fn mem_dram_nvme(dram_cap: f64, nvme_cap: f64) -> MemoryConfig {
    let mut capacity = BTreeMap::new();
    capacity.insert("host_dram".into(), dram_cap);
    capacity.insert("nvme".into(), nvme_cap);
    MemoryConfig {
        tiers: vec!["host_dram".into(), "nvme".into()],
        capacity,
        write: Some(inference_lab::config::WritePolicy::WriteThrough {}),
        eviction: Some(inference_lab::config::EvictionPolicy::Lru {}),
        ..Default::default()
    }
}

/// Experiment 1: sweep tier configuration at fixed load.
/// 1 prefill node + 1 decode node (8xB200 each, TP8).
fn experiment_tier_capacity() -> Vec<RunConfig> {
    let num_sessions = 1500;
    let prefill = 1;
    let decode = 1;
    let kv_link_bw = 5.0e10;

    let mut configs = Vec::new();

    // HBM only
    configs.push(RunConfig {
        label: "hbm_only".into(),
        prefill_workers: prefill,
        decode_workers: decode,
        memory: mem_hbm_only(),
        num_sessions,
        kv_link_bw,
        hardware_override: None,
    });

    // DRAM sweep: 100GB, 500GB, 1TB, 2TB
    for &dram_gb in &[100.0, 500.0, 1000.0, 2000.0] {
        configs.push(RunConfig {
            label: format!("dram_{dram_gb:.0}gb"),
            prefill_workers: prefill,
            decode_workers: decode,
            memory: mem_dram(dram_gb * 1e9),
            num_sessions,
            kv_link_bw,
            hardware_override: None,
        });
    }

    // DRAM + NVMe: full 2TB DRAM + NVMe sweep
    for &nvme_tb in &[1.0, 5.0, 15.0, 30.0] {
        configs.push(RunConfig {
            label: format!("dram_2tb_nvme_{nvme_tb:.0}tb"),
            prefill_workers: prefill,
            decode_workers: decode,
            memory: mem_dram_nvme(2.0e12, nvme_tb * 1e12),
            num_sessions,
            kv_link_bw,
            hardware_override: None,
        });
    }

    configs
}

fn b200_with_pcie_bw(pcie_bw: f64) -> inference_lab::config::HardwareConfig {
    let mut hw = b200();
    if let Some(mem) = hw.memory.as_mut() {
        for link in &mut mem.links {
            if link.name == "pcie" {
                link.bandwidth = pcie_bw;
            }
        }
    }
    hw
}

/// Experiment 2: sweep PCIe bandwidth (tier link) at fixed high capacity. 1+1 nodes.
fn experiment_tier_bandwidth() -> Vec<RunConfig> {
    let num_sessions = 750;
    let kv_link_bw = 5.0e10;

    let bw_values = [0.5e9, 1.0e9, 2.0e9, 5.0e9, 14.0e9, 32.0e9, 64.0e9, 128.0e9];

    bw_values
        .iter()
        .map(|&bw| RunConfig {
            label: format!("pcie_{:.1}gbs", bw / 1e9),
            prefill_workers: 1,
            decode_workers: 1,
            memory: mem_dram(2.0e12),
            num_sessions,
            kv_link_bw,
            hardware_override: Some(b200_with_pcie_bw(bw)),
        })
        .collect()
}

/// Experiment 3: sweep prefill workers × tier config.
fn experiment_prefill_ratio() -> Vec<RunConfig> {
    let num_sessions = 1000;
    let decode = 8;
    let kv_link_bw = 5.0e10;

    let prefill_counts = [2, 4, 8, 16];
    let tier_configs: Vec<(&str, MemoryConfig)> = vec![
        ("hbm_only", mem_hbm_only()),
        ("dram_2tb", mem_dram(2.0e12)),
        ("dram_2tb_nvme_30tb", mem_dram_nvme(2.0e12, 30.0e12)),
    ];

    let mut configs = Vec::new();
    for &p in &prefill_counts {
        for (tier_name, tier_mem) in &tier_configs {
            configs.push(RunConfig {
                label: format!("p{p}_{tier_name}"),
                prefill_workers: p,
                decode_workers: decode,
                memory: tier_mem.clone(),
                num_sessions,
                kv_link_bw,
                hardware_override: None,
            });
        }
    }
    configs
}

/// Experiment 4: sweep load with generous tiers. 1+1 nodes.
fn experiment_load_sweep() -> Vec<RunConfig> {
    let prefill = 1;
    let decode = 1;
    let kv_link_bw = 5.0e10;

    let session_counts = [50, 100, 200, 300, 400, 500, 750, 1000, 1500, 2000];

    session_counts
        .iter()
        .map(|&n| RunConfig {
            label: format!("sessions_{n}"),
            prefill_workers: prefill,
            decode_workers: decode,
            memory: mem_dram_nvme(2.0e12, 30.0e12),
            num_sessions: n,
            kv_link_bw,
            hardware_override: None,
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let experiment = args.get(1).map(|s| s.as_str()).unwrap_or("all");
    let json_output = args.iter().any(|a| a == "--json");

    let manifests = load_manifests();

    let configs: Vec<RunConfig> = match experiment {
        "tier_capacity" => experiment_tier_capacity(),
        "tier_bandwidth" => experiment_tier_bandwidth(),
        "prefill_ratio" => experiment_prefill_ratio(),
        "load_sweep" => experiment_load_sweep(),
        "all" => {
            let mut all = Vec::new();
            all.extend(experiment_tier_capacity());
            all.extend(experiment_tier_bandwidth());
            all.extend(experiment_prefill_ratio());
            all.extend(experiment_load_sweep());
            all
        }
        other => {
            eprintln!("Unknown experiment: {other}");
            eprintln!("Options: tier_capacity, tier_bandwidth, prefill_ratio, load_sweep, all");
            std::process::exit(1);
        }
    };

    eprintln!("Running {} configurations...", configs.len());

    let results: Vec<RunResult> = configs
        .par_iter()
        .map(|cfg| {
            eprintln!("  {}", cfg.label);
            run_one(cfg, &manifests)
        })
        .collect();

    if json_output {
        println!("{}", serde_json::to_string_pretty(&results).unwrap());
    } else {
        println!(
            "{:<30} {:>8} {:>10} {:>10} {:>10} {:>10} {:>8}",
            "label", "sessions", "out_tok/s", "ttft_mean", "ttft_p99", "tpot", "hit_rate"
        );
        for r in &results {
            println!(
                "{:<30} {:>8} {:>10.0} {:>10.1} {:>10.1} {:>10.2} {:>8.3}",
                r.label,
                r.num_sessions,
                r.output_tok_s,
                r.ttft_mean_ms,
                r.ttft_p99_ms,
                r.tpot_mean_ms,
                r.cache_hit_rate,
            );
        }
    }
}
