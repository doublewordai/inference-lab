//! Demonstrates the KV-cache hierarchy and PCIe-bandwidth contention model.
//!
//! Sets up a tiered cache (HBM + host RAM), pre-warms a shared prefix into
//! host RAM by churning HBM, then submits N requests that all hit the same
//! prefix simultaneously. Time-to-ready is measured for each request, with
//! and without contention.
//!
//! Run with: `cargo run --example hierarchy_demo --no-default-features`

use inference_lab::config::{
    DenseModel, HardwareConfig, KVTier, ModelConfig, ModelCosts, Precision, SchedulerConfig,
};
use inference_lab::kv_cache::KVCacheManager;
use inference_lab::request::Request;
use inference_lab::scheduler::Scheduler;

fn run_for_batch(num_concurrent: usize, share_prefix: bool) -> Vec<f64> {
    let hardware = HardwareConfig {
        name: "demo".into(),
        flops_fp4: None,
        flops_fp8: None,
        flops_bf16: Some(1e15),
        flops_fp16: Some(1e15),
        memory_bandwidth: 3e12,
        memory_capacity: 80_000_000_000,
        // Modest HBM. With block-ref sharing, batches that all share the
        // same prefix only need one physical copy regardless of N.
        kv_cache_capacity: 2_000_000_000,
        gpu_memory_utilization: 0.9,
        // Single host-RAM tier, plenty of capacity, 1 GB/s PCIe so wait
        // time is observable.
        kv_tiers: vec![KVTier {
            name: "host_ram".into(),
            capacity_bytes: 100_000_000_000,
            bandwidth_to_hbm: 1e9,
        }],
    };
    let model = ModelConfig::Dense(DenseModel {
        name: "demo".into(),
        num_parameters: 7_000_000_000,
        num_active_parameters: None,
        num_layers: 32,
        hidden_dim: 4096,
        num_heads: 32,
        num_kv_heads: None,
        max_seq_len: 8192,
        precision: Precision::Bf16,
    });
    let scheduler_cfg = SchedulerConfig {
        max_num_batched_tokens: 8192,
        max_num_seqs: 256,
        policy: "fcfs".into(),
        enable_chunked_prefill: true,
        long_prefill_token_threshold: 0,
        max_num_partial_prefills: 1,
        block_size: 16,
        enable_preemption_free: false,
        enable_cascade_attention: false,
    };
    let config_hardware = hardware.clone();
    let config_model = model.clone();
    let config_scheduler = scheduler_cfg.clone();
    let block_size = scheduler_cfg.block_size;

    let kv_cache_manager = KVCacheManager::new(
        config_hardware.kv_cache_capacity,
        block_size,
        config_model.kv_storage_bytes(1),
        true,
    )
    .with_tiers(&config_hardware.kv_tiers);

    let mut scheduler = Scheduler::new(
        config_scheduler,
        config_hardware,
        config_model,
        kv_cache_manager,
    )
    .unwrap();

    // Pre-warm: pretend a long prefix lives in host RAM. We do this by
    // hand-poking the manager: allocate blocks for the prefix so its hashes
    // land in HBM, free them, then evict them by allocating fresh blocks.
    let prefix_blocks: u32 = 64; // 64 * 16 = 1024 tokens
    // Each request gets its own prefix; if `share_prefix` is set, all of
    // them get the same one. We pre-warm host RAM with all the prefixes
    // we're going to use.
    let prefix_hashes_per_req: Vec<Vec<u64>> = (0..num_concurrent)
        .map(|r| {
            let base = if share_prefix {
                1_000_000
            } else {
                1_000_000 + (r as u64) * 1_000_000
            };
            (0..prefix_blocks as u64).map(|i| base + i).collect()
        })
        .collect();
    {
        let mgr = scheduler.kv_cache_manager_mut();
        // Allocate-and-free each prefix once to register them in HBM, then
        // churn them down to host RAM.
        for hashes in &prefix_hashes_per_req {
            let mut seed = Request::new("seed".into(), 0, 0.0, prefix_blocks * block_size, 1);
            seed.prompt_block_hashes = hashes.clone();
            let blocks = mgr.allocate_blocks(&seed, prefix_blocks * block_size).unwrap();
            mgr.free_blocks(&blocks);
        }
        let churn_blocks = mgr.total_blocks() as u32;
        let mut churn = Request::new("churn".into(), 0, 0.0, churn_blocks * block_size, 1);
        churn.prompt_block_hashes = (0..churn_blocks as u64).map(|i| 9_000_000_000 + i).collect();
        mgr.allocate_blocks(&churn, churn_blocks * block_size).unwrap();
        let churn_blocks_alloc: Vec<u32> = (0..churn_blocks).collect();
        mgr.free_blocks(&churn_blocks_alloc);
    }

    for i in 0..num_concurrent {
        let mut req = Request::new(
            format!("req-{i}"),
            0,
            0.0,
            (prefix_blocks + 1) * block_size,
            1,
        );
        let mut hashes = prefix_hashes_per_req[i].clone();
        hashes.push(20_000_000_000 + i as u64);
        req.prompt_block_hashes = hashes;
        scheduler.add_request(req);
    }

    // Drive the simulation, recording the moment each request transitions
    // out of pending_transfers (its transfer has completed).
    let dt = 0.01;
    let mut t = 0.0;
    let _ = scheduler.schedule(t);
    let mut completion_times = vec![None; num_concurrent];
    let mut was_in_pending = vec![false; num_concurrent];
    while completion_times.iter().any(|c| c.is_none()) && t < 60.0 {
        t += dt;
        let _ = scheduler.schedule(t);
        for (i, slot) in completion_times.iter_mut().enumerate() {
            if slot.is_some() {
                continue;
            }
            let id = format!("req-{i}");
            let in_pending = scheduler
                .pending_transfers()
                .iter()
                .any(|r| r.request_id == id);
            if was_in_pending[i] && !in_pending {
                *slot = Some(t);
            }
            was_in_pending[i] = in_pending;
        }
    }
    completion_times
        .into_iter()
        .map(|t| t.unwrap_or(f64::NAN))
        .collect()
}

fn main() {
    let prefix_mb = 1024 * 8192 * 32 * 2 / 1_000_000;
    println!(
        "Promotion latency on a 1 GB/s host-RAM tier. Prefix per request: 1024 tokens ({} MB).",
        prefix_mb
    );
    println!();

    println!("Case A: all requests share the same prefix (sim joins them on one transfer).");
    println!("{:>6}  {:>14}  {:>14}", "batch", "first_done(s)", "last_done(s)");
    println!("{}", "-".repeat(40));
    for &n in &[1usize, 2, 4, 8, 16] {
        let times = run_for_batch(n, /*share_prefix=*/ true);
        let first = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let last = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("{:>6}  {:>14.3}  {:>14.3}", n, first, last);
    }

    println!();
    println!("Case B: each request has its own prefix (transfers contend on shared PCIe).");
    println!("{:>6}  {:>14}  {:>14}", "batch", "first_done(s)", "last_done(s)");
    println!("{}", "-".repeat(40));
    for &n in &[1usize, 2, 4, 8, 16] {
        let times = run_for_batch(n, /*share_prefix=*/ false);
        let first = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let last = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("{:>6}  {:>14.3}  {:>14.3}", n, first, last);
    }
}
