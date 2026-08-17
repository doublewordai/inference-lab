//! Demonstrates the KV-cache hierarchy and PCIe-bandwidth contention model.
//!
//! Sets up a tiered cache (HBM + host RAM), pre-warms a shared prefix into
//! host RAM by churning HBM, then submits N requests that all hit the same
//! prefix simultaneously. Time-to-ready is measured for each request, with
//! and without contention.
//!
//! Run with: `cargo run --example hierarchy_demo --no-default-features`

use inference_lab::catalog;
use inference_lab::config::SchedulerConfig;
use inference_lab::kv_cache::KVCacheManager;
use inference_lab::request::Request;
use inference_lab::scheduler::Scheduler;
use inference_lab::scheduler::SchedulingPolicy;

fn run_for_batch(num_concurrent: usize, share_prefix: bool) -> Vec<f64> {
    // A dense 70B-class model from the catalog; only its KV footprint matters here.
    let model = catalog::model("llama-3-70b").expect("catalog preset");
    let scheduler_cfg = SchedulerConfig {
        max_num_batched_tokens: 8192,
        max_num_seqs: 256,
        policy: SchedulingPolicy::FCFS,
        enable_chunked_prefill: true,
        long_prefill_token_threshold: 0,
        max_num_partial_prefills: 1,
        block_size: 16,
        gpu_memory_utilization: 0.9,
        kv_cache_capacity: 0,
        max_model_len: None,
        enable_preemption_free: false,
        enable_cascade_attention: false,
    };
    let config_model = model.clone();
    let config_scheduler = scheduler_cfg.clone();
    let block_size = scheduler_cfg.block_size;

    let kv_model = config_model.clone();
    let mut kv_cache_manager = KVCacheManager::new(
        2_000_000_000, // 2 GB of HBM for KV: modest on purpose
        block_size,
        move |t| kv_model.kv_storage_bytes(t),
        config_model.per_sequence_state_bytes(),
        true,
    )
    // Single host-RAM tier, plenty of capacity, 1 GB/s PCIe so wait time
    // is observable.
    .with_private_tiers(&[("host_ram", 100_000_000_000, 1e9)]);

    // Pre-warm: pretend a long prefix lives in host RAM. We do this by
    // hand-poking the manager before handing it to the scheduler: allocate
    // blocks for the prefix so its hashes land in HBM, free them, then evict
    // them by allocating fresh blocks.
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
        let mgr = &mut kv_cache_manager;
        // Allocate-and-free each prefix once to register them in HBM, then
        // churn them down to host RAM.
        for hashes in &prefix_hashes_per_req {
            let mut seed = Request::new("seed".into(), 0, 0.0, prefix_blocks * block_size, 1);
            seed.prompt_block_hashes = hashes.clone();
            let n = mgr
                .allocate_blocks(&seed, prefix_blocks * block_size)
                .unwrap();
            seed.kv_blocks.extend(n);
            mgr.free_blocks(&seed);
        }
        let churn_blocks = mgr.total_blocks() as u32;
        let mut churn = Request::new("churn".into(), 0, 0.0, churn_blocks * block_size, 1);
        churn.prompt_block_hashes = (0..churn_blocks as u64)
            .map(|i| 9_000_000_000 + i)
            .collect();
        let n = mgr
            .allocate_blocks(&churn, churn_blocks * block_size)
            .unwrap();
        churn.kv_blocks.extend(n);
        mgr.free_blocks(&churn);
    }

    let mut scheduler = Scheduler::new(config_scheduler, kv_cache_manager);

    for (i, prefix_hashes) in prefix_hashes_per_req.iter().enumerate() {
        let mut req = Request::new(
            format!("req-{i}"),
            0,
            0.0,
            (prefix_blocks + 1) * block_size,
            1,
        );
        let mut hashes = prefix_hashes.clone();
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
    println!(
        "{:>6}  {:>14}  {:>14}",
        "batch", "first_done(s)", "last_done(s)"
    );
    println!("{}", "-".repeat(40));
    for &n in &[1usize, 2, 4, 8, 16] {
        let times = run_for_batch(n, /*share_prefix=*/ true);
        let first = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let last = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("{:>6}  {:>14.3}  {:>14.3}", n, first, last);
    }

    println!();
    println!("Case B: each request has its own prefix (transfers contend on shared PCIe).");
    println!(
        "{:>6}  {:>14}  {:>14}",
        "batch", "first_done(s)", "last_done(s)"
    );
    println!("{}", "-".repeat(40));
    for &n in &[1usize, 2, 4, 8, 16] {
        let times = run_for_batch(n, /*share_prefix=*/ false);
        let first = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let last = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("{:>6}  {:>14.3}  {:>14.3}", n, first, last);
    }
}
