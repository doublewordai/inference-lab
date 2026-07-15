//! Wide pure-decode roofline sweep: the disagg decode pool in isolation.
//!
//! Requests arrive already prefilled (skip_prefill), so there is no prefill
//! compute sharing the GPU -- exactly the no-competition regime that disagg
//! gives. KV caps lifted (huge memory) so the *compute* roofline binds, like the
//! old "caps lifted" sweeps. Sweep batch 1 -> 32768 and check the adaptive
//! GoodputBudget policy tracks the best fixed gamma across the whole curve,
//! including the compute-bound far end where speculation should switch off.
//!
//! V4-Flash / B200, ISL=3840 (long context for attention) OSL=512, alpha=0.75,
//! c_draft=0.10 -- the ISL>>OSL regime that gave the colocated tail.
//!
//! Run: `cargo run --release --example spec_decode_roofline --no-default-features`

use inference_lab::config::{
    AcceptanceModel, ClusterSpec, DeepseekV4Model, GammaPolicy, HardwareConfig, ModelConfig,
    ParallelConfig, Precision, SchedulerConfig, SpeculativeConfig,
};
use inference_lab::simulation::{simulate_closed_loop, Topology};
use rayon::prelude::*;
use std::collections::BTreeMap;

fn b200_unlimited_kv() -> HardwareConfig {
    HardwareConfig {
        name: "B200".into(),
        flops_fp4: Some(9.0e15),
        flops_fp8: Some(4.5e15),
        flops_bf16: Some(2.25e15),
        flops_fp16: Some(2.25e15),
        memory_bandwidth: 8.0e12,
        memory_capacity: 1_000_000_000_000_000, // 1 PB -> KV cap never binds
        kv_cache_capacity: 0,
        gpu_memory_utilization: 0.9,
        kv_tiers: Vec::new(),
    }
}

fn deepseek_v4_flash() -> ModelConfig {
    ModelConfig::DeepseekV4(DeepseekV4Model {
        name: "DeepSeek-V4-Flash".into(),
        num_layers: 43,
        hidden_dim: 4096,
        num_heads: 64,
        max_seq_len: 1_048_576,
        kv_latent_dim: 512,
        qk_rope_head_dim: 64,
        kv_precision: Precision::Fp8,
        num_active_expert_params: 7_574_913_024,
        num_active_non_expert_params: 5_660_947_776,
        num_resident_expert_params: 278_107_521_024,
        num_resident_non_expert_params: 6_225_000_000,
        expert_precision: Precision::Fp4,
        non_expert_precision: Precision::Fp8,
        window_size: 128,
        num_dense_layers: 2,
        num_near_layers: 21,
        num_far_layers: 20,
        near_compress_ratio: 4,
        far_compress_ratio: 128,
        index_topk: 512,
        index_n_heads: 64,
        index_head_dim: 128,
        index_kv_precision: None,
        num_experts_per_tok: 6,
        num_routed_experts: 256,
        num_moe_layers: 43,
    })
}

fn topology() -> Topology {
    let cluster = ClusterSpec {
        hardware: b200_unlimited_kv(),
        parallel: ParallelConfig {
            tp: 1,
            ep: 1,
            dp_attention: false,
        },
        comms: None,
        num_workers: 1,
        node: 0,
    };
    let sched = SchedulerConfig {
        max_num_batched_tokens: 100_000_000, // lift token budget too
        max_num_seqs: 10_000_000,
        enable_chunked_prefill: false,
        long_prefill_token_threshold: 0,
        max_num_partial_prefills: 1,
        block_size: 64,
        policy: "fcfs".into(),
        enable_preemption_free: false,
        enable_cascade_attention: false,
    };
    Topology::aggregated(cluster, deepseek_v4_flash(), sched).expect("topo")
}

#[derive(Clone, Copy)]
enum Policy {
    NoSpec,
    Fixed(u32),
    Budget(u32),
}

fn spec_for(p: Policy, alpha: f64, c_draft: f64) -> Option<SpeculativeConfig> {
    let acceptance = AcceptanceModel::Constant { alpha };
    match p {
        Policy::NoSpec => None,
        Policy::Fixed(g) => Some(SpeculativeConfig {
            gamma: g,
            acceptance,
            policy: GammaPolicy::Fixed,
            draft_cost_frac: c_draft,
            measured_cost: None,
            switch: Default::default(),
            drafter: None,
        }),
        Policy::Budget(g) => Some(SpeculativeConfig {
            gamma: g,
            acceptance,
            policy: GammaPolicy::GoodputBudget,
            draft_cost_frac: c_draft,
            measured_cost: None,
            switch: Default::default(),
            drafter: None,
        }),
    }
}

fn run(conc: u32, isl: u32, osl: u32, p: Policy, alpha: f64, c_draft: f64) -> (f64, f64) {
    let total = (conc * 2).max(1000);
    let warmup = conc / 2;
    let res = simulate_closed_loop(
        topology(),
        conc,
        isl,
        osl,
        total,
        warmup,
        spec_for(p, alpha, c_draft),
        7,
        true,
    )
    .expect("run");
    let dbatch = res
        .mean_batch_per_pool
        .first()
        .copied()
        .flatten()
        .unwrap_or(0.0);
    (res.throughput() * osl as f64, dbatch)
}

fn main() {
    let isl = 3840u32;
    let osl = 512u32;
    let alpha = 0.75;
    let c_draft = 0.10;
    let gamma_max = 8u32;
    let fixed = [1u32, 2, 3, 4, 5, 6, 7, 8];
    let concs = [
        1u32, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
    ];

    // Column layout per conc: 0 = nospec, 1..=8 = fixed g1..g8, 9 = BUDGET.
    let mut tasks: Vec<(u32, usize, Policy)> = Vec::new();
    for &conc in &concs {
        tasks.push((conc, 0, Policy::NoSpec));
        for (i, &g) in fixed.iter().enumerate() {
            tasks.push((conc, 1 + i, Policy::Fixed(g)));
        }
        tasks.push((conc, 9, Policy::Budget(gamma_max)));
    }

    // Bounded pool: a single conc=16384 sim peaks ~20 GB (live batch = conc, KV caps
    // lifted), so cap concurrency to keep peak memory well under box RAM. Most of the
    // wall-clock is the few largest concs anyway, so this still parallelizes the
    // expensive part. Tune via RAYON_NUM_THREADS; defaults to 8 here.
    let nthreads = std::env::var("ROOFLINE_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8usize);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(nthreads)
        .build()
        .expect("pool");
    let results: Vec<(u32, usize, f64, f64)> = pool.install(|| {
        tasks
            .par_iter()
            .map(|&(conc, col, p)| {
                let (gp, db) = run(conc, isl, osl, p, alpha, c_draft);
                (conc, col, gp, db)
            })
            .collect()
    });

    let mut gp: BTreeMap<u32, [f64; 10]> = BTreeMap::new();
    let mut dbatch: BTreeMap<u32, f64> = BTreeMap::new();
    for (conc, col, g, db) in results {
        gp.entry(conc).or_insert([0.0; 10])[col] = g;
        if col == 0 {
            dbatch.insert(conc, db);
        }
    }

    println!("Pure-decode roofline sweep (disagg decode pool in isolation, KV caps lifted)");
    println!("V4-Flash B200 TP1, ISL={isl} OSL={osl} alpha={alpha} c_draft={c_draft}\n");
    print!(
        "{:>7} {:>8} {:>9} {:>9} {:>9} {:>7} {:>7}",
        "conc", "dbatch", "nospec", "best-fix", "BUDGET", "argmax", "d%"
    );
    println!();
    for &conc in &concs {
        let row = gp[&conc];
        let ns = row[0];
        let bud = row[9];
        let mut best = ns;
        let mut bestg = 0u32;
        for (i, &g) in fixed.iter().enumerate() {
            if row[1 + i] > best {
                best = row[1 + i];
                bestg = g;
            }
        }
        let d = 100.0 * (bud - best) / best;
        let argmax = if bestg == 0 {
            "nospec".into()
        } else {
            format!("g{bestg}")
        };
        println!(
            "{conc:>7} {:>8.0} {ns:>9.0} {best:>9.0} {bud:>9.0} {argmax:>7} {d:>+7.2}",
            dbatch[&conc]
        );
    }
}
