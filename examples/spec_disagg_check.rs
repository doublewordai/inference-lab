//! Disaggregated control for the ISL>>OSL over-speculation tail.
//!
//! Colocated (single pool), the decode-only GoodputBudget policy over-speculates
//! by up to ~6% at large batch when ISL >> OSL, because deep verify competes with
//! the prefill sharing the GPU. Prediction: disaggregate prefill onto its own
//! pool and that competition disappears -- the decode pool runs pure decode, so
//! the decode-only policy should match the best fixed gamma exactly.
//!
//! Setup: prefill pool with `P_WORKERS` workers (provisioned so prefill is not
//! the bottleneck -> the decode batch fills toward `conc`), one decode worker
//! running spec. Same V4-Flash / B200, same ISL=3840/OSL=512, alpha=0.75,
//! c_draft=0.10 as the colocated C1 sweep.
//!
//! Run: `cargo run --release --example spec_disagg_check --no-default-features`

use inference_lab::config::DisaggTopology;
use inference_lab::config::{
    AcceptanceModel, ClusterSpec, CommsConfig, DeepseekV4Model, GammaPolicy, HardwareConfig,
    ModelConfig, Node, ParallelConfig, Precision, SchedulerConfig, SpeculativeConfig,
};
use inference_lab::simulation::{simulate_closed_loop, Topology};
use rayon::prelude::*;
use std::collections::BTreeMap;

fn b200() -> HardwareConfig {
    HardwareConfig {
        name: "B200".into(),
        flops_fp4: Some(9.0e15),
        flops_fp8: Some(4.5e15),
        flops_bf16: Some(2.25e15),
        flops_fp16: Some(2.25e15),
        memory_bandwidth: 8.0e12,
        memory_capacity: 206_158_430_208,
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

fn scheduler() -> SchedulerConfig {
    SchedulerConfig {
        max_num_batched_tokens: 8192,
        max_num_seqs: 32768,
        enable_chunked_prefill: true,
        long_prefill_token_threshold: 0,
        max_num_partial_prefills: 1,
        block_size: 64,
        policy: "fcfs".into(),
        enable_preemption_free: true,
        enable_cascade_attention: false,
    }
}

fn cluster(num_workers: u32) -> ClusterSpec {
    ClusterSpec {
        hardware: b200(),
        parallel: ParallelConfig {
            tp: 1,
            ep: 1,
            dp_attention: false,
        },
        comms: Some(CommsConfig {
            link_bw: 9.0e11,
            allreduce_latency: 5e-6,
            alltoall_latency: 8e-6,
        }),
        num_workers,
        node: 0,
    }
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

/// Returns (goodput_tok_s, decode_batch).
fn run(
    conc: u32,
    isl: u32,
    osl: u32,
    p_workers: u32,
    p: Policy,
    alpha: f64,
    c_draft: f64,
) -> (f64, f64) {
    let topo = DisaggTopology {
        nodes: vec![Node {
            name: "b200".into(),
            num_gpus: 72,
            intra_node_link_bw: 9.0e11,
        }],
        inter_node_link_bw: None,
        prefill: ClusterSpec {
            num_workers: p_workers,
            ..cluster(p_workers)
        },
        decode: cluster(1),
        kv_link_bw: 9.0e11,
    };
    let topology = Topology::from_disagg(&topo, deepseek_v4_flash(), scheduler()).expect("topo");
    let total = (conc * 16).max(2000);
    let warmup = conc * 4;
    let res = simulate_closed_loop(
        topology,
        conc,
        isl,
        osl,
        total,
        warmup,
        spec_for(p, alpha, c_draft),
        7,
        false,
    )
    .expect("run");
    let goodput = res.throughput() * osl as f64;
    let dbatch = res
        .mean_batch_per_pool
        .get(1)
        .copied()
        .flatten()
        .unwrap_or(0.0);
    (goodput, dbatch)
}

fn main() {
    let isl = 3840u32;
    let osl = 512u32;
    let alpha = 0.75;
    let c_draft = 0.10;
    let gamma_max = 8u32;
    let fixed = [1u32, 2, 3, 4, 5, 6, 7, 8];
    // One dedicated prefill worker per user -> prefill never queues, decode batch
    // fills toward conc (verified by the `dbatch` column) until the single decode
    // worker saturates. Low concs probe the no-spec/shallow shoulder; high concs
    // probe decode-pool saturation.
    let concs = [1u32, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

    // Column layout per conc: 0 = nospec, 1..=8 = fixed g1..g8, 9 = BUDGET.
    // Build a flat task list over (conc, col, policy) and run every sim in parallel;
    // the decode pool is a single worker so per-sim live state stays small.
    let mut tasks: Vec<(u32, usize, Policy)> = Vec::new();
    for &conc in &concs {
        tasks.push((conc, 0, Policy::NoSpec));
        for (i, &g) in fixed.iter().enumerate() {
            tasks.push((conc, 1 + i, Policy::Fixed(g)));
        }
        tasks.push((conc, 9, Policy::Budget(gamma_max)));
    }
    let results: Vec<(u32, usize, f64, f64)> = tasks
        .par_iter()
        .map(|&(conc, col, p)| {
            let p_workers = conc; // dedicated prefill per user
            let (gp, db) = run(conc, isl, osl, p_workers, p, alpha, c_draft);
            (conc, col, gp, db)
        })
        .collect();

    let mut gp: BTreeMap<u32, [f64; 10]> = BTreeMap::new();
    let mut dbatch: BTreeMap<u32, f64> = BTreeMap::new();
    for (conc, col, g, db) in results {
        gp.entry(conc).or_insert([0.0; 10])[col] = g;
        if col == 0 {
            dbatch.insert(conc, db);
        }
    }

    println!("Disagg control: V4-Flash, B200, decode pool TP1, prefill pool = conc workers");
    println!("ISL={isl} OSL={osl} alpha={alpha} c_draft={c_draft}. goodput = committed tok/s.\n");
    print!("{:>5} {:>8}", "conc", "dbatch");
    print!(" {:>9}", "nospec");
    for g in fixed {
        print!(" {:>9}", format!("fix g{g}"));
    }
    print!(
        " {:>9} {:>9} {:>7} {:>7}",
        "best-fix", "BUDGET", "argmax", "d%"
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
        print!("{conc:>5} {:>8.1}", dbatch[&conc]);
        print!(" {ns:>9.0}");
        for i in 0..8 {
            print!(" {:>9.0}", row[1 + i]);
        }
        let argmax = if bestg == 0 {
            "nospec".into()
        } else {
            format!("g{bestg}")
        };
        print!(" {best:>9.0} {bud:>9.0} {argmax:>7} {d:>+7.2}");
        println!();
    }
    println!("\nPrediction: with prefill disaggregated, the decode pool runs pure decode,");
    println!("so BUDGET should match best-fix (d% ~ 0) even at ISL >> OSL.");
}
