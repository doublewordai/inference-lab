//! Roofline sweep across the 6 GB300 dynamo-sglang pareto points from
//! InferenceX (PRs #1157, #1232; ISL=8192 OSL=1024 FP4). For each point we
//! predict TPOT, mean E2E latency, and throughput-per-GPU under a closed-loop
//! steady-state model:
//!
//! ```text
//!   prefill_tput  = num_p_workers / prefill_time          (requests/sec)
//!   decode_cap    = conc / (osl × tpot)                   (requests/sec)
//!   throughput    = min(prefill_tput, decode_cap)         (requests/sec)
//!   mean_e2e      = conc / throughput                     (Little's law)
//!   tput_per_gpu  = throughput × (isl + osl) / total_gpus
//! ```
//!
//! Run with: `cargo run --example dsv4_decode_tpot_sweep --no-default-features`
//!
//! Hypothesis behind the comparison: the gap to roofline at batch=1 is mostly
//! fixed per-iteration overhead (kernel launches, sparse-attn scoring, MLA
//! bookkeeping). At higher batch the absolute gap should stay roughly flat
//! while the multiplicative gap shrinks.

use inference_lab::config::{
    ClusterSpec, CommsConfig, DeepseekV4Model, DisaggTopology, HardwareConfig, ModelConfig, Node,
    ParallelConfig, SchedulerConfig,
};
use inference_lab::simulation::{
    predict_decode_tpot, predict_prefill_time, simulate_closed_loop, Topology,
};

fn b300_per_gpu() -> HardwareConfig {
    HardwareConfig {
        name: "B300".into(),
        compute_flops: 1.5e16,
        memory_bandwidth: 8.0e12,
        memory_capacity: 309_237_645_312,
        kv_cache_capacity: 0,
        gpu_memory_utilization: 0.9,
        bytes_per_param: 1,
        kv_tiers: Vec::new(),
    }
}

fn dsv4_pro() -> ModelConfig {
    ModelConfig::DeepseekV4(DeepseekV4Model {
        name: "DeepSeek-V4-Pro".into(),
        num_parameters: 1_600_000_000_000,
        num_active_parameters: 49_000_000_000,
        num_layers: 61,
        hidden_dim: 7168,
        num_heads: 128,
        max_seq_len: 1_048_576,
        kv_latent_dim: 512,
        kv_bytes_per_value: 1,
        effective_bytes_per_active_param: 0.625,
        effective_bytes_per_resident_param: 0.625,
        window_size: 128,
        num_dense_layers: 1,
        num_near_layers: 30,
        num_far_layers: 30,
        near_compress_ratio: 4,
        far_compress_ratio: 128,
        index_topk: 1024,
        index_n_heads: 64,
        index_head_dim: 128,
        index_kv_bytes_per_value: None,
        num_experts_per_tok: 6,
        num_moe_layers: 61,
    })
}

fn nvlink5_comms() -> CommsConfig {
    CommsConfig {
        link_bw: 9.0e11,
        allreduce_latency: 5.0e-6,
        alltoall_latency: 8.0e-6,
    }
}

fn cluster(tp: u32, ep: u32, dp_attention: bool) -> ClusterSpec {
    ClusterSpec {
        hardware: b300_per_gpu(),
        parallel: ParallelConfig { tp, ep, dp_attention },
        comms: Some(nvlink5_comms()),
        num_workers: 1,
        node: 0,
    }
}

struct Point {
    label: &'static str,
    conc: u32,
    p_workers: u32,
    p_tp: u32,
    p_ep: u32,
    p_dp: bool,
    d_tp: u32,
    d_ep: u32,
    d_dp: bool,
    real_tpot_ms: f64,
    real_e2e_s: f64,
    real_tput_per_gpu: f64,
}

fn main() {
    let model = dsv4_pro();

    let isl: u32 = 8192;
    let osl: u32 = 1024;
    // Average decode-phase seq len = ISL + OSL/2.
    let avg_seq_len = isl + osl / 2;

    // Real numbers from data/inferencex/dynamo-sglang bmk JSONs (PR #1157,
    // #1232). One decode worker per point; prefill scales with `p_workers`.
    let points = vec![
        // p_dp / d_dp track sglang's `--enable-dp-attention` per worker, taken
        // from the bmk JSON filenames (dpfalse/dptrue suffixes).
        Point {
            label: "conc=1     P 1×TP4/EP1   D 1×TP4/EP1  ",
            conc: 1, p_workers: 1, p_tp: 4, p_ep: 1, p_dp: false, d_tp: 4, d_ep: 1, d_dp: false,
            real_tpot_ms: 11.90, real_e2e_s: 12.09, real_tput_per_gpu: 86.1,
        },
        Point {
            label: "conc=512   P 1×TP4/EP4   D 1×TP16/EP16",
            conc: 512, p_workers: 1, p_tp: 4, p_ep: 4, p_dp: true, d_tp: 16, d_ep: 16, d_dp: true,
            real_tpot_ms: 22.55, real_e2e_s: 75.79, real_tput_per_gpu: 2573.1,
        },
        Point {
            label: "conc=512   P 1×TP4/EP4   D 1×TP8/EP8  ",
            conc: 512, p_workers: 1, p_tp: 4, p_ep: 4, p_dp: true, d_tp: 8, d_ep: 8, d_dp: true,
            real_tpot_ms: 27.80, real_e2e_s: 76.12, real_tput_per_gpu: 4256.2,
        },
        Point {
            label: "conc=1024  P 2×TP4/EP4   D 1×TP8/EP8  ",
            conc: 1024, p_workers: 2, p_tp: 4, p_ep: 4, p_dp: true, d_tp: 8, d_ep: 8, d_dp: true,
            real_tpot_ms: 35.60, real_e2e_s: 77.76, real_tput_per_gpu: 6171.2,
        },
        Point {
            label: "conc=2048  P 4×TP4/EP4   D 1×TP8/EP8  ",
            conc: 2048, p_workers: 4, p_tp: 4, p_ep: 4, p_dp: true, d_tp: 8, d_ep: 8, d_dp: true,
            real_tpot_ms: 46.09, real_e2e_s: 79.17, real_tput_per_gpu: 8399.5,
        },
        Point {
            label: "conc=16384 P 14×TP4/EP4  D 1×TP16/EP16",
            conc: 16384, p_workers: 14, p_tp: 4, p_ep: 4, p_dp: true, d_tp: 16, d_ep: 16, d_dp: true,
            real_tpot_ms: 69.35, real_e2e_s: 179.20, real_tput_per_gpu: 9502.4,
        },
    ];

    let scheduler_cfg = SchedulerConfig {
        max_num_batched_tokens: 32768,
        max_num_seqs: 32768,
        enable_chunked_prefill: false,
        long_prefill_token_threshold: 0,
        max_num_partial_prefills: 1,
        block_size: 64,
        policy: "fcfs".into(),
        enable_preemption_free: true,
        enable_cascade_attention: false,
    };

    println!("DSv4-Pro pareto sweep: roofline + DES vs InferenceX (GB300, dynamo-sglang, 8k/1k FP4)");
    println!("Roofline = analytic max(compute, memory) + collectives.");
    println!("DES      = topology-aware discrete-event simulator (closed-loop).\n");

    println!(
        "{:<43}  {:>22}  {:>22}  {:>22}  {:>20}  {:>13}",
        "operating point", "TPOT (ms)", "E2E (s)", "tput/gpu (tok/s)",
        "decode batch", "prefill batch",
    );
    println!(
        "{:<43}  {:>6} {:>6} {:>7}  {:>6} {:>6} {:>7}  {:>6} {:>6} {:>7}  {:>6} {:>6} {:>6}  {:>6} {:>6}",
        "", "real", "roof", "des", "real", "roof", "des", "real", "roof", "des",
        "conc", "L-law", "des", "cap", "des",
    );
    println!("{}", "-".repeat(43 + 2 + 22 + 2 + 22 + 2 + 22 + 2 + 20 + 2 + 13));

    for p in &points {
        let p_cluster = cluster(p.p_tp, p.p_ep, p.p_dp);
        let d_cluster = cluster(p.d_tp, p.d_ep, p.d_dp);

        // --- roofline (analytic, self-consistent) ---
        // Joint fixed point on decode batch N and throughput λ:
        //   N = min(λ × osl × TPOT(N), conc)
        //   λ = min(p_workers / T_P, conc / (T_P + osl × TPOT(N)))
        // Start at N = conc (the pessimistic, fully-loaded-decode batch) and
        // iterate. Converges in a handful of steps because TPOT(N) is roughly
        // linear in N once we're memory-bound on KV reads.
        let prefill_s = predict_prefill_time(&p_cluster, &model, isl);
        let prefill_tput = p.p_workers as f64 / prefill_s;
        let mut n_decode_lit = p.conc as f64;
        let mut tpot_s = predict_decode_tpot(&d_cluster, &model, p.conc, avg_seq_len);
        let mut throughput = prefill_tput;
        for _ in 0..32 {
            let batch = n_decode_lit.round().max(1.0) as u32;
            tpot_s = predict_decode_tpot(&d_cluster, &model, batch, avg_seq_len);
            let user_cycle_tput = p.conc as f64 / (prefill_s + osl as f64 * tpot_s);
            throughput = prefill_tput.min(user_cycle_tput);
            let next = (throughput * osl as f64 * tpot_s).min(p.conc as f64);
            if (next - n_decode_lit).abs() < 0.5 {
                n_decode_lit = next;
                break;
            }
            n_decode_lit = next;
        }
        let mean_e2e_roof = p.conc as f64 / throughput;
        let total_gpus = p.p_workers * p.p_tp + p.d_tp;
        let tput_per_gpu_roof = throughput * (isl + osl) as f64 / total_gpus as f64;

        // --- DES (closed-loop disagg) ---
        let topology = DisaggTopology {
            nodes: vec![Node {
                name: "gb300-nvl72".into(),
                num_gpus: 72,
                intra_node_link_bw: 9.0e11,
            }],
            inter_node_link_bw: None,
            prefill: ClusterSpec {
                num_workers: p.p_workers,
                ..p_cluster.clone()
            },
            decode: d_cluster.clone(),
            kv_link_bw: 9.0e11,
        };
        let topo = Topology::from_disagg(&topology, model.clone(), scheduler_cfg.clone())
            .expect("topology");
        // Run 2× conc completions: first conc are warmup (transient first
        // cycle), remaining conc are measurement (steady state).
        let total_completions = (2 * p.conc).max(64);
        let warmup = p.conc;
        let des = simulate_closed_loop(topo, p.conc, isl, osl, total_completions, warmup)
            .expect("des");
        let des_tpot_ms = des.mean_tpot() * 1000.0;
        let des_e2e_s = des.mean_e2e();
        let des_tput_per_gpu = des.throughput() * (isl + osl) as f64 / total_gpus as f64;
        // Roof assumes the decode pool runs at batch=conc; in practice
        // Little's law on the decode pool alone gives a much smaller batch
        // when prefill (or anything upstream) is the bottleneck.
        let des_decode_batch = des
            .mean_batch_per_pool
            .get(1)
            .copied()
            .flatten()
            .unwrap_or(0.0);
        let des_prefill_batch = des
            .mean_batch_per_pool
            .first()
            .copied()
            .flatten()
            .unwrap_or(0.0);
        // Prefill batch is structurally pinned, not L-law-determined. Each
        // prefill iter packs in `floor(max_num_batched_tokens / isl)` sequences
        // (here 32768/8192 = 4). L-law on prefill is degenerate: iter_time
        // scales linearly with B (compute-bound), so N_pool = λ × B × T_P
        // collapses to the throughput identity λ = p_workers/T_P with B free.
        // The structural cap is what binds. With only 1 request total
        // (conc=1) we're below the cap, so report min(cap, conc/p_workers).
        let pack_cap = (scheduler_cfg.max_num_batched_tokens as f64 / isl as f64).floor();
        let n_prefill_struct = pack_cap.min(p.conc as f64 / p.p_workers as f64).max(1.0);

        println!(
            "{:<43}  {:>6.2} {:>6.2} {:>7.2}  {:>6.2} {:>6.2} {:>7.2}  {:>6.0} {:>6.0} {:>7.0}  {:>6} {:>6.0} {:>6.0}  {:>6.2} {:>6.2}",
            p.label,
            p.real_tpot_ms, tpot_s * 1000.0, des_tpot_ms,
            p.real_e2e_s, mean_e2e_roof, des_e2e_s,
            p.real_tput_per_gpu, tput_per_gpu_roof, des_tput_per_gpu,
            p.conc, n_decode_lit, des_decode_batch,
            n_prefill_struct, des_prefill_batch,
        );
    }

    println!();
    println!("Both `roof` and `des` use the same per-iteration ComputeEngine arithmetic.");
    println!("They differ only in how the closed-loop pipeline is composed: roof = analytic");
    println!("Little's law on T_P + T_D scalars; des = full event-driven topology with");
    println!("queue dynamics across {n_points} configs.", n_points = points.len());
}
