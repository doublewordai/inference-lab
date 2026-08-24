//! Lane P: predicted throughput vs γ at *measured* phase times, stated before
//! the accept-length measurement (the brief's step 1).
//!
//! Same acceptance/verify model as lane J's `glm_isambard_spec_pricing`
//! (verify batch B·(γ+1); one MTP layer per draft token; committed =
//! Σ_{i≤γ} α^i), restricted to the stack we actually run — EP16 on the
//! coalesce2-LL + FULL-decode-graphs build — and priced at today's measured
//! per-phase floors instead of the design targets:
//!
//! * 196 µs — lane E's coalesce2 microbench p50 mean (dispatch 154 / combine 238)
//! * 853 µs — lane M's real-serving effective floor under graphs (128 ms
//!   comms/step ÷ 150 phase-calls; each phase kernel launches 2×/layer/step)
//! * 100 µs — lane B's design target, kept as the reference column
//!
//! Two extra tables: a calibration row against lane M's measured decode point
//! (B=512, ISL≈1.5k, HBM KV — measured 201 tok/s/GPU steady), and a batch
//! sweep locating the crossover batch B* past which speculation stops paying
//! (draft tokens consume hide-knee budget: verify tokens/expert = B·(γ+1)·8/256
//! vs the ~13 knee).

use inference_lab::catalog;
use inference_lab::compute::ComputeEngine;
use inference_lab::config::{HardwareConfig, MoeOverlap, ParallelConfig};
use inference_lab::request::{Request, SessionSpec};

const CONTEXT_CAP: u32 = 32_768;
const CORE_KV_BYTES_PER_TOKEN: u64 = 78 * (512 + 64);
const HBM_KV_BYTES_PER_GPU: f64 = 20e9;
const GRACE_ACTIVE_KV_BW: f64 = 311e9;

const LATENCIES_US: [u32; 3] = [100, 196, 853];
const GAMMAS: [u32; 5] = [0, 1, 2, 3, 4];
const ALPHAS: [f64; 4] = [0.60, 0.70, 0.80, 0.90];

fn contexts(path: &str, batch: usize) -> Vec<u32> {
    let sessions = SessionSpec::load(path).unwrap_or_else(|error| panic!("{error}"));
    let mut all: Vec<u32> = sessions
        .iter()
        .flat_map(|session| session.steps.iter())
        .map(|step| step.input.min(CONTEXT_CAP))
        .collect();
    all.sort_unstable();
    (0..batch)
        .map(|i| all[((2 * i + 1) * all.len() / (2 * batch)).min(all.len() - 1)])
        .collect()
}

fn requests(contexts: &[u32]) -> Vec<Request> {
    contexts
        .iter()
        .enumerate()
        .map(|(i, &context)| {
            let mut request = Request::new(format!("r{i}"), 0, 0.0, context, 8);
            request.num_computed_tokens = context;
            request
        })
        .collect()
}

fn hardware(latency_us: u32) -> HardwareConfig {
    let mut hardware = catalog::hardware("gh200").unwrap();
    hardware
        .fabric
        .as_mut()
        .unwrap()
        .scale_out
        .as_mut()
        .unwrap()
        .latency = latency_us as f64 * 1e-6;
    hardware
}

fn engine(hardware: HardwareConfig, overlap: MoeOverlap, kv_bw: Option<f64>) -> ComputeEngine {
    let model = catalog::model("glm-5.2-fp8").unwrap();
    let parallel = ParallelConfig {
        tp: 16,
        ep: 16,
        dp_attention: true,
        moe_overlap: overlap,
        megakernel: None,
    };
    let e = ComputeEngine::new(hardware, parallel, model);
    match kv_bw {
        Some(bw) => e.with_kv_memory_bandwidth(bw),
        None => e,
    }
}

fn expected_committed(gamma: u32, acceptance: f64) -> f64 {
    (0..=gamma).map(|i| acceptance.powi(i as i32)).sum()
}

/// Step time (s) for B sequences verifying (γ+1) tokens each, plus γ MTP
/// draft passes priced at one representative layer each.
fn step_time(eng: &ComputeEngine, refs: &[&Request], batch: usize, gamma: u32, layers: f64) -> f64 {
    let verify = gamma + 1;
    let verify_cost = eng.step_cost(refs, &vec![verify; batch]).time;
    let one_token = eng.step_cost(refs, &vec![1; batch]).time;
    verify_cost + gamma as f64 * one_token / layers
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: glm_isambard_p_spec_prediction <tracelab.jsonl>");
    let layers = catalog::model("glm-5.2-fp8").unwrap().num_layers() as f64;

    // ---- Table 1: operating point (B=400, 32k tracelab contexts, Grace KV) ----
    let batch = 400_usize;
    let ctx = contexts(&path, batch);
    let reqs = requests(&ctx);
    let refs: Vec<&Request> = reqs.iter().collect();
    let core_per_gpu: f64 = ctx
        .iter()
        .map(|&x| x as u64 * CORE_KV_BYTES_PER_TOKEN)
        .sum::<u64>() as f64
        / 16.0;
    let kv_bw = if core_per_gpu <= HBM_KV_BYTES_PER_GPU {
        None
    } else {
        Some(GRACE_ACTIVE_KV_BW)
    };

    println!("table,mode,latency_us,batch,gamma,acceptance,verify_tok_per_expert,committed,step_ms,tps_per_gpu");
    for (mode_name, overlap) in [
        ("serial", MoeOverlap::Serial),
        ("hidden", MoeOverlap::Hidden),
    ] {
        for latency_us in LATENCIES_US {
            let eng = engine(hardware(latency_us), overlap, kv_bw);
            for gamma in GAMMAS {
                let step = step_time(&eng, &refs, batch, gamma, layers);
                let tok_per_expert = batch as f64 * (gamma + 1) as f64 * 8.0 / 256.0;
                for alpha in ALPHAS {
                    if gamma == 0 && alpha != ALPHAS[0] {
                        continue;
                    }
                    let committed = if gamma == 0 {
                        1.0
                    } else {
                        expected_committed(gamma, alpha)
                    };
                    let a = if gamma == 0 { 1.0 } else { alpha };
                    let tps = batch as f64 * committed / (16.0 * step);
                    println!(
                        "operating,{mode_name},{latency_us},{batch},{gamma},{a:.2},{tok_per_expert:.2},{committed:.4},{:.3},{tps:.1}",
                        step * 1e3
                    );
                }
            }
        }
    }

    // ---- Table 2: calibration vs lane M's measured point ----
    // c512, ISL 1024 + ~512 generated → uniform 1536-token contexts, KV in HBM.
    let cal_batch = 512_usize;
    let cal_ctx = vec![1_536_u32; cal_batch];
    let cal_reqs = requests(&cal_ctx);
    let cal_refs: Vec<&Request> = cal_reqs.iter().collect();
    for latency_us in LATENCIES_US {
        let eng = engine(hardware(latency_us), MoeOverlap::Serial, None);
        let step = step_time(&eng, &cal_refs, cal_batch, 0, layers);
        println!(
            "calibration,serial,{latency_us},{cal_batch},0,1.00,16.00,1.0000,{:.3},{:.1}",
            step * 1e3,
            cal_batch as f64 / (16.0 * step)
        );
    }

    // ---- Table 3: batch sweep — where does spec stop paying? ----
    // Spec pays at batch B iff tps(γ, α, B) > tps(0, B). Committed uses the
    // mid acceptance α=0.80; crossover reported as the spec:no-spec ratio.
    for (mode_name, overlap) in [
        ("serial", MoeOverlap::Serial),
        ("hidden", MoeOverlap::Hidden),
    ] {
        for latency_us in LATENCIES_US {
            for sweep_batch in [25_usize, 50, 100, 200, 400, 800] {
                let sctx = contexts(&path, sweep_batch);
                let sreqs = requests(&sctx);
                let srefs: Vec<&Request> = sreqs.iter().collect();
                let score: f64 = sctx
                    .iter()
                    .map(|&x| x as u64 * CORE_KV_BYTES_PER_TOKEN)
                    .sum::<u64>() as f64
                    / 16.0;
                let skv = if score <= HBM_KV_BYTES_PER_GPU {
                    None
                } else {
                    Some(GRACE_ACTIVE_KV_BW)
                };
                let eng = engine(hardware(latency_us), overlap, skv);
                let base = step_time(&eng, &srefs, sweep_batch, 0, layers);
                let base_tps = sweep_batch as f64 / (16.0 * base);
                for gamma in [1_u32, 2, 3, 4] {
                    let step = step_time(&eng, &srefs, sweep_batch, gamma, layers);
                    let committed = expected_committed(gamma, 0.80);
                    let tps = sweep_batch as f64 * committed / (16.0 * step);
                    println!(
                    "batch_sweep,{mode_name},{latency_us},{sweep_batch},{gamma},0.80,{:.2},{committed:.4},{:.3},{tps:.1}",
                    sweep_batch as f64 * (gamma + 1) as f64 * 8.0 / 256.0,
                    step * 1e3
                );
                    let _ = (base_tps, base);
                }
                println!(
                "batch_sweep,{mode_name},{latency_us},{sweep_batch},0,1.00,{:.2},1.0000,{:.3},{base_tps:.1}",
                sweep_batch as f64 * 8.0 / 256.0,
                base * 1e3
            );
            }
        }
    }
}
