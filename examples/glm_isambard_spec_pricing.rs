//! Fair speculative-decode pricing for the glm-isambard campaign (lane J).
//!
//! Lane B's `pp-decode-hedge` priced speculation on the PP4×EP4 hedge only;
//! the EP16/EP32 columns carried no draft tokens. This extends the same
//! acceptance model to every decode layout so the comparison is fair:
//!
//! * γ draft tokens multiply the verify batch to `B·(γ+1)` tokens, which
//!   multiplies both the expert GEMM work and the dispatch/combine wire bytes
//!   (draft tokens count against the hide-knee on the CXI path) while leaving
//!   the per-call collective floor fixed — so speculation amortizes the fixed
//!   per-step cost.
//! * a single MTP drafting layer prices each draft token (γ autoregressive
//!   passes over B tokens, one representative layer each), the same
//!   convention lane B used for the PP hedge.
//! * expected committed tokens per sequence per step = Σ_{i=0..γ} α^i, the
//!   standard accepted-length model; output tok/s/GPU = B·committed /
//!   (N·t_step).
//!
//! Emits one long-format table over {EP16, EP32, PP4×EP4} × γ × acceptance ×
//! phase-floor × {serial, hidden}. Placement per width follows the operating
//! point (EP16 → Grace-active C2C, EP32 → HBM); PP4×EP4 reads active KV from
//! Grace, matching lane B's hedge. Composition of PP4×EP4 is analytical (the
//! simulator has no pipeline scheduler): four quarter-model microbatches are
//! one full-model EP4 pass over B/4, all 4×3 cross-node activation sends are
//! serialized conservatively, and BF16 activation wire is priced at 25 GB/s.

use inference_lab::catalog;
use inference_lab::compute::ComputeEngine;
use inference_lab::config::{HardwareConfig, MoeOverlap, ParallelConfig};
use inference_lab::request::{Request, SessionSpec};

const BATCH: usize = 400;
const CONTEXT_CAP: u32 = 32_768;
const MICRO_BATCHES: usize = 4;
const HIDDEN_SIZE: f64 = 6_144.0;
const ACTIVATION_BYTES: f64 = 2.0;
const NIC_BANDWIDTH: f64 = 25e9;
const CORE_KV_BYTES_PER_TOKEN: u64 = 78 * (512 + 64);
const HBM_KV_BYTES_PER_GPU: f64 = 20e9;
const GRACE_KV_BYTES_PER_GPU: f64 = 120e9;
// Active-KV reads during decode attention are a random gather over scattered
// per-token KV, which lane G measured at 311 GB/s on the Grace C2C — below the
// 420 GB/s streaming/link rate (that stays the figure for bulk transfers, e.g.
// lane D's p2p). Decode prices the gather rate.
const GRACE_ACTIVE_KV_BW: f64 = 311e9;
#[allow(dead_code)]
const GRACE_LINK_BW: f64 = 420e9;

const LATENCIES_US: [u32; 6] = [20, 50, 100, 200, 650, 5_800];
// (gamma, acceptance): γ=0 is the no-spec baseline; γ∈{1,2,4}×α∈{0.70,0.85}.
const SPEC: [(u32, f64); 7] = [
    (0, 1.0),
    (1, 0.70),
    (1, 0.85),
    (2, 0.70),
    (2, 0.85),
    (4, 0.70),
    (4, 0.85),
];

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

/// KV placement for `n` GPUs holding `batch` sequences' core KV: HBM if it
/// fits the 20 GB allowance, else Grace's 120 GB C2C store.
fn placement(core_per_gpu: f64) -> (&'static str, Option<f64>) {
    if core_per_gpu <= HBM_KV_BYTES_PER_GPU {
        ("hbm", None)
    } else if core_per_gpu <= GRACE_KV_BYTES_PER_GPU {
        ("grace", Some(GRACE_ACTIVE_KV_BW))
    } else {
        ("no_fit", Some(GRACE_ACTIVE_KV_BW))
    }
}

fn engine(
    hardware: HardwareConfig,
    tp: u32,
    overlap: MoeOverlap,
    kv_bw: Option<f64>,
) -> ComputeEngine {
    let model = catalog::model("glm-5.2-fp8").unwrap();
    let parallel = ParallelConfig {
        tp,
        ep: tp,
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

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: glm_isambard_spec_pricing <tracelab.jsonl>");
    let layers = catalog::model("glm-5.2-fp8").unwrap().num_layers() as f64;

    // EP decode batch (B sequences) and PP quarter-model microbatch (B/4).
    let ep_ctx = contexts(&path, BATCH);
    let ep_reqs = requests(&ep_ctx);
    let ep_refs: Vec<&Request> = ep_reqs.iter().collect();
    let pp_ctx = contexts(&path, BATCH / MICRO_BATCHES);
    let pp_reqs = requests(&pp_ctx);
    let pp_refs: Vec<&Request> = pp_reqs.iter().collect();

    // Core KV per GPU (campaign 44,928 B/token approximation) drives placement.
    let core_kv: f64 = ep_ctx
        .iter()
        .map(|&x| x as u64 * CORE_KV_BYTES_PER_TOKEN)
        .sum::<u64>() as f64;

    println!(
        "system,n_gpus,latency_us,gamma,acceptance,mode,verify_tokens_per_expert,knee_holds,committed,placement,step_ms,tps_per_gpu"
    );

    for &(n, ep_place_n) in &[(16_u32, 16.0_f64), (32, 32.0)] {
        let (place, kv_bw) = placement(core_kv / ep_place_n);
        for latency_us in LATENCIES_US {
            for (mode_name, overlap) in [
                ("serial", MoeOverlap::Serial),
                ("hidden", MoeOverlap::Hidden),
            ] {
                let eng = engine(hardware(latency_us), n, overlap, kv_bw);
                // Single-token step at this width/mode prices one MTP draft
                // layer as 1/layers of a decode step (matches the PP hedge).
                let one_token = eng.step_cost(&ep_refs, &vec![1; BATCH]).time;
                for (gamma, acceptance) in SPEC {
                    let verify = gamma + 1;
                    let verify_cost = eng.step_cost(&ep_refs, &vec![verify; BATCH]).time;
                    let mtp = gamma as f64 * one_token / layers;
                    let step = verify_cost + mtp;
                    let committed = expected_committed(gamma, acceptance);
                    let tps = BATCH as f64 * committed / (n as f64 * step);
                    let tok_per_expert = BATCH as f64 * verify as f64 * 8.0 / 256.0;
                    println!(
                        "ep{n},{n},{latency_us},{gamma},{acceptance:.2},{mode_name},{tok_per_expert:.4},{},{committed:.6},{place},{:.6},{tps:.6}",
                        verify == 1,
                        step * 1e3,
                    );
                }
            }
        }
    }

    // PP4×EP4 hedge: 16 GPUs, active KV in Grace. serial/hidden = the EP4
    // in-node MoE overlap; the pipeline send floor and activation wire are
    // added on top of the verify pass (identical composition to lane B).
    for latency_us in LATENCIES_US {
        for (mode_name, overlap) in [
            ("serial", MoeOverlap::Serial),
            ("hidden", MoeOverlap::Hidden),
        ] {
            let eng = engine(hardware(latency_us), 4, overlap, Some(GRACE_ACTIVE_KV_BW));
            let one_token = eng
                .step_cost(&pp_refs, &vec![1; BATCH / MICRO_BATCHES])
                .time;
            for (gamma, acceptance) in SPEC {
                let verify = gamma + 1;
                let target = eng
                    .step_cost(&pp_refs, &vec![verify; BATCH / MICRO_BATCHES])
                    .time;
                let mtp = gamma as f64 * MICRO_BATCHES as f64 * one_token / layers;
                let pipeline_floor = (3 * MICRO_BATCHES) as f64 * latency_us as f64 * 1e-6;
                let pipeline_wire =
                    3.0 * BATCH as f64 * verify as f64 * HIDDEN_SIZE * ACTIVATION_BYTES
                        / NIC_BANDWIDTH;
                let step = target + mtp + pipeline_floor + pipeline_wire;
                let committed = expected_committed(gamma, acceptance);
                let tps = BATCH as f64 * committed / (16.0 * step);
                let tok_per_expert = BATCH as f64 * verify as f64 * 8.0 / 256.0;
                println!(
                    "pp4_ep4,16,{latency_us},{gamma},{acceptance:.2},{mode_name},{tok_per_expert:.4},{},{committed:.6},grace,{:.6},{tps:.6}",
                    verify == 1,
                    step * 1e3,
                );
            }
        }
    }
}
