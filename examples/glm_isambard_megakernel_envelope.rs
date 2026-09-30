//! Megakernel envelope for the glm-isambard campaign (lane J, feeds lane H).
//!
//! Prices a wave-pipelined fused MoE layer against the two baselines a
//! decode megakernel has to beat, at the campaign knee shapes (per rank per
//! MoE layer, GLM-5.2-FP8, EP16 / EP32, B = 12.5 / 25 / 50 tokens/rank):
//!
//! * **WR** = expert weight read = `(E/N)·w_e / BW_hbm`, the one unavoidable
//!   cost of a decode MoE layer (all local experts active at the knee), and
//!   the thing the megakernel hides the wire behind.
//! * **wire** = dispatch + combine bandwidth time (both phases).
//! * **floor** = one dispatch/combine call latency = lane E's per-phase
//!   transport (the campaign "phase floor").
//!
//! Per-layer models:
//!
//! * **Serial (E-fixed)** — the realistic single-batch decode layer on lane
//!   E's fixed transport: `WR + wire + 2·floor` (dispatch, GEMM, combine, no
//!   overlap). This is the simulator's `Serial` MoE mode.
//! * **TBO (two-batch overlap)** — two half-batch microbatches ping-pong so
//!   each one's wire hides behind the other's compute: `2·max(WR, wire/2) +
//!   2·floor`. The weight read is paid **twice** — each microbatch re-reads
//!   the full local expert set (at the knee every expert is hit), because the
//!   two grouped GEMMs run at different pipeline stages and 603 MB ≫ L2. That
//!   doubled read is the decode-specific penalty the megakernel removes; it is
//!   also why TBO, a prefill (compute-bound) technique, is not clearly a win
//!   at the weight-read-bound decode knee.
//! * **Megakernel** — one fused persistent kernel per layer: the wire hides
//!   behind WR (like the simulator's `Hidden` mode) and the exposed per-layer
//!   floor is a fill/drain signal pair + epilogue, not the collective call
//!   floor: `max(WR, wire) + 2·signal + epilogue` (`MoeOverlap::Megakernel`).
//!
//! The megakernel's per-layer time is governed by `μ = 2·signal + epilogue`
//! (its fill/drain+epilogue budget). We sweep μ so lane H can read directly
//! what μ passes its gates:
//!   (i)  single-layer p50 ≤ 220 µs;
//!   (ii) ≥ 1.5× the serial baseline on E-fixed transport.
//! A coalescing sweep (wire × c) shows the wire-bound risk: the megakernel is
//! WR-bound only while coalesced wire ≤ WR, else `max` picks the wire and no
//! μ can reach the gate — the lane-E coupling lane H flagged.
//!
//! Two tables are emitted, split on a `##STEP` sentinel line: the per-layer
//! envelope, then step-level output tok/s/GPU (serial / TBO / megakernel at a
//! grid of μ) at the same shapes.

use inference_lab::catalog;
use inference_lab::compute::ComputeEngine;
use inference_lab::config::{MegakernelParams, MoeOverlap, ParallelConfig};
use inference_lab::request::Request;

const CONTEXT: u32 = 32_768;
const GATE_I_US: f64 = 220.0;
const COMM_SM_FRACTION: f64 = 0.25; // lane H gate (iii): comm CTAs ≤ 25% of SMs.
                                    // Active-KV random-gather rate (lane G measured), used for the step-level KV
                                    // reads; the per-layer envelope is KV-free (expert weight read + wire only).
const GRACE_ACTIVE_KV_BW: f64 = 311e9;
const HBM_KV_BYTES_PER_GPU: f64 = 20e9;
const CORE_KV_BYTES_PER_TOKEN: u64 = 78 * (512 + 64);

// (per-rank batch label, tokens/rank). 25 = EP16 knee, 12.5 = EP32 knee, 50 = 2×.
const RANK_BATCHES: [(&str, f64); 3] = [("12.5", 12.5), ("25", 25.0), ("50", 50.0)];
const FLOORS_US: [u32; 4] = [20, 50, 100, 200];
// Megakernel fill/drain + epilogue budgets μ (µs) to sweep.
const MU_US: [f64; 6] = [2.0, 10.0, 20.0, 50.0, 100.0, 150.0];
// Coalescing factors on the raw dispatch+combine wire (1.0 = uncoalesced).
const COALESCE: [f64; 2] = [1.0, 0.5];

fn requests(n_seq: usize) -> Vec<Request> {
    (0..n_seq)
        .map(|i| {
            let mut r = Request::new(format!("r{i}"), 0, 0.0, CONTEXT, 1);
            r.num_computed_tokens = CONTEXT;
            r
        })
        .collect()
}

fn engine(
    n: u32,
    overlap: MoeOverlap,
    floor_us: u32,
    kv_grace: bool,
    megakernel: Option<MegakernelParams>,
) -> ComputeEngine {
    let model = catalog::model("glm-5.2-fp8").unwrap();
    let mut hardware = catalog::hardware("gh200").unwrap();
    hardware
        .fabric
        .as_mut()
        .unwrap()
        .scale_out
        .as_mut()
        .unwrap()
        .latency = floor_us as f64 * 1e-6;
    let parallel = ParallelConfig {
        tp: n,
        ep: n,
        dp_attention: true,
        moe_overlap: overlap,
        megakernel,
    };
    let e = ComputeEngine::new(hardware, parallel, model);
    if kv_grace {
        e.with_kv_memory_bandwidth(GRACE_ACTIVE_KV_BW)
    } else {
        e
    }
}

fn main() {
    let model = catalog::model("glm-5.2-fp8").unwrap();
    let moe_layers = model.moe_layers() as f64;

    // -- Per-layer envelope table --
    println!(
        "system,n,b_rank,b_global,floor_us,wr_us,wire_raw_us,serial_us,tbo_us,coalesce,mu_us,mega_us,wr_bound,gate_i_pass,beats_serial,mega_vs_serial_x,beats_tbo"
    );
    for n in [16_u32, 32] {
        for (label, per_rank) in RANK_BATCHES {
            let b = (n as f64 * per_rank).round() as usize;
            let reqs = requests(b);
            let refs: Vec<&Request> = reqs.iter().collect();
            let toks = vec![1u32; b];
            // Grace-active KV at EP16 (36 GB/GPU > 20 GB HBM); HBM at EP32.
            let core_per_gpu =
                b as f64 * CONTEXT as f64 * CORE_KV_BYTES_PER_TOKEN as f64 / n as f64;
            let grace = core_per_gpu > HBM_KV_BYTES_PER_GPU;
            for floor_us in FLOORS_US {
                let eng = engine(n, MoeOverlap::Hidden, floor_us, grace, None);
                let wr = eng.routed_kernel_seconds(&refs, &toks, COMM_SM_FRACTION) / moe_layers;
                let (_serial_moe, wire, _lat) = eng.moe_collective_seconds(b as u32);
                let wire = wire / moe_layers;
                let floor = floor_us as f64 * 1e-6;
                let wr_us = wr * 1e6;
                let wire_us = wire * 1e6;
                let floor_us_f = floor * 1e6;
                let serial_us = wr_us + wire_us + 2.0 * floor_us_f;
                for c in COALESCE {
                    let wire_c = wire_us * c;
                    // TBO: wire (per microbatch = wire_c/2) hidden behind the
                    // paired microbatch's compute; weight read paid twice.
                    let tbo_us = 2.0 * wr_us.max(wire_c / 2.0) + 2.0 * floor_us_f;
                    let mega_base = wr_us.max(wire_c);
                    let wr_bound = wr_us >= wire_c;
                    for mu in MU_US {
                        let mega_us = mega_base + mu;
                        println!(
                            "ep{n},{n},{label},{b},{floor_us},{wr_us:.3},{wire_us:.3},{serial_us:.3},{tbo_us:.3},{c:.2},{mu:.1},{mega_us:.3},{wr_bound},{},{},{:.4},{}",
                            mega_us <= GATE_I_US,
                            mega_us < serial_us,
                            serial_us / mega_us,
                            mega_us < tbo_us,
                        );
                    }
                }
            }
        }
    }

    // -- Step-level output tok/s/GPU: serial / TBO / megakernel(μ) --
    println!("##STEP");
    println!("system,n,b_rank,b_global,floor_us,mode,mu_us,step_ms,tps_per_gpu");
    for n in [16_u32, 32] {
        for (label, per_rank) in RANK_BATCHES {
            let b = (n as f64 * per_rank).round() as usize;
            let reqs = requests(b);
            let refs: Vec<&Request> = reqs.iter().collect();
            let toks = vec![1u32; b];
            let core_per_gpu =
                b as f64 * CONTEXT as f64 * CORE_KV_BYTES_PER_TOKEN as f64 / n as f64;
            let grace = core_per_gpu > HBM_KV_BYTES_PER_GPU;
            for floor_us in FLOORS_US {
                let serial = engine(n, MoeOverlap::Serial, floor_us, grace, None)
                    .step_cost(&refs, &toks)
                    .time;
                // TBO = Hidden overlap + the second microbatch's weight read.
                let hidden_eng = engine(n, MoeOverlap::Hidden, floor_us, grace, None);
                let tbo = hidden_eng.step_cost(&refs, &toks).time
                    + hidden_eng.routed_kernel_seconds(&refs, &toks, 0.0);
                let emit = |mode: &str, mu: f64, step: f64| {
                    let mu_s = if mu.is_nan() {
                        "".to_string()
                    } else {
                        format!("{mu:.1}")
                    };
                    println!(
                        "ep{n},{n},{label},{b},{floor_us},{mode},{mu_s},{:.6},{:.6}",
                        step * 1e3,
                        b as f64 / (n as f64 * step),
                    );
                };
                emit("serial", f64::NAN, serial);
                emit("tbo", f64::NAN, tbo);
                for mu in MU_US {
                    // μ = 2·signal + epilogue: put it all on the signal term.
                    let params = MegakernelParams {
                        signal_latency: mu * 1e-6 / 2.0,
                        epilogue: 0.0,
                        comm_sm_fraction: COMM_SM_FRACTION,
                    };
                    let step = engine(n, MoeOverlap::Megakernel, floor_us, grace, Some(params))
                        .step_cost(&refs, &toks)
                        .time;
                    emit("megakernel", mu, step);
                }
            }
        }
    }
}
