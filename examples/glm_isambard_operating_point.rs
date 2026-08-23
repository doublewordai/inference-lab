//! Reproduce the GLM-5.2 Isambard operating-point roofline used by the
//! glm-isambard campaign. The first argument is a TraceLab session JSONL.

use inference_lab::catalog;
use inference_lab::compute::ComputeEngine;
use inference_lab::config::{HardwareConfig, MoeOverlap, ParallelConfig};
use inference_lab::request::{Request, SessionSpec};

const CONTEXT_CAP: u32 = 32_768;
const CORE_KV_BYTES_PER_TOKEN: u64 = 78 * (512 + 64); // MLA latent + RoPE only.
const HBM_KV_BYTES_PER_GPU: f64 = 20e9;
const GRACE_KV_BYTES_PER_GPU: f64 = 120e9;
const MEASURED_C2C_BW: f64 = 420e9;

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
            let mut request = Request::new(format!("r{i}"), 0, 0.0, context, 2);
            request.num_computed_tokens = context;
            request
        })
        .collect()
}

fn set_scale_out_latency(hardware: &mut HardwareConfig, latency: f64) {
    hardware
        .fabric
        .as_mut()
        .unwrap()
        .scale_out
        .as_mut()
        .unwrap()
        .latency = latency;
}

fn placement(core_per_gpu: f64, has_grace: bool) -> (&'static str, bool, f64) {
    if core_per_gpu <= HBM_KV_BYTES_PER_GPU {
        ("hbm", true, f64::NAN)
    } else if has_grace && core_per_gpu <= GRACE_KV_BYTES_PER_GPU {
        ("grace", true, MEASURED_C2C_BW)
    } else {
        ("no_fit", false, f64::NAN)
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: glm_isambard_operating_point <tracelab.jsonl>");
    let model = catalog::model("glm-5.2-fp8").unwrap();
    let sim_kv_bytes_per_token = model.kv_storage_bytes(1);
    eprintln!(
        "context_cap={CONTEXT_CAP} core_kv_B_per_tok={CORE_KV_BYTES_PER_TOKEN} sim_kv_B_per_tok={sim_kv_bytes_per_token} measured_c2c_Bps={MEASURED_C2C_BW}"
    );
    println!(
        "system,n,batch,latency_us,mode,placement,placement_fits_core,knee_holds,tokens_per_expert,mean_context,core_kv_gb_per_gpu,sim_kv_gb_per_gpu,step_ms,output_tps_per_gpu"
    );

    for (system, preset, widths) in [
        ("gh200", "gh200", &[16_u32, 32, 64][..]),
        ("h100", "h100", &[16_u32][..]),
    ] {
        for &n in widths {
            for batch in [400_usize, 800] {
                let contexts = contexts(&path, batch);
                let mean_context = contexts.iter().map(|&x| x as f64).sum::<f64>() / batch as f64;
                let core_kv = contexts
                    .iter()
                    .map(|&x| x as u64 * CORE_KV_BYTES_PER_TOKEN)
                    .sum::<u64>() as f64;
                let sim_kv = contexts
                    .iter()
                    .map(|&x| model.kv_storage_bytes(x))
                    .sum::<u64>() as f64;
                let (place, fits, kv_bw) = placement(core_kv / n as f64, system == "gh200");
                let reqs = requests(&contexts);
                let refs: Vec<&Request> = reqs.iter().collect();
                let tokens = vec![1; batch];
                for latency_us in [20_u32, 50, 100, 200, 650, 5_800] {
                    for (mode_name, moe_overlap) in [
                        ("serial", MoeOverlap::Serial),
                        ("hidden", MoeOverlap::Hidden),
                    ] {
                        let mut hardware = catalog::hardware(preset).unwrap();
                        set_scale_out_latency(&mut hardware, latency_us as f64 * 1e-6);
                        let parallel = ParallelConfig {
                            tp: n,
                            ep: n,
                            dp_attention: true,
                            moe_overlap,
                        };
                        let mut engine = ComputeEngine::new(hardware, parallel, model.clone());
                        if place == "grace" {
                            engine = engine.with_kv_memory_bandwidth(kv_bw);
                        }
                        let cost = engine.step_cost(&refs, &tokens);
                        let output_tps_per_gpu = batch as f64 / (cost.time * n as f64);
                        println!(
                            "{system},{n},{batch},{latency_us},{mode_name},{place},{fits},{},{:.4},{mean_context:.3},{:.6},{:.6},{:.6},{output_tps_per_gpu:.6}",
                            batch <= 400,
                            batch as f64 * 8.0 / 256.0,
                            core_kv / n as f64 / 1e9,
                            sim_kv / n as f64 / 1e9,
                            cost.time * 1e3,
                        );
                    }
                }
            }
        }
    }

    // Stage-2 calibration: EP16, c512, ISL/OSL 1024, 5.8 ms per phase.
    let context = vec![1_024; 512];
    let reqs = requests(&context);
    let refs: Vec<&Request> = reqs.iter().collect();
    let mut hardware = catalog::hardware("gh200").unwrap();
    set_scale_out_latency(&mut hardware, 5_800e-6);
    let parallel = ParallelConfig {
        tp: 16,
        ep: 16,
        dp_attention: true,
        moe_overlap: MoeOverlap::Serial,
    };
    let cost = ComputeEngine::new(hardware, parallel, model).step_cost(&refs, &vec![1; 512]);
    eprintln!(
        "calibration,ep16,c512,context1024,phase_us=5800,step_ms={:.6},output_tps_per_gpu={:.6}",
        cost.time * 1e3,
        512.0 / (16.0 * cost.time)
    );
}
