//! Analytical PP4 x EP4 decode hedge for the glm-isambard campaign.
//!
//! The simulator has no pipeline scheduler, so this composes its existing
//! roofline terms. Four microbatches through one quarter-model stage cost the
//! same kernel work as one full-model EP4 pass over one microbatch. Pipeline
//! transfers are conservatively serialized: 4 microbatches x 3 stage
//! boundaries. One representative model layer prices each MTP draft token.

use inference_lab::catalog;
use inference_lab::compute::{ComputeEngine, StepCost};
use inference_lab::config::{HardwareConfig, MoeOverlap, ParallelConfig};
use inference_lab::request::{Request, SessionSpec};

const BATCH: usize = 400;
const MICRO_BATCHES: usize = 4;
const HIDDEN_SIZE: f64 = 6_144.0;
const ACTIVATION_BYTES: f64 = 2.0;
const NIC_BANDWIDTH: f64 = 25e9;
const MEASURED_C2C_BANDWIDTH: f64 = 420e9;

fn contexts(path: &str, batch: usize) -> Vec<u32> {
    let sessions = SessionSpec::load(path).unwrap_or_else(|error| panic!("{error}"));
    let mut all: Vec<u32> = sessions
        .iter()
        .flat_map(|session| session.steps.iter())
        .map(|step| step.input.min(32_768))
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

fn cost(
    hardware: HardwareConfig,
    tp: u32,
    overlap: MoeOverlap,
    requests: &[Request],
    width: u32,
) -> StepCost {
    let model = catalog::model("glm-5.2-fp8").unwrap();
    let parallel = ParallelConfig {
        tp,
        ep: tp,
        dp_attention: true,
        moe_overlap: overlap,
    };
    // B=400 at the capped 32k TraceLab context occupies about 38 GB/GPU in
    // the simulator's KV layout, above the campaign's 20 GB HBM allowance.
    let engine = ComputeEngine::new(hardware, parallel, model)
        .with_kv_memory_bandwidth(MEASURED_C2C_BANDWIDTH);
    let refs: Vec<&Request> = requests.iter().collect();
    engine.step_cost(&refs, &vec![width; requests.len()])
}

fn expected_committed(gamma: u32, acceptance: f64) -> f64 {
    (0..=gamma).map(|i| acceptance.powi(i as i32)).sum()
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: glm_isambard_pp_decode_hedge <tracelab.jsonl>");
    let ep_contexts = contexts(&path, BATCH);
    let ep_requests = requests(&ep_contexts);
    let pp_contexts = contexts(&path, BATCH / MICRO_BATCHES);
    let pp_requests = requests(&pp_contexts);
    let layers = catalog::model("glm-5.2-fp8").unwrap().num_layers() as f64;

    println!(
        "latency_us,gamma,acceptance,verify_tokens_per_expert,knee_holds,ep16_serial_tps_per_gpu,ep16_hidden_tps_per_gpu,pp4_ep4_step_ms,pp4_ep4_tps_per_gpu,speedup_vs_ep16_serial,speedup_vs_ep16_hidden,target_ms,mtp_ms,pipeline_floor_ms,pipeline_wire_ms,expected_committed"
    );
    for latency_us in [20_u32, 50, 100, 200, 650, 5_800] {
        let ep_serial = cost(
            hardware(latency_us),
            16,
            MoeOverlap::Serial,
            &ep_requests,
            1,
        );
        let ep_hidden = cost(
            hardware(latency_us),
            16,
            MoeOverlap::Hidden,
            &ep_requests,
            1,
        );
        let ep_serial_tps = BATCH as f64 / (16.0 * ep_serial.time);
        let ep_hidden_tps = BATCH as f64 / (16.0 * ep_hidden.time);
        let pp_one_token = cost(hardware(latency_us), 4, MoeOverlap::Hidden, &pp_requests, 1);

        for (gamma, acceptance) in [
            (0, 1.0),
            (1, 0.70),
            (1, 0.85),
            (2, 0.70),
            (2, 0.85),
            (4, 0.70),
            (4, 0.85),
        ] {
            let verify_width = gamma + 1;
            let target = cost(
                hardware(latency_us),
                4,
                MoeOverlap::Hidden,
                &pp_requests,
                verify_width,
            );
            // The one MTP layer is approximated by one representative layer
            // of the one-token target, for every microbatch and draft token.
            let mtp = gamma as f64 * MICRO_BATCHES as f64 * pp_one_token.time / layers;
            let messages = (3 * MICRO_BATCHES) as f64;
            let pipeline_floor = messages * latency_us as f64 * 1e-6;
            let pipeline_wire =
                3.0 * BATCH as f64 * verify_width as f64 * HIDDEN_SIZE * ACTIVATION_BYTES
                    / NIC_BANDWIDTH;
            let pp_time = target.time + mtp + pipeline_floor + pipeline_wire;
            let committed = expected_committed(gamma, acceptance);
            let pp_tps = BATCH as f64 * committed / (16.0 * pp_time);
            println!(
                "{latency_us},{gamma},{acceptance:.2},{:.4},{},{ep_serial_tps:.6},{ep_hidden_tps:.6},{:.6},{pp_tps:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{committed:.6}",
                BATCH as f64 * verify_width as f64 * 8.0 / 256.0,
                verify_width == 1,
                pp_time * 1e3,
                pp_tps / ep_serial_tps,
                pp_tps / ep_hidden_tps,
                target.time * 1e3,
                mtp * 1e3,
                pipeline_floor * 1e3,
                pipeline_wire * 1e3,
            );
        }
    }
}
