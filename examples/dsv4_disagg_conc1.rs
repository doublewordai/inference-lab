//! DeepSeek-V4-Pro on a 1×P + 1×D Dynamo-sglang disagg topology, GB300
//! NVL72, ISL=8192 OSL=1024. The InferenceX conc=1 datapoint to compare
//! against:
//!
//! ```text
//!   P: 1× TP4 EP1 (no DPA)
//!   D: 1× TP4 EP1 (no DPA)
//!   TPOT  ~11.9 ms
//!   TTFT  ~1.0  s   (E2E ~12.1 s for OSL=1024)
//! ```
//!
//! Run with:
//!   `cargo run --example dsv4_disagg_conc1 --no-default-features`

use inference_lab::config::{
    ClusterSpec, DeepseekV4Model, DisaggTopology, HardwareConfig, ModelConfig, Node,
    SchedulerConfig,
};
use inference_lab::simulation::{
    predict_single_request, predict_single_request_aggregated, RequestTiming,
};
use inference_lab::request::Request;

fn b300_per_gpu() -> HardwareConfig {
    // Per-GPU B300 spec sheet numbers. FP4 dense tensor core, HBM3e.
    // (Refine when we have firm spec-sheet numbers; these are
    // placeholders in line with NVIDIA's announced B300.)
    HardwareConfig {
        name: "B300".into(),
        compute_flops: 1.5e16,            // 15 PFLOPS FP4 per GPU
        memory_bandwidth: 8.0e12,         // 8 TB/s per GPU HBM3e
        memory_capacity: 309_237_645_312, // 288 GB per GPU
        tp: 4,
        kv_cache_capacity: 0,
        gpu_memory_utilization: 0.9,
        bytes_per_param: 1, // overridden by DeepseekV4Model's mixed-precision fields
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
    })
}

fn topology() -> DisaggTopology {
    let cluster = ClusterSpec {
        hardware: b300_per_gpu(),
        num_workers: 1,
        node: 0,
    };
    DisaggTopology {
        nodes: vec![Node {
            name: "gb300-nvl72".into(),
            num_gpus: 72,
            // NVLink5 unidirectional all-to-all per GPU.
            intra_node_link_bw: 9.0e11,
        }],
        inter_node_link_bw: None,
        prefill: cluster.clone(),
        decode: cluster,
        // KV hand-off rides the NVL72 NVLink fabric.
        kv_link_bw: 9.0e11,
    }
}

fn scheduler() -> SchedulerConfig {
    SchedulerConfig {
        max_num_batched_tokens: 16384,
        max_num_seqs: 256,
        // No chunked prefill: in disagg the prefill worker runs the full
        // sequence in one shot.
        enable_chunked_prefill: false,
        long_prefill_token_threshold: 0,
        max_num_partial_prefills: 1,
        block_size: 64,
        policy: "fcfs".into(),
        enable_preemption_free: true,
        enable_cascade_attention: false,
    }
}

fn print_breakdown(name: &str, t: &RequestTiming) {
    let prefill_span = t.prefill_done_time - t.arrival_time;
    let handoff_span = t.handoff_done_time - t.prefill_done_time;
    let first_decode_span = t.first_token_time - t.handoff_done_time;
    let decode_span = t.completion_time - t.first_token_time;
    println!("--- {name} ---");
    println!("  arrival → prefill done : {:8.2} ms", prefill_span * 1000.0);
    println!("  prefill → handoff done : {:8.2} ms", handoff_span * 1000.0);
    println!("  handoff → first token  : {:8.2} ms", first_decode_span * 1000.0);
    println!(
        "  first → last token     : {:8.2} ms ({} output tokens)",
        decode_span * 1000.0,
        t.num_output_tokens
    );
    println!();
    println!("  TTFT  : {:7.2} ms", t.ttft() * 1000.0);
    println!("  TPOT  : {:7.3} ms", t.tpot() * 1000.0);
    println!("  E2E   : {:7.2} s",  t.e2e());
    println!();
}

fn aggregated_cluster() -> ClusterSpec {
    let mut hw = b300_per_gpu();
    hw.tp = 8;
    ClusterSpec {
        hardware: hw,
        num_workers: 1,
        node: 0,
    }
}

fn main() {
    let model = dsv4_pro();
    let topology = topology();
    let scheduler_cfg = scheduler();

    let isl = 8192u32;
    let osl = 1024u32;

    let req_disagg = Request::new("conc1-disagg".to_string(), 0, 0.0, isl, osl);
    let disagg = predict_single_request(&topology, model.clone(), scheduler_cfg.clone(), req_disagg)
        .expect("disagg prediction failed");

    let mut agg_scheduler = scheduler_cfg.clone();
    agg_scheduler.enable_chunked_prefill = true;
    let req_agg = Request::new("conc1-agg".to_string(), 0, 0.0, isl, osl);
    let aggregated = predict_single_request_aggregated(
        &aggregated_cluster(),
        model,
        agg_scheduler,
        req_agg,
    )
    .expect("aggregated prediction failed");

    println!("DeepSeek-V4-Pro conc=1, ISL={isl} OSL={osl}, FP4, 8 GPUs total\n");
    println!(
        "Roofline: max(compute_FLOPs/aggregate_FLOPS, weight+KV_bytes/aggregate_BW) per iteration."
    );
    println!("No kernel efficiency factor, no MoE batch=1 overhead, no comms term.\n");
    print_breakdown("disagg roofline (1×P TP4 + 1×D TP4)", &disagg);
    print_breakdown("aggregated roofline (1× TP8, chunked prefill)", &aggregated);

    // From data/inferencex/dynamo-sglang/bmk_..._conc1_gb300-cw_3.json (PR #1157).
    println!("--- InferenceX measurement (GB300, Dynamo+sglang, disagg) ---");
    println!("  TPOT  : 11.90 ms");
    println!("  TTFT  :  1.03 s");
    println!("  E2E   : 12.09 s\n");

    let real_tpot_ms = 11.90;
    let real_e2e_s = 12.09;
    println!(
        "TPOT  measurement / disagg roofline      = {:5.1}× ({:6.3} ms / {:6.3} ms)",
        real_tpot_ms / (disagg.tpot() * 1000.0),
        real_tpot_ms,
        disagg.tpot() * 1000.0
    );
    println!(
        "TPOT  measurement / aggregated roofline  = {:5.1}× ({:6.3} ms / {:6.3} ms)",
        real_tpot_ms / (aggregated.tpot() * 1000.0),
        real_tpot_ms,
        aggregated.tpot() * 1000.0
    );
    println!(
        "E2E   disagg / aggregated roofline       = {:5.2}× ({:6.2} s / {:6.2} s)",
        disagg.e2e() / aggregated.e2e(),
        disagg.e2e(),
        aggregated.e2e()
    );
    println!(
        "E2E   measurement / aggregated roofline  = {:5.1}× ({:6.2} s / {:6.2} s)",
        real_e2e_s / aggregated.e2e(),
        real_e2e_s,
        aggregated.e2e()
    );
}
