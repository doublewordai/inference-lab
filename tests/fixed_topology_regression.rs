//! Fixed topologies must keep producing exactly the results they did before
//! the engine learned to add, drain and kill workers at run time. Each
//! scenario runs a deterministic hand-built workload (no sampled lengths, no
//! transcendental maths on the dense test model) and fingerprints every
//! completion's timings bit for bit; the expected fingerprints were recorded
//! on the engine as it was before the runtime lifecycle and router hooks.

use inference_lab::config::{
    ClusterSpec, DisaggTopology, FabricConfig, FabricLink, HardwareConfig, LayerClass,
    MemoryConfig, ModelSpec, ParallelConfig, Precision, RouterConfig, SchedulerConfig,
    WeightStream,
};
use inference_lab::request::Request;
use inference_lab::scheduler::SchedulingPolicy;
use inference_lab::simulation::{Engine, RequestTiming, Topology};

fn hardware() -> HardwareConfig {
    HardwareConfig {
        name: "test".into(),
        flops_fp4: None,
        flops_fp8: None,
        flops_bf16: Some(1e15),
        flops_fp16: Some(1e15),
        memory_bandwidth: 1e12,
        memory_capacity: 80_000_000_000,
        memory: None,
        fabric: None,
    }
}

fn model() -> ModelSpec {
    ModelSpec {
        name: "test-dense".into(),
        hidden_dim: 1024,
        max_seq_len: 4096,
        attention_precision: Precision::Bf16,
        activation_bytes: 2,
        weights: vec![WeightStream {
            scale: None,
            precision: Precision::Bf16,
            active_params: 1_000_000_000,
            resident_params: 1_000_000_000,
            routing: None,
        }],
        layers: vec![LayerClass::Attention {
            count: 8,
            heads: 8,
            head_dim: 128,
            kv_heads: 8,
            kv_shared: false,
            window: 0,
            kv_precision: Precision::Bf16,
        }],
    }
}

/// Chunked prefill over a small KV cache: prefix hits, evictions and
/// preemptions all happen.
fn scheduler(kv_blocks: u64) -> SchedulerConfig {
    let m = model();
    SchedulerConfig {
        max_num_batched_tokens: 256,
        max_num_seqs: 16,
        max_waiting: 0,
        enable_chunked_prefill: true,
        long_prefill_token_threshold: 0,
        max_num_partial_prefills: 1,
        block_size: 16,
        gpu_memory_utilization: 0.9,
        kv_cache_capacity: kv_blocks * m.kv_storage_bytes(16),
        max_model_len: None,
        policy: SchedulingPolicy::FCFS,
        enable_preemption_free: false,
        enable_cascade_attention: false,
        mix_prefill_with_decode: true,
        overlap_schedule: false,
        prefill_delayer_max_passes: 0,
        idle_pass_s: 0.0,
        long_prefill_decode_interval: 0,
        long_prefill_min_tokens: 262_144,
        balance_set: None,
    }
}

fn cluster(num_workers: u32) -> ClusterSpec {
    toml_cluster(hardware(), ParallelConfig::default(), num_workers)
}

fn toml_cluster(hardware: HardwareConfig, parallel: ParallelConfig, n: u32) -> ClusterSpec {
    let mut c: ClusterSpec = toml::from_str("hardware = \"b200\"").unwrap();
    c.hardware = hardware;
    c.parallel = parallel;
    c.num_workers = n;
    c.memory = MemoryConfig::default();
    c
}

/// 120 requests from six sessions: each turn re-sends its session's growing
/// prefix plus two blocks of its own, arriving in tight bursts.
fn workload() -> Vec<Request> {
    (0..120u64)
        .map(|i| {
            let session = i % 6;
            let turn = i / 6;
            let shared = 4 + turn;
            let mut hashes: Vec<u64> = (1..=shared).map(|b| session * 1_000 + b).collect();
            hashes.extend([1_000_000 + 2 * i, 1_000_001 + 2 * i]);
            let tokens = hashes.len() as u32 * 16 - 3;
            let output = 4 + (i as u32 * 7) % 23;
            let mut r = Request::new(format!("r{i:03}"), 0, i as f64 * 0.004, tokens, output);
            r.prompt_block_hashes = hashes;
            r
        })
        .collect()
}

/// FNV-1a over every completion's fields, timings as raw bits, in request
/// order, plus the run's step count and final clock.
fn fingerprint(mut engine: Engine) -> (usize, u64) {
    for r in workload() {
        engine.submit(r);
    }
    let mut done: Vec<RequestTiming> = Vec::new();
    let mut steps = 0u64;
    while !engine.is_idle() {
        done.extend(engine.step().unwrap().completions);
        steps += 1;
    }
    done.sort_by(|a, b| a.request_id.cmp(&b.request_id));
    let mut text = format!("steps={steps} t={:016x}\n", engine.current_time().to_bits());
    for t in &done {
        text.push_str(&format!(
            "{} {:016x} {:016x} {:016x} {:016x} {:016x} {} {} {} {:?} {:?} {} {} {:?}\n",
            t.request_id,
            t.arrival_time.to_bits(),
            t.prefill_done_time.to_bits(),
            t.handoff_done_time.to_bits(),
            t.first_token_time.to_bits(),
            t.completion_time.to_bits(),
            t.num_prompt_tokens,
            t.num_output_tokens,
            t.num_cached_tokens,
            t.decode_cached_tokens,
            t.worker,
            t.num_preemptions,
            t.rejected,
            t.lookup,
        ));
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (done.len(), h)
}

fn check(name: &str, engine: Engine, expected: (usize, u64)) {
    let got = fingerprint(engine);
    assert_eq!(
        got, expected,
        "{name}: fixed-topology results changed (completions, fingerprint)"
    );
}

#[test]
fn aggregated_kv_aware_replicas_are_unchanged() {
    let topo = Topology::aggregated(cluster(3), model(), scheduler(96))
        .unwrap()
        .with_router(&RouterConfig::KvAware { load_weight: 1.0 });
    check("aggregated", Engine::new(topo), (120, 7933720976728682643));
}

#[test]
fn aggregated_tiered_prefix_affinity_is_unchanged() {
    const MEM: &str = r#"
gpus_per_node = 2
[[stores]]
name = "host"
per = "node"
capacity = 1e9
[[links]]
name = "pcie"
from = "gpu"
to = "host"
bandwidth = 5e10
"#;
    let mut c = cluster(2);
    c.hardware.memory = Some(toml::from_str(MEM).unwrap());
    c.memory = toml::from_str("tiers = [\"host\"]").unwrap();
    let topo = Topology::aggregated(c, model(), scheduler(64))
        .unwrap()
        .with_router(&RouterConfig::PrefixAffinity {
            max_load_ratio: Some(1.5),
        });
    check("tiered", Engine::new(topo), (120, 12333570640610404653));
}

#[test]
fn disaggregated_kv_aware_pools_are_unchanged() {
    let topo = DisaggTopology {
        prefill: cluster(2),
        decode: cluster(2),
        kv_link_bw: Some(2e10),
    };
    let topo = Topology::from_disagg(&topo, model(), scheduler(128))
        .unwrap()
        .with_routers(
            &RouterConfig::KvAware { load_weight: 1.0 },
            &RouterConfig::KvAwareDecode { load_weight: 64.0 },
        );
    check("disagg", Engine::new(topo), (120, 8325524301835990393));
}

#[test]
fn dp_attention_ranks_with_time_correction_are_unchanged() {
    let mut hw = hardware();
    hw.fabric = Some(FabricConfig {
        gpus_per_node: 8,
        scale_up: FabricLink {
            bandwidth: 4e11,
            latency: 2e-6,
            in_network_reduction: false,
        },
        scale_out: None,
    });
    let parallel = ParallelConfig {
        tp: 2,
        ep: 1,
        dp_attention: true,
    };
    let topo = Topology::aggregated(toml_cluster(hw, parallel, 1), model(), scheduler(192))
        .unwrap()
        .with_router(&RouterConfig::LeastLoaded {});
    let mut engine = Engine::new(topo);
    engine.set_time_correction(1.3, 0.001);
    check("dp_attention", engine, (120, 15907054320178273588));
}

#[test]
fn speculative_replicas_are_unchanged() {
    let topo = Topology::aggregated(cluster(2), model(), scheduler(160))
        .unwrap()
        .with_router(&RouterConfig::KvAware { load_weight: 1.0 });
    let mut engine = Engine::new(topo);
    let spec: inference_lab::config::SpeculativeConfig =
        toml::from_str("gamma = 3\n[acceptance]\nkind = \"constant\"\nalpha = 0.7\n").unwrap();
    engine.enable_speculative(spec, 7).unwrap();
    check("speculative", engine, (120, 2450992995670437319));
}
