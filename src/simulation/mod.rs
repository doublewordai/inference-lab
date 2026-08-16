//! Discrete-event simulation. The unified [`Engine`] is a pure state machine
//! over a [`Topology`] of worker pools and inter-pool links; drivers above it
//! pump events through `next_event_time` / `submit` / `step`. Three drivers
//! exist:
//!   * [`Simulator`] — synchronous batch sim driving a [`RequestGenerator`]
//!     against the engine. CLI and WASM use this.
//!   * `crate::serve::engine::RealtimeEngine` — async driver that turns
//!     external HTTP requests into engine submissions and paces wall-clock
//!     to simulated iter time.
//!   * [`simulate_closed_loop`] — convenience entry point used by the
//!     pareto-sweep example for steady-state runs.

pub mod engine;
pub mod roofline;
pub mod simulator;
pub mod spec;

pub use engine::{
    Engine, IterationInfo, RequestProgress, RequestTiming, StepKind, StepOutcome, Topology,
};
pub use roofline::{predict_decode_tpot, predict_prefill_time};
pub use simulator::{ProgressInfo, Simulator, TimeSeriesPoint};
pub use spec::{DepthSample, DraftPlan, PlanCosts, SpecPlanner};

use crate::config::SpeculativeConfig;
use crate::request::Request;

/// A closed-loop workload of fixed-shape requests for [`simulate_closed_loop`].
#[derive(Debug, Clone)]
pub struct ClosedLoop {
    /// Concurrent users; each issues its next request when its previous one
    /// completes.
    pub concurrency: u32,
    /// Prompt tokens per request.
    pub isl: u32,
    /// Output tokens per request.
    pub osl: u32,
    /// Stop once this many requests have completed.
    pub num_completions: u32,
    /// Drop the earliest completions from the result (steady state only).
    pub warmup_completions: u32,
    /// Optional speculative decoding (decode steps only, so on a disagg
    /// topology it affects only the decode pool).
    pub spec: Option<SpeculativeConfig>,
    pub seed: u64,
    /// Requests arrive already prefilled (pure decode work): the
    /// disaggregated decode pool in isolation, no prefill compute sharing
    /// the GPU. Pair with lifted KV caps to sweep the compute roofline.
    pub skip_prefill: bool,
}

/// Run `workload` through `topology`.
pub fn simulate_closed_loop(
    topology: Topology,
    workload: &ClosedLoop,
) -> Result<ClosedLoopResult, String> {
    let ClosedLoop {
        concurrency: conc,
        isl,
        osl,
        num_completions,
        warmup_completions,
        seed,
        skip_prefill,
        ..
    } = *workload;
    let mut engine = Engine::new(topology);
    if let Some(s) = &workload.spec {
        engine.enable_speculative(s.clone(), seed)?;
    }
    let mk = |id: u32, arrival: f64| {
        let mut req = Request::new(format!("req-{id}"), 0, arrival, isl, osl);
        if skip_prefill {
            req.mark_prefilled(arrival);
        }
        req
    };
    // Seed initial conc users at t=0.
    for i in 0..conc {
        engine.submit(mk(i, 0.0));
    }
    let mut next_id: u32 = conc;
    let mut all = Vec::with_capacity(num_completions as usize);
    while (all.len() as u32) < num_completions {
        if engine.next_event_time().is_none() {
            return Err("queue drained before reaching num_completions".to_string());
        }
        let outcome = engine.step()?;
        for timing in outcome.completions {
            let now = timing.completion_time;
            all.push(timing);
            if next_id < num_completions + conc {
                engine.submit(mk(next_id, now));
                next_id += 1;
            }
        }
    }

    // Sort by completion time so warmup-by-count drops the *earliest*
    // completions, not the first-finalised ones. The two diverge in a closed
    // loop because second-cycle requests can complete before late first-cycle
    // ones.
    all.sort_by(|a, b| a.completion_time.partial_cmp(&b.completion_time).unwrap());
    let total_time_full = engine.current_time();
    if warmup_completions == 0 || all.len() <= warmup_completions as usize {
        return Ok(ClosedLoopResult {
            timings: all,
            total_time: total_time_full,
            mean_batch_per_pool: engine.pool_batch_means(),
        });
    }
    let mean_batch_per_pool = engine.pool_batch_means();
    let kept: Vec<_> = all.into_iter().skip(warmup_completions as usize).collect();
    let total_time = if let (Some(first), Some(last)) = (kept.first(), kept.last()) {
        (last.completion_time - first.completion_time).max(1e-9)
    } else {
        total_time_full
    };
    Ok(ClosedLoopResult {
        timings: kept,
        total_time,
        mean_batch_per_pool,
    })
}

pub struct ClosedLoopResult {
    pub timings: Vec<RequestTiming>,
    pub total_time: f64,
    pub mean_batch_per_pool: Vec<Option<f64>>,
}

impl ClosedLoopResult {
    pub fn mean_ttft(&self) -> f64 {
        mean(self.timings.iter().map(|t| t.ttft()))
    }
    pub fn mean_tpot(&self) -> f64 {
        mean(self.timings.iter().filter_map(|t| t.tpot()))
    }
    pub fn mean_e2e(&self) -> f64 {
        mean(self.timings.iter().map(|t| t.e2e()))
    }
    pub fn throughput(&self) -> f64 {
        if self.total_time <= 0.0 {
            0.0
        } else {
            self.timings.len() as f64 / self.total_time
        }
    }
}

fn mean<I: Iterator<Item = f64>>(iter: I) -> f64 {
    let (sum, n) = iter.fold((0.0, 0u64), |(s, n), v| (s + v, n + 1));
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        AcceptanceModel, ClusterSpec, DenseModel, DisaggTopology, GammaPolicy, HardwareConfig,
        MeasuredCostConfig, ModelConfig, ModelCosts, ParallelConfig, Precision, SchedulerConfig,
        SpeculativeConfig,
    };
    use crate::scheduler::SchedulingPolicy;

    fn small_dense_parts() -> (ClusterSpec, ModelConfig, SchedulerConfig) {
        let hardware = HardwareConfig {
            name: "test".into(),
            flops_fp4: None,
            flops_fp8: None,
            flops_bf16: Some(1e15),
            flops_fp16: Some(1e15),
            memory_bandwidth: 1e12,
            memory_capacity: 80_000_000_000,
            kv_cache_capacity: 0,
            gpu_memory_utilization: 0.9,
            kv_tiers: Vec::new(),
        };
        let model = ModelConfig::Dense(DenseModel {
            name: "test-dense".into(),
            num_parameters: 1_000_000_000,
            num_active_parameters: None,
            num_layers: 8,
            hidden_dim: 1024,
            num_heads: 8,
            num_kv_heads: None,
            head_dim: None,
            max_seq_len: 4096,
            sliding_window: 0,
            num_sliding_layers: 0,
            precision: Precision::Bf16,
        });
        let sched = SchedulerConfig {
            max_num_batched_tokens: 8192,
            max_num_seqs: 256,
            enable_chunked_prefill: false,
            long_prefill_token_threshold: 0,
            max_num_partial_prefills: 1,
            block_size: 16,
            policy: SchedulingPolicy::FCFS,
            enable_preemption_free: false,
            enable_cascade_attention: false,
        };
        let cluster = ClusterSpec {
            hardware,
            parallel: ParallelConfig {
                tp: 1,
                ep: 1,
                dp_attention: false,
            },
            comms: None,
            num_workers: 1,
        };
        (cluster, model, sched)
    }

    fn small_dense_topology() -> Topology {
        let (cluster, model, sched) = small_dense_parts();
        Topology::aggregated(cluster, model, sched).expect("topo")
    }

    /// Run `engine` until `n` requests complete; returns their timings.
    fn run_until(engine: &mut Engine, n: usize) -> Vec<RequestTiming> {
        let mut done = Vec::new();
        while done.len() < n {
            assert!(engine.next_event_time().is_some(), "queue drained");
            done.extend(engine.step().unwrap().completions);
        }
        done
    }

    #[test]
    fn disagg_handoffs_share_the_link_bandwidth() {
        let (cluster, model, sched) = small_dense_parts();
        // KV of a 64-token prompt takes exactly 1.0 s alone on the link.
        let kv_bytes = model.kv_storage_bytes(64) as f64;
        let topo = DisaggTopology {
            prefill: cluster.clone(),
            decode: cluster,
            kv_link_bw: kv_bytes,
        };
        let topology = Topology::from_disagg(&topo, model, sched).unwrap();
        let mut engine = Engine::new(topology);
        // Two prompts prefill in the same step, so both hand-offs start
        // together and share the link: each takes 2.0 s, not 1.0 s.
        engine.submit(Request::new("a".into(), 0, 0.0, 64, 2));
        engine.submit(Request::new("b".into(), 0, 0.0, 64, 2));
        let timings = run_until(&mut engine, 2);
        for t in &timings {
            let handoff = t.handoff_done_time - t.prefill_done_time;
            assert!(
                (handoff - 2.0).abs() < 1e-6,
                "{}: handoff {handoff}",
                t.request_id
            );
        }
        // A third request whose prefill lands mid-way through the pair's
        // transfer slows them (three-way share) and is itself slowed.
        let mut engine =
            Engine::new(Topology::from_disagg(&topo_of(64), model_of(), sched_of()).unwrap());
        engine.submit(Request::new("a".into(), 0, 0.0, 64, 2));
        engine.submit(Request::new("b".into(), 0, 0.0, 64, 2));
        // The prefill step of a+b takes t_p; c arrives at t_p + 1.0 (both
        // half done, 1.0 s of link work each left) and prefills alone in
        // roughly t_p / 2, joining at ~t_p + 1 + t_p/2.
        let outcome_time = {
            let mut probe =
                Engine::new(Topology::from_disagg(&topo_of(64), model_of(), sched_of()).unwrap());
            probe.submit(Request::new("a".into(), 0, 0.0, 64, 2));
            probe.submit(Request::new("b".into(), 0, 0.0, 64, 2));
            let t = run_until(&mut probe, 2);
            t[0].prefill_done_time
        };
        engine.submit(Request::new("c".into(), 0, outcome_time + 1.0, 64, 2));
        let timings = run_until(&mut engine, 3);
        let get = |id: &str| timings.iter().find(|t| t.request_id == id).unwrap();
        let (a, c) = (get("a"), get("c"));
        // a and b were half done when c joined; the rest ran three-way.
        assert!(a.handoff_done_time - a.prefill_done_time > 2.0 + 1e-6);
        // c never had the link to itself.
        assert!(c.handoff_done_time - c.prefill_done_time > 1.0 + 1e-6);
        // Conservation: the link moved 3 transfers' bytes in the span from
        // the first submit to the last completion at full rate throughout.
        let first_start = timings
            .iter()
            .map(|t| t.prefill_done_time)
            .fold(f64::MAX, f64::min);
        let last_end = timings
            .iter()
            .map(|t| t.handoff_done_time)
            .fold(0.0, f64::max);
        assert!((last_end - first_start - 3.0).abs() < 1e-6);
    }

    fn topo_of(prompt: u32) -> DisaggTopology {
        let (cluster, model, _) = small_dense_parts();
        DisaggTopology {
            prefill: cluster.clone(),
            decode: cluster,
            kv_link_bw: model.kv_storage_bytes(prompt) as f64,
        }
    }
    fn model_of() -> ModelConfig {
        small_dense_parts().1
    }
    fn sched_of() -> SchedulerConfig {
        small_dense_parts().2
    }

    /// Drive a tiny prefilled closed loop with a measured cost table and
    /// return the mean chosen draft depth — exercises the table loading and
    /// the c_curve override in the policy decision path end to end.
    fn mean_draft_with_table(csv: &str) -> f64 {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("costs.csv");
        std::fs::write(&path, csv).unwrap();
        let spec = SpeculativeConfig {
            gamma: 4,
            acceptance: AcceptanceModel::Constant { alpha: 0.9 },
            policy: GammaPolicy::GoodputBudget,
            measured_cost: Some(MeasuredCostConfig {
                path: path.to_str().unwrap().into(),
                ref_seq_len: None,
            }),
            switch: Default::default(),
            drafter: None,
        };
        let mut engine = Engine::new(small_dense_topology());
        engine.enable_speculative(spec, 7).unwrap();
        for i in 0..4u32 {
            let mut req = Request::new(format!("r{i}"), 0, 0.0, 64, 32);
            req.mark_prefilled(0.0); // pure decode
            engine.submit(req);
        }
        let mut done = 0usize;
        while done < 4 {
            assert!(engine.next_event_time().is_some(), "queue drained");
            done += engine.step().unwrap().completions.len();
        }
        let series = engine.spec_depth_series();
        let (s, n) = series
            .iter()
            .fold((0.0f64, 0.0f64), |(s, n), d| (s + d.mean_draft, n + 1.0));
        s / n.max(1.0)
    }

    #[test]
    fn measured_cost_table_steers_draft_choice() {
        // File column is the verify width (ndt = g + 1; ndt=1 = plain decode).
        // Deep drafts measured as ruinously expensive -> policy stays at 0,
        // regardless of what the analytic roofline would have said.
        let spec_off = "batch_size,num_draft_tokens,step_seconds\n\
                        4,1,0.001\n4,2,1.0\n4,3,1.0\n4,4,1.0\n4,5,1.0\n";
        // Deep drafts measured as free -> policy pins gamma_max.
        let spec_on = "batch_size,num_draft_tokens,step_seconds\n\
                       4,1,0.001\n4,2,0.001\n4,3,0.001\n4,4,0.001\n4,5,0.001\n";
        assert!(mean_draft_with_table(spec_off) < 0.01);
        assert!(mean_draft_with_table(spec_on) > 3.9);
    }
}
