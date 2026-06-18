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

pub use engine::{
    Engine, IterationInfo, RequestProgress, RequestTiming, StepKind, StepOutcome, Topology,
};
pub use roofline::{predict_decode_tpot, predict_prefill_time};
pub use simulator::{ProgressInfo, Simulator, TimeSeriesPoint};

use crate::config::SpeculativeConfig;
use crate::request::Request;

/// Run a closed-loop workload of `conc` users issuing fixed-shape requests
/// (`isl` prompt tokens, `osl` output tokens) through `topology`. Stops once
/// `num_completions` requests have finished. The first `warmup_completions`
/// are dropped from the returned set (so the steady state is what's reported).
/// `spec` optionally enables speculative decoding (applies only to decode
/// steps, so on a disagg topology it affects only the decode pool).
pub fn simulate_closed_loop(
    topology: Topology,
    conc: u32,
    isl: u32,
    osl: u32,
    num_completions: u32,
    warmup_completions: u32,
    spec: Option<SpeculativeConfig>,
    seed: u64,
    skip_prefill: bool,
) -> Result<ClosedLoopResult, String> {
    let mut engine = Engine::new(topology);
    if let Some(s) = spec {
        engine.enable_speculative(s, seed);
    }
    // `skip_prefill` makes requests arrive already prefilled (num_computed = isl),
    // i.e. as pure-decode work -- the disaggregated decode pool in isolation, no
    // prefill compute sharing the GPU. Pair with lifted KV caps to sweep the
    // compute roofline.
    let mk = |id: u32, arrival: f64| {
        let mut req = Request::new(format!("req-{id}"), 0, arrival, isl, osl);
        if skip_prefill {
            req.num_computed_tokens = isl;
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
    let kept: Vec<_> = all
        .into_iter()
        .skip(warmup_completions as usize)
        .collect();
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
        AcceptanceModel, ClusterSpec, DenseModel, GammaPolicy, HardwareConfig, MeasuredCostConfig,
        ModelConfig, ParallelConfig, Precision, SchedulerConfig, SpeculativeConfig,
    };

    fn small_dense_topology() -> Topology {
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
            precision: Precision::Bf16,
        });
        let sched = SchedulerConfig {
            max_num_batched_tokens: 8192,
            max_num_seqs: 256,
            enable_chunked_prefill: false,
            long_prefill_token_threshold: 0,
            max_num_partial_prefills: 1,
            block_size: 16,
            policy: "fcfs".into(),
            enable_preemption_free: false,
            enable_cascade_attention: false,
        };
        let cluster = ClusterSpec {
            hardware,
            parallel: ParallelConfig { tp: 1, ep: 1, dp_attention: false },
            comms: None,
            num_workers: 1,
            node: 0,
        };
        Topology::aggregated(cluster, model, sched).expect("topo")
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
            draft_cost_frac: 0.0,
            measured_cost: Some(MeasuredCostConfig {
                path: path.to_str().unwrap().into(),
                ref_seq_len: None,
            }),
            switch: Default::default(),
            drafter: None,
        };
        let mut engine = Engine::new(small_dense_topology());
        engine.enable_speculative(spec, 7);
        for i in 0..4u32 {
            let mut req = Request::new(format!("r{i}"), 0, 0.0, 64, 32);
            req.num_computed_tokens = 64; // arrive prefilled: pure decode
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
            .fold((0.0f64, 0.0f64), |(s, n), &(_, md, _)| (s + md, n + 1.0));
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
