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

use crate::request::Request;

/// Run a closed-loop workload of `conc` users issuing fixed-shape requests
/// (`isl` prompt tokens, `osl` output tokens) through `topology`. Stops once
/// `num_completions` requests have finished. The first `warmup_completions`
/// are dropped from the returned set (so the steady state is what's reported).
pub fn simulate_closed_loop(
    topology: Topology,
    conc: u32,
    isl: u32,
    osl: u32,
    num_completions: u32,
    warmup_completions: u32,
) -> Result<ClosedLoopResult, String> {
    let mut engine = Engine::new(topology);
    // Seed initial conc users at t=0.
    for i in 0..conc {
        let req = Request::new(format!("req-{i}"), 0, 0.0, isl, osl);
        engine.submit(req);
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
                let req = Request::new(format!("req-{next_id}"), 0, now, isl, osl);
                engine.submit(req);
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
