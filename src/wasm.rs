//! WASM entry points. The JS-facing structs here are the contract with the
//! web UI: their field names are stable and independent of the Rust-side
//! [`MetricsSummary`] / [`TimeSeriesPoint`] layout.

use js_sys::Function;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::config::Config;
use crate::metrics::{LatencySamples, MetricsSummary};
use crate::simulation::{ProgressInfo, Simulator, TimeSeriesPoint};

/// Flat metrics block consumed by the UI.
#[derive(Serialize)]
struct MetricsData {
    ttft_min: f64,
    ttft_mean: f64,
    ttft_p50: f64,
    ttft_p90: f64,
    ttft_p99: f64,
    e2e_min: f64,
    e2e_mean: f64,
    e2e_p50: f64,
    e2e_p90: f64,
    e2e_p99: f64,
    per_token_min: f64,
    per_token_mean: f64,
    per_token_p50: f64,
    per_token_p90: f64,
    per_token_p99: f64,
    input_tokens_per_sec: f64,
    output_tokens_per_sec: f64,
    requests_per_sec: f64,
    avg_kv_cache_util: f64,
    avg_flops_util: f64,
    avg_bandwidth_util: f64,
    total_preemptions: u64,
    avg_preemptions_per_request: f64,
    completed_requests: u64,
    total_requests: u64,
    total_time: f64,
}

impl MetricsData {
    fn from_summary(m: &MetricsSummary, total_time: f64) -> Self {
        let l = &m.latency_metrics;
        Self {
            ttft_min: l.ttft_ms.min,
            ttft_mean: l.ttft_ms.mean,
            ttft_p50: l.ttft_ms.p50,
            ttft_p90: l.ttft_ms.p90,
            ttft_p99: l.ttft_ms.p99,
            e2e_min: l.e2e_ms.min,
            e2e_mean: l.e2e_ms.mean,
            e2e_p50: l.e2e_ms.p50,
            e2e_p90: l.e2e_ms.p90,
            e2e_p99: l.e2e_ms.p99,
            per_token_min: l.per_token_ms.min,
            per_token_mean: l.per_token_ms.mean,
            per_token_p50: l.per_token_ms.p50,
            per_token_p90: l.per_token_ms.p90,
            per_token_p99: l.per_token_ms.p99,
            input_tokens_per_sec: m.throughput_metrics.input_tokens_per_sec,
            output_tokens_per_sec: m.throughput_metrics.output_tokens_per_sec,
            requests_per_sec: m.throughput_metrics.requests_per_sec,
            avg_kv_cache_util: m.utilization.avg_kv_cache_util,
            avg_flops_util: m.utilization.avg_flops_util,
            avg_bandwidth_util: m.utilization.avg_bandwidth_util,
            total_preemptions: m.preemptions.total,
            avg_preemptions_per_request: m.preemptions.per_request_mean,
            completed_requests: m.requests.completed,
            total_requests: m.requests.total,
            total_time,
        }
    }
}

/// Column-oriented time series consumed by the UI. `ttft_p50` / `tpot_p50`
/// are the names the UI reads; they carry the per-interval means.
#[derive(Serialize)]
struct TimeSeriesData {
    times: Vec<f64>,
    arrivals: Vec<u64>,
    running: Vec<usize>,
    waiting: Vec<usize>,
    kv_cache_util: Vec<f64>,
    num_prefilling: Vec<usize>,
    num_decoding: Vec<usize>,
    prefill_tokens: Vec<u32>,
    decode_tokens: Vec<u32>,
    input_throughput: Vec<f64>,
    output_throughput: Vec<f64>,
    ttft_p50: Vec<f64>,
    tpot_p50: Vec<f64>,
}

impl TimeSeriesData {
    fn from_points(ts: &[TimeSeriesPoint]) -> Self {
        Self {
            times: ts.iter().map(|p| p.time).collect(),
            arrivals: ts.iter().map(|p| p.arrivals).collect(),
            running: ts.iter().map(|p| p.running).collect(),
            waiting: ts.iter().map(|p| p.waiting).collect(),
            kv_cache_util: ts.iter().map(|p| p.kv_cache_util).collect(),
            num_prefilling: ts.iter().map(|p| p.num_prefilling).collect(),
            num_decoding: ts.iter().map(|p| p.num_decoding).collect(),
            prefill_tokens: ts.iter().map(|p| p.prefill_tokens).collect(),
            decode_tokens: ts.iter().map(|p| p.decode_tokens).collect(),
            input_throughput: ts.iter().map(|p| p.input_throughput).collect(),
            output_throughput: ts.iter().map(|p| p.output_throughput).collect(),
            ttft_p50: ts.iter().map(|p| p.ttft_interval_mean_ms).collect(),
            tpot_p50: ts.iter().map(|p| p.tpot_interval_mean_ms).collect(),
        }
    }
}

/// Latency samples in milliseconds with their observation times.
#[derive(Serialize)]
struct LatencySamplesData {
    ttft_samples: Vec<f64>,
    e2e_samples: Vec<f64>,
    tpot_samples: Vec<f64>,
    ttft_timestamps: Vec<f64>,
    e2e_timestamps: Vec<f64>,
    tpot_timestamps: Vec<f64>,
}

impl LatencySamplesData {
    fn from_samples(s: &LatencySamples<'_>) -> Self {
        let ms = |v: &[f64]| v.iter().map(|&x| x * 1000.0).collect::<Vec<_>>();
        Self {
            ttft_samples: ms(s.ttft.values),
            e2e_samples: ms(s.e2e.values),
            tpot_samples: ms(s.tpot.values),
            ttft_timestamps: s.ttft.timestamps.to_vec(),
            e2e_timestamps: s.e2e.timestamps.to_vec(),
            tpot_timestamps: s.tpot.timestamps.to_vec(),
        }
    }
}

#[derive(Serialize)]
struct DistributionData {
    input_lengths: Vec<u32>,
    output_lengths: Vec<u32>,
}

#[derive(Serialize)]
struct SimulationResult {
    metrics: MetricsData,
    time_series: TimeSeriesData,
    distributions: DistributionData,
    latency_samples: LatencySamplesData,
}

/// Progress event for `run_simulation_streaming`. Samples and lengths are
/// deltas since the previous event.
#[derive(Serialize)]
struct ProgressUpdate {
    current_time: f64,
    completed_requests: u64,
    total_requests: u64,
    running: usize,
    waiting: usize,
    kv_cache_util: f64,
    time_series: TimeSeriesData,
    metrics: MetricsData,
    latency_samples: LatencySamplesData,
    distribution_samples: DistributionData,
}

impl ProgressUpdate {
    fn from_progress(p: &ProgressInfo<'_>) -> Self {
        Self {
            current_time: p.current_time,
            completed_requests: p.completed_requests,
            total_requests: p.total_requests,
            running: p.running,
            waiting: p.waiting,
            kv_cache_util: p.kv_cache_util,
            time_series: TimeSeriesData::from_points(p.time_series),
            metrics: MetricsData::from_summary(&p.metrics, p.current_time),
            latency_samples: LatencySamplesData::from_samples(&p.latency_samples),
            distribution_samples: DistributionData {
                input_lengths: p.input_lengths.to_vec(),
                output_lengths: p.output_lengths.to_vec(),
            },
        }
    }
}

fn build_simulator(config_json: &str) -> Result<Simulator, JsValue> {
    console_error_panic_hook::set_once();
    let mut config: Config = serde_json::from_str(config_json)
        .map_err(|e| JsValue::from_str(&format!("Config parse error: {e}")))?;
    config.finalize();
    Simulator::new(config, None).map_err(|e| JsValue::from_str(&format!("Simulator error: {e}")))
}

fn final_result(simulator: &mut Simulator) -> Result<JsValue, JsValue> {
    let total_time = simulator.current_time();
    let summary = simulator.summary();
    let result = SimulationResult {
        metrics: MetricsData::from_summary(&summary, total_time),
        time_series: TimeSeriesData::from_points(simulator.time_series()),
        distributions: DistributionData {
            input_lengths: simulator.input_lengths().to_vec(),
            output_lengths: simulator.output_lengths().to_vec(),
        },
        latency_samples: LatencySamplesData::from_samples(&simulator.latency_samples()),
    };
    Ok(serde_wasm_bindgen::to_value(&result)?)
}

/// Run a simulation to completion and return the full result.
#[wasm_bindgen]
pub fn run_simulation(config_json: &str) -> Result<JsValue, JsValue> {
    let mut simulator = build_simulator(config_json)?;
    simulator
        .run_with_callback(|_| {})
        .map_err(|e| JsValue::from_str(&e))?;
    final_result(&mut simulator)
}

/// Run a simulation, invoking `progress_callback` with a `ProgressUpdate`
/// roughly every simulated second, and return the full result.
#[wasm_bindgen]
pub fn run_simulation_streaming(
    config_json: &str,
    progress_callback: &Function,
) -> Result<JsValue, JsValue> {
    let mut simulator = build_simulator(config_json)?;
    simulator
        .run_with_callback(|progress| {
            if let Ok(js_value) =
                serde_wasm_bindgen::to_value(&ProgressUpdate::from_progress(&progress))
            {
                let _ = progress_callback.call1(&JsValue::null(), &js_value);
            }
        })
        .map_err(|e| JsValue::from_str(&format!("Simulation error: {e}")))?;
    final_result(&mut simulator)
}
