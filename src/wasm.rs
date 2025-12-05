use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use js_sys::Function;

#[derive(Serialize, Deserialize)]
struct ProgressUpdate {
    current_time: f64,
    completed_requests: u64,
    total_requests: u64,
    running: usize,
    waiting: usize,
    kv_cache_util: f64,
    time_series: Option<TimeSeriesData>,
    metrics: Option<MetricsData>,
    latency_samples: Option<LatencySamples>,
    distribution_samples: Option<DistributionSamples>,
}

#[derive(Serialize, Deserialize)]
struct LatencySamples {
    ttft_samples: Vec<f64>,
    e2e_samples: Vec<f64>,
    tpot_samples: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
struct DistributionSamples {
    input_lengths: Vec<u32>,
    output_lengths: Vec<u32>,
}

#[wasm_bindgen]
pub fn run_simulation(config_json: &str) -> Result<JsValue, JsValue> {
    // Set up better panic messages
    console_error_panic_hook::set_once();

    let mut config: crate::Config = serde_json::from_str(config_json)
        .map_err(|e| JsValue::from_str(&format!("Config parse error: {}", e)))?;

    // Compute derived fields (same as from_file)
    config.hardware.compute_threshold();
    config.model.compute_kv_cache_size(config.hardware.bytes_per_param);

    // Compute KV cache capacity if not explicitly set
    let model_size_bytes = config.model.num_parameters * config.hardware.bytes_per_param as u64;
    config.hardware.compute_kv_cache_capacity(model_size_bytes);

    config.scheduler.set_default_prefill_threshold(config.model.max_seq_len);

    let mut simulator = crate::Simulator::new(config)
        .map_err(|e| JsValue::from_str(&format!("Simulator error: {}", e)))?;

    // Run simulation
    simulator.run();

    // Extract metrics
    let summary = simulator.get_metrics_summary();
    let time_series = simulator.get_time_series_data();
    let input_lengths = simulator.get_input_lengths();
    let output_lengths = simulator.get_output_lengths();
    let total_time = simulator.get_current_time();
    let ((ttft_samples, ttft_timestamps), (e2e_samples, e2e_timestamps), (tpot_samples, tpot_timestamps)) =
        simulator.get_latency_samples();

    let result = SimulationResult {
        metrics: MetricsData {
            ttft_mean: summary.ttft_mean,
            ttft_p50: summary.ttft_p50,
            ttft_p90: summary.ttft_p90,
            ttft_p99: summary.ttft_p99,
            e2e_mean: summary.e2e_mean,
            e2e_p50: summary.e2e_p50,
            e2e_p90: summary.e2e_p90,
            e2e_p99: summary.e2e_p99,
            per_token_mean: summary.per_token_mean,
            per_token_p50: summary.per_token_p50,
            per_token_p90: summary.per_token_p90,
            per_token_p99: summary.per_token_p99,
            input_tokens_per_sec: summary.input_tokens_per_sec,
            input_tokens_per_sec_p50: summary.input_tokens_per_sec_p50,
            input_tokens_per_sec_p90: summary.input_tokens_per_sec_p90,
            input_tokens_per_sec_p99: summary.input_tokens_per_sec_p99,
            output_tokens_per_sec: summary.output_tokens_per_sec,
            output_tokens_per_sec_p50: summary.output_tokens_per_sec_p50,
            output_tokens_per_sec_p90: summary.output_tokens_per_sec_p90,
            output_tokens_per_sec_p99: summary.output_tokens_per_sec_p99,
            requests_per_sec: summary.requests_per_sec,
            requests_per_sec_p50: summary.requests_per_sec_p50,
            requests_per_sec_p90: summary.requests_per_sec_p90,
            requests_per_sec_p99: summary.requests_per_sec_p99,
            avg_kv_cache_util: summary.avg_kv_cache_util,
            avg_flops_util: summary.avg_flops_util,
            avg_bandwidth_util: summary.avg_bandwidth_util,
            total_preemptions: summary.total_preemptions,
            avg_preemptions_per_request: summary.preemptions_per_request_mean,
            completed_requests: summary.completed_requests,
            total_requests: summary.total_requests,
            total_time,
        },
        time_series: TimeSeriesData {
            times: time_series.iter().map(|p| p.time).collect(),
            arrivals: time_series.iter().map(|p| p.arrivals).collect(),
            running: time_series.iter().map(|p| p.running).collect(),
            waiting: time_series.iter().map(|p| p.waiting).collect(),
            kv_cache_util: time_series.iter().map(|p| p.kv_cache_util).collect(),
            num_prefilling: time_series.iter().map(|p| p.num_prefilling).collect(),
            num_decoding: time_series.iter().map(|p| p.num_decoding).collect(),
            prefill_tokens: time_series.iter().map(|p| p.prefill_tokens).collect(),
            decode_tokens: time_series.iter().map(|p| p.decode_tokens).collect(),
            input_throughput: time_series.iter().map(|p| p.input_throughput).collect(),
            output_throughput: time_series.iter().map(|p| p.output_throughput).collect(),
            ttft_p50: time_series.iter().map(|p| p.ttft_p50).collect(),
            tpot_p50: time_series.iter().map(|p| p.tpot_p50).collect(),
        },
        distributions: DistributionData {
            input_lengths: input_lengths.to_vec(),
            output_lengths: output_lengths.to_vec(),
        },
        latency_samples: LatencySamplesData {
            ttft_samples: ttft_samples.iter().map(|&x| x * 1000.0).collect(),  // Convert to ms
            e2e_samples: e2e_samples.iter().map(|&x| x * 1000.0).collect(),    // Convert to ms
            tpot_samples: tpot_samples.iter().map(|&x| x * 1000.0).collect(),  // Convert to ms
            ttft_timestamps: ttft_timestamps.to_vec(),
            e2e_timestamps: e2e_timestamps.to_vec(),
            tpot_timestamps: tpot_timestamps.to_vec(),
        },
    };

    Ok(serde_wasm_bindgen::to_value(&result)?)
}

#[wasm_bindgen]
pub fn run_simulation_streaming(config_json: &str, progress_callback: &Function) -> Result<JsValue, JsValue> {
    // Set up better panic messages
    console_error_panic_hook::set_once();

    let mut config: crate::Config = serde_json::from_str(config_json)
        .map_err(|e| JsValue::from_str(&format!("Config parse error: {}", e)))?;

    // Compute derived fields (same as from_file)
    config.hardware.compute_threshold();
    config.model.compute_kv_cache_size(config.hardware.bytes_per_param);

    // Compute KV cache capacity if not explicitly set
    let model_size_bytes = config.model.num_parameters * config.hardware.bytes_per_param as u64;
    config.hardware.compute_kv_cache_capacity(model_size_bytes);

    config.scheduler.set_default_prefill_threshold(config.model.max_seq_len);

    let mut simulator = crate::Simulator::new(config)
        .map_err(|e| JsValue::from_str(&format!("Simulator error: {}", e)))?;

    // Run simulation with progress callback
    let callback_result = simulator.run_with_callback(|progress| {
        // Convert progress to JS value and call callback
        let progress_update = ProgressUpdate {
            current_time: progress.current_time,
            completed_requests: progress.completed_requests,
            total_requests: progress.total_requests,
            running: progress.running,
            waiting: progress.waiting,
            kv_cache_util: progress.kv_cache_util,
            time_series: progress.time_series.map(|ts| TimeSeriesData {
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
                ttft_p50: ts.iter().map(|p| p.ttft_p50).collect(),
                tpot_p50: ts.iter().map(|p| p.tpot_p50).collect(),
            }),
            metrics: progress.metrics.map(|m| MetricsData {
                ttft_mean: m.ttft_mean,
                ttft_p50: m.ttft_p50,
                ttft_p90: m.ttft_p90,
                ttft_p99: m.ttft_p99,
                e2e_mean: m.e2e_mean,
                e2e_p50: m.e2e_p50,
                e2e_p90: m.e2e_p90,
                e2e_p99: m.e2e_p99,
                per_token_mean: m.per_token_mean,
                per_token_p50: m.per_token_p50,
                per_token_p90: m.per_token_p90,
                per_token_p99: m.per_token_p99,
                input_tokens_per_sec: m.input_tokens_per_sec,
                input_tokens_per_sec_p50: m.input_tokens_per_sec_p50,
                input_tokens_per_sec_p90: m.input_tokens_per_sec_p90,
                input_tokens_per_sec_p99: m.input_tokens_per_sec_p99,
                output_tokens_per_sec: m.output_tokens_per_sec,
                output_tokens_per_sec_p50: m.output_tokens_per_sec_p50,
                output_tokens_per_sec_p90: m.output_tokens_per_sec_p90,
                output_tokens_per_sec_p99: m.output_tokens_per_sec_p99,
                requests_per_sec: m.requests_per_sec,
                requests_per_sec_p50: m.requests_per_sec_p50,
                requests_per_sec_p90: m.requests_per_sec_p90,
                requests_per_sec_p99: m.requests_per_sec_p99,
                avg_kv_cache_util: m.avg_kv_cache_util,
                avg_flops_util: m.avg_flops_util,
                avg_bandwidth_util: m.avg_bandwidth_util,
                total_preemptions: m.total_preemptions,
                avg_preemptions_per_request: m.preemptions_per_request_mean,
                completed_requests: m.completed_requests,
                total_requests: m.total_requests,
                total_time: progress.current_time,
            }),
            latency_samples: progress.latency_samples.map(|((ttft, _), (e2e, _), (tpot, _))| {
                LatencySamples {
                    ttft_samples: ttft.iter().map(|&x| x * 1000.0).collect(), // Convert to ms
                    e2e_samples: e2e.iter().map(|&x| x * 1000.0).collect(),
                    tpot_samples: tpot.iter().map(|&x| x * 1000.0).collect(),
                }
            }),
            distribution_samples: progress.distribution_samples.map(|(input, output)| {
                DistributionSamples {
                    input_lengths: input.to_vec(),
                    output_lengths: output.to_vec(),
                }
            }),
        };

        if let Ok(js_value) = serde_wasm_bindgen::to_value(&progress_update) {
            let this = JsValue::null();
            let _ = progress_callback.call1(&this, &js_value);
        }
    });

    if let Err(e) = callback_result {
        return Err(JsValue::from_str(&format!("Simulation error: {}", e)));
    }

    // Extract final metrics
    let summary = simulator.get_metrics_summary();
    let time_series = simulator.get_time_series_data();
    let input_lengths = simulator.get_input_lengths();
    let output_lengths = simulator.get_output_lengths();
    let total_time = simulator.get_current_time();
    let ((ttft_samples, ttft_timestamps), (e2e_samples, e2e_timestamps), (tpot_samples, tpot_timestamps)) =
        simulator.get_latency_samples();

    let result = SimulationResult {
        metrics: MetricsData {
            ttft_mean: summary.ttft_mean,
            ttft_p50: summary.ttft_p50,
            ttft_p90: summary.ttft_p90,
            ttft_p99: summary.ttft_p99,
            e2e_mean: summary.e2e_mean,
            e2e_p50: summary.e2e_p50,
            e2e_p90: summary.e2e_p90,
            e2e_p99: summary.e2e_p99,
            per_token_mean: summary.per_token_mean,
            per_token_p50: summary.per_token_p50,
            per_token_p90: summary.per_token_p90,
            per_token_p99: summary.per_token_p99,
            input_tokens_per_sec: summary.input_tokens_per_sec,
            input_tokens_per_sec_p50: summary.input_tokens_per_sec_p50,
            input_tokens_per_sec_p90: summary.input_tokens_per_sec_p90,
            input_tokens_per_sec_p99: summary.input_tokens_per_sec_p99,
            output_tokens_per_sec: summary.output_tokens_per_sec,
            output_tokens_per_sec_p50: summary.output_tokens_per_sec_p50,
            output_tokens_per_sec_p90: summary.output_tokens_per_sec_p90,
            output_tokens_per_sec_p99: summary.output_tokens_per_sec_p99,
            requests_per_sec: summary.requests_per_sec,
            requests_per_sec_p50: summary.requests_per_sec_p50,
            requests_per_sec_p90: summary.requests_per_sec_p90,
            requests_per_sec_p99: summary.requests_per_sec_p99,
            avg_kv_cache_util: summary.avg_kv_cache_util,
            avg_flops_util: summary.avg_flops_util,
            avg_bandwidth_util: summary.avg_bandwidth_util,
            total_preemptions: summary.total_preemptions,
            avg_preemptions_per_request: summary.preemptions_per_request_mean,
            completed_requests: summary.completed_requests,
            total_requests: summary.total_requests,
            total_time,
        },
        time_series: TimeSeriesData {
            times: time_series.iter().map(|p| p.time).collect(),
            arrivals: time_series.iter().map(|p| p.arrivals).collect(),
            running: time_series.iter().map(|p| p.running).collect(),
            waiting: time_series.iter().map(|p| p.waiting).collect(),
            kv_cache_util: time_series.iter().map(|p| p.kv_cache_util).collect(),
            num_prefilling: time_series.iter().map(|p| p.num_prefilling).collect(),
            num_decoding: time_series.iter().map(|p| p.num_decoding).collect(),
            prefill_tokens: time_series.iter().map(|p| p.prefill_tokens).collect(),
            decode_tokens: time_series.iter().map(|p| p.decode_tokens).collect(),
            input_throughput: time_series.iter().map(|p| p.input_throughput).collect(),
            output_throughput: time_series.iter().map(|p| p.output_throughput).collect(),
            ttft_p50: time_series.iter().map(|p| p.ttft_p50).collect(),
            tpot_p50: time_series.iter().map(|p| p.tpot_p50).collect(),
        },
        distributions: DistributionData {
            input_lengths: input_lengths.to_vec(),
            output_lengths: output_lengths.to_vec(),
        },
        latency_samples: LatencySamplesData {
            ttft_samples: ttft_samples.iter().map(|&x| x * 1000.0).collect(),  // Convert to ms
            e2e_samples: e2e_samples.iter().map(|&x| x * 1000.0).collect(),    // Convert to ms
            tpot_samples: tpot_samples.iter().map(|&x| x * 1000.0).collect(),  // Convert to ms
            ttft_timestamps: ttft_timestamps.to_vec(),
            e2e_timestamps: e2e_timestamps.to_vec(),
            tpot_timestamps: tpot_timestamps.to_vec(),
        },
    };

    Ok(serde_wasm_bindgen::to_value(&result)?)
}

#[derive(Serialize, Deserialize)]
struct SimulationResult {
    metrics: MetricsData,
    time_series: TimeSeriesData,
    distributions: DistributionData,
    latency_samples: LatencySamplesData,
}

#[derive(Serialize, Deserialize)]
struct MetricsData {
    ttft_mean: f64,
    ttft_p50: f64,
    ttft_p90: f64,
    ttft_p99: f64,
    e2e_mean: f64,
    e2e_p50: f64,
    e2e_p90: f64,
    e2e_p99: f64,
    per_token_mean: f64,
    per_token_p50: f64,
    per_token_p90: f64,
    per_token_p99: f64,
    input_tokens_per_sec: f64,
    input_tokens_per_sec_p50: f64,
    input_tokens_per_sec_p90: f64,
    input_tokens_per_sec_p99: f64,
    output_tokens_per_sec: f64,
    output_tokens_per_sec_p50: f64,
    output_tokens_per_sec_p90: f64,
    output_tokens_per_sec_p99: f64,
    requests_per_sec: f64,
    requests_per_sec_p50: f64,
    requests_per_sec_p90: f64,
    requests_per_sec_p99: f64,
    avg_kv_cache_util: f64,
    avg_flops_util: f64,
    avg_bandwidth_util: f64,
    total_preemptions: u64,
    avg_preemptions_per_request: f64,
    completed_requests: u64,
    total_requests: u64,
    total_time: f64,
}

#[derive(Serialize, Deserialize)]
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

#[derive(Serialize, Deserialize)]
struct DistributionData {
    input_lengths: Vec<u32>,
    output_lengths: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
struct LatencySamplesData {
    ttft_samples: Vec<f64>,      // in ms
    e2e_samples: Vec<f64>,       // in ms
    tpot_samples: Vec<f64>,      // in ms
    ttft_timestamps: Vec<f64>,   // completion time for each sample
    e2e_timestamps: Vec<f64>,    // completion time for each sample
    tpot_timestamps: Vec<f64>,   // generation time for each token
}
