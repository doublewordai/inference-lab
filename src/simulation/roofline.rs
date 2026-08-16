//! Analytic roofline scalars: one-iteration time predictions that ignore
//! scheduler queueing, KV pressure, and topology composition. Useful as
//! upper-bound references for the full event-driven [`super::Engine`].

use crate::config::{ClusterSpec, ModelConfig};
use crate::request::Request;

/// Roofline TPOT (seconds per output token) for a decode-only cluster running
/// a steady-state batch of `batch_size` requests, each at average sequence
/// length `avg_seq_len`. One iteration produces one output token per running
/// request, so iteration time *is* TPOT.
pub fn predict_decode_tpot(
    decode_cluster: &ClusterSpec,
    model: &ModelConfig,
    batch_size: u32,
    avg_seq_len: u32,
) -> f64 {
    let engine = decode_cluster.compute_engine(model.clone());
    let requests: Vec<Request> = (0..batch_size)
        .map(|i| {
            let mut req = Request::new(format!("r{i}"), 0, 0.0, avg_seq_len, 2);
            req.mark_prefilled(0.0);
            req
        })
        .collect();
    let req_refs: Vec<&Request> = requests.iter().collect();
    let tokens_per_request: Vec<u32> = vec![1; batch_size as usize];
    engine.calculate_iteration_time(&req_refs, &tokens_per_request)
}

/// Roofline prefill latency (seconds) for a single ISL-length request running
/// alone on a prefill-only cluster. One monolithic iteration, no chunking.
pub fn predict_prefill_time(prefill_cluster: &ClusterSpec, model: &ModelConfig, isl: u32) -> f64 {
    let engine = prefill_cluster.compute_engine(model.clone());
    let req = Request::new("prefill".to_string(), 0, 0.0, isl, 1);
    let req_refs: Vec<&Request> = vec![&req];
    let tokens_per_request: Vec<u32> = vec![isl];
    engine.calculate_iteration_time(&req_refs, &tokens_per_request)
}
