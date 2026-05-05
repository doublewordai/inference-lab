//! Disaggregated prefill/decode simulator.
//!
//! Each worker pool runs independently — its own `Scheduler`, `ComputeEngine`,
//! and `KVCacheManager`. Requests flow:
//!
//! ```text
//! arrival → prefill_router → P_worker → KV handoff (Link) → D_worker → done
//! ```
//!
//! Routing is round-robin §1. KV hand-off is
//! pushed onto a shared bandwidth-limited `Link` (per §2). Decode workers run
//! continuous batching via the existing single-cluster scheduler logic — the
//! disagg layer is just the orchestrator.
//!
//! For v1 the supported topology is one prefill worker + one decode worker;
//! conc=1 validation runs through the same code path that scales up to
//! multi-worker pools later.

use crate::compute::ComputeEngine;
use crate::config::{ClusterSpec, DisaggTopology, ModelConfig, ModelCosts, SchedulerConfig};
use crate::kv_cache::{KVCacheManager, Link};
use crate::request::Request;
use crate::scheduler::Scheduler;

/// Per-request timing breakdown.
#[derive(Debug, Clone)]
pub struct RequestTiming {
    pub request_id: String,
    pub arrival_time: f64,
    /// Time the request's prefill phase finished on the prefill worker.
    pub prefill_done_time: f64,
    /// Time the KV hand-off transfer completed and the request entered the
    /// decode worker.
    pub handoff_done_time: f64,
    /// Time the first output token was produced (= TTFT relative to arrival).
    pub first_token_time: f64,
    /// Time the request completed.
    pub completion_time: f64,
    pub num_prompt_tokens: u32,
    pub num_output_tokens: u32,
}

impl RequestTiming {
    pub fn ttft(&self) -> f64 {
        self.first_token_time - self.arrival_time
    }
    pub fn e2e(&self) -> f64 {
        self.completion_time - self.arrival_time
    }
    /// Mean time per output token (TPOT). Computed across the decode phase
    /// only (token 1 onwards), matching the InferenceX definition.
    pub fn tpot(&self) -> f64 {
        if self.num_output_tokens <= 1 {
            return 0.0;
        }
        let decode_span = self.completion_time - self.first_token_time;
        decode_span / (self.num_output_tokens - 1) as f64
    }
}

/// One prefill or decode worker. Owns its own scheduler, compute engine,
/// and (transitively) KV cache manager.
struct Worker {
    scheduler: Scheduler,
    compute_engine: ComputeEngine,
}

impl Worker {
    fn new(
        cluster: &ClusterSpec,
        model: ModelConfig,
        scheduler_config: SchedulerConfig,
    ) -> Result<Self, String> {
        let mut cluster = cluster.clone();
        // The single-cluster KV cache sizing assumes finalize() has run; for
        // the disagg simulator we own the lifecycle, so do it locally.
        let model_size_bytes = model.weight_residency_bytes();
        cluster.compute_kv_cache_capacity(model_size_bytes);

        let kv_cache_manager = KVCacheManager::new(
            cluster.hardware.kv_cache_capacity,
            scheduler_config.block_size,
            model.kv_storage_bytes(1),
            true,
        );

        let scheduler = Scheduler::new(
            scheduler_config,
            cluster.hardware.clone(),
            model.clone(),
            kv_cache_manager,
        )?;
        let mut compute_engine = ComputeEngine::new(
            cluster.hardware.clone(),
            cluster.parallel.clone(),
            model,
        );
        if let Some(comms) = cluster.comms {
            compute_engine = compute_engine.with_comms(comms);
        }
        Ok(Self {
            scheduler,
            compute_engine,
        })
    }
}

/// Run a single request end-to-end through a 1×P + 1×D disagg topology and
/// return its timing breakdown. This is the conc=1 path — the same machinery
/// extends to multi-request workloads once we wire arrivals + a global event
/// loop on top.
pub fn predict_single_request(
    topology: &DisaggTopology,
    model: ModelConfig,
    scheduler_config: SchedulerConfig,
    request: Request,
) -> Result<RequestTiming, String> {
    let arrival_time = request.arrival_time;
    let num_prompt_tokens = request.num_prompt_tokens;
    let request_id = request.request_id.clone();

    let mut prefill_worker =
        Worker::new(&topology.prefill, model.clone(), scheduler_config.clone())?;
    let mut decode_worker = Worker::new(&topology.decode, model.clone(), scheduler_config.clone())?;
    let mut kv_link = Link::new(topology.kv_link_bw);

    // --- Prefill phase ---
    prefill_worker.scheduler.add_request(request);
    let mut now = arrival_time;
    let mut prefill_done_time = None;
    while prefill_done_time.is_none() {
        let decision = prefill_worker.scheduler.schedule(now);
        if decision.scheduled_new.is_empty() && decision.scheduled_running.is_empty() {
            return Err("prefill scheduler made no progress".into());
        }
        // Build the batch we just scheduled and ask the engine for the
        // iteration time.
        let mut batch_indices: Vec<usize> = decision.scheduled_new.iter().copied().collect();
        batch_indices.extend(decision.scheduled_running.iter().copied());
        let mut tokens_per_request: Vec<u32> = decision.tokens_for_new.clone();
        tokens_per_request.extend(decision.tokens_for_running.iter().copied());
        let running = prefill_worker.scheduler.running();
        let batch_refs: Vec<&Request> = batch_indices.iter().map(|&i| &running[i]).collect();
        let iter_time = prefill_worker
            .compute_engine
            .calculate_iteration_time(&batch_refs, &tokens_per_request);
        now += iter_time;

        // Apply the iteration's token deltas. Mirror simulator.rs steps 5-6:
        // record_generated_tokens before checking is_prefill, since the
        // bookkeeping tracks num_computed_tokens crossing num_prompt_tokens.
        for (i, &idx) in decision.scheduled_new.iter().enumerate() {
            if let Some(req) = prefill_worker.scheduler.running_mut().get_mut(idx) {
                req.record_generated_tokens(decision.tokens_for_new[i], now);
            }
        }
        for (i, &idx) in decision.scheduled_running.iter().enumerate() {
            if let Some(req) = prefill_worker.scheduler.running_mut().get_mut(idx) {
                req.record_generated_tokens(decision.tokens_for_running[i], now);
            }
        }

        // Check whether the request finished prefill this step. In disagg the
        // prefill worker doesn't continue into decode — once num_computed
        // hits num_prompt we pull the request out, free its KV, and submit a
        // hand-off transfer.
        let done = prefill_worker
            .scheduler
            .running()
            .iter()
            .position(|r| r.num_computed_tokens >= r.num_prompt_tokens);
        if let Some(idx) = done {
            prefill_done_time = Some(now);
            let mut req = prefill_worker.scheduler.running_mut().remove(idx);
            // Free KV blocks on the prefill side; they're transferred over the
            // link, not retained on the prefill worker.
            prefill_worker
                .scheduler
                .kv_cache_manager_mut()
                .free_blocks(&req.kv_blocks);
            req.kv_blocks.clear();
            // Submit the hand-off transfer. KV bytes for the full prompt.
            let kv_bytes = model.kv_storage_bytes(num_prompt_tokens);
            kv_link.submit(req.request_id.clone(), kv_bytes, now);

            // --- KV hand-off ---
            // Single-request case: drain the link to completion.
            let handoff_eta = kv_link.estimate_remaining(&req.request_id);
            now += handoff_eta;
            let _ = kv_link.advance(now);
            let handoff_done_time = now;

            // --- Decode phase ---
            // Reset the request so the decode scheduler sees it as fresh-but-prefilled.
            // num_computed_tokens stays at num_prompt_tokens; the existing
            // scheduler will treat is_prefill()==false and only schedule
            // decode tokens.
            decode_worker.scheduler.add_request(req);

            let mut first_token_time: Option<f64> = None;
            let completion_time;
            loop {
                let decision = decode_worker.scheduler.schedule(now);
                if decision.scheduled_new.is_empty()
                    && decision.scheduled_running.is_empty()
                    && decision.completed.is_empty()
                {
                    return Err("decode scheduler stuck before request completed".into());
                }
                let mut batch_indices: Vec<usize> =
                    decision.scheduled_new.iter().copied().collect();
                batch_indices.extend(decision.scheduled_running.iter().copied());
                let mut tokens_per_request: Vec<u32> = decision.tokens_for_new.clone();
                tokens_per_request.extend(decision.tokens_for_running.iter().copied());
                let iter_time = if !batch_indices.is_empty() {
                    let running = decode_worker.scheduler.running();
                    let batch_refs: Vec<&Request> =
                        batch_indices.iter().map(|&i| &running[i]).collect();
                    decode_worker
                        .compute_engine
                        .calculate_iteration_time(&batch_refs, &tokens_per_request)
                } else {
                    0.0
                };
                now += iter_time;
                for (i, &idx) in decision.scheduled_new.iter().enumerate() {
                    if let Some(req) = decode_worker.scheduler.running_mut().get_mut(idx) {
                        req.record_generated_tokens(decision.tokens_for_new[i], now);
                    }
                }
                for (i, &idx) in decision.scheduled_running.iter().enumerate() {
                    if let Some(req) = decode_worker.scheduler.running_mut().get_mut(idx) {
                        req.record_generated_tokens(decision.tokens_for_running[i], now);
                    }
                }
                if first_token_time.is_none() {
                    if let Some(r) = decode_worker
                        .scheduler
                        .running()
                        .iter()
                        .find(|r| r.first_token_time.is_some())
                    {
                        first_token_time = r.first_token_time;
                    } else if let Some(r) = decision
                        .completed
                        .iter()
                        .find(|r| r.first_token_time.is_some())
                    {
                        first_token_time = r.first_token_time;
                    }
                }
                if !decision.completed.is_empty() {
                    let req = &decision.completed[0];
                    completion_time = now;
                    return Ok(RequestTiming {
                        request_id,
                        arrival_time,
                        prefill_done_time: prefill_done_time.unwrap(),
                        handoff_done_time,
                        first_token_time: first_token_time.unwrap_or(now),
                        completion_time,
                        num_prompt_tokens,
                        num_output_tokens: req.num_output_tokens,
                    });
                }
                if now > arrival_time + 600.0 {
                    return Err("decode loop exceeded 10 minutes — likely stuck".into());
                }
            }
        }
    }
    Err("prefill loop exited without completion".into())
}

/// Convenience helper used by tests / examples to construct a fresh request.
pub fn make_request(id: &str, isl: u32, osl: u32, arrival: f64) -> Request {
    Request::new(id.to_string(), 0, arrival, isl, osl)
}

/// Roofline TPOT (seconds per output token) for a decode-only cluster running
/// a steady-state batch of `batch_size` requests, each at average sequence
/// length `avg_seq_len`. One iteration of `ComputeEngine` produces one output
/// token per running request, so iteration time *is* TPOT.
///
/// This is the same arithmetic as `predict_single_request`'s decode loop,
/// just packaged for batch sweeps without the prefill/handoff plumbing.
/// EP-aware weight sharding is *not* explicitly modelled — `ComputeEngine`
/// treats weights as `aggregate_active_param_bytes / aggregate_BW`, which
/// approximates EP=TP routed-expert sharding (per-GPU 1/TP slice of the
/// active params).
pub fn predict_decode_tpot(
    decode_cluster: &ClusterSpec,
    model: &ModelConfig,
    batch_size: u32,
    avg_seq_len: u32,
) -> f64 {
    let mut cluster = decode_cluster.clone();
    let model_size_bytes = model.weight_residency_bytes();
    cluster.compute_kv_cache_capacity(model_size_bytes);
    let mut engine = ComputeEngine::new(
        cluster.hardware.clone(),
        cluster.parallel.clone(),
        model.clone(),
    );
    if let Some(comms) = cluster.comms {
        engine = engine.with_comms(comms);
    }

    // Construct `batch_size` decode-phase requests. num_computed_tokens is
    // already at num_prompt_tokens so each is fully prefilled and asks for
    // one output token this iteration.
    let requests: Vec<Request> = (0..batch_size)
        .map(|i| {
            let mut req = Request::new(format!("r{i}"), 0, 0.0, avg_seq_len, 1);
            req.num_computed_tokens = avg_seq_len;
            req
        })
        .collect();

    let req_refs: Vec<&Request> = requests.iter().collect();
    let tokens_per_request: Vec<u32> = vec![1; batch_size as usize];
    engine.calculate_iteration_time(&req_refs, &tokens_per_request)
}

/// Run a single request through a single (aggregated) cluster — no disagg, no
/// KV handoff. Same `ComputeEngine` / `Scheduler` machinery, just one worker
/// doing prefill and decode back-to-back. Useful as a baseline comparison
/// against [`predict_single_request`] under the same hardware budget.
///
/// Note: at conc=1 chunked prefill collapses to the same arithmetic as a
/// monolithic prefill — there's no decode batch to interleave with.
pub fn predict_single_request_aggregated(
    cluster: &ClusterSpec,
    model: ModelConfig,
    scheduler_config: SchedulerConfig,
    request: Request,
) -> Result<RequestTiming, String> {
    let arrival_time = request.arrival_time;
    let num_prompt_tokens = request.num_prompt_tokens;
    let request_id = request.request_id.clone();

    let mut worker = Worker::new(cluster, model, scheduler_config)?;
    worker.scheduler.add_request(request);

    let mut now = arrival_time;
    let mut prefill_done_time: Option<f64> = None;
    let mut first_token_time: Option<f64> = None;

    loop {
        let decision = worker.scheduler.schedule(now);
        if decision.scheduled_new.is_empty()
            && decision.scheduled_running.is_empty()
            && decision.completed.is_empty()
        {
            return Err("aggregated scheduler stuck before request completed".into());
        }
        let mut batch_indices: Vec<usize> = decision.scheduled_new.iter().copied().collect();
        batch_indices.extend(decision.scheduled_running.iter().copied());
        let mut tokens_per_request: Vec<u32> = decision.tokens_for_new.clone();
        tokens_per_request.extend(decision.tokens_for_running.iter().copied());
        let iter_time = if !batch_indices.is_empty() {
            let running = worker.scheduler.running();
            let batch_refs: Vec<&Request> = batch_indices.iter().map(|&i| &running[i]).collect();
            worker
                .compute_engine
                .calculate_iteration_time(&batch_refs, &tokens_per_request)
        } else {
            0.0
        };
        now += iter_time;

        // Detect prefill→decode boundary by looking at num_computed_tokens
        // *before* the iteration's deltas land below.
        let was_prefill = worker
            .scheduler
            .running()
            .iter()
            .any(|r| r.num_computed_tokens < r.num_prompt_tokens);

        for (i, &idx) in decision.scheduled_new.iter().enumerate() {
            if let Some(req) = worker.scheduler.running_mut().get_mut(idx) {
                req.record_generated_tokens(decision.tokens_for_new[i], now);
            }
        }
        for (i, &idx) in decision.scheduled_running.iter().enumerate() {
            if let Some(req) = worker.scheduler.running_mut().get_mut(idx) {
                req.record_generated_tokens(decision.tokens_for_running[i], now);
            }
        }

        if prefill_done_time.is_none() {
            let still_prefill = worker
                .scheduler
                .running()
                .iter()
                .any(|r| r.num_computed_tokens < r.num_prompt_tokens);
            if was_prefill && !still_prefill {
                prefill_done_time = Some(now);
            }
        }
        if first_token_time.is_none() {
            if let Some(r) = worker
                .scheduler
                .running()
                .iter()
                .find(|r| r.first_token_time.is_some())
            {
                first_token_time = r.first_token_time;
            }
        }

        if !decision.completed.is_empty() {
            let req = &decision.completed[0];
            // Aggregated has no separate handoff; align fields to keep the
            // breakdown printable.
            let prefill_done_time = prefill_done_time.unwrap_or(first_token_time.unwrap_or(now));
            let first_token_time = first_token_time.unwrap_or(now);
            return Ok(RequestTiming {
                request_id,
                arrival_time,
                prefill_done_time,
                handoff_done_time: prefill_done_time,
                first_token_time,
                completion_time: now,
                num_prompt_tokens,
                num_output_tokens: req.num_output_tokens,
            });
        }
        if now > arrival_time + 600.0 {
            return Err("aggregated loop exceeded 10 minutes — likely stuck".into());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ClusterSpec, HardwareConfig, Node, ParallelConfig, SchedulerConfig};

    fn b300_per_gpu() -> HardwareConfig {
        HardwareConfig {
            name: "B300".into(),
            // FP4 dense tensor core ~15 PFLOPS per B300 (placeholder — see
            // step 1d notes; refine when we have spec sheet numbers).
            compute_flops: 1.5e16,
            memory_bandwidth: 8.0e12,
            memory_capacity: 309_237_645_312, // 288 GB
            kv_cache_capacity: 0,
            gpu_memory_utilization: 0.9,
            bytes_per_param: 1,
            kv_tiers: Vec::new(),
        }
    }

    fn small_model() -> ModelConfig {
        // Tiny dense model so the test runs quickly; the V4-Pro arithmetic
        // and the InferenceX comparison live in a separate config-driven
        // example, not in this unit test.
        use crate::config::DenseModel;
        ModelConfig::Dense(DenseModel {
            name: "test".into(),
            num_parameters: 1_000_000_000,
            num_active_parameters: None,
            num_layers: 16,
            hidden_dim: 1024,
            num_heads: 16,
            num_kv_heads: None,
            max_seq_len: 4096,
            bytes_per_param: Some(1),
        })
    }

    fn small_topology() -> DisaggTopology {
        let cluster = ClusterSpec {
            hardware: b300_per_gpu(),
            parallel: ParallelConfig { tp: 4, ep: 1 },
            comms: None,
            num_workers: 1,
            node: 0,
        };
        DisaggTopology {
            nodes: vec![Node {
                name: "rack-0".into(),
                num_gpus: 8,
                intra_node_link_bw: 9.0e11,
            }],
            inter_node_link_bw: None,
            prefill: cluster.clone(),
            decode: cluster,
            kv_link_bw: 9.0e11,
        }
    }

    fn small_scheduler() -> SchedulerConfig {
        SchedulerConfig {
            max_num_batched_tokens: 16384,
            max_num_seqs: 256,
            policy: "fcfs".into(),
            enable_chunked_prefill: false,
            long_prefill_token_threshold: 0,
            max_num_partial_prefills: 1,
            block_size: 64,
            enable_preemption_free: true,
            enable_cascade_attention: false,
        }
    }

    #[test]
    fn predict_single_request_smoke() {
        let model = small_model();
        let topology = small_topology();
        let scheduler = small_scheduler();
        let req = make_request("r1", 1024, 32, 0.0);
        let timing = predict_single_request(&topology, model, scheduler, req).unwrap();
        // Sanity: TTFT is positive, E2E > TTFT, prefill < TTFT, handoff <= TTFT.
        assert!(timing.ttft() > 0.0);
        assert!(timing.e2e() >= timing.ttft());
        assert!(timing.prefill_done_time > timing.arrival_time);
        assert!(timing.handoff_done_time >= timing.prefill_done_time);
        assert!(timing.first_token_time >= timing.handoff_done_time);
        assert_eq!(timing.num_output_tokens, 32);
    }
}
