//! Realtime OpenAI-compatible serve driver. Wraps the unified
//! [`crate::simulation::Engine`] with a tokio loop that paces sim-time to
//! wall-time and forwards per-iter token generation back to HTTP clients.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::time::Duration;

use crate::config::{Deployment, WorkloadConfig};
use crate::request::Request;
use crate::simulation::{Engine, StepKind, Topology};

use super::capacity::Capacity;
use super::types::{EngineRequest, TokenEvent};

/// Also cycled by serve::fault for the deterministic partial output ahead of a death.
pub(crate) const PLACEHOLDER_WORDS: &[&str] = &[
    "the", "of", "and", "to", "in", "a", "is", "that", "for", "it", "was", "on", "are", "be",
    "with", "as", "at", "this", "have", "from", "or", "an", "by", "not", "but", "what", "all",
    "were", "when", "we", "there", "can", "which", "their", "if", "do", "will", "each", "about",
    "how", "up", "out", "them", "then", "she", "many", "some", "so", "these", "would",
];

struct LiveRequest {
    tx: mpsc::Sender<TokenEvent>,
    word_index: usize,
    first_token_sent: bool,
}

pub struct RealtimeEngine {
    engine: Engine,
    /// Output lengths are sampled from this workload's `output_len_dist`
    /// (capped at the request's `max_tokens`); without one every request
    /// runs to `max_tokens`.
    workload: Option<WorkloadConfig>,
    rx: mpsc::Receiver<EngineRequest>,
    live_requests: HashMap<String, LiveRequest>,
    /// Wall-clock-anchored offset from sim-time to real-time. Set on the
    /// first event we process. We use it to translate engine event-times
    /// back into `tokio::time::Instant`s for `sleep_until`.
    epoch: Option<tokio::time::Instant>,
    /// Queue depth published to the HTTP handlers, and the control knobs
    /// they write back. See [`super::capacity`].
    capacity: Arc<Capacity>,
    /// The concurrency cap currently written through to the schedulers, so a
    /// control change is detected as a difference against `capacity`.
    applied_max_num_seqs: u32,
}

impl RealtimeEngine {
    pub fn new(
        deployment: &Deployment,
        workload: Option<WorkloadConfig>,
        rx: mpsc::Receiver<EngineRequest>,
        capacity: Arc<Capacity>,
    ) -> Result<Self, String> {
        let topology = Topology::aggregated(
            deployment.cluster(),
            deployment.model.clone(),
            deployment.scheduler.clone(),
        )?
        .with_routers(&deployment.router, deployment.decode_router());
        let engine = Engine::new(topology);
        let applied_max_num_seqs = engine.max_num_seqs();
        Ok(Self {
            engine,
            workload,
            rx,
            live_requests: HashMap::new(),
            epoch: None,
            capacity,
            applied_max_num_seqs,
        })
    }

    pub async fn run(mut self) {
        log::info!("RealtimeEngine started");
        self.epoch = Some(tokio::time::Instant::now());
        self.publish_depth(0);

        loop {
            // 0. Pick up any capacity knob turned since the last pass. Done
            //    at the top of every iteration as well as on the `changed`
            //    wakeup, so a knob turned while we were mid-step is still
            //    applied at the next opportunity rather than waiting for a
            //    second notification.
            self.apply_capacity_controls();

            // 1. Drain any pending HTTP arrivals into the engine.
            loop {
                match self.rx.try_recv() {
                    Ok(req) => self.admit_request(req),
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        if self.live_requests.is_empty() && self.engine.is_idle() {
                            log::info!(
                                "RealtimeEngine shutting down: no senders, no live requests"
                            );
                            return;
                        }
                        break;
                    }
                }
            }

            // 2. Decide what to do next: wait for the next sim event, OR for
            //    a new HTTP request, OR for a capacity change, whichever
            //    fires first. If nothing is in flight at all, just block.
            let next_ev = self.engine.next_event_time();
            match next_ev {
                None => {
                    // Engine fully idle. Block until a request arrives, a
                    // knob turns, or senders drop.
                    tokio::select! {
                        biased;
                        received = self.rx.recv() => match received {
                            Some(req) => self.admit_request(req),
                            None => {
                                log::info!("RealtimeEngine shutting down: receiver closed");
                                return;
                            }
                        },
                        // Loop and re-evaluate; `apply_capacity_controls`
                        // at the top of the next pass does the work.
                        _ = self.capacity.changed() => {}
                    }
                }
                Some(t_sim) => {
                    let wake = self.sim_to_wall(t_sim);
                    tokio::select! {
                        biased;
                        Some(req) = self.rx.recv() => {
                            self.admit_request(req);
                            // Loop and re-evaluate.
                        }
                        _ = self.capacity.changed() => {}
                        _ = tokio::time::sleep_until(wake) => {
                            let arrived = self.advance_one_step().await;
                            self.publish_depth(arrived);
                        }
                    }
                }
            }
        }
    }

    /// Write a changed concurrency cap through to every scheduler.
    ///
    /// This is the serve-mode scale up/down: it changes how fast the engine
    /// drains work, and so the offered load at which the queue backs up and
    /// admission starts refusing. A cap of 0 would wedge the engine (nothing
    /// could ever be admitted), so it is ignored here as well as rejected at
    /// the control route.
    fn apply_capacity_controls(&mut self) {
        let desired = self.capacity.max_num_seqs();
        if desired == 0 || desired == self.applied_max_num_seqs {
            return;
        }
        log::info!(
            "capacity: max_num_seqs {} -> {}",
            self.applied_max_num_seqs,
            desired
        );
        self.engine.set_max_num_seqs(desired);
        self.applied_max_num_seqs = desired;
    }

    /// Publish the engine's queue depth for the admission check, handing
    /// back the reservations of `arrived` requests that have now landed in a
    /// scheduler's waiting queue and are therefore counted by
    /// `aggregate_waiting`.
    fn publish_depth(&self, arrived: usize) {
        self.capacity.publish(
            self.engine.aggregate_waiting(),
            self.engine.aggregate_running(),
            arrived,
        );
    }

    /// Emit one engine step's token events to their HTTP clients.
    ///
    /// Sends BLOCK on a full client channel rather than dropping (this used to be
    /// `try_send`, which silently discarded everything past the channel's 64 slots — the
    /// terminating `Done` included, so a client saw the stream stop dead at exactly 64
    /// frames with no finish_reason, no usage and no [DONE], byte-indistinguishable from
    /// a mid-stream death). Blocking makes a slow reader back-pressure generation, which
    /// is what a real engine does; a client that has actually gone away closes the
    /// channel, and [`Self::deliver`] reports that so the request is dropped promptly.
    ///
    /// Returns how many arrivals this step moved into a scheduler's waiting
    /// queue (0 or 1) — the caller hands those admission reservations back
    /// to [`Capacity`], which now counts them via `aggregate_waiting`.
    async fn advance_one_step(&mut self) -> usize {
        let outcome = match self.engine.step() {
            Ok(o) => o,
            Err(e) => {
                log::error!("engine step failed: {e}");
                return 0;
            }
        };
        let arrived = usize::from(matches!(outcome.kind, StepKind::Arrival));

        if matches!(outcome.kind, StepKind::Iteration) {
            if let Some(iter) = outcome.iteration {
                for prog in &iter.progress {
                    // `num_output` is what the client sees: 1 when this step
                    // completed prefill (the first token comes out of the
                    // prefill pass), `1 + accepted` per decode step, 0 for a
                    // prefill chunk that did not finish the prompt.
                    if prog.num_output == 0 || !self.live_requests.contains_key(&prog.request_id) {
                        continue;
                    }

                    // First token marks the prefill→decode boundary.
                    let first = !self.live_requests[&prog.request_id].first_token_sent;
                    if first {
                        self.live_requests
                            .get_mut(&prog.request_id)
                            .expect("presence checked above")
                            .first_token_sent = true;
                        if !self.deliver(&prog.request_id, TokenEvent::FirstToken).await {
                            continue;
                        }
                    }

                    for _ in 0..prog.num_output {
                        let live = match self.live_requests.get_mut(&prog.request_id) {
                            Some(l) => l,
                            None => break,
                        };
                        let word = PLACEHOLDER_WORDS[live.word_index % PLACEHOLDER_WORDS.len()];
                        live.word_index += 1;
                        let event = TokenEvent::Token {
                            text: format!("{} ", word),
                        };
                        if !self.deliver(&prog.request_id, event).await {
                            break;
                        }
                    }
                }
            }
        }

        for done in outcome.completions {
            if let Some(live) = self.live_requests.remove(&done.request_id) {
                let _ = live
                    .tx
                    .send(TokenEvent::Done {
                        prompt_tokens: done.num_prompt_tokens,
                        completion_tokens: done.num_output_tokens,
                    })
                    .await;
            }
        }

        arrived
    }

    /// Send one event to a live request, waiting for room. Returns false (and forgets the
    /// request) once its client is gone, so a disconnected stream stops costing sends.
    async fn deliver(&mut self, request_id: &str, event: TokenEvent) -> bool {
        let Some(live) = self.live_requests.get(request_id) else {
            return false;
        };
        if live.tx.send(event).await.is_ok() {
            return true;
        }
        self.live_requests.remove(request_id);
        false
    }

    fn admit_request(&mut self, engine_req: EngineRequest) {
        let target_output_tokens = match &self.workload {
            Some(w) => w
                .output_len_dist
                .sample(&mut rand::rng())
                .min(engine_req.max_output_tokens),
            None => engine_req.max_output_tokens,
        };

        let now = self.engine.current_time();
        let request = Request::new_with_target(
            engine_req.request_id.clone(),
            0,
            now,
            engine_req.prompt_tokens,
            engine_req.max_output_tokens,
            target_output_tokens,
        );

        self.live_requests.insert(
            engine_req.request_id,
            LiveRequest {
                tx: engine_req.tx,
                word_index: 0,
                first_token_sent: false,
            },
        );

        self.engine.submit(request);
    }

    /// Convert a simulated time-since-epoch into a wall-clock Instant.
    fn sim_to_wall(&self, t_sim: f64) -> tokio::time::Instant {
        let epoch = self.epoch.expect("epoch set in run()");
        epoch + Duration::from_secs_f64(t_sim.max(0.0))
    }
}
