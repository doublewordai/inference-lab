//! Realtime OpenAI-compatible serve driver. Wraps the unified
//! [`crate::simulation::Engine`] with a tokio loop that paces sim-time to
//! wall-time and forwards per-iter token generation back to HTTP clients.

use std::collections::HashMap;
use tokio::sync::mpsc;
use tokio::time::Duration;

use crate::config::{Deployment, WorkloadConfig};
use crate::request::Request;
use crate::simulation::{Engine, StepKind, Topology};

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
}

impl RealtimeEngine {
    pub fn new(
        deployment: &Deployment,
        workload: Option<WorkloadConfig>,
        rx: mpsc::Receiver<EngineRequest>,
    ) -> Result<Self, String> {
        let topology = Topology::aggregated(
            deployment.cluster(),
            deployment.model.clone(),
            deployment.scheduler.clone(),
        )?;
        Ok(Self {
            engine: Engine::new(topology),
            workload,
            rx,
            live_requests: HashMap::new(),
            epoch: None,
        })
    }

    pub async fn run(mut self) {
        log::info!("RealtimeEngine started");
        self.epoch = Some(tokio::time::Instant::now());

        loop {
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
            //    a new HTTP request, whichever fires first. If nothing is in
            //    flight at all, just block on the receiver.
            let next_ev = self.engine.next_event_time();
            match next_ev {
                None => {
                    // Engine fully idle. Block until a request arrives or
                    // senders drop.
                    match self.rx.recv().await {
                        Some(req) => self.admit_request(req),
                        None => {
                            log::info!("RealtimeEngine shutting down: receiver closed");
                            return;
                        }
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
                        _ = tokio::time::sleep_until(wake) => {
                            self.advance_one_step().await;
                        }
                    }
                }
            }
        }
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
    async fn advance_one_step(&mut self) {
        let outcome = match self.engine.step() {
            Ok(o) => o,
            Err(e) => {
                log::error!("engine step failed: {e}");
                return;
            }
        };

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
