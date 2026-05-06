//! Realtime OpenAI-compatible serve driver. Wraps the unified
//! [`crate::simulation::Engine`] with a tokio loop that paces sim-time to
//! wall-time and forwards per-iter token generation back to HTTP clients.

use std::collections::HashMap;
use tokio::sync::mpsc;
use tokio::time::Duration;

use crate::config::{ClusterSpec, Config};
use crate::request::Request;
use crate::simulation::{Engine, StepKind, Topology};

use super::types::{EngineRequest, TokenEvent};

const PLACEHOLDER_WORDS: &[&str] = &[
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
    config: Config,
    rx: mpsc::Receiver<EngineRequest>,
    live_requests: HashMap<String, LiveRequest>,
    /// Wall-clock-anchored offset from sim-time to real-time. Set on the
    /// first event we process. We use it to translate engine event-times
    /// back into `tokio::time::Instant`s for `sleep_until`.
    epoch: Option<tokio::time::Instant>,
}

impl RealtimeEngine {
    pub fn new(config: Config, rx: mpsc::Receiver<EngineRequest>) -> Result<Self, String> {
        let cluster = ClusterSpec {
            hardware: config.hardware.clone(),
            parallel: config.parallel.clone(),
            comms: None,
            num_workers: 1,
            node: 0,
        };
        let topology =
            Topology::aggregated(cluster, config.model.clone(), config.scheduler.clone())?;
        Ok(Self {
            engine: Engine::new(topology),
            config,
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
                            self.advance_one_step();
                        }
                    }
                }
            }
        }
    }

    fn advance_one_step(&mut self) {
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
                    let live = match self.live_requests.get_mut(&prog.request_id) {
                        Some(l) => l,
                        None => continue,
                    };

                    // First token marks prefill→decode boundary.
                    if !live.first_token_sent && !prog.was_prefill {
                        live.first_token_sent = true;
                        let _ = live.tx.try_send(TokenEvent::FirstToken);
                    }

                    // Emit decode tokens only. Prefill iterations advance
                    // num_computed_tokens but don't yield user-visible text.
                    if !prog.was_prefill {
                        for _ in 0..prog.num_tokens {
                            let word =
                                PLACEHOLDER_WORDS[live.word_index % PLACEHOLDER_WORDS.len()];
                            live.word_index += 1;
                            let _ = live.tx.try_send(TokenEvent::Token {
                                text: format!("{} ", word),
                            });
                        }
                    }
                }
            }
        }

        for done in outcome.completions {
            if let Some(live) = self.live_requests.remove(&done.request_id) {
                let _ = live.tx.try_send(TokenEvent::Done {
                    prompt_tokens: done.num_prompt_tokens,
                    completion_tokens: done.num_output_tokens,
                });
            }
        }
    }

    fn admit_request(&mut self, engine_req: EngineRequest) {
        let mut rng = rand::thread_rng();
        let target_output_tokens = self
            .config
            .workload
            .output_len_dist
            .sample(&mut rng)
            .min(engine_req.max_output_tokens);

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

