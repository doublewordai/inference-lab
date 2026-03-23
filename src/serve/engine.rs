use std::collections::HashMap;
use tokio::sync::mpsc;

use crate::config::Config;
use crate::compute::ComputeEngine;
use crate::kv_cache::KVCacheManager;
use crate::request::Request;
use crate::scheduler::Scheduler;

use super::types::{EngineRequest, TokenEvent};

const PLACEHOLDER_WORDS: &[&str] = &[
    "the", "of", "and", "to", "in", "a", "is", "that", "for", "it",
    "was", "on", "are", "be", "with", "as", "at", "this", "have", "from",
    "or", "an", "by", "not", "but", "what", "all", "were", "when", "we",
    "there", "can", "which", "their", "if", "do", "will", "each", "about", "how",
    "up", "out", "them", "then", "she", "many", "some", "so", "these", "would",
];

struct LiveRequest {
    tx: mpsc::Sender<TokenEvent>,
    word_index: usize,
    first_token_sent: bool,
}

pub struct RealtimeEngine {
    scheduler: Scheduler,
    compute_engine: ComputeEngine,
    config: Config,
    rx: mpsc::Receiver<EngineRequest>,
    live_requests: HashMap<String, LiveRequest>,
    current_time: f64,
}

impl RealtimeEngine {
    pub fn new(config: Config, rx: mpsc::Receiver<EngineRequest>) -> Result<Self, String> {
        let kv_cache_manager = KVCacheManager::new(
            config.hardware.kv_cache_capacity,
            config.scheduler.block_size,
            config.model.kv_cache_bytes_per_token,
            false, // no prefix caching for serve mode
        );

        let scheduler = Scheduler::new(
            config.scheduler.clone(),
            config.hardware.clone(),
            config.model.clone(),
            kv_cache_manager,
        )?;

        let compute_engine = ComputeEngine::new(
            config.hardware.clone(),
            config.model.clone(),
        );

        Ok(Self {
            scheduler,
            compute_engine,
            config,
            rx,
            live_requests: HashMap::new(),
            current_time: 0.0,
        })
    }

    pub async fn run(mut self) {
        log::info!("RealtimeEngine started");

        loop {
            // 1. Drain new requests from channel (non-blocking)
            loop {
                match self.rx.try_recv() {
                    Ok(engine_req) => {
                        self.admit_request(engine_req);
                    }
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        // All senders dropped - shut down if no live requests
                        if self.live_requests.is_empty() {
                            log::info!("RealtimeEngine shutting down: no senders, no live requests");
                            return;
                        }
                        break;
                    }
                }
            }

            // 2. If idle (nothing running/waiting), sleep briefly and continue
            if self.scheduler.num_running() == 0 && self.scheduler.num_waiting() == 0 {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                continue;
            }

            // 3. Schedule
            let decision = self.scheduler.schedule(self.current_time);

            // 4. Handle completed requests (before batch check, since completed
            //    requests cause num_scheduled==0)
            for completed in &decision.completed {
                let request_id = &completed.request_id;
                if let Some(live) = self.live_requests.remove(request_id) {
                    let _ = live.tx.try_send(TokenEvent::Done {
                        prompt_tokens: completed.num_prompt_tokens,
                        completion_tokens: completed.num_output_tokens,
                    });
                }
            }

            // 5. Build batch and compute iteration time
            let batch_size = decision.num_scheduled();
            if batch_size == 0 {
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                continue;
            }

            // Collect references to scheduled requests and their token counts
            let running = self.scheduler.running();
            let mut batch_requests: Vec<&Request> = Vec::new();
            let mut tokens_per_request: Vec<u32> = Vec::new();

            for (i, &idx) in decision.scheduled_new.iter().enumerate() {
                batch_requests.push(&running[idx]);
                tokens_per_request.push(decision.tokens_for_new[i]);
            }
            for (i, &idx) in decision.scheduled_running.iter().enumerate() {
                batch_requests.push(&running[idx]);
                tokens_per_request.push(decision.tokens_for_running[i]);
            }

            let iteration_time = self.compute_engine.calculate_iteration_time(
                &batch_requests,
                &tokens_per_request,
            );

            // 5. Sleep for real-time pacing
            tokio::time::sleep(std::time::Duration::from_secs_f64(iteration_time)).await;

            // 6. Advance simulated time
            self.current_time += iteration_time;

            // 7. Update request states and emit tokens
            // Collect token updates to apply
            let mut token_updates: Vec<(usize, u32)> = Vec::new();
            for (i, &idx) in decision.scheduled_new.iter().enumerate() {
                token_updates.push((idx, decision.tokens_for_new[i]));
            }
            for (i, &idx) in decision.scheduled_running.iter().enumerate() {
                token_updates.push((idx, decision.tokens_for_running[i]));
            }

            // Apply token generation and emit events
            for (idx, num_tokens) in &token_updates {
                let request = &mut self.scheduler.running_mut()[*idx];
                let was_prefill = request.is_prefill();
                request.record_generated_tokens(*num_tokens, self.current_time);

                let request_id = request.request_id.clone();

                if let Some(live) = self.live_requests.get_mut(&request_id) {
                    // Detect first token (prefill → decode transition)
                    if !live.first_token_sent && request.first_token_time.is_some() {
                        live.first_token_sent = true;
                        let _ = live.tx.try_send(TokenEvent::FirstToken);
                    }

                    // Emit tokens only for decode iterations (scheduler never
                    // crosses prefill/decode boundary in a single iteration)
                    if !was_prefill {
                        for _ in 0..*num_tokens {
                            let word = PLACEHOLDER_WORDS[live.word_index % PLACEHOLDER_WORDS.len()];
                            live.word_index += 1;
                            let _ = live.tx.try_send(TokenEvent::Token {
                                text: format!("{} ", word),
                            });
                        }
                    }
                }
            }

        }
    }

    fn admit_request(&mut self, engine_req: EngineRequest) {
        // Sample target output tokens from config distribution
        let mut rng = rand::thread_rng();
        let target_output_tokens = self
            .config
            .workload
            .output_len_dist
            .sample(&mut rng)
            .min(engine_req.max_output_tokens);

        let request = Request::new_with_target(
            engine_req.request_id.clone(),
            0, // priority
            self.current_time,
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

        self.scheduler.add_request(request);
    }
}
