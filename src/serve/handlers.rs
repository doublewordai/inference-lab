use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Json,
    },
};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::types::*;

pub struct AppState {
    pub engine_tx: mpsc::Sender<EngineRequest>,
    pub model_name: String,
    pub tokenizer: Option<tokenizers::Tokenizer>,
}

pub async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    Json(ModelList {
        object: "list",
        data: vec![ModelEntry {
            id: state.model_name.clone(),
            object: "model",
            created: now,
            owned_by: "inference-lab",
        }],
    })
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    // Tokenize the messages to get prompt token count
    let prompt_tokens = count_prompt_tokens(&state, &req.messages);

    // Create per-request channel
    let (tx, mut rx) = mpsc::channel::<TokenEvent>(64);

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    let engine_req = EngineRequest {
        request_id: request_id.clone(),
        prompt_tokens,
        max_output_tokens: req.max_tokens,
        tx,
    };

    // Send to engine
    state.engine_tx.send(engine_req).await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "engine unavailable"})),
        )
    })?;

    if req.stream {
        // Streaming response
        let model_name = req.model.clone();
        let id = request_id.clone();

        let (stream_tx, stream_rx) = mpsc::channel::<Result<Event, Infallible>>(64);

        tokio::spawn(async move {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            // Send initial chunk with role
            let initial_chunk = ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk",
                created: now,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: Some("assistant"),
                        content: None,
                    },
                    finish_reason: None,
                }],
            };
            let _ = stream_tx
                .send(Ok(Event::default()
                    .data(serde_json::to_string(&initial_chunk).unwrap())))
                .await;

            // Stream tokens
            while let Some(event) = rx.recv().await {
                match event {
                    TokenEvent::FirstToken => {
                        // No output needed; first content token follows
                    }
                    TokenEvent::Token { text } => {
                        let chunk = ChatCompletionChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk",
                            created: now,
                            model: model_name.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: Some(text),
                                },
                                finish_reason: None,
                            }],
                        };
                        let _ = stream_tx
                            .send(Ok(Event::default()
                                .data(serde_json::to_string(&chunk).unwrap())))
                            .await;
                    }
                    TokenEvent::Done { .. } => {
                        let chunk = ChatCompletionChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk",
                            created: now,
                            model: model_name.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta {
                                    role: None,
                                    content: None,
                                },
                                finish_reason: Some("stop"),
                            }],
                        };
                        let _ = stream_tx
                            .send(Ok(Event::default()
                                .data(serde_json::to_string(&chunk).unwrap())))
                            .await;
                        let _ = stream_tx
                            .send(Ok(Event::default().data("[DONE]")))
                            .await;
                        break;
                    }
                    TokenEvent::Error { message } => {
                        let _ = stream_tx
                            .send(Ok(Event::default()
                                .data(format!("{{\"error\": \"{}\"}}", message))))
                            .await;
                        break;
                    }
                }
            }
        });

        let stream = ReceiverStream::new(stream_rx);
        Ok(Sse::new(stream).into_response())
    } else {
        // Non-streaming: collect all tokens
        let mut content = String::new();
        let mut completion_tokens = 0u32;
        let mut final_prompt_tokens = prompt_tokens;

        while let Some(event) = rx.recv().await {
            match event {
                TokenEvent::FirstToken => {}
                TokenEvent::Token { text } => {
                    content.push_str(&text);
                }
                TokenEvent::Done {
                    prompt_tokens: pt,
                    completion_tokens: ct,
                } => {
                    final_prompt_tokens = pt;
                    completion_tokens = ct;
                    break;
                }
                TokenEvent::Error { message } => {
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": message})),
                    ));
                }
            }
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion",
            created: now,
            model: req.model,
            choices: vec![Choice {
                index: 0,
                message: ChoiceMessage {
                    role: "assistant",
                    content: content.trim_end().to_string(),
                },
                finish_reason: "stop",
            }],
            usage: Usage {
                prompt_tokens: final_prompt_tokens,
                completion_tokens,
                total_tokens: final_prompt_tokens + completion_tokens,
            },
        };

        Ok(Json(response).into_response())
    }
}

fn count_prompt_tokens(state: &AppState, messages: &[ChatMessage]) -> u32 {
    if let Some(ref tokenizer) = state.tokenizer {
        // Concatenate all message content for tokenization
        let text: String = messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        match tokenizer.encode(text, false) {
            Ok(encoding) => encoding.get_ids().len() as u32,
            Err(_) => estimate_tokens_from_chars(messages),
        }
    } else {
        estimate_tokens_from_chars(messages)
    }
}

fn estimate_tokens_from_chars(messages: &[ChatMessage]) -> u32 {
    // Rough estimate: ~4 chars per token
    let total_chars: usize = messages.iter().map(|m| m.content.len() + m.role.len() + 2).sum();
    (total_chars as f64 / 4.0).ceil() as u32
}
