use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Json,
    },
};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::types::*;

pub struct AppState {
    pub engines: HashMap<String, mpsc::Sender<EngineRequest>>,
    pub model_names: Vec<String>,
    pub tokenizer: Option<Arc<tokenizers::Tokenizer>>,
}

pub async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status": "ok"}))
}

pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let data = state
        .model_names
        .iter()
        .map(|name| ModelEntry {
            id: name.clone(),
            object: "model",
            created: now,
            owned_by: "inference-lab",
        })
        .collect();

    Json(ModelList {
        object: "list",
        data,
    })
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let prompt_tokens = count_prompt_tokens(&state, &req.messages);
    let include_usage = req
        .stream_options
        .as_ref()
        .map(|options| options.include_usage)
        .unwrap_or(false);
    let (request_id, mut rx) = submit_engine_request(
        &state,
        &req.model,
        prompt_tokens,
        req.max_tokens,
        "chatcmpl",
    )
    .await?;

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
                usage: None,
            };
            let _ = stream_tx
                .send(Ok(
                    Event::default().data(serde_json::to_string(&initial_chunk).unwrap())
                ))
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
                            usage: None,
                        };
                        let _ = stream_tx
                            .send(Ok(
                                Event::default().data(serde_json::to_string(&chunk).unwrap())
                            ))
                            .await;
                    }
                    TokenEvent::Done {
                        prompt_tokens,
                        completion_tokens,
                    } => {
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
                            usage: include_usage.then_some(Usage {
                                prompt_tokens,
                                completion_tokens,
                                total_tokens: prompt_tokens + completion_tokens,
                            }),
                        };
                        let _ = stream_tx
                            .send(Ok(
                                Event::default().data(serde_json::to_string(&chunk).unwrap())
                            ))
                            .await;
                        let _ = stream_tx.send(Ok(Event::default().data("[DONE]"))).await;
                        break;
                    }
                    TokenEvent::Error { message } => {
                        let _ = stream_tx
                            .send(Ok(
                                Event::default().data(format!("{{\"error\": \"{}\"}}", message))
                            ))
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

pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let prompt_tokens = count_text_prompt_tokens(&state, &req.prompt);
    let include_usage = req
        .stream_options
        .as_ref()
        .map(|options| options.include_usage)
        .unwrap_or(false);
    let (request_id, mut rx) =
        submit_engine_request(&state, &req.model, prompt_tokens, req.max_tokens, "cmpl").await?;

    if req.stream {
        let model_name = req.model.clone();
        let id = request_id.clone();
        let (stream_tx, stream_rx) = mpsc::channel::<Result<Event, Infallible>>(64);

        tokio::spawn(async move {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            while let Some(event) = rx.recv().await {
                match event {
                    TokenEvent::FirstToken => {}
                    TokenEvent::Token { text } => {
                        let chunk = CompletionChunk {
                            id: id.clone(),
                            object: "text_completion",
                            created: now,
                            model: model_name.clone(),
                            choices: vec![CompletionChunkChoice {
                                text,
                                index: 0,
                                finish_reason: None,
                            }],
                            usage: None,
                        };
                        let _ = stream_tx
                            .send(Ok(
                                Event::default().data(serde_json::to_string(&chunk).unwrap())
                            ))
                            .await;
                    }
                    TokenEvent::Done {
                        prompt_tokens,
                        completion_tokens,
                    } => {
                        let chunk = CompletionChunk {
                            id: id.clone(),
                            object: "text_completion",
                            created: now,
                            model: model_name.clone(),
                            choices: vec![CompletionChunkChoice {
                                text: String::new(),
                                index: 0,
                                finish_reason: Some("stop"),
                            }],
                            usage: include_usage.then_some(Usage {
                                prompt_tokens,
                                completion_tokens,
                                total_tokens: prompt_tokens + completion_tokens,
                            }),
                        };
                        let _ = stream_tx
                            .send(Ok(
                                Event::default().data(serde_json::to_string(&chunk).unwrap())
                            ))
                            .await;
                        let _ = stream_tx.send(Ok(Event::default().data("[DONE]"))).await;
                        break;
                    }
                    TokenEvent::Error { message } => {
                        let _ = stream_tx
                            .send(Ok(
                                Event::default().data(format!("{{\"error\": \"{}\"}}", message))
                            ))
                            .await;
                        break;
                    }
                }
            }
        });

        Ok(Sse::new(ReceiverStream::new(stream_rx)).into_response())
    } else {
        let mut content = String::new();
        let mut completion_tokens = 0u32;
        let mut final_prompt_tokens = prompt_tokens;

        while let Some(event) = rx.recv().await {
            match event {
                TokenEvent::FirstToken => {}
                TokenEvent::Token { text } => content.push_str(&text),
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

        let response = CompletionResponse {
            id: request_id,
            object: "text_completion",
            created: now,
            model: req.model,
            choices: vec![CompletionChoice {
                text: content.trim_end().to_string(),
                index: 0,
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

async fn submit_engine_request(
    state: &AppState,
    model: &str,
    prompt_tokens: u32,
    max_output_tokens: u32,
    request_prefix: &str,
) -> Result<(String, mpsc::Receiver<TokenEvent>), (StatusCode, Json<serde_json::Value>)> {
    let engine_tx = state.engines.get(model).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "message": format!("Model '{}' not found. Available models: {}", model, state.model_names.join(", ")),
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            })),
        )
    })?;

    let (tx, rx) = mpsc::channel::<TokenEvent>(64);
    let request_id = format!("{}-{}", request_prefix, uuid::Uuid::new_v4());

    let engine_req = EngineRequest {
        request_id: request_id.clone(),
        prompt_tokens,
        max_output_tokens,
        tx,
    };

    engine_tx.send(engine_req).await.map_err(|_| {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "engine unavailable"})),
        )
    })?;

    Ok((request_id, rx))
}

fn count_prompt_tokens(state: &AppState, messages: &[ChatMessage]) -> u32 {
    let text = messages_to_prompt_text(messages);
    count_text_prompt_tokens(state, &text)
}

fn count_text_prompt_tokens(state: &AppState, prompt: &str) -> u32 {
    if let Some(ref tokenizer) = state.tokenizer {
        match tokenizer.encode(prompt, false) {
            Ok(encoding) => encoding.get_ids().len() as u32,
            Err(_) => estimate_tokens_from_chars(prompt),
        }
    } else {
        estimate_tokens_from_chars(prompt)
    }
}

fn messages_to_prompt_text(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n")
}

fn estimate_tokens_from_chars(text: &str) -> u32 {
    // Rough estimate: ~4 chars per token
    (text.len() as f64 / 4.0).ceil() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::response::Response;

    fn test_state(engine_tx: mpsc::Sender<EngineRequest>) -> Arc<AppState> {
        Arc::new(AppState {
            engines: HashMap::from([("test-model".to_string(), engine_tx)]),
            model_names: vec!["test-model".to_string()],
            tokenizer: None,
        })
    }

    async fn response_json(response: Response) -> serde_json::Value {
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    async fn response_sse_events(response: Response) -> Vec<String> {
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        String::from_utf8(body.to_vec())
            .unwrap()
            .split("\n\n")
            .filter_map(|chunk| chunk.strip_prefix("data: ").map(str::to_string))
            .collect()
    }

    #[tokio::test]
    async fn completions_returns_openai_style_response() {
        let (engine_tx, mut engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);

        tokio::spawn(async move {
            let engine_req = engine_rx.recv().await.unwrap();
            assert_eq!(engine_req.prompt_tokens, 3);
            assert_eq!(engine_req.max_output_tokens, 4);
            assert!(engine_req.request_id.starts_with("cmpl-"));

            let _ = engine_req.tx.send(TokenEvent::FirstToken).await;
            let _ = engine_req
                .tx
                .send(TokenEvent::Token {
                    text: "Hello".to_string(),
                })
                .await;
            let _ = engine_req
                .tx
                .send(TokenEvent::Token {
                    text: " world".to_string(),
                })
                .await;
            let _ = engine_req
                .tx
                .send(TokenEvent::Done {
                    prompt_tokens: 3,
                    completion_tokens: 2,
                })
                .await;
        });

        let response = completions(
            State(state),
            Json(CompletionRequest {
                model: "test-model".to_string(),
                prompt: "hello world".to_string(),
                stream: false,
                max_tokens: 4,
                stream_options: None,
            }),
        )
        .await
        .unwrap()
        .into_response();

        assert_eq!(response.status(), StatusCode::OK);

        let json = response_json(response).await;

        assert_eq!(json["object"], "text_completion");
        assert_eq!(json["model"], "test-model");
        assert_eq!(json["choices"][0]["text"], "Hello world");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["usage"]["prompt_tokens"], 3);
        assert_eq!(json["usage"]["completion_tokens"], 2);
        assert_eq!(json["usage"]["total_tokens"], 5);

        let id = json["id"].as_str().unwrap();
        assert!(id.starts_with("cmpl-"));
    }

    #[tokio::test]
    async fn chat_completions_returns_usage() {
        let (engine_tx, mut engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);

        tokio::spawn(async move {
            let engine_req = engine_rx.recv().await.unwrap();
            assert_eq!(engine_req.prompt_tokens, 5);
            assert!(engine_req.request_id.starts_with("chatcmpl-"));

            let _ = engine_req.tx.send(TokenEvent::FirstToken).await;
            let _ = engine_req
                .tx
                .send(TokenEvent::Token {
                    text: "Hello".to_string(),
                })
                .await;
            let _ = engine_req
                .tx
                .send(TokenEvent::Done {
                    prompt_tokens: 5,
                    completion_tokens: 1,
                })
                .await;
        });

        let response = chat_completions(
            State(state),
            Json(ChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![ChatMessage {
                    role: "user".to_string(),
                    content: "hello world".to_string(),
                }],
                stream: false,
                max_tokens: 4,
                stream_options: None,
            }),
        )
        .await
        .unwrap()
        .into_response();

        let json = response_json(response).await;

        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["message"]["content"], "Hello");
        assert_eq!(json["usage"]["prompt_tokens"], 5);
        assert_eq!(json["usage"]["completion_tokens"], 1);
        assert_eq!(json["usage"]["total_tokens"], 6);
    }

    #[tokio::test]
    async fn streaming_completions_include_usage_in_final_chunk() {
        let (engine_tx, mut engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);

        tokio::spawn(async move {
            let engine_req = engine_rx.recv().await.unwrap();

            let _ = engine_req.tx.send(TokenEvent::FirstToken).await;
            let _ = engine_req
                .tx
                .send(TokenEvent::Token {
                    text: "Hello".to_string(),
                })
                .await;
            let _ = engine_req
                .tx
                .send(TokenEvent::Done {
                    prompt_tokens: 3,
                    completion_tokens: 1,
                })
                .await;
        });

        let response = completions(
            State(state),
            Json(CompletionRequest {
                model: "test-model".to_string(),
                prompt: "hello world".to_string(),
                stream: true,
                max_tokens: 4,
                stream_options: Some(StreamOptions {
                    include_usage: true,
                }),
            }),
        )
        .await
        .unwrap()
        .into_response();

        let events = response_sse_events(response).await;
        let final_chunk: serde_json::Value = serde_json::from_str(
            events
                .iter()
                .rev()
                .find(|event| *event != "[DONE]")
                .unwrap(),
        )
        .unwrap();

        assert_eq!(final_chunk["choices"][0]["finish_reason"], "stop");
        assert_eq!(final_chunk["usage"]["prompt_tokens"], 3);
        assert_eq!(final_chunk["usage"]["completion_tokens"], 1);
        assert_eq!(final_chunk["usage"]["total_tokens"], 4);
    }

    #[tokio::test]
    async fn streaming_chat_completions_include_usage_in_final_chunk() {
        let (engine_tx, mut engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);

        tokio::spawn(async move {
            let engine_req = engine_rx.recv().await.unwrap();

            let _ = engine_req.tx.send(TokenEvent::FirstToken).await;
            let _ = engine_req
                .tx
                .send(TokenEvent::Token {
                    text: "Hello".to_string(),
                })
                .await;
            let _ = engine_req
                .tx
                .send(TokenEvent::Done {
                    prompt_tokens: 5,
                    completion_tokens: 1,
                })
                .await;
        });

        let response = chat_completions(
            State(state),
            Json(ChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![ChatMessage {
                    role: "user".to_string(),
                    content: "hello world".to_string(),
                }],
                stream: true,
                max_tokens: 4,
                stream_options: Some(StreamOptions {
                    include_usage: true,
                }),
            }),
        )
        .await
        .unwrap()
        .into_response();

        let events = response_sse_events(response).await;
        let final_chunk: serde_json::Value = serde_json::from_str(
            events
                .iter()
                .rev()
                .find(|event| *event != "[DONE]")
                .unwrap(),
        )
        .unwrap();

        assert_eq!(final_chunk["choices"][0]["finish_reason"], "stop");
        assert_eq!(final_chunk["usage"]["prompt_tokens"], 5);
        assert_eq!(final_chunk["usage"]["completion_tokens"], 1);
        assert_eq!(final_chunk["usage"]["total_tokens"], 6);
    }

    #[tokio::test]
    async fn streaming_completions_omit_usage_when_stream_options_missing() {
        let (engine_tx, mut engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);

        tokio::spawn(async move {
            let engine_req = engine_rx.recv().await.unwrap();
            let _ = engine_req.tx.send(TokenEvent::FirstToken).await;
            let _ = engine_req
                .tx
                .send(TokenEvent::Done {
                    prompt_tokens: 3,
                    completion_tokens: 1,
                })
                .await;
        });

        let response = completions(
            State(state),
            Json(CompletionRequest {
                model: "test-model".to_string(),
                prompt: "hello world".to_string(),
                stream: true,
                max_tokens: 4,
                stream_options: None,
            }),
        )
        .await
        .unwrap()
        .into_response();

        let events = response_sse_events(response).await;
        let final_chunk: serde_json::Value = serde_json::from_str(
            events
                .iter()
                .rev()
                .find(|event| *event != "[DONE]")
                .unwrap(),
        )
        .unwrap();

        assert_eq!(final_chunk["choices"][0]["finish_reason"], "stop");
        assert!(final_chunk.get("usage").is_none());
    }

    #[tokio::test]
    async fn streaming_completions_omit_usage_when_include_usage_false() {
        let (engine_tx, mut engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);

        tokio::spawn(async move {
            let engine_req = engine_rx.recv().await.unwrap();
            let _ = engine_req.tx.send(TokenEvent::FirstToken).await;
            let _ = engine_req
                .tx
                .send(TokenEvent::Done {
                    prompt_tokens: 3,
                    completion_tokens: 1,
                })
                .await;
        });

        let response = completions(
            State(state),
            Json(CompletionRequest {
                model: "test-model".to_string(),
                prompt: "hello world".to_string(),
                stream: true,
                max_tokens: 4,
                stream_options: Some(StreamOptions {
                    include_usage: false,
                }),
            }),
        )
        .await
        .unwrap()
        .into_response();

        let events = response_sse_events(response).await;
        let final_chunk: serde_json::Value = serde_json::from_str(
            events
                .iter()
                .rev()
                .find(|event| *event != "[DONE]")
                .unwrap(),
        )
        .unwrap();

        assert_eq!(final_chunk["choices"][0]["finish_reason"], "stop");
        assert!(final_chunk.get("usage").is_none());
    }

    #[tokio::test]
    async fn streaming_chat_completions_omit_usage_when_stream_options_missing() {
        let (engine_tx, mut engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);

        tokio::spawn(async move {
            let engine_req = engine_rx.recv().await.unwrap();
            let _ = engine_req.tx.send(TokenEvent::FirstToken).await;
            let _ = engine_req
                .tx
                .send(TokenEvent::Done {
                    prompt_tokens: 5,
                    completion_tokens: 1,
                })
                .await;
        });

        let response = chat_completions(
            State(state),
            Json(ChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![ChatMessage {
                    role: "user".to_string(),
                    content: "hello world".to_string(),
                }],
                stream: true,
                max_tokens: 4,
                stream_options: None,
            }),
        )
        .await
        .unwrap()
        .into_response();

        let events = response_sse_events(response).await;
        let final_chunk: serde_json::Value = serde_json::from_str(
            events
                .iter()
                .rev()
                .find(|event| *event != "[DONE]")
                .unwrap(),
        )
        .unwrap();

        assert_eq!(final_chunk["choices"][0]["finish_reason"], "stop");
        assert!(final_chunk.get("usage").is_none());
    }

    #[tokio::test]
    async fn streaming_chat_completions_omit_usage_when_include_usage_false() {
        let (engine_tx, mut engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);

        tokio::spawn(async move {
            let engine_req = engine_rx.recv().await.unwrap();
            let _ = engine_req.tx.send(TokenEvent::FirstToken).await;
            let _ = engine_req
                .tx
                .send(TokenEvent::Done {
                    prompt_tokens: 5,
                    completion_tokens: 1,
                })
                .await;
        });

        let response = chat_completions(
            State(state),
            Json(ChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![ChatMessage {
                    role: "user".to_string(),
                    content: "hello world".to_string(),
                }],
                stream: true,
                max_tokens: 4,
                stream_options: Some(StreamOptions {
                    include_usage: false,
                }),
            }),
        )
        .await
        .unwrap()
        .into_response();

        let events = response_sse_events(response).await;
        let final_chunk: serde_json::Value = serde_json::from_str(
            events
                .iter()
                .rev()
                .find(|event| *event != "[DONE]")
                .unwrap(),
        )
        .unwrap();

        assert_eq!(final_chunk["choices"][0]["finish_reason"], "stop");
        assert!(final_chunk.get("usage").is_none());
    }

    #[tokio::test]
    async fn completions_returns_model_not_found_error() {
        let state = Arc::new(AppState {
            engines: HashMap::new(),
            model_names: vec!["other-model".to_string()],
            tokenizer: None,
        });

        let error = match completions(
            State(state),
            Json(CompletionRequest {
                model: "missing-model".to_string(),
                prompt: "hello".to_string(),
                stream: false,
                max_tokens: 4,
                stream_options: None,
            }),
        )
        .await
        {
            Ok(_) => panic!("expected missing-model request to fail"),
            Err(error) => error,
        };

        assert_eq!(error.0, StatusCode::NOT_FOUND);
        let body = serde_json::to_value(error.1 .0).unwrap();
        assert_eq!(body["error"]["code"], "model_not_found");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("missing-model"));
    }

    #[test]
    fn prompt_token_estimation_supports_raw_text() {
        let state = AppState {
            engines: HashMap::new(),
            model_names: Vec::new(),
            tokenizer: None,
        };

        assert_eq!(count_text_prompt_tokens(&state, "hello world"), 3);
    }
}
