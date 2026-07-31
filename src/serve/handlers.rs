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
    /// Honor echo-directives (serve::directive). Explicitly opt-in: a scripted-response
    /// bypass reachable by untrusted clients would be a response-spoofing vector.
    pub enable_directives: bool,
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
    let prompt_tokens = count_prompt_tokens(&state, &req.messages, req.tools.as_ref());
    let include_usage = req
        .stream_options
        .as_ref()
        .map(|options| options.include_usage)
        .unwrap_or(false);

    // Echo-directive mode (see serve::directive): a scripted response bypasses the engine —
    // deterministic content/tool_calls, immediate return. The model must still exist so an
    // unknown-model request fails the same way as the normal path. Gated on the
    // --enable-directives opt-in; when off, directive text is inert prompt content.
    if state.enable_directives {
        if let Some(directive) = super::directive::find_directive(&req.messages) {
            if !state.engines.contains_key(&req.model) {
                return Err(model_not_found(&state, &req.model));
            }
            let completion_tokens = count_text_prompt_tokens(&state, &directive.completion_text());
            return Ok(scripted_chat_response(
                &req,
                &directive,
                prompt_tokens,
                completion_tokens,
            ));
        }
    }

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
                        tool_calls: None,
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
                                    tool_calls: None,
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
                                    tool_calls: None,
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
                    content: Some(content.trim_end().to_string()),
                    tool_calls: None,
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

fn model_not_found(state: &AppState, model: &str) -> (StatusCode, Json<serde_json::Value>) {
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
}

/// Build the scripted (echo-directive) response — non-streaming JSON, or the same content
/// as a minimal, well-formed SSE sequence (role chunk -> content/tool_call chunks ->
/// finish+usage chunk -> [DONE]) when the request asked to stream.
fn scripted_chat_response(
    req: &ChatCompletionRequest,
    directive: &super::directive::Directive,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> axum::response::Response {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let usage = Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    };
    let tool_calls = directive.response_tool_calls();
    let finish_reason = directive.finish_reason();
    // Pure tool-call turns carry content: null, like a real model server.
    let content = match directive.text.as_deref() {
        Some(t) if !t.is_empty() => Some(t.to_string()),
        _ if tool_calls.is_empty() => Some(String::new()),
        _ => None,
    };

    if !req.stream {
        let response = ChatCompletionResponse {
            id,
            object: "chat.completion",
            created: now,
            model: req.model.clone(),
            choices: vec![Choice {
                index: 0,
                message: ChoiceMessage {
                    role: "assistant",
                    content,
                    tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                },
                finish_reason,
            }],
            usage,
        };
        return Json(response).into_response();
    }

    let chunk = |delta: ChunkDelta, finish: Option<&'static str>, usage: Option<Usage>| {
        ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created: now,
            model: req.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta,
                finish_reason: finish,
            }],
            usage,
        }
    };
    let mut chunks = vec![chunk(
        ChunkDelta {
            role: Some("assistant"),
            content: None,
            tool_calls: None,
        },
        None,
        None,
    )];
    if let Some(text) = content.filter(|c| !c.is_empty()) {
        chunks.push(chunk(
            ChunkDelta {
                role: None,
                content: Some(text),
                tool_calls: None,
            },
            None,
            None,
        ));
    }
    if !tool_calls.is_empty() {
        let stream_calls = tool_calls
            .into_iter()
            .enumerate()
            .map(|(i, tc)| StreamToolCall {
                index: i as u32,
                id: tc.id,
                r#type: tc.r#type,
                function: tc.function,
            })
            .collect();
        chunks.push(chunk(
            ChunkDelta {
                role: None,
                content: None,
                tool_calls: Some(stream_calls),
            },
            None,
            None,
        ));
    }
    // Scripted turns exist for usage assertions, so the final chunk ALWAYS carries usage,
    // even without stream_options.include_usage — matching the many real providers
    // (OpenRouter, vllm-with-flag, ...) that emit it unconditionally. The engine path keeps
    // strict spec behavior; this deliberate divergence surfaced a real gateway gap
    // (anthropic-ingress streaming not injecting include_usage) that spec-strict fakes hide.
    chunks.push(chunk(
        ChunkDelta {
            role: None,
            content: None,
            tool_calls: None,
        },
        Some(finish_reason),
        Some(usage),
    ));

    let mut events: Vec<Result<Event, Infallible>> = chunks
        .into_iter()
        .map(|c| Ok(Event::default().data(serde_json::to_string(&c).unwrap())))
        .collect();
    events.push(Ok(Event::default().data("[DONE]")));
    Sse::new(tokio_stream::iter(events)).into_response()
}

async fn submit_engine_request(
    state: &AppState,
    model: &str,
    prompt_tokens: u32,
    max_output_tokens: u32,
    request_prefix: &str,
) -> Result<(String, mpsc::Receiver<TokenEvent>), (StatusCode, Json<serde_json::Value>)> {
    let engine_tx = state
        .engines
        .get(model)
        .ok_or_else(|| model_not_found(state, model))?;

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

fn count_prompt_tokens(
    state: &AppState,
    messages: &[ChatMessage],
    tools: Option<&serde_json::Value>,
) -> u32 {
    let text = prompt_text(messages, tools);
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

/// The text whose token count stands in for the engine's prompt_tokens. Real chat
/// templates render tool DEFINITIONS and assistant tool CALLS into the prompt, so both
/// count — as their JSON serialization, which is also what prefix-metering gateways
/// (dwctl's cache classifier) tokenize. Leaving them out made the fake's count fall
/// BELOW such gateways' on tool-heavy bodies, tripping their split-vs-prompt guards
/// (the 2026-07-30 cache-parity tool_exchange failures).
fn prompt_text(messages: &[ChatMessage], tools: Option<&serde_json::Value>) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(messages.len() + 1);
    if let Some(tools) = tools {
        parts.push(format!("tools: {}", tools));
    }
    for m in messages {
        // Null/omitted content (assistant tool-call turns) contributes an empty body.
        let content = m.content.as_ref().map(|c| c.text()).unwrap_or_default();
        match &m.tool_calls {
            Some(calls) => {
                // Separate non-empty content from the tool-call JSON so the two tokenize
                // as distinct chunks (direct concatenation could merge tokens across the
                // boundary and makes the counted text ambiguous).
                let sep = if content.is_empty() { "" } else { "\n" };
                parts.push(format!("{}: {}{}{}", m.role, content, sep, calls));
            }
            None => parts.push(format!("{}: {}", m.role, content)),
        }
    }
    parts.join("\n")
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
        // Directives ON: the directive tests exercise the scripted path; engine-path tests
        // send no directive text, so the flag is inert for them.
        Arc::new(AppState {
            engines: HashMap::from([("test-model".to_string(), engine_tx)]),
            model_names: vec!["test-model".to_string()],
            tokenizer: None,
            enable_directives: true,
        })
    }

    fn test_state_directives_off(engine_tx: mpsc::Sender<EngineRequest>) -> Arc<AppState> {
        Arc::new(AppState {
            engines: HashMap::from([("test-model".to_string(), engine_tx)]),
            model_names: vec!["test-model".to_string()],
            tokenizer: None,
            enable_directives: false,
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

    #[test]
    fn content_accepts_string_array_and_null_shapes() {
        // Real OpenAI-compatible servers accept all three content shapes; gateways in front
        // of this server emit array-form content (multimodal parts, single-text-part arrays)
        // and `content: null` on assistant tool-call turns. Extra fields on a part (e.g. a
        // gateway's cache_control that slipped through) must be tolerated, and non-text parts
        // contribute nothing to the prompt.
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "system", "content": "plain"},
                {"role": "user", "content": [
                    {"type": "text", "text": "part one, ", "cache_control": {"type": "ephemeral"}},
                    {"type": "image_url", "image_url": {"url": "http://example/img.png"}},
                    {"type": "text", "text": "part two"}
                ]},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ]}
            ]
        }))
        .expect("string, array, and null content all deserialize");

        let text = prompt_text(&req.messages, None);
        // The assistant tool-call turn contributes its JSON serialization: real chat
        // templates render tool calls into the prompt, and prefix-metering gateways
        // (dwctl's cache classifier) tokenize the same JSON — leaving it out made the
        // fake's prompt_tokens undercount tool-heavy bodies.
        assert_eq!(
            text,
            "system: plain\nuser: part one, part two\nassistant: \
             [{\"function\":{\"arguments\":\"{}\",\"name\":\"f\"},\"id\":\"t1\",\"type\":\"function\"}]"
        );
    }

    #[test]
    fn tool_calls_with_content_include_separator() {
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [{
                "role": "assistant",
                "content": "thinking",
                "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
            }]
        }))
        .unwrap();
        let text = prompt_text(&req.messages, None);
        assert!(text.contains("assistant: thinking\n["), "got: {text}");
    }

    #[test]
    fn tools_definitions_count_toward_prompt_text() {
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        let text = prompt_text(&req.messages, req.tools.as_ref());
        assert!(
            text.starts_with("tools: "),
            "tool definitions lead the counted prompt"
        );
        assert!(
            text.contains("\"name\":\"f\""),
            "definition JSON is included"
        );
        assert!(text.ends_with("user: hi"));
    }

    #[test]
    fn single_text_part_array_counts_like_plain_string() {
        // A single-text-part array must produce byte-identical prompt text to the plain
        // string form, so token counts (and anything keyed on them) agree across the two
        // encodings of the same message.
        let plain = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text("hello world".to_string())),
            tool_calls: None,
        }];
        let array: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello world"}]}]
        }))
        .unwrap();

        assert_eq!(
            prompt_text(&plain, None),
            prompt_text(&array.messages, None)
        );
    }

    #[tokio::test]
    async fn directive_returns_scripted_tool_calls_without_engine() {
        // Channel with NO consumer: if the directive path touched the engine, the handler
        // would hang or error — completing at all proves the bypass.
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);

        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "please look this up <<respond:{\"tool_calls\":[{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/w/step1.txt\"}}]}>>"}
            ]
        }))
        .unwrap();
        let response = chat_completions(State(state), Json(req))
            .await
            .unwrap()
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
        let json = response_json(response).await;

        let choice = &json["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls");
        assert!(choice["message"]["content"].is_null());
        let tc = &choice["message"]["tool_calls"][0];
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "Read");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(
                tc["function"]["arguments"].as_str().unwrap()
            )
            .unwrap()["file_path"],
            "/w/step1.txt"
        );
        assert!(json["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
        assert!(json["usage"]["completion_tokens"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn directive_text_response_and_last_directive_wins() {
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);

        // The tool-result message carries the freshest directive — it must win over the
        // task prompt's original one (the chained agent-loop flow).
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "task <<respond:{\"tool_calls\":[{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/w/s1\"}}]}>>"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "t1", "type": "function", "function": {"name": "Read", "arguments": "{\"file_path\":\"/w/s1\"}"}}
                ]},
                {"role": "tool", "content": "file body <<respond:{\"text\":\"all done\"}>>"}
            ]
        }))
        .unwrap();
        let response = chat_completions(State(state), Json(req))
            .await
            .unwrap()
            .into_response();
        let json = response_json(response).await;
        let choice = &json["choices"][0];
        assert_eq!(choice["finish_reason"], "stop");
        assert_eq!(choice["message"]["content"], "all done");
        assert!(choice["message"].get("tool_calls").is_none());
    }

    #[tokio::test]
    async fn directive_streams_scripted_sequence_with_usage() {
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);

        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "stream": true,
            "stream_options": {"include_usage": true},
            "messages": [
                {"role": "user", "content": "<<respond:{\"text\":\"hi\",\"tool_calls\":[{\"name\":\"Read\",\"arguments\":{\"file_path\":\"/x\"}}]}>>"}
            ]
        }))
        .unwrap();
        let response = chat_completions(State(state), Json(req))
            .await
            .unwrap()
            .into_response();
        assert_eq!(response.status(), StatusCode::OK);
        let events = response_sse_events(response).await;
        assert_eq!(events.last().map(String::as_str), Some("[DONE]"));

        let frames: Vec<serde_json::Value> = events
            .iter()
            .filter(|e| *e != "[DONE]")
            .map(|e| serde_json::from_str(e).unwrap())
            .collect();
        // role -> content -> tool_calls -> finish+usage
        assert_eq!(frames[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(frames[1]["choices"][0]["delta"]["content"], "hi");
        let tc = &frames[2]["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tc["index"], 0);
        assert_eq!(tc["function"]["name"], "Read");
        let last = frames.last().unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
        assert!(last["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn directive_stream_carries_usage_without_stream_options() {
        // Real harnesses (the claude CLI) don't set stream_options.include_usage; scripted
        // turns must still return usage or every session asserts on zeros.
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "stream": true,
            "messages": [{"role": "user", "content": "<<respond:{\"text\":\"hi\"}>>"}]
        }))
        .unwrap();
        let response = chat_completions(State(state), Json(req))
            .await
            .unwrap()
            .into_response();
        let events = response_sse_events(response).await;
        let frames: Vec<serde_json::Value> = events
            .iter()
            .filter(|e| *e != "[DONE]")
            .map(|e| serde_json::from_str(e).unwrap())
            .collect();
        let last = frames.last().unwrap();
        assert!(last["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn directive_inert_when_disabled() {
        // With the flag off, directive text is ordinary prompt content: the request takes
        // the ENGINE path (proven by answering the engine request it submits).
        let (engine_tx, mut engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state_directives_off(engine_tx);

        let answer = tokio::spawn(async move {
            let engine_req = engine_rx.recv().await.expect("engine path must be taken");
            let _ = engine_req
                .tx
                .send(TokenEvent::Token {
                    text: "plain".into(),
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

        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "<<respond:{\"text\":\"spoof\"}>>"}]
        }))
        .unwrap();
        let response = chat_completions(State(state), Json(req))
            .await
            .unwrap()
            .into_response();
        let json = response_json(response).await;
        answer.await.unwrap();
        assert_eq!(json["choices"][0]["message"]["content"], "plain");
        assert!(json["choices"][0]["message"].get("tool_calls").is_none());
    }

    #[tokio::test]
    async fn directive_unknown_model_is_404() {
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "nope",
            "messages": [{"role": "user", "content": "<<respond:{\"text\":\"x\"}>>"}]
        }))
        .unwrap();
        let err = chat_completions(State(state), Json(req))
            .await
            .err()
            .unwrap();
        assert_eq!(err.0, StatusCode::NOT_FOUND);
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
                    content: Some(MessageContent::Text("hello world".to_string())),
                    tool_calls: None,
                }],
                stream: false,
                max_tokens: 4,
                stream_options: None,
                tools: None,
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
                    content: Some(MessageContent::Text("hello world".to_string())),
                    tool_calls: None,
                }],
                stream: true,
                max_tokens: 4,
                stream_options: Some(StreamOptions {
                    include_usage: true,
                }),
                tools: None,
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
                    content: Some(MessageContent::Text("hello world".to_string())),
                    tool_calls: None,
                }],
                stream: true,
                max_tokens: 4,
                stream_options: None,
                tools: None,
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
                    content: Some(MessageContent::Text("hello world".to_string())),
                    tool_calls: None,
                }],
                stream: true,
                max_tokens: 4,
                stream_options: Some(StreamOptions {
                    include_usage: false,
                }),
                tools: None,
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
            enable_directives: false,
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
            enable_directives: false,
        };

        assert_eq!(count_text_prompt_tokens(&state, "hello world"), 3);
    }
}
