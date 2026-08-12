use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, Sse},
        IntoResponse, Json,
    },
    Extension,
};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::fault;
use super::types::*;

pub struct AppState {
    pub engines: HashMap<String, mpsc::Sender<EngineRequest>>,
    pub model_names: Vec<String>,
    pub tokenizer: Option<Arc<tokenizers::Tokenizer>>,
    /// Honor echo-directives (serve::directive). Explicitly opt-in: a scripted-response
    /// bypass reachable by untrusted clients would be a response-spoofing vector.
    pub enable_directives: bool,
    /// Static per-model fault injection (serve::fault), keyed by model name. The
    /// fallback trigger for clients that can't set the fault header; validated at
    /// server startup.
    pub model_faults: HashMap<String, fault::FaultSpec>,
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
    connection: Option<Extension<Arc<fault::FaultConnection>>>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let prompt_tokens = count_prompt_tokens(&state, &req.messages, req.tools.as_ref());
    let include_usage = req
        .stream_options
        .as_ref()
        .map(|options| options.include_usage)
        .unwrap_or(false);

    // Fault injection (serve::fault): per-request header first, per-model static config
    // as the fallback. Takes precedence over echo-directives — a faulting stream is
    // about wire behavior, not content. With neither trigger present this block is
    // inert and the request takes the unchanged normal path.
    if let Some(spec) = resolve_fault(&state, &headers, req.stream, &req.model)? {
        // The model must still exist so an unknown-model request fails identically to
        // the normal path (directive-mode precedent).
        if !state.engines.contains_key(&req.model) {
            return Err(model_not_found(&state, &req.model));
        }
        return Ok(fault::fault_response(
            spec,
            fault::StreamParams {
                id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                model: req.model.clone(),
                prompt_tokens,
                include_usage,
                flavor: fault::Flavor::Chat,
            },
            connection.map(|Extension(conn)| conn),
        ));
    }

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
        let max_tokens = req.max_tokens;

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
            let mut streamed_tokens = 0u32;
            let mut completed: Option<(u32, u32)> = None;
            let mut errored = false;
            while let Some(event) = rx.recv().await {
                match event {
                    TokenEvent::FirstToken => {
                        // No output needed; first content token follows
                    }
                    TokenEvent::Token { text } => {
                        streamed_tokens += 1;
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
                        completed = Some((prompt_tokens, completion_tokens));
                        break;
                    }
                    TokenEvent::Error { message } => {
                        let _ = stream_tx
                            .send(Ok(
                                Event::default().data(format!("{{\"error\": \"{}\"}}", message))
                            ))
                            .await;
                        errored = true;
                        break;
                    }
                }
            }

            if !errored {
                let (final_prompt, final_completion) =
                    completed.unwrap_or((prompt_tokens, streamed_tokens));
                for event in terminal_chat_events(
                    &id,
                    &model_name,
                    now,
                    finish_reason(final_completion, max_tokens),
                    include_usage.then_some(Usage {
                        prompt_tokens: final_prompt,
                        completion_tokens: final_completion,
                        total_tokens: final_prompt + final_completion,
                    }),
                ) {
                    let _ = stream_tx.send(Ok(event)).await;
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
                finish_reason: finish_reason(completion_tokens, req.max_tokens),
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

/// Why generation stopped, by the rule real engines use: hitting the cap is `length`,
/// anything else is `stop`. The sim previously reported `stop` unconditionally, so a
/// length-capped stream was indistinguishable from a natural one.
fn finish_reason(completion_tokens: u32, max_tokens: u32) -> &'static str {
    if completion_tokens >= max_tokens {
        "length"
    } else {
        "stop"
    }
}

/// How a healthy chat stream ends: the finish_reason chunk, then — only when
/// `stream_options.include_usage` asked for it — usage as its OWN frame carrying
/// `"choices": []`, then `[DONE]`.
///
/// The separate usage frame is what both real continuation targets emit (dynamo AND
/// Fireworks); the sim used to hang `usage` off the finish_reason chunk, which no
/// real engine does.
fn terminal_chat_events(
    id: &str,
    model: &str,
    created: u64,
    finish_reason: &'static str,
    usage: Option<Usage>,
) -> Vec<Event> {
    let mut events = vec![sse_json(&ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: ChunkDelta {
                role: None,
                content: None,
                tool_calls: None,
            },
            finish_reason: Some(finish_reason),
        }],
        usage: None,
    })];
    if let Some(usage) = usage {
        events.push(sse_json(&ChatCompletionChunk {
            id: id.to_string(),
            object: "chat.completion.chunk",
            created,
            model: model.to_string(),
            choices: Vec::new(),
            usage: Some(usage),
        }));
    }
    events.push(Event::default().data("[DONE]"));
    events
}

/// `/v1/completions` counterpart of [`terminal_chat_events`] — same sequence, in
/// `text_completion` shape.
fn terminal_completion_events(
    id: &str,
    model: &str,
    created: u64,
    finish_reason: &'static str,
    usage: Option<Usage>,
) -> Vec<Event> {
    let mut events = vec![sse_json(&CompletionChunk {
        id: id.to_string(),
        object: "text_completion",
        created,
        model: model.to_string(),
        choices: vec![CompletionChunkChoice {
            text: String::new(),
            index: 0,
            finish_reason: Some(finish_reason),
        }],
        usage: None,
    })];
    if let Some(usage) = usage {
        events.push(sse_json(&CompletionChunk {
            id: id.to_string(),
            object: "text_completion",
            created,
            model: model.to_string(),
            choices: Vec::new(),
            usage: Some(usage),
        }));
    }
    events.push(Event::default().data("[DONE]"));
    events
}

fn sse_json<T: serde::Serialize>(value: &T) -> Event {
    Event::default().data(serde_json::to_string(value).expect("response types serialize"))
}

pub async fn completions(
    State(state): State<Arc<AppState>>,
    connection: Option<Extension<Arc<fault::FaultConnection>>>,
    headers: HeaderMap,
    Json(req): Json<CompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    // Batched prompts would each need their own choice; this sim serves batch-of-one,
    // like the engines behind the continuation targets. Reject the rest explicitly
    // instead of silently answering only the first.
    if req.prompt.batch_len() > 1 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": format!(
                        "batched prompts are not supported: got {} prompts, expected 1",
                        req.prompt.batch_len()
                    ),
                    "type": "invalid_request_error",
                    "code": "invalid_prompt"
                }
            })),
        ));
    }

    // Token-id prompts report their id COUNT verbatim. This is load-bearing: the
    // continuation billing merge derives `prompt = P_reported − seg` from it, so an
    // estimate here would corrupt every resumed request's accounting.
    let prompt_tokens = match req.prompt.token_ids() {
        Some(ids) => ids.len() as u32,
        None => count_text_prompt_tokens(&state, req.prompt.text()),
    };
    let include_usage = req
        .stream_options
        .as_ref()
        .map(|options| options.include_usage)
        .unwrap_or(false);

    // Fault injection (serve::fault), same rules as chat completions: resume legs ARE
    // streaming completions requests, so every death mode has to be reachable here or
    // chain-resume tests have nothing to kill.
    if let Some(spec) = resolve_fault(&state, &headers, req.stream, &req.model)? {
        if !state.engines.contains_key(&req.model) {
            return Err(model_not_found(&state, &req.model));
        }
        return Ok(fault::fault_response(
            spec,
            fault::StreamParams {
                id: format!("cmpl-{}", uuid::Uuid::new_v4()),
                model: req.model.clone(),
                prompt_tokens,
                include_usage,
                flavor: fault::Flavor::Completion,
            },
            connection.map(|Extension(conn)| conn),
        ));
    }

    let (request_id, mut rx) =
        submit_engine_request(&state, &req.model, prompt_tokens, req.max_tokens, "cmpl").await?;

    if req.stream {
        let model_name = req.model.clone();
        let id = request_id.clone();
        let max_tokens = req.max_tokens;
        let (stream_tx, stream_rx) = mpsc::channel::<Result<Event, Infallible>>(64);

        tokio::spawn(async move {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            let mut streamed_tokens = 0u32;
            let mut completed: Option<(u32, u32)> = None;
            let mut errored = false;
            while let Some(event) = rx.recv().await {
                match event {
                    TokenEvent::FirstToken => {}
                    TokenEvent::Token { text } => {
                        streamed_tokens += 1;
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
                        completed = Some((prompt_tokens, completion_tokens));
                        break;
                    }
                    TokenEvent::Error { message } => {
                        let _ = stream_tx
                            .send(Ok(
                                Event::default().data(format!("{{\"error\": \"{}\"}}", message))
                            ))
                            .await;
                        errored = true;
                        break;
                    }
                }
            }

            if !errored {
                let (final_prompt, final_completion) =
                    completed.unwrap_or((prompt_tokens, streamed_tokens));
                for event in terminal_completion_events(
                    &id,
                    &model_name,
                    now,
                    finish_reason(final_completion, max_tokens),
                    include_usage.then_some(Usage {
                        prompt_tokens: final_prompt,
                        completion_tokens: final_completion,
                        total_tokens: final_prompt + final_completion,
                    }),
                ) {
                    let _ = stream_tx.send(Ok(event)).await;
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

        // `echo` on a text prompt prepends the prompt to the completion (classic OpenAI
        // behavior); on an id prompt the sim cannot detokenize, so the ids come back
        // verbatim as `prompt_token_ids` instead (Fireworks behavior).
        let echoed_ids = req
            .echo
            .then(|| req.prompt.token_ids().map(<[u32]>::to_vec))
            .flatten();
        let text = match (req.echo, req.prompt.token_ids()) {
            (true, None) => format!("{}{}", req.prompt.text(), content.trim_end()),
            _ => content.trim_end().to_string(),
        };

        let response = CompletionResponse {
            id: request_id,
            object: "text_completion",
            created: now,
            model: req.model,
            choices: vec![CompletionChoice {
                text,
                index: 0,
                finish_reason: finish_reason(completion_tokens, req.max_tokens),
                prompt_token_ids: echoed_ids,
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

/// The fault directive for this request, if any. Header wins over static model config;
/// a malformed header is a 400 (never a silent no-op — a typo'd e2e test must fail
/// loudly). Fault modes are mid-STREAM deaths: an explicit header on a non-streaming
/// request is rejected, while static model config simply doesn't apply there (a fault
/// model must still serve normal non-streaming traffic).
///
/// The header is client-controlled and lets any caller stall or abort connections, so
/// it sits behind `--enable-directives` — the same "untrusted clients must not reach
/// this" trust boundary as echo-directives (rejected loudly when off, never ignored).
/// Static `[fault]` config is operator input validated at boot and is not gated.
fn resolve_fault(
    state: &AppState,
    headers: &HeaderMap,
    stream: bool,
    model: &str,
) -> Result<Option<fault::FaultSpec>, (StatusCode, Json<serde_json::Value>)> {
    if let Some(value) = headers.get(fault::FAULT_HEADER) {
        if !state.enable_directives {
            return Err(fault::invalid_fault(
                "fault injection via header requires --enable-directives",
            ));
        }
        let spec = value
            .to_str()
            .map_err(|_| "header value must be visible ASCII".to_string())
            .and_then(fault::FaultSpec::parse_header)
            .map_err(|e| fault::invalid_fault(&e))?;
        if !stream {
            return Err(fault::invalid_fault(
                "fault modes are mid-stream deaths; set \"stream\": true",
            ));
        }
        return Ok(Some(spec));
    }
    Ok(state.model_faults.get(model).filter(|_| stream).cloned())
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

/// Simulated chat-template overhead applied to the reported chat prompt count (percent).
///
/// Real engines report `prompt_tokens` from their chat-template RENDERING, which adds
/// role headers and section scaffolding on top of the raw content — a few percent on
/// typical prompts. This fake counts raw content, so a template-exact meter upstream
/// (dwctl's render-counting cache classifier) legitimately counts a fully-marked
/// prefix ABOVE our reported prompt and trips its creations-vs-prompt corrupt guard.
/// Inflating the reported prompt by a bounded margin keeps that guard armed but not
/// hair-triggered: template counts within +5% of raw pass; anything drifting further
/// still zeroes the split and turns the cache-parity board red. Deliberately NOT
/// sourced from tokenizer-svc — the fake must remain an independent count or the
/// comparison stops testing anything. Floor arithmetic, so tiny prompts (< 20 tokens)
/// round to no overhead. Applies to chat completions only; raw /v1/completions
/// prompts are not chat-templated by real engines either.
const PROMPT_TEMPLATE_OVERHEAD_PCT: u64 = 5;

fn count_prompt_tokens(
    state: &AppState,
    messages: &[ChatMessage],
    tools: Option<&serde_json::Value>,
) -> u32 {
    let text = prompt_text(messages, tools);
    let raw = u64::from(count_text_prompt_tokens(state, &text));
    (raw * (100 + PROMPT_TEMPLATE_OVERHEAD_PCT) / 100) as u32
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
            model_faults: HashMap::new(),
        })
    }

    fn test_state_directives_off(engine_tx: mpsc::Sender<EngineRequest>) -> Arc<AppState> {
        Arc::new(AppState {
            engines: HashMap::from([("test-model".to_string(), engine_tx)]),
            model_names: vec!["test-model".to_string()],
            tokenizer: None,
            enable_directives: false,
            model_faults: HashMap::new(),
        })
    }

    /// A `/v1/completions` request with the sim's defaults; tests override the fields
    /// they care about.
    fn completion_request(prompt: CompletionPrompt) -> CompletionRequest {
        CompletionRequest {
            model: "test-model".to_string(),
            prompt,
            stream: false,
            max_tokens: 4,
            stream_options: None,
            priority: None,
            echo: false,
        }
    }

    fn text_prompt(text: &str) -> CompletionPrompt {
        CompletionPrompt::Text(text.to_string())
    }

    /// Answer the engine request with a fixed completion, so a handler test can assert on
    /// the wire shape without a simulator.
    fn answer_engine(
        mut engine_rx: mpsc::Receiver<EngineRequest>,
        tokens: &'static [&'static str],
        prompt_tokens: u32,
        completion_tokens: u32,
    ) -> tokio::task::JoinHandle<EngineRequest> {
        tokio::spawn(async move {
            let engine_req = engine_rx.recv().await.expect("engine path must be taken");
            let _ = engine_req.tx.send(TokenEvent::FirstToken).await;
            for token in tokens {
                let _ = engine_req
                    .tx
                    .send(TokenEvent::Token {
                        text: token.to_string(),
                    })
                    .await;
            }
            let _ = engine_req
                .tx
                .send(TokenEvent::Done {
                    prompt_tokens,
                    completion_tokens,
                })
                .await;
            engine_req
        })
    }

    /// Parse the SSE data frames that are JSON (everything but `[DONE]`).
    fn json_frames(events: &[String]) -> Vec<serde_json::Value> {
        events
            .iter()
            .filter(|e| *e != "[DONE]")
            .map(|e| serde_json::from_str(e).unwrap())
            .collect()
    }

    async fn response_json(response: Response) -> serde_json::Value {
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    async fn response_sse_events(response: Response) -> Vec<String> {
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        sse_data_frames(&body)
    }

    /// Collect a body that ABORTS: fault modes end the stream with an error instead of a
    /// clean close, which `to_bytes` refuses outright, so read until the abort and keep
    /// whatever reached the wire first.
    async fn response_sse_events_until_abort(response: Response) -> Vec<String> {
        use tokio_stream::StreamExt;
        let mut stream = response.into_body().into_data_stream();
        let mut buf = Vec::new();
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => buf.extend_from_slice(&bytes),
                Err(_) => break,
            }
        }
        sse_data_frames(&buf)
    }

    fn sse_data_frames(body: &[u8]) -> Vec<String> {
        String::from_utf8_lossy(body)
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
        let response = chat_completions(State(state), None, HeaderMap::new(), Json(req))
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
        let response = chat_completions(State(state), None, HeaderMap::new(), Json(req))
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
        let response = chat_completions(State(state), None, HeaderMap::new(), Json(req))
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
        let response = chat_completions(State(state), None, HeaderMap::new(), Json(req))
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
        let response = chat_completions(State(state), None, HeaderMap::new(), Json(req))
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
        let err = chat_completions(State(state), None, HeaderMap::new(), Json(req))
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
            None,
            HeaderMap::new(),
            Json(completion_request(text_prompt("hello world"))),
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
            None,
            HeaderMap::new(),
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
            None,
            HeaderMap::new(),
            Json(CompletionRequest {
                stream: true,
                stream_options: Some(StreamOptions {
                    include_usage: true,
                }),
                ..completion_request(text_prompt("hello world"))
            }),
        )
        .await
        .unwrap()
        .into_response();

        let events = response_sse_events(response).await;
        assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
        let frames = json_frames(&events);

        // Real engines put usage in its OWN final frame with choices:[], AFTER the
        // finish_reason chunk — never hanging off the finish chunk itself.
        let finish = &frames[frames.len() - 2];
        assert_eq!(finish["choices"][0]["finish_reason"], "stop");
        assert!(finish.get("usage").is_none());

        let usage = frames.last().unwrap();
        assert_eq!(usage["object"], "text_completion");
        assert_eq!(usage["choices"].as_array().unwrap().len(), 0);
        assert_eq!(usage["usage"]["prompt_tokens"], 3);
        assert_eq!(usage["usage"]["completion_tokens"], 1);
        assert_eq!(usage["usage"]["total_tokens"], 4);
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
            None,
            HeaderMap::new(),
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
        assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
        let frames = json_frames(&events);

        let finish = &frames[frames.len() - 2];
        assert_eq!(finish["choices"][0]["finish_reason"], "stop");
        assert!(finish.get("usage").is_none());

        let usage = frames.last().unwrap();
        assert_eq!(usage["object"], "chat.completion.chunk");
        assert_eq!(usage["choices"].as_array().unwrap().len(), 0);
        assert_eq!(usage["usage"]["prompt_tokens"], 5);
        assert_eq!(usage["usage"]["completion_tokens"], 1);
        assert_eq!(usage["usage"]["total_tokens"], 6);
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
            None,
            HeaderMap::new(),
            Json(CompletionRequest {
                stream: true,
                ..completion_request(text_prompt("hello world"))
            }),
        )
        .await
        .unwrap()
        .into_response();

        let events = response_sse_events(response).await;
        let frames = json_frames(&events);

        // No include_usage means no usage frame at all — the stream ends finish -> [DONE].
        assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
        assert_eq!(frames.last().unwrap()["choices"][0]["finish_reason"], "stop");
        assert!(!frames.iter().any(|f| f.get("usage").is_some()));
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
            None,
            HeaderMap::new(),
            Json(CompletionRequest {
                stream: true,
                stream_options: Some(StreamOptions {
                    include_usage: false,
                }),
                ..completion_request(text_prompt("hello world"))
            }),
        )
        .await
        .unwrap()
        .into_response();

        let events = response_sse_events(response).await;
        let frames = json_frames(&events);

        assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
        assert_eq!(frames.last().unwrap()["choices"][0]["finish_reason"], "stop");
        assert!(!frames.iter().any(|f| f.get("usage").is_some()));
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
            None,
            HeaderMap::new(),
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
            None,
            HeaderMap::new(),
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

    #[test]
    fn chat_prompt_count_includes_template_overhead() {
        let state = Arc::new(AppState {
            engines: HashMap::new(),
            model_names: vec!["m".to_string()],
            tokenizer: None,
            enable_directives: false,
            model_faults: HashMap::new(),
        });
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            tool_calls: None,
            content: Some(MessageContent::Text("x".repeat(400))),
        }];
        let raw = u64::from(count_text_prompt_tokens(
            &state,
            &prompt_text(&messages, None),
        ));
        assert!(
            raw >= 20,
            "prompt must be large enough for the floor overhead to bite"
        );
        let expected = (raw * (100 + PROMPT_TEMPLATE_OVERHEAD_PCT) / 100) as u32;
        assert_eq!(count_prompt_tokens(&state, &messages, None), expected);
        assert!(u64::from(count_prompt_tokens(&state, &messages, None)) > raw);
    }

    #[tokio::test]
    async fn completions_returns_model_not_found_error() {
        let state = Arc::new(AppState {
            engines: HashMap::new(),
            model_names: vec!["other-model".to_string()],
            tokenizer: None,
            enable_directives: false,
            model_faults: HashMap::new(),
        });

        let error = match completions(
            State(state),
            None,
            HeaderMap::new(),
            Json(CompletionRequest {
                model: "missing-model".to_string(),
                ..completion_request(text_prompt("hello"))
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

    fn fault_header(value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(fault::FAULT_HEADER, value.parse().unwrap());
        headers
    }

    fn chat_request(stream: bool) -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "stream": stream,
            "stream_options": {"include_usage": true},
            "messages": [{"role": "user", "content": "hello world"}]
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn fault_header_on_non_streaming_request_is_400() {
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);
        let err = chat_completions(
            State(state),
            None,
            fault_header("cut_between_frames"),
            Json(chat_request(false)),
        )
        .await
        .err()
        .unwrap();
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        let body = serde_json::to_value(err.1 .0).unwrap();
        assert_eq!(body["error"]["code"], "invalid_fault_directive");
    }

    #[tokio::test]
    async fn malformed_fault_header_is_400_not_a_silent_noop() {
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);
        let err = chat_completions(
            State(state),
            None,
            fault_header("cut_betwen_frames"), // typo'd mode
            Json(chat_request(true)),
        )
        .await
        .err()
        .unwrap();
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn fault_header_with_unknown_model_is_404() {
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);
        let mut req = chat_request(true);
        req.model = "nope".to_string();
        let err = chat_completions(State(state), None, fault_header("no_usage"), Json(req))
            .await
            .err()
            .unwrap();
        assert_eq!(err.0, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn fault_header_streams_the_death_without_touching_the_engine() {
        // Channel with NO consumer: completing at all proves the engine bypass, like the
        // directive tests.
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);
        let response = chat_completions(
            State(state),
            None,
            fault_header("no_usage;after_chunks=2;delay_ms=0"),
            Json(chat_request(true)),
        )
        .await
        .unwrap()
        .into_response();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[axum::http::header::CONTENT_TYPE],
            "text/event-stream"
        );

        let events = response_sse_events(response).await;
        // role + 2 content + finish + [DONE]
        assert_eq!(events.len(), 5);
        assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
        let finish: serde_json::Value = serde_json::from_str(&events[3]).unwrap();
        assert_eq!(finish["choices"][0]["finish_reason"], "stop");
        // The whole point of no_usage: include_usage was requested and never honored.
        assert!(finish.get("usage").is_none());
    }

    #[tokio::test]
    async fn fault_header_without_enable_directives_is_400() {
        // The header is client-controlled and can stall/abort connections, so it sits
        // behind the same trust gate as echo-directives — and fails loudly, so an e2e
        // test pointed at a mis-deployed sim can't chase a phantom pass.
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state_directives_off(engine_tx);
        let err = chat_completions(
            State(state),
            None,
            fault_header("cut_between_frames"),
            Json(chat_request(true)),
        )
        .await
        .err()
        .unwrap();
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        let body = serde_json::to_value(err.1 .0).unwrap();
        assert_eq!(body["error"]["code"], "invalid_fault_directive");
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("--enable-directives"));
    }

    #[tokio::test]
    async fn static_model_fault_applies_without_header() {
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineRequest>(1);
        // Directives OFF: static [fault] config is operator input validated at boot,
        // not client input — it is deliberately NOT behind the directives gate.
        let mut state = test_state_directives_off(engine_tx);
        let spec = fault::FaultSpec::parse_header("no_done;after_chunks=1;delay_ms=0").unwrap();
        Arc::get_mut(&mut state)
            .unwrap()
            .model_faults
            .insert("test-model".to_string(), spec);

        let response = chat_completions(
            State(state),
            None,
            HeaderMap::new(),
            Json(chat_request(true)),
        )
        .await
        .unwrap()
        .into_response();
        let events = response_sse_events(response).await;
        // role + 1 content + finish + usage, and no [DONE] — the configured death.
        assert_eq!(events.len(), 4);
        assert!(!events.iter().any(|e| e == "[DONE]"));
        let finish: serde_json::Value = serde_json::from_str(&events[2]).unwrap();
        assert_eq!(finish["choices"][0]["finish_reason"], "stop");
        assert!(finish.get("usage").is_none());
        let usage: serde_json::Value = serde_json::from_str(&events[3]).unwrap();
        assert_eq!(usage["choices"].as_array().unwrap().len(), 0);
        assert!(usage["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn fault_header_takes_precedence_over_directive() {
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx); // directives enabled
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "stream": true,
            "messages": [{"role": "user", "content": "<<respond:{\"text\":\"scripted\"}>>"}]
        }))
        .unwrap();
        let response = chat_completions(
            State(state),
            None,
            fault_header("cancelled_499;after_chunks=1;delay_ms=0"),
            Json(req),
        )
        .await
        .unwrap()
        .into_response();
        let events = response_sse_events(response).await;
        assert_eq!(
            events.last().map(String::as_str),
            Some(
                r#"{"error":{"code":499,"message":"CancelledError: ","type":"request_cancelled"}}"#
            )
        );
        assert!(!events.iter().any(|e| e.contains("scripted")));
    }

    // --- Fault injection on streaming /v1/completions (resume-leg deaths) ---

    /// A streaming completions request carrying whatever a resume leg carries.
    fn streaming_completion_request() -> CompletionRequest {
        CompletionRequest {
            stream: true,
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
            priority: Some(0),
            ..completion_request(text_prompt("hello world"))
        }
    }

    #[tokio::test]
    async fn completions_fault_cut_between_frames_emits_n_frames_then_dies() {
        // Chain-resume tests kill a RESUME LEG mid-stream, and resume legs are streaming
        // completions requests. No engine consumer: completing at all proves the bypass.
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);
        let response = completions(
            State(state),
            None,
            fault_header("cut_between_frames;after_chunks=2;delay_ms=0"),
            Json(streaming_completion_request()),
        )
        .await
        .unwrap()
        .into_response();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[axum::http::header::CONTENT_TYPE],
            "text/event-stream"
        );

        let events = response_sse_events_until_abort(response).await;
        // Exactly 2 content frames, no role frame, and no terminator of any kind.
        assert_eq!(events.len(), 2);
        assert!(!events.iter().any(|e| e == "[DONE]"));
        for frame in json_frames(&events) {
            assert_eq!(frame["object"], "text_completion");
            assert!(frame["choices"][0]["text"].is_string());
            assert!(frame["choices"][0].get("finish_reason").is_none());
        }
    }

    #[tokio::test]
    async fn completions_fault_error_envelope_200_streams_the_provider_error() {
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);
        let response = completions(
            State(state),
            None,
            fault_header("error_envelope_200;after_chunks=1;delay_ms=0"),
            Json(streaming_completion_request()),
        )
        .await
        .unwrap()
        .into_response();

        assert_eq!(response.status(), StatusCode::OK);
        let events = response_sse_events(response).await;
        assert_eq!(events.len(), 3); // 1 content + error envelope + [DONE]
        assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
        let envelope: serde_json::Value = serde_json::from_str(&events[1]).unwrap();
        assert_eq!(envelope["error"]["code"], 502);
    }

    #[tokio::test]
    async fn completions_fault_stall_holds_the_connection_open() {
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);
        let response = completions(
            State(state),
            None,
            fault_header("stall;after_chunks=1;delay_ms=0"),
            Json(streaming_completion_request()),
        )
        .await
        .unwrap()
        .into_response();

        // The body never completes, so collecting it must time out rather than end.
        let body = response.into_body();
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_millis(250),
                to_bytes(body, usize::MAX)
            )
            .await
            .is_err(),
            "stall must leave the connection open with no terminator"
        );
    }

    #[tokio::test]
    async fn completions_fault_on_non_streaming_request_is_400() {
        // Faults are mid-STREAM deaths; a non-streaming request asking for one is a
        // loud 400 on both endpoints, never a silently healthy response.
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);
        let err = completions(
            State(state),
            None,
            fault_header("cut_between_frames"),
            Json(completion_request(text_prompt("hello"))),
        )
        .await
        .err()
        .unwrap();
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        let body = serde_json::to_value(err.1 .0).unwrap();
        assert_eq!(body["error"]["code"], "invalid_fault_directive");
    }

    #[tokio::test]
    async fn completions_fault_header_without_enable_directives_is_400() {
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state_directives_off(engine_tx);
        let err = completions(
            State(state),
            None,
            fault_header("cut_between_frames"),
            Json(streaming_completion_request()),
        )
        .await
        .err()
        .unwrap();
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(serde_json::to_value(err.1 .0).unwrap()["error"]["message"]
            .as_str()
            .unwrap()
            .contains("--enable-directives"));
    }

    #[tokio::test]
    async fn completions_fault_reports_id_prompt_token_count() {
        // The death happens on a resume leg whose prompt is token ids, and the usage that
        // does arrive must still carry the id-exact prompt count.
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);
        let response = completions(
            State(state),
            None,
            fault_header("no_done;after_chunks=2;delay_ms=0"),
            Json(CompletionRequest {
                prompt: CompletionPrompt::Ids(vec![9, 8, 7, 6, 5]),
                ..streaming_completion_request()
            }),
        )
        .await
        .unwrap()
        .into_response();

        let events = response_sse_events(response).await;
        let frames = json_frames(&events);
        let usage = frames.last().unwrap();
        assert_eq!(usage["choices"].as_array().unwrap().len(), 0);
        assert_eq!(usage["usage"]["prompt_tokens"], 5);
    }

    // --- Token-id prompts ---

    #[tokio::test]
    async fn id_prompt_reports_exact_id_count_as_prompt_tokens() {
        // Load-bearing: the continuation billing merge computes `prompt = P_reported − seg`
        // from this number, so it must be the id COUNT, never a char estimate. The four
        // ids here estimate to a very different number through the text path.
        let (engine_tx, engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);
        let answered = answer_engine(engine_rx, &["Hello", " world"], 4, 2);

        let response = completions(
            State(state),
            None,
            HeaderMap::new(),
            Json(completion_request(CompletionPrompt::Ids(vec![1, 2, 3, 4]))),
        )
        .await
        .unwrap()
        .into_response();

        let engine_req = answered.await.unwrap();
        assert_eq!(
            engine_req.prompt_tokens, 4,
            "the engine must be told the id count"
        );

        let json = response_json(response).await;
        assert_eq!(json["usage"]["prompt_tokens"], 4);
        // Text is simulated exactly as for a string prompt: this deployment mounts no
        // tokenizer, so ids are never detokenized — only the accounting is id-exact.
        assert_eq!(json["choices"][0]["text"], "Hello world");
    }

    #[tokio::test]
    async fn id_prompt_batch_of_one_counts_the_inner_list() {
        let (engine_tx, engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);
        let answered = answer_engine(engine_rx, &["x"], 6, 1);

        let response = completions(
            State(state),
            None,
            HeaderMap::new(),
            Json(completion_request(CompletionPrompt::IdBatch(vec![vec![
                11, 12, 13, 14, 15, 16,
            ]]))),
        )
        .await
        .unwrap()
        .into_response();

        assert_eq!(answered.await.unwrap().prompt_tokens, 6);
        assert_eq!(response_json(response).await["usage"]["prompt_tokens"], 6);
    }

    #[test]
    fn prompt_deserializes_every_accepted_shape() {
        let parse = |value: serde_json::Value| -> CompletionRequest {
            serde_json::from_value(serde_json::json!({"model": "m", "prompt": value}))
                .expect("prompt shape must deserialize")
        };
        // string
        assert!(parse(serde_json::json!("hi")).prompt.token_ids().is_none());
        // int[]
        assert_eq!(
            parse(serde_json::json!([1, 2, 3])).prompt.token_ids(),
            Some(&[1u32, 2, 3][..])
        );
        // int[][] (batch of one)
        assert_eq!(
            parse(serde_json::json!([[4, 5]])).prompt.token_ids(),
            Some(&[4u32, 5][..])
        );
        // string[] — accepted like a real engine; still a text prompt
        let texts = parse(serde_json::json!(["hi"]));
        assert!(texts.prompt.token_ids().is_none());
        assert_eq!(texts.prompt.text(), "hi");
    }

    #[tokio::test]
    async fn batched_prompt_larger_than_one_is_rejected() {
        // Batches would need one choice per prompt; answering only the first silently
        // would be worse than a clear 400.
        let (engine_tx, _engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);
        let err = completions(
            State(state),
            None,
            HeaderMap::new(),
            Json(completion_request(CompletionPrompt::IdBatch(vec![
                vec![1, 2],
                vec![3, 4],
            ]))),
        )
        .await
        .err()
        .unwrap();
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        let body = serde_json::to_value(err.1 .0).unwrap();
        assert_eq!(body["error"]["code"], "invalid_prompt");
        assert!(body["error"]["message"].as_str().unwrap().contains("got 2"));
    }

    #[tokio::test]
    async fn echo_on_id_prompt_returns_the_received_ids() {
        let (engine_tx, engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);
        let answered = answer_engine(engine_rx, &["out"], 3, 1);

        let response = completions(
            State(state),
            None,
            HeaderMap::new(),
            Json(CompletionRequest {
                echo: true,
                ..completion_request(CompletionPrompt::Ids(vec![7, 8, 9]))
            }),
        )
        .await
        .unwrap()
        .into_response();

        answered.await.unwrap();
        let json = response_json(response).await;
        assert_eq!(
            json["choices"][0]["prompt_token_ids"],
            serde_json::json!([7, 8, 9])
        );
        assert_eq!(json["choices"][0]["text"], "out");
    }

    #[tokio::test]
    async fn echo_is_absent_and_ids_omitted_by_default() {
        let (engine_tx, engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);
        let answered = answer_engine(engine_rx, &["out"], 3, 1);

        let response = completions(
            State(state),
            None,
            HeaderMap::new(),
            Json(completion_request(CompletionPrompt::Ids(vec![7, 8, 9]))),
        )
        .await
        .unwrap()
        .into_response();

        answered.await.unwrap();
        let json = response_json(response).await;
        assert!(json["choices"][0].get("prompt_token_ids").is_none());
    }

    #[tokio::test]
    async fn priority_and_stream_options_are_tolerated_on_completions() {
        // Every resume leg carries both; a serde error on either would 400 the resume
        // before it reached the engine.
        let req: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "prompt": [1, 2, 3],
            "max_tokens": 8,
            "priority": 0,
            "stream": true,
            "stream_options": {"include_usage": true}
        }))
        .expect("priority + stream_options must deserialize");
        assert_eq!(req.priority, Some(0));
        assert!(req.stream_options.unwrap().include_usage);

        // A non-zero priority is equally fine, and otherwise ignored by the sim.
        let req: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test-model",
            "prompt": "hi",
            "priority": 100
        }))
        .unwrap();
        assert_eq!(req.priority, Some(100));
    }

    // --- Length-capped termination ---

    #[tokio::test]
    async fn completions_length_capped_stream_terminates_with_length_then_usage_then_done() {
        // The bug this covers: a max_tokens-capped stream used to stop dead with no
        // finish_reason, no usage and no [DONE] — byte-indistinguishable from a
        // mid-stream death, which poisons every test using max_tokens as a stop.
        let (engine_tx, engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);
        let answered = answer_engine(engine_rx, &["a ", "b ", "c "], 4, 3);

        let response = completions(
            State(state),
            None,
            HeaderMap::new(),
            Json(CompletionRequest {
                stream: true,
                max_tokens: 3, // completion_tokens == max_tokens -> length
                stream_options: Some(StreamOptions {
                    include_usage: true,
                }),
                ..completion_request(CompletionPrompt::Ids(vec![1, 2, 3, 4]))
            }),
        )
        .await
        .unwrap()
        .into_response();

        answered.await.unwrap();
        let events = response_sse_events(response).await;
        assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
        let frames = json_frames(&events);

        let finish = &frames[frames.len() - 2];
        assert_eq!(finish["choices"][0]["finish_reason"], "length");
        assert!(finish.get("usage").is_none());

        let usage = frames.last().unwrap();
        assert_eq!(usage["choices"].as_array().unwrap().len(), 0);
        assert_eq!(usage["usage"]["prompt_tokens"], 4);
        assert_eq!(usage["usage"]["completion_tokens"], 3);
    }

    #[tokio::test]
    async fn chat_length_capped_stream_terminates_with_length_then_usage_then_done() {
        let (engine_tx, engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);
        let answered = answer_engine(engine_rx, &["a ", "b "], 5, 2);

        let response = chat_completions(
            State(state),
            None,
            HeaderMap::new(),
            Json(ChatCompletionRequest {
                max_tokens: 2,
                ..chat_request(true)
            }),
        )
        .await
        .unwrap()
        .into_response();

        answered.await.unwrap();
        let events = response_sse_events(response).await;
        assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
        let frames = json_frames(&events);

        let finish = &frames[frames.len() - 2];
        assert_eq!(finish["choices"][0]["finish_reason"], "length");
        assert!(finish.get("usage").is_none());
        assert_eq!(frames.last().unwrap()["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn natural_stop_still_reports_stop() {
        // The other side of the finish_reason rule: under the cap is an ordinary stop.
        let (engine_tx, engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);
        let answered = answer_engine(engine_rx, &["a "], 4, 1);

        let response = completions(
            State(state),
            None,
            HeaderMap::new(),
            Json(CompletionRequest {
                stream: true,
                max_tokens: 64,
                ..completion_request(text_prompt("hello world"))
            }),
        )
        .await
        .unwrap()
        .into_response();

        answered.await.unwrap();
        let frames = json_frames(&response_sse_events(response).await);
        assert_eq!(frames.last().unwrap()["choices"][0]["finish_reason"], "stop");
    }

    #[tokio::test]
    async fn non_streaming_length_cap_reports_length() {
        let (engine_tx, engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);
        let answered = answer_engine(engine_rx, &["a ", "b "], 4, 2);

        let response = completions(
            State(state),
            None,
            HeaderMap::new(),
            Json(CompletionRequest {
                max_tokens: 2,
                ..completion_request(text_prompt("hello world"))
            }),
        )
        .await
        .unwrap()
        .into_response();

        answered.await.unwrap();
        assert_eq!(
            response_json(response).await["choices"][0]["finish_reason"],
            "length"
        );
    }

    #[tokio::test]
    async fn stream_terminates_even_when_the_engine_never_reports_done() {
        // The engine used to drop events on a full channel, losing the terminating Done
        // and leaving the stream to just stop. Whatever the cause, a stream must never
        // end without a terminator — explicit fault modes are how deaths are requested.
        let (engine_tx, mut engine_rx) = mpsc::channel::<EngineRequest>(1);
        let state = test_state(engine_tx);

        let answered = tokio::spawn(async move {
            let engine_req = engine_rx.recv().await.unwrap();
            for token in ["a ", "b "] {
                let _ = engine_req
                    .tx
                    .send(TokenEvent::Token {
                        text: token.to_string(),
                    })
                    .await;
            }
            // Channel closes with no Done: the request vanished mid-generation.
            drop(engine_req);
        });

        let response = completions(
            State(state),
            None,
            HeaderMap::new(),
            Json(CompletionRequest {
                stream: true,
                max_tokens: 2,
                stream_options: Some(StreamOptions {
                    include_usage: true,
                }),
                ..completion_request(text_prompt("hello world"))
            }),
        )
        .await
        .unwrap()
        .into_response();

        answered.await.unwrap();
        let events = response_sse_events(response).await;
        assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
        let frames = json_frames(&events);

        // Counts fall back to what actually went out on the wire.
        let finish = &frames[frames.len() - 2];
        assert_eq!(finish["choices"][0]["finish_reason"], "length");
        assert_eq!(frames.last().unwrap()["usage"]["completion_tokens"], 2);
    }

    #[test]
    fn prompt_token_estimation_supports_raw_text() {
        let state = AppState {
            engines: HashMap::new(),
            model_names: Vec::new(),
            tokenizer: None,
            enable_directives: false,
            model_faults: HashMap::new(),
        };

        assert_eq!(count_text_prompt_tokens(&state, "hello world"), 3);
    }
}
