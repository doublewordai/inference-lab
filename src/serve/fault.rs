//! On-demand mid-stream death modes for streaming chat completions.
//!
//! This is the test double for the mid-stream continuation workstream: every way a real
//! downstream kills a stream mid-generation, reproducible deterministically per request.
//! A faulting stream emits real partial output first (the initial role frame plus
//! `after_chunks` content-bearing delta frames of deterministic placeholder text), then
//! dies in the requested way — so there is always something to resume.
//!
//! Trigger (precedence order):
//! 1. Request header `x-inference-lab-fault: <mode>[;key=value]...` — keys `after_chunks`
//!    (default 3), `delay_ms` (default 10), `utf8` (`cut_mid_frame` only). A malformed
//!    value is a 400, never a silent no-op: a typo'd e2e test must fail loudly.
//! 2. Per-model `[fault]` config (`config::FaultConfig`) — static fallback for clients
//!    that can't carry the header; applies to every streaming chat completion.
//!
//! Determinism: no randomness anywhere in this module. Same spec -> same frames, same
//! death at the same byte. Content words cycle [`PLACEHOLDER_WORDS`]; pacing is a fixed
//! `delay_ms` between frames.
//!
//! The fault path bypasses the simulation engine entirely (precedent: serve::directive)
//! and writes raw SSE bytes through an [`axum::body::Body::from_stream`] body, because
//! several modes need byte-level control axum's `Sse` can't give: torn frames, missing
//! terminators, and bodies that ABORT (yielding `Err` makes hyper drop the connection
//! without the terminating 0-length chunk — a mid-stream FIN, not a clean end). The
//! `reset` mode additionally arms `SO_LINGER=0` on the accepted socket (via
//! [`FaultConnection`], threaded through request extensions by the serve accept loop) so
//! that drop becomes an abortive close — TCP RST — instead of a FIN.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::http::{header, StatusCode};
use axum::response::Response;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::engine::PLACEHOLDER_WORDS;

/// Request header carrying a per-request fault spec.
pub const FAULT_HEADER: &str = "x-inference-lab-fault";

const DEFAULT_AFTER_CHUNKS: u32 = 3;
const DEFAULT_DELAY_MS: u64 = 10;

/// Every way a stream can die. Names are the wire syntax (header + `[fault]` config).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultMode {
    /// Emit N complete SSE frames, then close the connection (FIN, no chunked terminator).
    CutBetweenFrames,
    /// Close midway through an SSE frame's bytes (torn JSON). With `utf8=true`, the cut
    /// lands inside a multi-byte UTF-8 character in the delta text.
    CutMidFrame,
    /// Abortive close (TCP RST via SO_LINGER=0) instead of FIN, mid-stream.
    Reset,
    /// Emit N frames then send nothing forever, connection open (client idle timeout).
    Stall,
    /// After N frames, an OpenRouter-style error JSON envelope as an SSE data frame,
    /// then [DONE]. HTTP status stays 200.
    ErrorEnvelope200,
    /// After N frames, a 4xx-shaped error object inside the stream (the
    /// tesseracted-nemotron incident signature), then a clean close without [DONE].
    Error400InSse,
    /// Full stream including the finish_reason frame, but close without [DONE].
    NoDone,
    /// Full stream and [DONE], but the usage that was asked for never arrives.
    NoUsage,
    /// dynamo-style frontend cancellation: N frames then the exact 499 error body.
    Cancelled499,
    /// Die cut_between_frames-style while streaming `reasoning_content` deltas.
    MidReasoning,
    /// Die cut_between_frames-style mid tool-call: name announced, `arguments` cut off
    /// partway through its (JSON-string) fragments.
    MidToolCall,
}

impl FaultMode {
    pub const ALL: &'static [FaultMode] = &[
        FaultMode::CutBetweenFrames,
        FaultMode::CutMidFrame,
        FaultMode::Reset,
        FaultMode::Stall,
        FaultMode::ErrorEnvelope200,
        FaultMode::Error400InSse,
        FaultMode::NoDone,
        FaultMode::NoUsage,
        FaultMode::Cancelled499,
        FaultMode::MidReasoning,
        FaultMode::MidToolCall,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            FaultMode::CutBetweenFrames => "cut_between_frames",
            FaultMode::CutMidFrame => "cut_mid_frame",
            FaultMode::Reset => "reset",
            FaultMode::Stall => "stall",
            FaultMode::ErrorEnvelope200 => "error_envelope_200",
            FaultMode::Error400InSse => "error_400_in_sse",
            FaultMode::NoDone => "no_done",
            FaultMode::NoUsage => "no_usage",
            FaultMode::Cancelled499 => "cancelled_499",
            FaultMode::MidReasoning => "mid_reasoning",
            FaultMode::MidToolCall => "mid_tool_call",
        }
    }

    fn parse(s: &str) -> Result<Self, String> {
        Self::ALL
            .iter()
            .find(|m| m.as_str() == s)
            .copied()
            .ok_or_else(|| {
                let names: Vec<&str> = Self::ALL.iter().map(|m| m.as_str()).collect();
                format!(
                    "unknown fault mode '{}'; expected one of: {}",
                    s,
                    names.join(", ")
                )
            })
    }
}

/// A fully validated fault directive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaultSpec {
    pub mode: FaultMode,
    /// Content-bearing delta frames emitted before the fault fires (the initial
    /// role-only frame is always sent and not counted).
    pub after_chunks: u32,
    /// Fixed pacing between emitted frames — deterministic, never sampled.
    pub delay_ms: u64,
    /// `cut_mid_frame` only: cut inside a multi-byte UTF-8 character.
    pub utf8: bool,
}

impl FaultSpec {
    fn new(mode: FaultMode) -> Self {
        Self {
            mode,
            after_chunks: DEFAULT_AFTER_CHUNKS,
            delay_ms: DEFAULT_DELAY_MS,
            utf8: false,
        }
    }

    fn validate(self) -> Result<Self, String> {
        if self.utf8 && self.mode != FaultMode::CutMidFrame {
            return Err(format!(
                "utf8=true only applies to cut_mid_frame (got mode '{}')",
                self.mode.as_str()
            ));
        }
        Ok(self)
    }

    /// Parse the `x-inference-lab-fault` header value:
    /// `<mode>[;after_chunks=<u32>][;delay_ms=<u64>][;utf8=<bool>]`.
    pub fn parse_header(value: &str) -> Result<Self, String> {
        let mut parts = value.split(';').map(str::trim);
        let mode_str = parts.next().unwrap_or_default();
        if mode_str.is_empty() {
            return Err("empty fault header; expected <mode>[;key=value]...".to_string());
        }
        let mut spec = Self::new(FaultMode::parse(mode_str)?);
        for part in parts {
            if part.is_empty() {
                continue;
            }
            let (key, val) = part
                .split_once('=')
                .ok_or_else(|| format!("expected key=value, got '{}'", part))?;
            match key.trim() {
                "after_chunks" => {
                    spec.after_chunks = val
                        .trim()
                        .parse()
                        .map_err(|_| format!("after_chunks must be an integer, got '{}'", val))?;
                }
                "delay_ms" => {
                    spec.delay_ms = val
                        .trim()
                        .parse()
                        .map_err(|_| format!("delay_ms must be an integer, got '{}'", val))?;
                }
                "utf8" => {
                    spec.utf8 = val
                        .trim()
                        .parse()
                        .map_err(|_| format!("utf8 must be true or false, got '{}'", val))?;
                }
                other => return Err(format!("unknown fault parameter '{}'", other)),
            }
        }
        spec.validate()
    }

    /// Validate a static `[fault]` model config (server-startup path — a bad mode name
    /// must fail the boot, not every request).
    pub fn from_config(cfg: &crate::config::FaultConfig) -> Result<Self, String> {
        let mut spec = Self::new(FaultMode::parse(&cfg.mode)?);
        if let Some(n) = cfg.after_chunks {
            spec.after_chunks = n;
        }
        if let Some(ms) = cfg.delay_ms {
            spec.delay_ms = ms;
        }
        if let Some(u) = cfg.utf8 {
            spec.utf8 = u;
        }
        spec.validate()
    }
}

/// Handle to the accepted TCP socket, threaded into request extensions by the serve
/// accept loop so `reset` can arm an abortive close on the exact connection carrying
/// the response.
///
/// Holds its own `dup(2)` of the accepted fd rather than the raw fd number: the fault
/// body task is detached, so if the client disconnects mid-script hyper closes the
/// original `TcpStream` while this handle is still alive — a raw fd could by then have
/// been reused for an unrelated socket. The dup keeps the *same underlying socket*
/// alive and owned, so `arm_reset` can never touch anyone else's connection.
#[derive(Debug)]
pub struct FaultConnection {
    #[cfg(unix)]
    fd: std::os::fd::OwnedFd,
}

impl FaultConnection {
    pub fn new(socket: &tokio::net::TcpStream) -> std::io::Result<Self> {
        #[cfg(unix)]
        {
            use std::os::fd::AsFd;
            Ok(Self {
                fd: socket.as_fd().try_clone_to_owned()?,
            })
        }
        #[cfg(not(unix))]
        {
            let _ = socket;
            Ok(Self {})
        }
    }

    /// Arm an abortive close: with linger zeroed, the kernel sends RST instead of FIN
    /// when the socket's last fd closes. `SO_LINGER` lives on the socket, not the fd,
    /// so setting it through our dup covers hyper's `TcpStream` too. If the client is
    /// already gone this is a harmless setsockopt on a dead-but-owned socket.
    pub fn arm_reset(&self) -> std::io::Result<()> {
        #[cfg(unix)]
        {
            socket2::SockRef::from(&self.fd).set_linger(Some(Duration::ZERO))
        }
        #[cfg(not(unix))]
        {
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "reset fault mode requires a unix platform",
            ))
        }
    }
}

/// The 400 for a malformed/ignored fault directive. Loud on purpose: silently serving a
/// normal stream would send an e2e test chasing a phantom pass.
pub fn invalid_fault(message: &str) -> (StatusCode, axum::Json<serde_json::Value>) {
    (
        StatusCode::BAD_REQUEST,
        axum::Json(json!({
            "error": {
                "message": format!("invalid fault directive: {}", message),
                "type": "invalid_request_error",
                "code": "invalid_fault_directive"
            }
        })),
    )
}

/// Which endpoint's wire shape the faulting stream must speak.
///
/// Both are reachable: a mid-stream-continuation RESUME LEG is a streaming
/// `/v1/completions` request, so chain-resume tests kill a `Completion`-flavored stream.
/// The shapes differ in more than a field name — completions have no role frame and no
/// delta object — so frame counts differ by one between the two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flavor {
    Chat,
    Completion,
}

impl Flavor {
    fn object(&self) -> &'static str {
        match self {
            Flavor::Chat => "chat.completion.chunk",
            Flavor::Completion => "text_completion",
        }
    }
}

/// Per-request context the fault stream needs from the handler.
pub struct StreamParams {
    pub id: String,
    pub model: String,
    pub prompt_tokens: u32,
    pub include_usage: bool,
    pub flavor: Flavor,
}

/// Build the faulting SSE response. Headers match what axum's `Sse` sets, so up to the
/// death the response is indistinguishable from the normal streaming path.
pub fn fault_response(
    spec: FaultSpec,
    params: StreamParams,
    conn: Option<Arc<FaultConnection>>,
) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(16);
    tokio::spawn(run_script(spec, params, conn, tx));
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .expect("static parts are valid")
}

// --- Wire shapes ---
//
// Error bodies below are representative captures of each downstream's signature. The
// cancelled_499 body is exact (dynamo frontend). The OpenRouter envelope and the vLLM
// 400 shape are representative; exact shapes sync with the death-taxonomy workstream
// as it lands.

/// OpenRouter-style mid-stream error envelope, delivered on a 200 stream.
const OPENROUTER_ERROR_ENVELOPE: &str = r#"{"error":{"message":"Provider returned error","code":502,"metadata":{"provider_name":"inference-lab","raw":"simulated upstream provider failure (fault: error_envelope_200)"}}}"#;

/// vLLM-style 400-inside-SSE error object (the tesseracted-nemotron signature).
const ERROR_400_IN_SSE: &str = r#"{"object":"error","message":"This request would exceed the model's maximum context length (fault: error_400_in_sse)","type":"BadRequestError","param":null,"code":400}"#;

/// Exact dynamo frontend-cancellation body.
const CANCELLED_499: &str =
    r#"{"error":{"code":499,"message":"CancelledError: ","type":"request_cancelled"}}"#;

/// The word carrying a multi-byte UTF-8 character ('é' = 0xC3 0xA9) for the
/// `cut_mid_frame;utf8=true` variant.
const UTF8_CUT_WORD: &str = "café ";

/// Fixed partial tool call streamed by `mid_tool_call`. The first delta frame announces
/// id/name with empty arguments (real-server shape); subsequent frames append fragments
/// of [`TOOL_CALL_ARG_OPEN`] + placeholder words, so however many frames are emitted the
/// accumulated `arguments` is always an unterminated JSON prefix.
const TOOL_CALL_NAME: &str = "get_current_weather";
const TOOL_CALL_ID: &str = "call_fault_0";
const TOOL_CALL_ARG_OPEN: &str = r#"{"location":"San Francisco, CA","notes":""#;

/// `/v1/completions` has no delta object, so the reasoning/tool-call modes express the
/// same partial payloads as RAW TEXT — which is exactly how they appear on a real
/// base-model completions stream (an unterminated `<think>` block, or a tool call the
/// model is still writing out as JSON). Both accumulate to something unparseable, which
/// is the property those modes exist to produce.
const COMPLETION_REASONING_OPEN: &str = "<think>";
const COMPLETION_TOOL_CALL_OPEN: &str =
    r#"{"name":"get_current_weather","arguments":{"location":"San Francisco, CA","notes":""#;

fn sse_frame(json: &serde_json::Value) -> Bytes {
    Bytes::from(format!("data: {}\n\n", json))
}

fn sse_done() -> Bytes {
    Bytes::from_static(b"data: [DONE]\n\n")
}

fn sse_raw(body: &str) -> Bytes {
    Bytes::from(format!("data: {}\n\n", body))
}

/// A chat.completion.chunk with the given delta. Null fields are omitted, matching the
/// serializer conventions of the normal path (`types::ChunkDelta`).
fn chunk(params: &StreamParams, created: u64, delta: serde_json::Value) -> serde_json::Value {
    envelope(params, created, json!([{"index": 0, "delta": delta}]))
}

/// A text_completion chunk carrying the given text.
fn text_chunk(params: &StreamParams, created: u64, text: String) -> serde_json::Value {
    envelope(params, created, json!([{"index": 0, "text": text}]))
}

fn envelope(params: &StreamParams, created: u64, choices: serde_json::Value) -> serde_json::Value {
    json!({
        "id": params.id,
        "object": params.flavor.object(),
        "created": created,
        "model": params.model,
        "choices": choices,
    })
}

/// The finish_reason frame — usage is NOT attached here; it follows as its own frame (see
/// [`usage_frame`]), which is what both real continuation targets emit.
fn finish_chunk(params: &StreamParams, created: u64) -> serde_json::Value {
    let choice = match params.flavor {
        Flavor::Chat => json!([{"index": 0, "delta": {}, "finish_reason": "stop"}]),
        Flavor::Completion => json!([{"index": 0, "text": "", "finish_reason": "stop"}]),
    };
    envelope(params, created, choice)
}

/// The separate `"choices": []` usage frame that closes a healthy stream when
/// `stream_options.include_usage` was set.
fn usage_frame(params: &StreamParams, created: u64, chunks_sent: u32) -> serde_json::Value {
    let mut value = envelope(params, created, json!([]));
    // 1 placeholder word per chunk stands in for 1 completion token, like the engine
    // path's word-per-token stream.
    value["usage"] = json!({
        "prompt_tokens": params.prompt_tokens,
        "completion_tokens": chunks_sent,
        "total_tokens": params.prompt_tokens + chunks_sent,
    });
    value
}

/// The i-th content-bearing frame for this mode and endpoint flavor.
fn content_frame(
    spec: &FaultSpec,
    params: &StreamParams,
    created: u64,
    i: u32,
) -> serde_json::Value {
    let word = || {
        format!(
            "{} ",
            PLACEHOLDER_WORDS[i as usize % PLACEHOLDER_WORDS.len()]
        )
    };
    match (spec.mode, params.flavor) {
        (FaultMode::MidReasoning, Flavor::Chat) => {
            chunk(params, created, json!({"reasoning_content": word()}))
        }
        (FaultMode::MidReasoning, Flavor::Completion) => text_chunk(
            params,
            created,
            if i == 0 {
                COMPLETION_REASONING_OPEN.to_string()
            } else {
                word()
            },
        ),
        (FaultMode::MidToolCall, Flavor::Chat) if i == 0 => chunk(
            params,
            created,
            json!({"tool_calls": [{
                "index": 0,
                "id": TOOL_CALL_ID,
                "type": "function",
                "function": {"name": TOOL_CALL_NAME, "arguments": ""}
            }]}),
        ),
        (FaultMode::MidToolCall, Flavor::Chat) => {
            // Fragment 1 opens the arguments object; later fragments extend the
            // never-terminated notes string. JSON-string-escaped placeholder words are
            // plain ASCII, so raw concatenation is safe.
            let fragment = if i == 1 {
                TOOL_CALL_ARG_OPEN.to_string()
            } else {
                word()
            };
            chunk(
                params,
                created,
                json!({"tool_calls": [{"index": 0, "function": {"arguments": fragment}}]}),
            )
        }
        (FaultMode::MidToolCall, Flavor::Completion) => text_chunk(
            params,
            created,
            if i == 0 {
                COMPLETION_TOOL_CALL_OPEN.to_string()
            } else {
                word()
            },
        ),
        (_, Flavor::Chat) => chunk(params, created, json!({"content": word()})),
        (_, Flavor::Completion) => text_chunk(params, created, word()),
    }
}

/// The frame that gets torn by `cut_mid_frame`, and where to cut it.
///
/// Plain variant: cut at the byte midpoint — inside the frame's JSON for any realistic
/// frame length. `utf8` variant: the delta text carries 'é' and the cut lands between
/// its two UTF-8 bytes (one past the leading byte of the first multi-byte sequence).
fn torn_frame_prefix(spec: &FaultSpec, params: &StreamParams, created: u64) -> Bytes {
    let content = if spec.utf8 {
        UTF8_CUT_WORD.to_string()
    } else {
        format!(
            "{} ",
            PLACEHOLDER_WORDS[spec.after_chunks as usize % PLACEHOLDER_WORDS.len()]
        )
    };
    let full = sse_frame(&match params.flavor {
        Flavor::Chat => chunk(params, created, json!({"content": content})),
        Flavor::Completion => text_chunk(params, created, content),
    });
    let cut = if spec.utf8 {
        full.iter()
            .position(|b| (b & 0xC0) == 0xC0)
            .map(|p| p + 1)
            .expect("UTF8_CUT_WORD guarantees a multi-byte character in the frame")
    } else {
        full.len() / 2
    };
    full.slice(..cut)
}

async fn send(tx: &mpsc::Sender<Result<Bytes, std::io::Error>>, bytes: Bytes) -> bool {
    tx.send(Ok(bytes)).await.is_ok()
}

/// How long the body waits before aborting, so hyper flushes everything already queued.
/// hyper only writes buffered frames out when the body stream returns Pending; an `Err`
/// polled immediately behind data discards it, and the death would land EARLIER on the
/// wire than scripted (observed empirically: torn frames never left the server without
/// this). The producer sleeping keeps the channel empty -> body Pending -> flush.
const FLUSH_GRACE: Duration = Duration::from_millis(25);

/// Abort the response body: hyper drops the connection without the chunked-encoding
/// terminator, so the client sees a mid-stream close (FIN — or RST if `reset` armed
/// linger first) rather than a clean end.
async fn abort(tx: &mpsc::Sender<Result<Bytes, std::io::Error>>, mode: FaultMode) {
    tokio::time::sleep(FLUSH_GRACE).await;
    let _ = tx
        .send(Err(std::io::Error::new(
            std::io::ErrorKind::ConnectionAborted,
            format!("injected fault: {}", mode.as_str()),
        )))
        .await;
}

/// Produce the faulting stream. Public within serve so tests can drive it directly and
/// assert on the exact wire bytes.
pub(super) async fn run_script(
    spec: FaultSpec,
    params: StreamParams,
    conn: Option<Arc<FaultConnection>>,
    tx: mpsc::Sender<Result<Bytes, std::io::Error>>,
) {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let delay = Duration::from_millis(spec.delay_ms);

    // Initial role frame, exactly like the normal streaming path (not counted in
    // after_chunks). Chat only: `/v1/completions` streams have no role frame, so a
    // completions fault's first frame on the wire is already content.
    if params.flavor == Flavor::Chat {
        let role = chunk(&params, created, json!({"role": "assistant"}));
        if !send(&tx, sse_frame(&role)).await {
            return;
        }
    }

    for i in 0..spec.after_chunks {
        tokio::time::sleep(delay).await;
        if !send(&tx, sse_frame(&content_frame(&spec, &params, created, i))).await {
            return;
        }
    }
    tokio::time::sleep(delay).await;

    match spec.mode {
        FaultMode::CutBetweenFrames | FaultMode::MidReasoning | FaultMode::MidToolCall => {
            abort(&tx, spec.mode).await;
        }
        FaultMode::CutMidFrame => {
            if !send(&tx, torn_frame_prefix(&spec, &params, created)).await {
                return;
            }
            abort(&tx, spec.mode).await;
        }
        FaultMode::Reset => {
            match &conn {
                Some(conn) => {
                    if let Err(e) = conn.arm_reset() {
                        log::warn!(
                            "reset fault: failed to arm SO_LINGER=0 ({e}); closing with FIN"
                        );
                    }
                }
                // Only reachable when the handler is driven outside the serve accept
                // loop (unit tests); the real server always threads the handle.
                None => log::warn!("reset fault: no connection handle; closing with FIN"),
            }
            abort(&tx, spec.mode).await;
        }
        FaultMode::Stall => {
            // Send nothing forever, connection open. Resolves (and releases the task)
            // only when the client gives up and the body is dropped.
            tx.closed().await;
        }
        FaultMode::ErrorEnvelope200 => {
            if send(&tx, sse_raw(OPENROUTER_ERROR_ENVELOPE)).await {
                let _ = send(&tx, sse_done()).await;
            }
            // Clean close (chunked terminator sent).
        }
        FaultMode::Error400InSse => {
            // Error object then clean close — no finish_reason, no [DONE].
            let _ = send(&tx, sse_raw(ERROR_400_IN_SSE)).await;
        }
        FaultMode::NoDone => {
            // Everything a healthy stream ends with, except [DONE]: finish_reason frame,
            // then the separate usage frame if it was asked for.
            if !send(&tx, sse_frame(&finish_chunk(&params, created))).await {
                return;
            }
            if params.include_usage {
                let usage = usage_frame(&params, created, spec.after_chunks);
                let _ = send(&tx, sse_frame(&usage)).await;
            }
        }
        FaultMode::NoUsage => {
            // finish_reason and [DONE], but the usage frame never arrives even though
            // include_usage asked for it.
            if send(&tx, sse_frame(&finish_chunk(&params, created))).await {
                let _ = send(&tx, sse_done()).await;
            }
        }
        FaultMode::Cancelled499 => {
            // Exact dynamo frontend body, then clean close.
            let _ = send(&tx, sse_raw(CANCELLED_499)).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> StreamParams {
        StreamParams {
            id: "chatcmpl-test".to_string(),
            model: "test-model".to_string(),
            prompt_tokens: 7,
            include_usage: true,
            flavor: Flavor::Chat,
        }
    }

    /// `/v1/completions` flavor — the shape a mid-stream-continuation RESUME LEG dies in.
    fn completion_params() -> StreamParams {
        StreamParams {
            id: "cmpl-test".to_string(),
            model: "test-model".to_string(),
            prompt_tokens: 7,
            include_usage: true,
            flavor: Flavor::Completion,
        }
    }

    fn spec(mode: FaultMode) -> FaultSpec {
        FaultSpec {
            mode,
            after_chunks: 3,
            delay_ms: 0,
            utf8: false,
        }
    }

    /// Drive the script to completion and collect every item it produced.
    async fn collect(spec: FaultSpec, params: StreamParams) -> Vec<Result<Bytes, std::io::Error>> {
        let (tx, mut rx) = mpsc::channel(16);
        let task = tokio::spawn(run_script(spec, params, None, tx));
        let mut items = Vec::new();
        while let Some(item) = rx.recv().await {
            items.push(item);
        }
        task.await.unwrap();
        items
    }

    fn frames(items: &[Result<Bytes, std::io::Error>]) -> Vec<String> {
        items
            .iter()
            .map(|r| String::from_utf8(r.as_ref().expect("expected Ok frame").to_vec()).unwrap())
            .collect()
    }

    fn frame_json(frame: &str) -> serde_json::Value {
        let data = frame
            .strip_prefix("data: ")
            .and_then(|f| f.strip_suffix("\n\n"))
            .expect("well-formed SSE frame");
        serde_json::from_str(data).expect("frame data is JSON")
    }

    // --- Header parsing ---

    #[test]
    fn parse_mode_only_uses_defaults() {
        let spec = FaultSpec::parse_header("cut_between_frames").unwrap();
        assert_eq!(spec.mode, FaultMode::CutBetweenFrames);
        assert_eq!(spec.after_chunks, DEFAULT_AFTER_CHUNKS);
        assert_eq!(spec.delay_ms, DEFAULT_DELAY_MS);
        assert!(!spec.utf8);
    }

    #[test]
    fn parse_full_grammar() {
        let spec = FaultSpec::parse_header("cut_mid_frame; after_chunks=12; delay_ms=0; utf8=true")
            .unwrap();
        assert_eq!(spec.mode, FaultMode::CutMidFrame);
        assert_eq!(spec.after_chunks, 12);
        assert_eq!(spec.delay_ms, 0);
        assert!(spec.utf8);
    }

    #[test]
    fn parse_every_mode_name_round_trips() {
        for mode in FaultMode::ALL {
            assert_eq!(FaultSpec::parse_header(mode.as_str()).unwrap().mode, *mode);
        }
    }

    #[test]
    fn parse_rejects_unknown_mode_and_key_and_values() {
        assert!(FaultSpec::parse_header("explode").is_err());
        assert!(FaultSpec::parse_header("").is_err());
        assert!(FaultSpec::parse_header("stall;bogus=1").is_err());
        assert!(FaultSpec::parse_header("stall;after_chunks=lots").is_err());
        assert!(FaultSpec::parse_header("stall;after_chunks").is_err());
    }

    #[test]
    fn parse_rejects_utf8_outside_cut_mid_frame() {
        assert!(FaultSpec::parse_header("stall;utf8=true").is_err());
        assert!(FaultSpec::parse_header("cut_mid_frame;utf8=true").is_ok());
    }

    #[test]
    fn from_config_validates_and_defaults() {
        let cfg = crate::config::FaultConfig {
            mode: "no_usage".to_string(),
            after_chunks: Some(5),
            delay_ms: None,
            utf8: None,
        };
        let spec = FaultSpec::from_config(&cfg).unwrap();
        assert_eq!(spec.mode, FaultMode::NoUsage);
        assert_eq!(spec.after_chunks, 5);
        assert_eq!(spec.delay_ms, DEFAULT_DELAY_MS);

        let bad = crate::config::FaultConfig {
            mode: "nope".to_string(),
            after_chunks: None,
            delay_ms: None,
            utf8: None,
        };
        assert!(FaultSpec::from_config(&bad).is_err());
    }

    // --- Script wire behavior ---

    #[tokio::test]
    async fn cut_between_frames_emits_n_frames_then_aborts() {
        let items = collect(spec(FaultMode::CutBetweenFrames), params()).await;
        // role + 3 content frames + Err
        assert_eq!(items.len(), 5);
        assert!(items.last().unwrap().is_err());
        let frames = frames(&items[..4]);
        assert_eq!(
            frame_json(&frames[0])["choices"][0]["delta"]["role"],
            "assistant"
        );
        for f in &frames[1..] {
            assert!(frame_json(f)["choices"][0]["delta"]["content"].is_string());
        }
    }

    #[tokio::test]
    async fn cut_mid_frame_tears_the_next_frame() {
        let items = collect(spec(FaultMode::CutMidFrame), params()).await;
        assert_eq!(items.len(), 6); // role + 3 + torn prefix + Err
        assert!(items.last().unwrap().is_err());
        let torn = items[4].as_ref().unwrap();
        let torn_str = String::from_utf8(torn.to_vec()).unwrap();
        assert!(torn_str.starts_with("data: {"));
        assert!(!torn_str.ends_with("\n\n"), "must not be a complete frame");
        assert!(
            serde_json::from_str::<serde_json::Value>(&torn_str[6..]).is_err(),
            "torn JSON must not parse"
        );
    }

    #[tokio::test]
    async fn cut_mid_frame_utf8_variant_cuts_inside_a_character() {
        let mut s = spec(FaultMode::CutMidFrame);
        s.utf8 = true;
        let items = collect(s, params()).await;
        let torn = items[items.len() - 2].as_ref().unwrap();
        // The prefix must end exactly one byte into a multi-byte UTF-8 sequence: last
        // byte is a leading byte (11xxxxxx) with its continuation missing.
        assert_eq!(torn.last().unwrap() & 0xC0, 0xC0);
        assert!(String::from_utf8(torn.to_vec()).is_err());
    }

    #[tokio::test]
    async fn reset_without_handle_still_aborts() {
        let items = collect(spec(FaultMode::Reset), params()).await;
        assert!(items.last().unwrap().is_err());
    }

    #[tokio::test]
    async fn arm_reset_sets_linger_on_a_real_socket() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (client, (server, _)) = tokio::join!(tokio::net::TcpStream::connect(addr), async {
            listener.accept().await.unwrap()
        });
        let _client = client.unwrap();
        let conn = FaultConnection::new(&server).unwrap();
        conn.arm_reset().unwrap();
    }

    /// The regression the dup'd fd exists for: arming reset after hyper has already
    /// closed its `TcpStream` must hit our own (still-open) dup, not a recycled fd.
    #[tokio::test]
    async fn arm_reset_is_safe_after_original_stream_closes() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (client, (server, _)) = tokio::join!(tokio::net::TcpStream::connect(addr), async {
            listener.accept().await.unwrap()
        });
        let _client = client.unwrap();
        let conn = FaultConnection::new(&server).unwrap();
        drop(server); // client disconnect path: hyper drops the stream first
        conn.arm_reset().unwrap();
    }

    #[tokio::test]
    async fn stall_emits_frames_then_holds_until_client_leaves() {
        let (tx, mut rx) = mpsc::channel(16);
        let task = tokio::spawn(run_script(spec(FaultMode::Stall), params(), None, tx));
        for _ in 0..4 {
            assert!(rx.recv().await.unwrap().is_ok()); // role + 3 content frames
        }
        // Nothing further is ever produced...
        assert!(matches!(
            rx.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
        // ...until the client (receiver) goes away, which releases the task.
        drop(rx);
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("stall task must end when the body is dropped")
            .unwrap();
    }

    #[tokio::test]
    async fn error_envelope_200_emits_openrouter_shape_then_done() {
        let items = collect(spec(FaultMode::ErrorEnvelope200), params()).await;
        let frames = frames(&items); // clean close: all Ok
        assert_eq!(frames.len(), 6); // role + 3 + error + [DONE]
        let err = frame_json(&frames[4]);
        assert_eq!(err["error"]["code"], 502);
        assert_eq!(err["error"]["metadata"]["provider_name"], "inference-lab");
        assert_eq!(frames[5], "data: [DONE]\n\n");
    }

    #[tokio::test]
    async fn error_400_in_sse_emits_error_object_without_done() {
        let items = collect(spec(FaultMode::Error400InSse), params()).await;
        let frames = frames(&items);
        assert_eq!(frames.len(), 5); // role + 3 + error; no [DONE]
        let err = frame_json(&frames[4]);
        assert_eq!(err["object"], "error");
        assert_eq!(err["code"], 400);
        assert_eq!(err["type"], "BadRequestError");
    }

    #[tokio::test]
    async fn no_done_finishes_with_usage_but_never_done() {
        let items = collect(spec(FaultMode::NoDone), params()).await;
        let frames = frames(&items);
        assert_eq!(frames.len(), 6); // role + 3 + finish + usage; no [DONE]
        let finish = frame_json(&frames[4]);
        assert_eq!(finish["choices"][0]["finish_reason"], "stop");
        assert!(
            finish.get("usage").is_none(),
            "usage rides its own frame, never the finish_reason chunk"
        );
        // Usage arrives as the separate choices:[] frame both real continuation targets
        // (dynamo, Fireworks) emit.
        let usage = frame_json(&frames[5]);
        assert_eq!(usage["choices"].as_array().unwrap().len(), 0);
        assert_eq!(usage["usage"]["prompt_tokens"], 7);
        assert_eq!(usage["usage"]["completion_tokens"], 3);
        assert!(!frames.iter().any(|f| f.contains("[DONE]")));
    }

    #[tokio::test]
    async fn no_usage_sends_done_but_drops_requested_usage() {
        let items = collect(spec(FaultMode::NoUsage), params()).await; // include_usage: true
        let frames = frames(&items);
        assert_eq!(frames.len(), 6); // role + 3 + finish + [DONE]; the usage frame is missing
        let finish = frame_json(&frames[4]);
        assert_eq!(finish["choices"][0]["finish_reason"], "stop");
        assert!(!frames.iter().any(|f| f.contains("\"usage\"")));
        assert_eq!(frames[5], "data: [DONE]\n\n");
    }

    // --- Completion flavor (resume-leg deaths) ---

    #[tokio::test]
    async fn completion_flavor_emits_text_frames_with_no_role_frame() {
        // A resume leg is a streaming /v1/completions request: text_completion frames
        // carrying `text`, and no role frame at all — so `after_chunks=N` puts exactly N
        // content frames on the wire before the death.
        let items = collect(spec(FaultMode::CutBetweenFrames), completion_params()).await;
        assert_eq!(items.len(), 4); // 3 content frames + Err, no role frame
        assert!(items.last().unwrap().is_err());
        for f in frames(&items[..3]) {
            let json = frame_json(&f);
            assert_eq!(json["object"], "text_completion");
            assert!(json["choices"][0]["text"].is_string());
            assert!(json["choices"][0].get("delta").is_none());
        }
    }

    #[tokio::test]
    async fn completion_flavor_no_done_ends_with_finish_then_usage() {
        let items = collect(spec(FaultMode::NoDone), completion_params()).await;
        let frames = frames(&items);
        assert_eq!(frames.len(), 5); // 3 content + finish + usage; no [DONE]
        let finish = frame_json(&frames[3]);
        assert_eq!(finish["object"], "text_completion");
        assert_eq!(finish["choices"][0]["finish_reason"], "stop");
        assert_eq!(finish["choices"][0]["text"], "");
        let usage = frame_json(&frames[4]);
        assert_eq!(usage["choices"].as_array().unwrap().len(), 0);
        assert_eq!(usage["usage"]["completion_tokens"], 3);
    }

    #[tokio::test]
    async fn completion_flavor_error_envelope_and_499_bodies_are_unchanged() {
        // The error bodies are downstream signatures, not endpoint shapes: they must be
        // byte-identical on both endpoints so a client's death classifier works either way.
        let envelope = collect(spec(FaultMode::ErrorEnvelope200), completion_params()).await;
        let envelope_frames = frames(&envelope);
        assert_eq!(envelope_frames.len(), 5); // 3 content + error + [DONE]
        assert_eq!(frame_json(&envelope_frames[3])["error"]["code"], 502);
        assert_eq!(envelope_frames[4], "data: [DONE]\n\n");

        let cancelled = collect(spec(FaultMode::Cancelled499), completion_params()).await;
        assert_eq!(
            frames(&cancelled).last().unwrap(),
            "data: {\"error\":{\"code\":499,\"message\":\"CancelledError: \",\"type\":\"request_cancelled\"}}\n\n"
        );
    }

    #[tokio::test]
    async fn completion_flavor_cut_mid_frame_tears_a_text_frame() {
        let items = collect(spec(FaultMode::CutMidFrame), completion_params()).await;
        assert_eq!(items.len(), 5); // 3 content + torn prefix + Err
        assert!(items.last().unwrap().is_err());
        let torn = String::from_utf8(items[3].as_ref().unwrap().to_vec()).unwrap();
        assert!(torn.starts_with("data: {"));
        assert!(!torn.ends_with("\n\n"));
        assert!(serde_json::from_str::<serde_json::Value>(&torn[6..]).is_err());
    }

    #[tokio::test]
    async fn completion_flavor_reasoning_and_tool_call_accumulate_unterminated_text() {
        // No delta object exists on /v1/completions, so these modes stream the same
        // partial payloads as raw text — an unterminated <think> block and a tool call
        // the model never finished writing.
        let accumulate = |items: &[Result<Bytes, std::io::Error>]| -> String {
            frames(&items[..items.len() - 1])
                .iter()
                .map(|f| {
                    frame_json(f)["choices"][0]["text"]
                        .as_str()
                        .unwrap()
                        .to_string()
                })
                .collect()
        };

        let reasoning = collect(spec(FaultMode::MidReasoning), completion_params()).await;
        assert!(reasoning.last().unwrap().is_err());
        let text = accumulate(&reasoning);
        assert!(text.starts_with(COMPLETION_REASONING_OPEN), "got: {text}");
        assert!(!text.contains("</think>"), "reasoning must stay open");

        let tool_call = collect(spec(FaultMode::MidToolCall), completion_params()).await;
        assert!(tool_call.last().unwrap().is_err());
        let args = accumulate(&tool_call);
        assert!(args.starts_with('{'));
        assert!(
            serde_json::from_str::<serde_json::Value>(&args).is_err(),
            "partial tool call must be unterminated JSON, got: {args}"
        );
    }

    #[tokio::test]
    async fn cancelled_499_emits_the_exact_dynamo_body() {
        let items = collect(spec(FaultMode::Cancelled499), params()).await;
        let frames = frames(&items);
        assert_eq!(
            frames.last().unwrap(),
            "data: {\"error\":{\"code\":499,\"message\":\"CancelledError: \",\"type\":\"request_cancelled\"}}\n\n"
        );
    }

    #[tokio::test]
    async fn mid_reasoning_streams_reasoning_deltas_then_aborts() {
        let items = collect(spec(FaultMode::MidReasoning), params()).await;
        assert!(items.last().unwrap().is_err());
        let frames = frames(&items[..items.len() - 1]);
        for f in &frames[1..] {
            let delta = &frame_json(f)["choices"][0]["delta"];
            assert!(delta["reasoning_content"].is_string());
            assert!(delta.get("content").is_none());
        }
    }

    #[tokio::test]
    async fn mid_tool_call_dies_with_arguments_unterminated() {
        let items = collect(spec(FaultMode::MidToolCall), params()).await;
        assert!(items.last().unwrap().is_err());
        let frames = frames(&items[..items.len() - 1]);
        let first = frame_json(&frames[1]);
        let tc = &first["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tc["id"], TOOL_CALL_ID);
        assert_eq!(tc["function"]["name"], TOOL_CALL_NAME);
        // Accumulate the argument fragments the way a client would.
        let args: String = frames[1..]
            .iter()
            .map(|f| {
                frame_json(f)["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert!(args.starts_with('{'));
        assert!(
            serde_json::from_str::<serde_json::Value>(&args).is_err(),
            "partial tool-call arguments must be unterminated JSON, got: {args}"
        );
    }

    #[tokio::test]
    async fn after_chunks_zero_dies_right_after_the_role_frame() {
        let mut s = spec(FaultMode::CutBetweenFrames);
        s.after_chunks = 0;
        let items = collect(s, params()).await;
        assert_eq!(items.len(), 2); // role + Err
        assert!(items[0].is_ok());
        assert!(items[1].is_err());
    }

    #[tokio::test]
    async fn same_spec_produces_identical_frame_bodies() {
        // Determinism: everything except id/created (which the handler supplies) is
        // byte-identical run to run.
        let a = collect(spec(FaultMode::CutMidFrame), params()).await;
        let b = collect(spec(FaultMode::CutMidFrame), params()).await;
        let strip = |items: &[Result<Bytes, std::io::Error>]| -> Vec<Vec<u8>> {
            items
                .iter()
                .filter_map(|r| r.as_ref().ok())
                .map(|bytes| {
                    let s = String::from_utf8_lossy(bytes).into_owned();
                    // created varies with the clock; normalize it out.
                    let re_stripped = s
                        .split("\"created\":")
                        .enumerate()
                        .map(|(i, part)| {
                            if i == 0 {
                                part.to_string()
                            } else {
                                part.split_once(',').map(|x| x.1).unwrap_or("").to_string()
                            }
                        })
                        .collect::<String>();
                    re_stripped.into_bytes()
                })
                .collect()
        };
        assert_eq!(strip(&a), strip(&b));
    }
}
