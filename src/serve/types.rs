use serde::{Deserialize, Serialize};

// --- Request types ---

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Tool definitions. Kept as raw JSON purely for token accounting: a real engine's
    /// chat template renders tool schemas into the prompt, so they must count.
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: CompletionPrompt,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Scheduling hint. Accepted and ignored: every mid-stream-continuation resume leg
    /// carries one, and a serde error here would 400 the whole resume before it reached
    /// the engine. Typed (not swallowed as an unknown field) so a wrong-typed value still
    /// fails like a real engine's validation.
    #[serde(default)]
    pub priority: Option<i64>,
    /// Return the prompt alongside the completion. On id prompts this echoes the received
    /// ids back as `prompt_token_ids` (Fireworks behavior), which is what makes an
    /// id-prompt round trip verifiable end to end.
    #[serde(default)]
    pub echo: bool,
}

/// OpenAI-compatible `/v1/completions` `prompt`: text, a batch of texts, token ids, or a
/// batch of token-id lists.
///
/// Token-id prompts are the load-bearing case for mid-stream continuation: a resume leg
/// re-sends the prefix as ids so no detokenize/retokenize round trip can perturb it, and
/// the continuation billing merge computes `prompt = P_reported − seg` from the reported
/// `prompt_tokens`. So for id prompts `usage.prompt_tokens` MUST equal the number of ids
/// sent (Fireworks-verified), not an estimate — see [`Self::token_ids`].
///
/// Variant order matters: serde tries untagged variants top down, so the string forms are
/// attempted before the integer ones and `[[…]]` only ever lands on [`Self::IdBatch`].
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum CompletionPrompt {
    Text(String),
    Texts(Vec<String>),
    Ids(Vec<u32>),
    IdBatch(Vec<Vec<u32>>),
}

impl CompletionPrompt {
    /// Number of prompts in this request. Engines that accept batched prompts return one
    /// choice per entry; this sim serves batch-of-one only and rejects anything larger
    /// (see `handlers::completions`).
    pub fn batch_len(&self) -> usize {
        match self {
            CompletionPrompt::Text(_) => 1,
            CompletionPrompt::Texts(v) => v.len(),
            CompletionPrompt::Ids(_) => 1,
            CompletionPrompt::IdBatch(v) => v.len(),
        }
    }

    /// The token ids for an id-form prompt, whose LENGTH is the exact `prompt_tokens` this
    /// request must report. `None` for text prompts, which are counted by the tokenizer or
    /// the char estimator instead.
    pub fn token_ids(&self) -> Option<&[u32]> {
        match self {
            CompletionPrompt::Ids(ids) => Some(ids),
            // Batch-of-one is the only batch shape that reaches the engine.
            CompletionPrompt::IdBatch(batch) => {
                Some(batch.first().map(Vec::as_slice).unwrap_or(&[]))
            }
            _ => None,
        }
    }

    /// Prompt text for token counting. Empty for id prompts: this deployment mounts no
    /// tokenizer, so ids cannot be detokenized — only the ACCOUNTING is id-exact, while
    /// generated text is simulated exactly as it is for a string prompt.
    pub fn text(&self) -> &str {
        match self {
            CompletionPrompt::Text(t) => t,
            CompletionPrompt::Texts(v) => v.first().map(String::as_str).unwrap_or(""),
            _ => "",
        }
    }
}

fn default_max_tokens() -> u32 {
    256
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// Assistant tool calls, kept as raw JSON for token accounting (rendered into the
    /// prompt by real chat templates, and counted by gateways that meter cache prefixes).
    #[serde(default)]
    pub tool_calls: Option<serde_json::Value>,
    /// OpenAI-compatible `content`: a plain string, an array of content parts, `null`, or
    /// omitted entirely. The old bare-`String` field rejected everything but the string form
    /// with a 400, which broke any client sending part-form content (multimodal messages, or
    /// gateways emitting single-text-part arrays) before it ever reached the
    /// request→chat-template→tokenize chain. `#[serde(default)]` is deliberate: per the
    /// OpenAI spec, `content` is "required unless `tool_calls` is specified" — assistant
    /// tool-call turns may OMIT the field (not just send `null`), and real model servers
    /// accept that shape, so rejecting omission would reintroduce the same class of 400.
    #[serde(default)]
    pub content: Option<MessageContent>,
}

/// String-or-parts `content`, matching what OpenAI-compatible servers accept.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// One entry of array-form content. Only `text` contributes to the prompt; non-text parts
/// (e.g. `image_url`) and any extra fields on a part are accepted and ignored, like a real
/// model server.
#[derive(Debug, Clone, Deserialize)]
pub struct ContentPart {
    #[serde(default)]
    pub text: Option<String>,
}

impl MessageContent {
    /// The message text for prompt assembly / token counting. Part texts are concatenated
    /// with NO separator, so a single-text-part array counts identically to the same plain
    /// string — the shape equivalence real chat templates give. Borrows for the common
    /// string form (load tests send multi-hundred-KB prompts; cloning each message would
    /// double transient memory) and allocates only to join parts.
    pub fn text(&self) -> std::borrow::Cow<'_, str> {
        match self {
            MessageContent::Text(t) => std::borrow::Cow::Borrowed(t),
            MessageContent::Parts(parts) => {
                std::borrow::Cow::Owned(parts.iter().filter_map(|p| p.text.as_deref()).collect())
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

// --- Non-streaming response ---

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChoiceMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ChoiceMessage {
    pub role: &'static str,
    /// `null` (not omitted) on pure tool-call turns, matching real model servers.
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ResponseToolCall>>,
}

/// OpenAI response-shaped tool call (directive mode; see serve::directive).
#[derive(Debug, Clone, Serialize)]
pub struct ResponseToolCall {
    pub id: String,
    pub r#type: &'static str,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// JSON-encoded arguments object, per the OpenAI wire format.
    pub arguments: String,
}

/// Streaming tool-call delta: the whole call in one chunk at its index (a real server
/// fragments `arguments` across chunks; emitting it whole is valid and deterministic).
#[derive(Debug, Clone, Serialize)]
pub struct StreamToolCall {
    pub index: u32,
    pub id: String,
    pub r#type: &'static str,
    pub function: ToolCallFunction,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: &'static str,
    /// `echo` on an id prompt: the ids exactly as received, so a caller can verify the
    /// prefix survived the round trip unchanged (Fireworks behavior). Omitted otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_token_ids: Option<Vec<u32>>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// --- Streaming response ---

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: ChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<StreamToolCall>>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChunkChoice {
    pub text: String,
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<&'static str>,
}

// --- Models endpoint ---

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelEntry>,
}

#[derive(Debug, Serialize)]
pub struct ModelEntry {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

// --- Runtime capacity control ---

/// Body of `POST /control/capacity`. Every field is optional: `model`
/// defaults to every loaded model, and each knob is left alone when absent,
/// so one knob can be turned without restating the other.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityUpdate {
    /// Which model to retune. Absent = all of them.
    #[serde(default)]
    pub model: Option<String>,

    /// New waiting-queue bound. `0` disables saturation rejection, which is
    /// how this knob is normally used: not to tune the queue length, but to
    /// turn 529s on and off against a running server.
    #[serde(default)]
    pub max_waiting: Option<u32>,

    /// New per-worker concurrency cap — the scale up/down. This is the knob
    /// that changes how fast work actually drains, and so the offered load
    /// at which the queue backs up. Must be at least 1.
    #[serde(default)]
    pub max_num_seqs: Option<u32>,
}

/// One model's capacity, as reported by `GET`/`POST /control/capacity`.
#[derive(Debug, Serialize)]
pub struct CapacityState {
    pub model: String,
    /// The waiting-queue bound; `0` means unbounded (no 529s).
    pub max_waiting: u32,
    /// The concurrency cap now in force.
    pub max_num_seqs: u32,
    /// Requests queued right now, against which `max_waiting` is compared.
    pub waiting: usize,
    /// Requests running right now.
    pub running: usize,
}

// --- Engine communication ---

pub struct EngineRequest {
    pub request_id: String,
    pub prompt_tokens: u32,
    pub max_output_tokens: u32,
    pub tx: tokio::sync::mpsc::Sender<TokenEvent>,
}

#[derive(Debug, Clone)]
pub enum TokenEvent {
    FirstToken,
    Token {
        text: String,
    },
    Done {
        prompt_tokens: u32,
        completion_tokens: u32,
    },
    Error {
        message: String,
    },
}
