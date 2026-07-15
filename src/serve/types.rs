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
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
}

fn default_max_tokens() -> u32 {
    256
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// OpenAI-compatible `content`: a plain string, an array of content parts, or `null`
    /// (assistant tool-call turns send `content: null`). Real servers accept all three; the
    /// old bare-`String` field rejected array/null bodies with a 400, which broke any client
    /// sending part-form content (multimodal messages, or gateways emitting single-text-part
    /// arrays) before it ever reached the request→chat-template→tokenize chain.
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
    /// string — the shape equivalence real chat templates give.
    pub fn text(&self) -> String {
        match self {
            MessageContent::Text(t) => t.clone(),
            MessageContent::Parts(parts) => {
                parts.iter().filter_map(|p| p.text.as_deref()).collect()
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
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: &'static str,
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
