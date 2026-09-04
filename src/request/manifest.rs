//! Batchbench replay manifests.
//!
//! The simulator consumes the same JSONL schema as `batchbench-agent`: one
//! trajectory per line, with exact first-arrival offsets and per-request token
//! counts, delays, resets, and optional content-block identities.

use serde::Deserialize;
use serde_json::Value;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::BufRead;

const SUPPORTED_SCHEMA_VERSIONS: [u32; 2] = [1, 2];
const STABLE_HASH_NAMESPACE: u64 = 1 << 63;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayManifest {
    pub schema_version: u32,
    pub trajectory_id: String,
    pub requests: Vec<ReplayRequest>,
    #[serde(default)]
    pub start_after_ms: Option<u64>,
    #[serde(default, rename = "metadata")]
    _metadata: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayRequest {
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    #[serde(default)]
    pub reset_before: bool,
    #[serde(default)]
    pub delay_after_ms: u64,
    #[serde(default)]
    pub overhead_tokens: Option<u32>,
    #[serde(default)]
    #[serde(rename = "stream")]
    _stream: Option<bool>,
    #[serde(default)]
    #[serde(rename = "max_tokens")]
    _max_tokens: Option<u32>,
    #[serde(default)]
    pub blocks: Option<Vec<ReplayBlock>>,
}

impl ReplayRequest {
    pub(crate) fn expected_prompt_tokens(&self) -> Option<u64> {
        self.blocks.as_ref().map(|blocks| {
            blocks
                .iter()
                .map(|block| u64::from(block.tokens))
                .sum::<u64>()
                + u64::from(self.overhead_tokens.unwrap_or(0))
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayBlock {
    pub seed: String,
    pub tokens: u32,
    pub role: ReplayBlockRole,
    #[serde(default)]
    pub live: bool,
}

#[derive(Debug, Clone, Copy, Deserialize, Hash, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplayBlockRole {
    ToolDefinition,
    System,
    User,
    Assistant,
    Tool,
    ToolCall,
}

impl ReplayBlockRole {
    pub(crate) fn step_kind(self) -> Option<String> {
        match self {
            Self::User => Some("user".into()),
            Self::Tool | Self::ToolCall => Some("tool".into()),
            _ => None,
        }
    }
}

impl ReplayManifest {
    /// Load a batchbench JSONL replay manifest.
    pub fn load(path: &str) -> Result<Vec<Self>, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
        Self::read_jsonl(std::io::BufReader::new(file)).map_err(|e| format!("{path}: {e}"))
    }

    pub fn read_jsonl<R: BufRead>(reader: R) -> Result<Vec<Self>, String> {
        let mut manifests = Vec::new();
        for (index, line) in reader.lines().enumerate() {
            let line_number = index + 1;
            let line = line.map_err(|e| format!("line {line_number}: {e}"))?;
            if line.trim().is_empty() {
                continue;
            }
            let manifest: Self =
                serde_json::from_str(&line).map_err(|e| format!("line {line_number}: {e}"))?;
            manifest
                .validate()
                .map_err(|e| format!("line {line_number}: {e}"))?;
            manifests.push(manifest);
        }
        if manifests.is_empty() {
            return Err("no trajectories".into());
        }
        Ok(manifests)
    }

    fn validate(&self) -> Result<(), String> {
        if !SUPPORTED_SCHEMA_VERSIONS.contains(&self.schema_version) {
            return Err(format!(
                "schema_version {} is unsupported; expected 1 or 2",
                self.schema_version
            ));
        }
        if self.trajectory_id.trim().is_empty() {
            return Err("trajectory_id must not be empty".into());
        }
        if self.requests.is_empty() {
            return Err("requests must contain at least one request".into());
        }
        if self.schema_version == 1 && self.start_after_ms.is_some() {
            return Err("start_after_ms requires schema_version 2".into());
        }
        if self.schema_version == 1 && self.requests[0].reset_before {
            return Err("the first request cannot set reset_before in schema version 1".into());
        }

        let uses_blocks = self.requests[0].blocks.is_some();
        let mut seed_definitions = std::collections::HashMap::new();
        for (index, request) in self.requests.iter().enumerate() {
            let number = index + 1;
            if request.prompt_tokens == 0 {
                return Err(format!("request {number} prompt_tokens must be positive"));
            }
            if request.output_tokens == 0 {
                return Err(format!("request {number} output_tokens must be positive"));
            }
            if request._max_tokens == Some(0) {
                return Err(format!("request {number} max_tokens must be positive"));
            }
            if self.schema_version == 1
                && (request.overhead_tokens.is_some()
                    || request._stream.is_some()
                    || request._max_tokens.is_some()
                    || request.blocks.is_some())
            {
                return Err(format!(
                    "request {number} uses fields that require schema_version 2"
                ));
            }
            if request.blocks.is_some() != uses_blocks {
                return Err(format!(
                    "request {number} must {} blocks like the first request",
                    if uses_blocks { "define" } else { "omit" }
                ));
            }
            if let Some(blocks) = &request.blocks {
                if blocks.is_empty() {
                    return Err(format!("request {number} blocks must not be empty"));
                }
                for (block_index, block) in blocks.iter().enumerate() {
                    if block.seed.is_empty() {
                        return Err(format!(
                            "request {number} block {} seed must not be empty",
                            block_index + 1
                        ));
                    }
                    if block.live && block.role != ReplayBlockRole::Assistant {
                        return Err(format!(
                            "request {number} block {} is live but is not assistant",
                            block_index + 1
                        ));
                    }
                    match seed_definitions.insert(block.seed.as_str(), (block.tokens, block.role)) {
                        Some(previous) if previous != (block.tokens, block.role) => {
                            return Err(format!(
                                "request {number} block {} redefines seed {:?}",
                                block_index + 1,
                                block.seed
                            ));
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    }
}

/// The manifest's content identity for one prompt. Batchbench records only a
/// count for chat-template overhead, not its token positions, so the simulator
/// conservatively appends one deterministic template run after the blocks.
#[derive(Debug, Clone)]
pub(crate) struct PromptIdentity {
    blocks: Vec<ReplayBlock>,
    overhead_tokens: u32,
}

impl PromptIdentity {
    pub(crate) fn from_request(request: &ReplayRequest) -> Option<Self> {
        request.blocks.clone().map(|blocks| Self {
            overhead_tokens: request.overhead_tokens.unwrap_or_else(|| {
                request.prompt_tokens.saturating_sub(
                    blocks
                        .iter()
                        .fold(0_u32, |total, block| total.saturating_add(block.tokens)),
                )
            }),
            blocks,
        })
    }

    pub(crate) fn step_kind(&self) -> Option<String> {
        self.blocks
            .iter()
            .rev()
            .find_map(|block| block.role.step_kind())
    }

    /// Stable incremental hashes for every complete KV block in the prompt.
    /// Equal ordered manifest blocks therefore share cache identity across
    /// trajectories without reconstructing or retaining prompt text.
    pub(crate) fn full_block_hashes(&self, prompt_tokens: u32, block_size: u32) -> Vec<u64> {
        let block_size = block_size.max(1);
        let full_tokens = prompt_tokens / block_size * block_size;
        let mut state = StablePromptHasher::new(block_size, full_tokens);
        for block in &self.blocks {
            state.feed_content(block);
        }
        state.feed_template(self.overhead_tokens);
        state.feed_unattributed();
        state.hashes
    }
}

struct StablePromptHasher {
    block_size: u32,
    limit: u32,
    position: u32,
    hasher: DefaultHasher,
    hashes: Vec<u64>,
}

impl StablePromptHasher {
    fn new(block_size: u32, limit: u32) -> Self {
        Self {
            block_size,
            limit,
            position: 0,
            hasher: DefaultHasher::new(),
            hashes: Vec::with_capacity((limit / block_size) as usize),
        }
    }

    fn feed_content(&mut self, block: &ReplayBlock) {
        let marker = 0_u8;
        let mut offset = 0_u32;
        self.feed_run(block.tokens, |hasher, take| {
            marker.hash(hasher);
            block.role.hash(hasher);
            block.seed.hash(hasher);
            offset.hash(hasher);
            take.hash(hasher);
            offset += take;
        });
    }

    fn feed_template(&mut self, tokens: u32) {
        let marker = 1_u8;
        let mut offset = 0_u32;
        self.feed_run(tokens, |hasher, take| {
            marker.hash(hasher);
            offset.hash(hasher);
            take.hash(hasher);
            offset += take;
        });
    }

    fn feed_unattributed(&mut self) {
        let marker = 2_u8;
        let remaining = self.limit.saturating_sub(self.position);
        let mut offset = 0_u32;
        self.feed_run(remaining, |hasher, take| {
            marker.hash(hasher);
            offset.hash(hasher);
            take.hash(hasher);
            offset += take;
        });
    }

    fn feed_run(&mut self, tokens: u32, mut feed: impl FnMut(&mut DefaultHasher, u32)) {
        let mut remaining = tokens.min(self.limit.saturating_sub(self.position));
        while remaining > 0 {
            let in_block = self.position % self.block_size;
            let take = remaining.min(self.block_size - in_block);
            feed(&mut self.hasher, take);
            self.position += take;
            remaining -= take;
            if self.position.is_multiple_of(self.block_size) {
                self.hashes
                    .push(self.hasher.clone().finish() | STABLE_HASH_NAMESPACE);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &str = concat!(
        r#"{"schema_version":2,"trajectory_id":"a","start_after_ms":1250,"requests":["#,
        r#"{"prompt_tokens":40,"output_tokens":8,"overhead_tokens":8,"delay_after_ms":250,"blocks":["#,
        r#"{"seed":"sys","tokens":16,"role":"system"},{"seed":"u1","tokens":16,"role":"user"}]},"#,
        r#"{"prompt_tokens":56,"output_tokens":4,"overhead_tokens":8,"blocks":["#,
        r#"{"seed":"sys","tokens":16,"role":"system"},{"seed":"u1","tokens":16,"role":"user"},"#,
        r#"{"seed":"a1","tokens":8,"role":"assistant","live":true},{"seed":"tool","tokens":8,"role":"tool"}]}]}"#,
        "\n"
    );

    #[test]
    fn reads_batchbench_jsonl() {
        let plans = ReplayManifest::read_jsonl(MANIFEST.as_bytes()).unwrap();
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].start_after_ms, Some(1250));
        assert_eq!(plans[0].requests[0].delay_after_ms, 250);
        assert!(plans[0].requests[1].blocks.as_ref().unwrap()[2].live);
    }

    #[test]
    fn rejects_unknown_and_invalid_fields() {
        assert!(ReplayManifest::read_jsonl(
            MANIFEST
                .replace("\"trajectory_id\"", "\"surprise\":1,\"trajectory_id\"")
                .as_bytes()
        )
        .is_err());
        assert!(ReplayManifest::read_jsonl(
            MANIFEST.replace("\"assistant\"", "\"user\"").as_bytes()
        )
        .is_err());
    }

    #[test]
    fn common_manifest_prefixes_have_common_kv_hashes() {
        let plans = ReplayManifest::read_jsonl(MANIFEST.as_bytes()).unwrap();
        let first = PromptIdentity::from_request(&plans[0].requests[0]).unwrap();
        let mut other_request = plans[0].requests[0].clone();
        other_request.blocks.as_mut().unwrap()[1].seed = "different-user".into();
        let other = PromptIdentity::from_request(&other_request).unwrap();
        let first_hashes = first.full_block_hashes(40, 16);
        let other_hashes = other.full_block_hashes(40, 16);
        assert_eq!(first_hashes[0], other_hashes[0]);
        assert_ne!(first_hashes[1], other_hashes[1]);
    }
}
