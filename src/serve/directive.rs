//! Echo-directive mode: scripted responses for deterministic agentic testing.
//!
//! A real agent harness (the claude CLI, the Claude Agent SDK) advances its loop based on
//! whether the model returns tool calls — which a fake can't *decide* to do. Directive mode
//! removes the deciding: if any message's text contains
//!
//! ```text
//! <<respond:{"text":"...","tool_calls":[{"name":"Read","arguments":{"file_path":"/x"}}]}>>
//! ```
//!
//! the server skips the engine and returns exactly that content / tool_calls, with
//! deterministic usage. Chained loops fall out naturally: the harness's NEXT request carries
//! the tool RESULT it just produced, and that result's text embeds the NEXT directive — so a
//! scripted multi-turn agent session is just files whose contents point at each other.
//!
//! Rules:
//! - The LAST directive in message order wins (the freshest instruction — exactly the
//!   chained-tool-result flow above). Earlier directives are inert history.
//! - Opt-in per request: no directive, no behavior change (load tests are unaffected).
//! - Directives are parsed from the same text the prompt-token count sees, so gateways that
//!   reshape content (string vs parts) don't change behavior.

use serde::Deserialize;

use super::types::{ChatMessage, ResponseToolCall, ToolCallFunction};

/// What a directive asks the server to respond with.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Directive {
    /// Assistant text content. Omitted/empty with tool_calls present -> content: null,
    /// like a real model's pure tool-call turn.
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<DirectiveToolCall>,
    /// Override the finish_reason ("stop" / "tool_calls" are inferred when absent).
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct DirectiveToolCall {
    pub name: String,
    /// Tool arguments as a JSON object; serialized to the OpenAI `arguments` string form.
    #[serde(default)]
    pub arguments: serde_json::Value,
    /// Stable id for the call (auto-generated when absent).
    #[serde(default)]
    pub id: Option<String>,
}

impl Directive {
    pub fn finish_reason(&self) -> &'static str {
        match self.finish_reason.as_deref() {
            Some("stop") => "stop",
            Some("length") => "length",
            Some("tool_calls") => "tool_calls",
            // Unknown overrides fall back to the inferred reason rather than erroring:
            // a typo'd script should still produce a well-formed response.
            _ if !self.tool_calls.is_empty() => "tool_calls",
            _ => "stop",
        }
    }

    /// The OpenAI response-shaped tool calls, with ids filled in.
    pub fn response_tool_calls(&self) -> Vec<ResponseToolCall> {
        self.tool_calls
            .iter()
            .enumerate()
            .map(|(i, tc)| ResponseToolCall {
                id: tc
                    .id
                    .clone()
                    .unwrap_or_else(|| format!("call_directive_{i}")),
                r#type: "function",
                function: ToolCallFunction {
                    name: tc.name.clone(),
                    arguments: tc.arguments.to_string(),
                },
            })
            .collect()
    }

    /// The text whose token count stands in for "generated output" in usage. Includes the
    /// serialized tool calls so scripted turns bill like real ones (arguments are output).
    pub fn completion_text(&self) -> String {
        let mut out = self.text.clone().unwrap_or_default();
        for tc in &self.tool_calls {
            out.push_str(&tc.name);
            out.push_str(&tc.arguments.to_string());
        }
        out
    }
}

const MARKER: &str = "<<respond:";

/// Find the LAST directive across all messages. Returns None (normal engine path) when no
/// message carries one; malformed JSON after a marker is also None — a fake must never 500
/// on content that merely *resembles* a directive.
pub fn find_directive(messages: &[ChatMessage]) -> Option<Directive> {
    let mut found = None;
    for message in messages {
        let Some(content) = message.content.as_ref() else {
            continue;
        };
        let text = content.text();
        let mut search_from = 0;
        while let Some(pos) = text[search_from..].find(MARKER) {
            let json_start = search_from + pos + MARKER.len();
            // Parse exactly one JSON value from the marker onward; trailing text (including
            // the closing `>>`) is irrelevant. This survives `>>` inside JSON strings.
            let mut iter =
                serde_json::Deserializer::from_str(&text[json_start..]).into_iter::<Directive>();
            if let Some(Ok(d)) = iter.next() {
                found = Some(d);
            }
            search_from = json_start;
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::types::MessageContent;

    fn msg(role: &str, text: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: Some(MessageContent::Text(text.to_string())),
        }
    }

    #[test]
    fn no_directive_is_none() {
        let messages = vec![msg("user", "just a normal prompt << with brackets >>")];
        assert!(find_directive(&messages).is_none());
    }

    #[test]
    fn parses_text_directive() {
        let messages = vec![msg("user", r#"do it <<respond:{"text":"done"}>>"#)];
        let d = find_directive(&messages).unwrap();
        assert_eq!(d.text.as_deref(), Some("done"));
        assert!(d.tool_calls.is_empty());
        assert_eq!(d.finish_reason(), "stop");
    }

    #[test]
    fn parses_tool_call_directive_and_infers_finish_reason() {
        let messages = vec![msg(
            "user",
            r#"<<respond:{"tool_calls":[{"name":"Read","arguments":{"file_path":"/tmp/a"}}]}>>"#,
        )];
        let d = find_directive(&messages).unwrap();
        assert_eq!(d.finish_reason(), "tool_calls");
        let calls = d.response_tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "Read");
        assert_eq!(calls[0].function.arguments, r#"{"file_path":"/tmp/a"}"#);
        assert_eq!(calls[0].id, "call_directive_0");
    }

    #[test]
    fn last_directive_wins_across_messages() {
        let messages = vec![
            msg("user", r#"<<respond:{"text":"first"}>>"#),
            msg("assistant", "ok"),
            msg("tool", r#"result body <<respond:{"text":"second"}>>"#),
        ];
        let d = find_directive(&messages).unwrap();
        assert_eq!(d.text.as_deref(), Some("second"));
    }

    #[test]
    fn last_directive_wins_within_one_message() {
        let messages = vec![msg(
            "user",
            r#"<<respond:{"text":"a"}>> then <<respond:{"text":"b"}>>"#,
        )];
        assert_eq!(
            find_directive(&messages).unwrap().text.as_deref(),
            Some("b")
        );
    }

    #[test]
    fn survives_double_angle_inside_json_strings() {
        let messages = vec![msg("user", r#"<<respond:{"text":"a >> b"}>>"#)];
        assert_eq!(
            find_directive(&messages).unwrap().text.as_deref(),
            Some("a >> b")
        );
    }

    #[test]
    fn malformed_json_is_ignored() {
        let messages = vec![msg("user", "<<respond:{not json}>>")];
        assert!(find_directive(&messages).is_none());
    }

    #[test]
    fn malformed_then_valid_uses_valid() {
        let messages = vec![msg(
            "user",
            r#"<<respond:{oops}>> <<respond:{"text":"ok"}>>"#,
        )];
        assert_eq!(
            find_directive(&messages).unwrap().text.as_deref(),
            Some("ok")
        );
    }

    #[test]
    fn directive_in_parts_content_is_found() {
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Parts(vec![])),
        }];
        assert!(find_directive(&messages).is_none());
    }

    #[test]
    fn explicit_finish_reason_override() {
        let messages = vec![msg(
            "user",
            r#"<<respond:{"text":"x","finish_reason":"length"}>>"#,
        )];
        assert_eq!(find_directive(&messages).unwrap().finish_reason(), "length");
    }

    #[test]
    fn completion_text_covers_text_and_tool_calls() {
        let d = Directive {
            text: Some("hello".into()),
            tool_calls: vec![DirectiveToolCall {
                name: "Read".into(),
                arguments: serde_json::json!({"file_path": "/x"}),
                id: None,
            }],
            finish_reason: None,
        };
        let t = d.completion_text();
        assert!(t.contains("hello") && t.contains("Read") && t.contains("/x"));
    }
}
