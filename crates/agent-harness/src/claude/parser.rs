//! Claude Code's `stream-json` parser — its NDJSON wire format → the neutral
//! [`crate::RunEvent`] vocabulary, via [`parse_claude_line`].
//!
//! Wire format reference (verified against the official headless docs,
//! https://code.claude.com/docs/en/headless): with `--output-format
//! stream-json --verbose --include-partial-messages`, streamed text arrives as
//! `stream_event` lines whose `event.delta.type == "text_delta"`; reasoning as
//! `thinking_delta`. A tool call starts as `content_block_start` with a
//! `tool_use` block (its arguments stream separately as `input_json_delta`, so
//! `input` stays `None`); tool results arrive as a `user` message's
//! `tool_result` blocks. `system/init` carries the session id + model; the
//! final `result` line carries token usage. Aggregate `assistant` lines are
//! ignored — their text already streamed via the deltas, so honoring them too
//! would double it.

use serde_json::Value;

use crate::{
    ParsedLine, Question, QuestionOption, SessionInfo, ToolCallEnd, ToolCallStart, ToolKind,
    UsageInfo,
};

/// Classify a Claude Code tool name into the neutral [`ToolKind`]. Claude
/// reaches the consumer through `RunEvent`, so this classifier stays private
/// — the kind rides on the event, the consumer never sees `Read`/`Edit`/….
fn claude_tool_kind(name: &str) -> ToolKind {
    match name {
        "Read" | "NotebookRead" => ToolKind::Read,
        "Write" => ToolKind::Write,
        "Edit" | "MultiEdit" | "NotebookEdit" => ToolKind::Edit,
        "Grep" | "Glob" | "WebSearch" | "WebFetch" => ToolKind::Search,
        "Bash" | "BashOutput" => ToolKind::Execute,
        _ => ToolKind::Other,
    }
}

/// Parse one line of Claude Code's `stream-json` output into the
/// shared [`ParsedLine`] shape. See the module docs for the wire
/// format. Claude edits files directly via its tools (reflected on
/// disk by the file watcher), so it never emits suggested-edit
/// previews — `edits` is always empty here.
pub fn parse_claude_line(line: &str) -> ParsedLine {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return ParsedLine::default();
    }
    let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
        // Claude shouldn't emit non-JSON in stream-json mode; ignore.
        return ParsedLine::default();
    };
    let Some(obj) = value.as_object() else {
        return ParsedLine::default();
    };

    match obj.get("type").and_then(Value::as_str) {
        Some("stream_event") => {
            let Some(event) = obj.get("event").and_then(Value::as_object) else {
                return ParsedLine::default();
            };
            let event_type = event.get("type").and_then(Value::as_str);

            // Streamed assistant text.
            if event_type == Some("content_block_delta") {
                if let Some(delta) = event.get("delta").and_then(Value::as_object) {
                    if delta.get("type").and_then(Value::as_str) == Some("text_delta") {
                        if let Some(text) = delta.get("text").and_then(Value::as_str) {
                            if !text.is_empty() {
                                return ParsedLine {
                                    text: Some(text.to_owned()),
                                    ..ParsedLine::default()
                                };
                            }
                        }
                    }
                    // Extended-thinking deltas (Anthropic streaming):
                    // `delta.type == "thinking_delta"` carries the reasoning
                    // text in `delta.thinking`. Surfaced as a distinct
                    // Thinking event (harmless no-op when the model isn't
                    // thinking, since the arm never matches).
                    if delta.get("type").and_then(Value::as_str) == Some("thinking_delta") {
                        if let Some(thinking) = delta.get("thinking").and_then(Value::as_str) {
                            if !thinking.is_empty() {
                                return ParsedLine {
                                    thinking: Some(thinking.to_owned()),
                                    ..ParsedLine::default()
                                };
                            }
                        }
                    }
                }
            }

            // A tool call beginning → structured ToolStart (id + name) so
            // the UI renders a state-ful card, ended by the matching
            // tool_result (the `user` arm below).
            if event_type == Some("content_block_start") {
                if let Some(block) = event.get("content_block").and_then(Value::as_object) {
                    if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                        let name = block.get("name").and_then(Value::as_str).unwrap_or("tool");
                        // AskUserQuestion is surfaced as selectable chips (from
                        // the aggregate `assistant` line below), not a tool card
                        // — and headless Claude denies the tool, so its card
                        // would only ever show as a failure. Suppress it.
                        if name == "AskUserQuestion" {
                            return ParsedLine::default();
                        }
                        let id = block.get("id").and_then(Value::as_str).unwrap_or_default();
                        // In streaming mode the tool's arguments are NOT here:
                        // `content_block_start` carries an empty `input: {}`,
                        // and the real args arrive incrementally as
                        // `input_json_delta` fragments. Reconstructing them
                        // would mean accumulating partial JSON and delaying the
                        // card — so leave `input: None` (honest: the card still
                        // renders, just without args) rather than emit `{}`.
                        // The tool's *output* is captured at tool_result below.
                        return ParsedLine {
                            tool_start: Some(ToolCallStart {
                                tool_call_id: id.to_owned(),
                                name: name.to_owned(),
                                input: None,
                                tool_kind: claude_tool_kind(name),
                            }),
                            ..ParsedLine::default()
                        };
                    }
                }
            }

            ParsedLine::default()
        }
        Some("system") => {
            match obj.get("subtype").and_then(Value::as_str) {
                // init → session identity (id + model). Claude's first line
                // in stream-json mode carries both.
                Some("init") => ParsedLine {
                    session: Some(SessionInfo {
                        session_id: pick_str(obj, "session_id"),
                        model: pick_str(obj, "model"),
                    }),
                    ..ParsedLine::default()
                },
                // Surface retry progress.
                Some("api_retry") => {
                    let attempt = obj.get("attempt").and_then(Value::as_u64).unwrap_or(1);
                    ParsedLine {
                        activity: Some(format!("Retrying (attempt {attempt})…")),
                        ..ParsedLine::default()
                    }
                }
                // Other system events: ignored.
                _ => ParsedLine::default(),
            }
        }
        Some("user") => {
            // Tool results arrive as a `user` message carrying
            // tool_result blocks; each ends a tool call (matched by
            // tool_use_id; `ok` = not is_error). Grounded from a real
            // claude stream-json run.
            if let Some(content) = obj
                .get("message")
                .and_then(|m| m.get("content"))
                .and_then(Value::as_array)
            {
                for item in content {
                    let Some(block) = item.as_object() else {
                        continue;
                    };
                    if block.get("type").and_then(Value::as_str) == Some("tool_result") {
                        if let Some(id) = block.get("tool_use_id").and_then(Value::as_str) {
                            let is_error =
                                block.get("is_error").and_then(Value::as_bool).unwrap_or(false);
                            // `content` is the tool's result — a string, or an
                            // array of `{type:"text", text}` blocks.
                            let output = block.get("content").and_then(claude_tool_result_text);
                            return ParsedLine {
                                tool_end: Some(ToolCallEnd {
                                    tool_call_id: id.to_owned(),
                                    ok: !is_error,
                                    output,
                                }),
                                ..ParsedLine::default()
                            };
                        }
                    }
                }
            }
            ParsedLine::default()
        }
        // The final result line carries authoritative token usage. (Text
        // already streamed via deltas, so we take only the usage here.)
        Some("result") => {
            let usage = obj.get("usage").and_then(Value::as_object);
            let input_tokens = usage.and_then(|u| u.get("input_tokens")).and_then(Value::as_u64);
            let output_tokens = usage.and_then(|u| u.get("output_tokens")).and_then(Value::as_u64);
            // Claude reports prompt-cache counters alongside input/output;
            // surface them on Usage rather than dropping them — they're the
            // difference between a useful cost display and a misleading one.
            let cache_read_tokens = usage
                .and_then(|u| u.get("cache_read_input_tokens"))
                .and_then(Value::as_u64);
            let cache_write_tokens = usage
                .and_then(|u| u.get("cache_creation_input_tokens"))
                .and_then(Value::as_u64);
            if input_tokens.is_none() && output_tokens.is_none() {
                return ParsedLine::default();
            }
            // Claude reports input/output separately; derive the total. Cache
            // tokens are reported separately and are NOT folded into the total.
            let total_tokens = match (input_tokens, output_tokens) {
                (Some(i), Some(o)) => Some(i + o),
                _ => None,
            };
            ParsedLine {
                usage: Some(UsageInfo {
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cache_read_tokens,
                    cache_write_tokens,
                }),
                ..ParsedLine::default()
            }
        }
        // Aggregate assistant lines: text already streamed via deltas, so we
        // take ONLY a completed `AskUserQuestion` tool call from one — its full
        // input is present here in a single piece — and ignore everything else
        // (a double of already-streamed content).
        Some("assistant") => {
            let Some(content) = obj
                .get("message")
                .and_then(|m| m.get("content"))
                .and_then(Value::as_array)
            else {
                return ParsedLine::default();
            };
            for item in content {
                let Some(block) = item.as_object() else {
                    continue;
                };
                if block.get("type").and_then(Value::as_str) != Some("tool_use")
                    || block.get("name").and_then(Value::as_str) != Some("AskUserQuestion")
                {
                    continue;
                }
                let id = block.get("id").and_then(Value::as_str).unwrap_or_default();
                if let Some(questions) = block.get("input").and_then(parse_ask_user_question) {
                    return ParsedLine {
                        ask_question: Some((id.to_owned(), questions)),
                        ..ParsedLine::default()
                    };
                }
            }
            ParsedLine::default()
        }
        // Any other line type carries nothing actionable.
        _ => ParsedLine::default(),
    }
}

/// Parse Claude's experimental `AskUserQuestion` tool input into neutral
/// [`Question`]s. Defensive on purpose — the tool's shape is non-contractual,
/// so a question missing its text or all its options is dropped rather than
/// yielding a broken chip set; `None` means nothing usable was found (the run
/// then falls back to Claude's own prose re-ask). Field map: `question` →
/// `prompt`, `header`, `multiSelect` → `multi_select`, `options[].label` /
/// `.description`; `allowFreeText` defaults to false.
fn parse_ask_user_question(input: &Value) -> Option<Vec<Question>> {
    let raw = input.get("questions").and_then(Value::as_array)?;
    let mut questions = Vec::new();
    for question in raw {
        let Some(prompt) = question
            .get("question")
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
        else {
            continue;
        };
        let options: Vec<QuestionOption> = question
            .get("options")
            .and_then(Value::as_array)
            .map(|opts| {
                opts.iter()
                    .filter_map(|opt| {
                        let label = opt
                            .get("label")
                            .and_then(Value::as_str)
                            .filter(|text| !text.is_empty())?;
                        Some(QuestionOption {
                            label: label.to_owned(),
                            description: opt
                                .get("description")
                                .and_then(Value::as_str)
                                .filter(|text| !text.is_empty())
                                .map(str::to_owned),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        if options.is_empty() {
            continue;
        }
        questions.push(Question {
            header: question
                .get("header")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
                .map(str::to_owned),
            prompt: prompt.to_owned(),
            options,
            multi_select: question
                .get("multiSelect")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            allow_free_text: question
                .get("allowFreeText")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        });
    }
    (!questions.is_empty()).then_some(questions)
}

/// Non-empty string field of `obj`, else `None`.
fn pick_str(obj: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
    obj.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}

/// A Claude `tool_result.content` rendered as text: a string verbatim, or
/// the concatenated `text` of an array of content blocks. `None` if empty
/// or an unrecognized shape.
fn claude_tool_result_text(content: &Value) -> Option<String> {
    match content {
        Value::String(s) => (!s.is_empty()).then(|| s.clone()),
        Value::Array(items) => {
            let mut text = String::new();
            for item in items {
                if let Some(t) = item.get("text").and_then(Value::as_str) {
                    text.push_str(t);
                }
            }
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_delta(text: &str) -> String {
        serde_json::json!({
            "type": "stream_event",
            "event": { "type": "content_block_delta", "delta": { "type": "text_delta", "text": text } }
        })
        .to_string()
    }

    #[test]
    fn streams_text_deltas() {
        let parsed = parse_claude_line(&text_delta("Hello"));
        assert_eq!(parsed.text.as_deref(), Some("Hello"));
        assert!(parsed.activity.is_none());
        assert!(parsed.edits.is_empty());
    }

    #[test]
    fn empty_text_delta_yields_nothing() {
        let parsed = parse_claude_line(&text_delta(""));
        assert!(parsed.text.is_none());
    }

    fn thinking_delta(text: &str) -> String {
        serde_json::json!({
            "type": "stream_event",
            "event": { "type": "content_block_delta", "delta": { "type": "thinking_delta", "thinking": text } }
        })
        .to_string()
    }

    #[test]
    fn streams_thinking_deltas() {
        let parsed = parse_claude_line(&thinking_delta("Let me reason"));
        assert_eq!(parsed.thinking.as_deref(), Some("Let me reason"));
        assert!(parsed.text.is_none());
        assert!(parsed.activity.is_none());
    }

    #[test]
    fn tool_use_start_becomes_tool_start() {
        let line = serde_json::json!({
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "content_block": { "type": "tool_use", "name": "Edit", "id": "toolu_1" }
            }
        })
        .to_string();
        let parsed = parse_claude_line(&line);
        let start = parsed.tool_start.expect("tool_start");
        assert_eq!(start.tool_call_id, "toolu_1");
        assert_eq!(start.name, "Edit");
        assert!(parsed.activity.is_none());
    }

    #[test]
    fn tool_result_becomes_tool_end() {
        let ok_line = serde_json::json!({
            "type": "user",
            "message": { "role": "user", "content": [
                { "type": "tool_result", "tool_use_id": "toolu_1", "is_error": false, "content": "ok" }
            ]}
        })
        .to_string();
        let end = parse_claude_line(&ok_line).tool_end.expect("tool_end");
        assert_eq!(end.tool_call_id, "toolu_1");
        assert!(end.ok);
        assert_eq!(end.output.as_deref(), Some("ok")); // tool_result.content lifted

        let err_line = serde_json::json!({
            "type": "user",
            "message": { "content": [
                { "type": "tool_result", "tool_use_id": "toolu_2", "is_error": true }
            ]}
        })
        .to_string();
        assert!(!parse_claude_line(&err_line).tool_end.unwrap().ok);
    }

    #[test]
    fn api_retry_becomes_activity() {
        let line = serde_json::json!({
            "type": "system", "subtype": "api_retry", "attempt": 2, "max_retries": 5
        })
        .to_string();
        assert_eq!(
            parse_claude_line(&line).activity.as_deref(),
            Some("Retrying (attempt 2)…")
        );
    }

    #[test]
    fn system_init_yields_session() {
        let line = serde_json::json!({
            "type": "system", "subtype": "init", "session_id": "sess-abc", "model": "claude-x"
        })
        .to_string();
        let session = parse_claude_line(&line).session.expect("session");
        assert_eq!(session.session_id.as_deref(), Some("sess-abc"));
        assert_eq!(session.model.as_deref(), Some("claude-x"));
    }

    #[test]
    fn aggregate_and_empty_result_lines_are_ignored() {
        // aggregate assistant message — text already came via deltas
        let assistant = serde_json::json!({
            "type": "assistant",
            "message": { "content": [{ "type": "text", "text": "full turn" }] }
        })
        .to_string();
        let parsed = parse_claude_line(&assistant);
        assert!(parsed.text.is_none() && parsed.activity.is_none());
        // result line without usage → nothing
        assert!(parse_claude_line(
            &serde_json::json!({ "type": "result", "subtype": "success", "is_error": false }).to_string()
        )
        .is_empty());
    }

    #[test]
    fn result_with_usage_yields_usage() {
        let line = serde_json::json!({
            "type": "result",
            "subtype": "success",
            "usage": { "input_tokens": 120, "output_tokens": 30, "cache_read_input_tokens": 5 }
        })
        .to_string();
        let usage = parse_claude_line(&line).usage.expect("usage");
        assert_eq!(usage.input_tokens, Some(120));
        assert_eq!(usage.output_tokens, Some(30));
        assert_eq!(usage.total_tokens, Some(150)); // derived = input + output
    }

    #[test]
    fn non_json_is_ignored_not_echoed() {
        // Unlike bob, Claude's stream-json is always JSON; a stray
        // line should be dropped, not surfaced as assistant text.
        assert!(parse_claude_line("not json").text.is_none());
    }

    // Grounded in a live `claude -p --output-format stream-json` probe: the
    // aggregate `assistant` line carries the AskUserQuestion tool_use with its
    // full input in one piece.
    fn ask_question_assistant_line(id: &str) -> String {
        serde_json::json!({
            "type": "assistant",
            "message": { "content": [{
                "type": "tool_use",
                "id": id,
                "name": "AskUserQuestion",
                "input": { "questions": [{
                    "question": "Where should you go for your weekend trip?",
                    "header": "Destination",
                    "multiSelect": false,
                    "options": [
                        { "label": "The mountains", "description": "Hiking and fresh air." },
                        { "label": "The beach", "description": "Sun and sand." }
                    ]
                }]}
            }]}
        })
        .to_string()
    }

    #[test]
    fn ask_user_question_assistant_line_becomes_ask_question() {
        let parsed = parse_claude_line(&ask_question_assistant_line("toolu_q1"));
        let (request_id, questions) = parsed.ask_question.clone().expect("ask_question");
        assert_eq!(request_id, "toolu_q1");
        assert_eq!(questions.len(), 1);
        let question = &questions[0];
        assert_eq!(question.prompt, "Where should you go for your weekend trip?");
        assert_eq!(question.header.as_deref(), Some("Destination"));
        assert!(!question.multi_select);
        assert!(!question.allow_free_text);
        assert_eq!(question.options.len(), 2);
        assert_eq!(question.options[0].label, "The mountains");
        assert_eq!(question.options[0].description.as_deref(), Some("Hiking and fresh air."));
        // The aggregate line must not also double the streamed text / a card.
        assert!(parsed.text.is_none());
        assert!(parsed.tool_start.is_none());
    }

    #[test]
    fn ask_user_question_card_is_suppressed() {
        // The streamed tool_use start for AskUserQuestion is NOT a tool card —
        // it renders as chips, and headless Claude denies the tool.
        let line = serde_json::json!({
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "content_block": { "type": "tool_use", "name": "AskUserQuestion", "id": "toolu_q1" }
            }
        })
        .to_string();
        assert!(parse_claude_line(&line).is_empty());
    }

    #[test]
    fn malformed_ask_user_question_is_dropped() {
        // A question with no usable options is unusable → no event (the run
        // falls back to Claude's prose re-ask), never a broken chip set.
        let line = serde_json::json!({
            "type": "assistant",
            "message": { "content": [{
                "type": "tool_use", "id": "toolu_x", "name": "AskUserQuestion",
                "input": { "questions": [{ "question": "Pick one?", "options": [] }] }
            }]}
        })
        .to_string();
        assert!(parse_claude_line(&line).ask_question.is_none());
    }
}
