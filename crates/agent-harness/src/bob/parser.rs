//! bob's stream-json parser — bob's wire format → the neutral
//! [`crate::RunEvent`] vocabulary.
//!
//! The normalized event types and the generic `normalize_process_event`
//! skeleton live in [`crate::events`]; this module is bob's
//! adapter-side decoder on top of them. bob emits one JSON object per
//! line with a snake_case `type` discriminator — see [`parse_bob_line`]
//! for the grounded schema. Reasoning is streamed inline as
//! `<thinking>…</thinking>` and routed by the stateful [`BobStreamParser`].

use serde_json::Value;

use crate::{
    normalize_process_event, ByteRange, ParsedLine, ProcessEvent, RunEvent, SessionInfo,
    SuggestedEdit, ToolCallEnd, ToolCallStart, ToolKind, UsageInfo,
};

/// Classify a bob (Roo-style) tool name into the neutral [`ToolKind`].
/// Public so a consumer that interprets bob's *raw* events directly — rather
/// than through `RunEvent` — can share this one table instead of re-encoding
/// bob's vocabulary downstream.
pub fn bob_tool_kind(name: &str) -> ToolKind {
    match name {
        "read_file" => ToolKind::Read,
        "write_file" | "write_to_file" => ToolKind::Write,
        "apply_diff" | "insert_content" | "search_and_replace" => ToolKind::Edit,
        "search_files" | "list_files" | "list_code_definition_names" => ToolKind::Search,
        "execute_command" => ToolKind::Execute,
        _ => ToolKind::Other,
    }
}

/// bob's adapter-side normalization: parse bob's `--output-format
/// stream-json` stdout via [`parse_bob_line`].
pub fn normalize_bob_event(event: ProcessEvent) -> Vec<RunEvent> {
    normalize_process_event(event, parse_bob_line)
}

/// Parse one line of bob's `--output-format stream-json` into the shared
/// [`ParsedLine`]. Grounded in bob's *empirical* event schema (the
/// `bob-agents` reference + "bob shell usage" findings), not guessed: bob
/// emits one JSON object per line with a snake_case `type` discriminator —
/// `init` / `message{role,content,delta}` / `tool_use{tool_id,tool_name,
/// parameters}` / `tool_result{tool_id,status,output}` / `result{stats}`.
///
/// Mapping: an assistant `message` → text (the echoed `user` prompt is
/// skipped — a real fix vs. the old role-blind heuristic); `tool_use` →
/// a structured [`ToolCallStart`] (bob's edit tools — write_file /
/// apply_diff / insert_content — surface as tool-cards too; reconstructing
/// previewable diffs from their `parameters` is a separate follow-up);
/// `tool_result` → [`ToolCallEnd`] (ok unless `status == "error"`).
/// `init` / `result` are lifecycle (process start/exit drives
/// Started/Exited). A non-JSON line passes through as raw text.
/// Unrecognized shapes fall back to the legacy suggested-edits heuristic
/// so a bob build that emits edit arrays still surfaces them.
pub fn parse_bob_line(line: &str) -> ParsedLine {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return ParsedLine::default();
    }

    let payload: Value = match serde_json::from_str(trimmed) {
        Ok(value) => value,
        // Not JSON — bob occasionally prints prose / stderr-ish lines.
        // Pass the raw (untrimmed) line through as text.
        Err(_) => {
            return ParsedLine {
                text: Some(line.to_owned()),
                ..ParsedLine::default()
            }
        }
    };

    let Some(record) = payload.as_object() else {
        return ParsedLine::default();
    };

    match record.get("type").and_then(Value::as_str) {
        // Assistant text (`delta: true` marks a streaming chunk; both
        // chunk and full message carry the text in `content`). The echoed
        // user prompt (role "user") is not surfaced.
        Some("message") => {
            if record.get("role").and_then(Value::as_str) == Some("assistant") {
                if let Some(content) = pick_string(record, "content") {
                    return ParsedLine {
                        text: Some(content),
                        ..ParsedLine::default()
                    };
                }
            }
            ParsedLine::default()
        }
        // Tool call start → structured ToolStart (tool_id + tool_name).
        Some("tool_use") => {
            let name = pick_string(record, "tool_name").unwrap_or_else(|| "tool".to_owned());
            // bob delivers its final answer via the `attempt_completion`
            // tool (grounded in a real run) — surface its `result` as the
            // answer text, not a bare tool-card.
            if name == "attempt_completion" {
                return match record
                    .get("parameters")
                    .and_then(Value::as_object)
                    .and_then(|p| p.get("result"))
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                {
                    Some(result) => ParsedLine {
                        text: Some(result.to_owned()),
                        ..ParsedLine::default()
                    },
                    None => ParsedLine::default(),
                };
            }
            let tool_call_id = pick_string(record, "tool_id").unwrap_or_default();
            // The call's arguments, lifted verbatim (parameters object →
            // compact JSON) so the UI can show what the tool was asked to do.
            let input = record.get("parameters").map(value_to_display_string);
            let tool_kind = bob_tool_kind(&name);
            ParsedLine {
                tool_start: Some(ToolCallStart { tool_call_id, name, input, tool_kind }),
                ..ParsedLine::default()
            }
        }
        // Tool call end → ToolEnd, matched by tool_id; ok unless the
        // status is explicitly "error". `output` carries the tool's result.
        Some("tool_result") => {
            let tool_call_id = pick_string(record, "tool_id").unwrap_or_default();
            let ok = record.get("status").and_then(Value::as_str) != Some("error");
            let output = record
                .get("output")
                .map(value_to_display_string)
                .filter(|s| !s.is_empty());
            ParsedLine {
                tool_end: Some(ToolCallEnd { tool_call_id, ok, output }),
                ..ParsedLine::default()
            }
        }
        // init → session identity (id + model), arriving a beat after the
        // engine's `Started`.
        Some("init") => ParsedLine {
            session: Some(SessionInfo {
                session_id: pick_string(record, "session_id"),
                model: pick_string(record, "model"),
            }),
            ..ParsedLine::default()
        },
        // result → token usage. bob reports a single `total_tokens` in
        // `stats` (no input/output split); coins (`session_costs`) are
        // bob-specific and intentionally NOT lifted into the neutral Usage.
        Some("result") => {
            let total_tokens = record
                .get("stats")
                .and_then(Value::as_object)
                .and_then(|s| s.get("total_tokens"))
                .and_then(Value::as_u64);
            ParsedLine {
                usage: total_tokens.map(|t| UsageInfo {
                    total_tokens: Some(t),
                    ..UsageInfo::default()
                }),
                ..ParsedLine::default()
            }
        }
        // Anything else: unknown. Fall back to the legacy suggested-edits
        // heuristic so a bob build that emits edit arrays still surfaces them.
        _ => {
            let edits = parse_suggested_edits(record);
            if edits.is_empty() {
                ParsedLine::default()
            } else {
                let n = edits.len();
                ParsedLine {
                    edits,
                    activity: Some(format!("{n} suggested edit{}", if n == 1 { "" } else { "s" })),
                    ..ParsedLine::default()
                }
            }
        }
    }
}

/// Stateful wrapper over [`parse_bob_line`] for a single bob run. bob
/// streams its reasoning inline as `<thinking>…</thinking>` within the
/// assistant `message` content (grounded in a real run — the tags arrive
/// as their own deltas), so routing that reasoning to the Thinking
/// stream requires tracking the open/closed state *across* lines. The
/// per-line dispatch (text / tool events / answer) stays in
/// `parse_bob_line`; this only re-routes assistant text through the
/// thinking-tag state machine. One instance per run (see `BobHarness`).
#[derive(Debug, Default)]
pub struct BobStreamParser {
    in_thinking: bool,
    /// True while inside a `[using tool …]` narration echo whose closing `]`
    /// has not yet arrived — bob prints that echo inline right before each
    /// structured `tool_use`, so it's redundant with the ToolStart card and is
    /// dropped. The echo can span deltas, so the state carries across lines.
    suppressing_echo: bool,
}

impl BobStreamParser {
    /// Parse one stdout line, routing any assistant text through the
    /// `<thinking>` state machine into [`ParsedLine::text`] /
    /// [`ParsedLine::thinking`].
    pub fn parse_line(&mut self, line: &str) -> ParsedLine {
        let mut parsed = parse_bob_line(line);
        if let Some(content) = parsed.text.take() {
            let (visible, mut thinking) = self.route_thinking(&content);
            // Drop bob's `[using tool …]` narration echo (redundant with the
            // structured ToolStart card; may span deltas).
            let visible = visible.and_then(|t| self.narration_after_echo(t));
            // bob's *answer* is always the `attempt_completion` result (verified
            // in code AND ask mode — the assistant `message` carries only
            // `<thinking>` + preamble prose, e.g. "I'll read … then enrich …").
            // So an assistant message's leftover prose is narration, not the
            // answer: fold it into the collapsed thinking trace, leaving only the
            // attempt_completion text as the visible message. A `tool_use` /
            // non-message line (the answer) keeps its text.
            if line_is_assistant_message(line) {
                if let Some(t) = visible {
                    thinking = Some(match thinking {
                        Some(a) => a + &t,
                        None => t,
                    });
                }
            } else {
                parsed.text = visible;
            }
            parsed.thinking = match (thinking, parsed.thinking.take()) {
                (Some(a), Some(b)) => Some(a + &b),
                (a, b) => a.or(b),
            };
        }
        parsed
    }

    /// Split an assistant content chunk into (visible text, thinking),
    /// honoring `<thinking>`/`</thinking>` markers and the carried-over
    /// `in_thinking` state. Handles tags split across chunks and multiple
    /// tags within one chunk.
    fn route_thinking(&mut self, content: &str) -> (Option<String>, Option<String>) {
        const OPEN: &str = "<thinking>";
        const CLOSE: &str = "</thinking>";
        let mut text = String::new();
        let mut thinking = String::new();
        let mut rest = content;
        loop {
            if self.in_thinking {
                match rest.find(CLOSE) {
                    Some(i) => {
                        thinking.push_str(&rest[..i]);
                        self.in_thinking = false;
                        rest = &rest[i + CLOSE.len()..];
                    }
                    None => {
                        thinking.push_str(rest);
                        break;
                    }
                }
            } else {
                match rest.find(OPEN) {
                    Some(i) => {
                        text.push_str(&rest[..i]);
                        self.in_thinking = true;
                        rest = &rest[i + OPEN.len()..];
                    }
                    None => {
                        text.push_str(rest);
                        break;
                    }
                }
            }
        }
        (
            (!text.is_empty()).then_some(text),
            (!thinking.is_empty()).then_some(thinking),
        )
    }

    /// Drop bob's `[using tool …]` narration echo. bob prints it inline right
    /// before the structured `tool_use` event, so it's redundant with the
    /// ToolStart card. The echo can span deltas (its `]` arriving in a later
    /// chunk), so `suppressing_echo` carries the drop across lines. Ported from
    /// the pre-migration `BobChatMapper`.
    fn narration_after_echo(&mut self, text: String) -> Option<String> {
        if self.suppressing_echo {
            // Still inside an unterminated echo — drop until its `]`.
            if text.contains(']') {
                self.suppressing_echo = false;
            }
            return None;
        }
        if text.trim_start().starts_with("[using tool") {
            // Start of the echo. If its `]` isn't in this delta, keep dropping
            // subsequent deltas until it arrives.
            if !text.contains(']') {
                self.suppressing_echo = true;
            }
            return None;
        }
        Some(text)
    }
}

/// Is this stdout line bob's assistant `message` (its narration / preamble
/// prose) — as opposed to a `tool_use` (incl. `attempt_completion`, the answer),
/// a `tool_result`, or a lifecycle line? Used to fold the message prose into the
/// collapsed thinking trace: bob's answer is always the `attempt_completion`
/// result, never the message text. A cheap re-parse of the line envelope (bob's
/// stdout is bounded, not a hot loop).
fn line_is_assistant_message(line: &str) -> bool {
    serde_json::from_str::<Value>(line.trim())
        .ok()
        .map(|v| {
            v.get("type").and_then(Value::as_str) == Some("message")
                && v.get("role").and_then(Value::as_str) == Some("assistant")
        })
        .unwrap_or(false)
}

/// Render a JSON value as a display string for a tool's `input`/`output`:
/// a JSON string is taken verbatim (no surrounding quotes); any other shape
/// (object / array / number) is serialized compactly. Lets the adapter lift
/// bob's `parameters` / `tool_result.output` into the neutral layer without
/// imposing a schema on them.
fn value_to_display_string(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Non-empty string field, else `None` (mirrors TS `pickString`).
fn pick_string(record: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
    match record.get(key) {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}

/// String field allowing empty (mirrors TS `pickStringValue` — used
/// for replacements, which may legitimately be the empty string for
/// a deletion).
fn pick_string_value(record: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
    match record.get(key) {
        Some(Value::String(s)) => Some(s.clone()),
        _ => None,
    }
}

fn parse_suggested_edits(record: &serde_json::Map<String, Value>) -> Vec<SuggestedEdit> {
    let mut edits = Vec::new();
    if let Some(direct) = parse_suggested_edit(record) {
        edits.push(direct);
    }
    for key in ["edits", "suggestedEdits", "suggestions"] {
        let Some(Value::Array(items)) = record.get(key) else {
            continue;
        };
        for item in items {
            if let Some(obj) = item.as_object() {
                if let Some(parsed) = parse_suggested_edit(obj) {
                    edits.push(parsed);
                }
            }
        }
    }
    edits
}

fn parse_suggested_edit(record: &serde_json::Map<String, Value>) -> Option<SuggestedEdit> {
    let file_path = pick_string(record, "filePath")
        .or_else(|| pick_string(record, "path"))
        .or_else(|| pick_string(record, "file"))?;

    // Range may be nested under `range` or flat on the record.
    let range_record = match record.get("range").and_then(Value::as_object) {
        Some(nested) => nested,
        None => record,
    };
    let start = range_record.get("start").and_then(Value::as_u64)?;
    let end = range_record.get("end").and_then(Value::as_u64)?;

    let replacement = pick_string_value(record, "replacement")
        .or_else(|| pick_string_value(record, "replaceWith"))
        .or_else(|| pick_string_value(record, "insert"))
        .or_else(|| pick_string_value(record, "newText"))?;

    let title = pick_string(record, "title")
        .or_else(|| pick_string(record, "summary"))
        .or_else(|| pick_string(record, "description"));

    Some(SuggestedEdit {
        file_path,
        range: ByteRange { start, end },
        replacement,
        title,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_line_yields_nothing() {
        assert!(parse_bob_line("   ").is_empty());
    }

    #[test]
    fn non_json_passes_through_as_text() {
        let parsed = parse_bob_line("hello world");
        assert_eq!(parsed.text.as_deref(), Some("hello world"));
        assert!(parsed.edits.is_empty());
    }

    #[test]
    fn assistant_message_becomes_text() {
        let parsed =
            parse_bob_line(r#"{"type":"message","role":"assistant","content":"hi there"}"#);
        assert_eq!(parsed.text.as_deref(), Some("hi there"));
        assert!(parsed.activity.is_none());
    }

    #[test]
    fn user_message_is_skipped() {
        // The echoed user prompt must not surface as assistant text.
        let parsed = parse_bob_line(r#"{"type":"message","role":"user","content":"my prompt"}"#);
        assert!(parsed.is_empty());
    }

    #[test]
    fn assistant_delta_chunk_becomes_text() {
        // `delta: true` marks a streaming chunk; the text is still in
        // `content`.
        let parsed = parse_bob_line(
            r#"{"type":"message","role":"assistant","content":"chunk","delta":true}"#,
        );
        assert_eq!(parsed.text.as_deref(), Some("chunk"));
    }

    #[test]
    fn flat_suggested_edit_parses() {
        let line = r#"{"filePath":"notes/a.md","start":3,"end":7,"replacement":"X","title":"fix"}"#;
        let parsed = parse_bob_line(line);
        assert_eq!(parsed.edits.len(), 1);
        let edit = &parsed.edits[0];
        assert_eq!(edit.file_path, "notes/a.md");
        assert_eq!(edit.range, ByteRange { start: 3, end: 7 });
        assert_eq!(edit.replacement, "X");
        assert_eq!(edit.title.as_deref(), Some("fix"));
        // No text → activity reports the edit count.
        assert_eq!(parsed.activity.as_deref(), Some("1 suggested edit"));
    }

    #[test]
    fn nested_range_and_array_edits_parse() {
        let line = r#"{"edits":[{"path":"a.md","range":{"start":0,"end":1},"newText":""},
                                 {"file":"b.md","range":{"start":2,"end":4},"insert":"yo"}]}"#;
        let parsed = parse_bob_line(line);
        assert_eq!(parsed.edits.len(), 2);
        assert_eq!(parsed.edits[0].replacement, ""); // empty replacement = deletion, allowed
        assert_eq!(parsed.edits[1].replacement, "yo");
        assert_eq!(parsed.activity.as_deref(), Some("2 suggested edits"));
    }

    #[test]
    fn tool_use_becomes_tool_start() {
        let parsed = parse_bob_line(
            r#"{"type":"tool_use","tool_id":"tool-1","tool_name":"execute_command","parameters":{"command":"ls"}}"#,
        );
        let start = parsed.tool_start.expect("tool_start");
        assert_eq!(start.tool_call_id, "tool-1");
        assert_eq!(start.name, "execute_command");
        // The parameters object is lifted verbatim as the call's input.
        assert_eq!(start.input.as_deref(), Some(r#"{"command":"ls"}"#));
        assert!(parsed.activity.is_none());
    }

    #[test]
    fn edit_tools_surface_as_tool_start() {
        // bob's edit tools (apply_diff / insert_content / write_file) flow
        // through as tool-cards too.
        let start = parse_bob_line(
            r#"{"type":"tool_use","tool_id":"t9","tool_name":"apply_diff","parameters":{"path":"a.md"}}"#,
        )
        .tool_start
        .expect("tool_start");
        assert_eq!(start.name, "apply_diff");
    }

    #[test]
    fn tool_result_becomes_tool_end() {
        let ok = parse_bob_line(
            r#"{"type":"tool_result","tool_id":"tool-1","status":"success","output":"done"}"#,
        )
        .tool_end
        .expect("tool_end");
        assert_eq!(ok.tool_call_id, "tool-1");
        assert!(ok.ok);
        // The tool's result is lifted as the end event's output.
        assert_eq!(ok.output.as_deref(), Some("done"));

        let err = parse_bob_line(
            r#"{"type":"tool_result","tool_id":"tool-2","status":"error","output":"boom"}"#,
        )
        .tool_end
        .expect("tool_end");
        assert!(!err.ok);
    }

    #[test]
    fn init_yields_session_and_result_yields_usage() {
        // init → Session (id + model); no text/tool content.
        let init = parse_bob_line(r#"{"type":"init","session_id":"s1","model":"premium"}"#);
        let session = init.session.expect("session");
        assert_eq!(session.session_id.as_deref(), Some("s1"));
        assert_eq!(session.model.as_deref(), Some("premium"));
        assert!(init.text.is_none() && init.tool_start.is_none());

        // result → Usage (neutral tokens only; coins stay bob-specific).
        let result = parse_bob_line(
            r#"{"type":"result","status":"success","stats":{"total_tokens":1280,"session_costs":3,"tool_calls":2}}"#,
        );
        let usage = result.usage.expect("usage");
        assert_eq!(usage.total_tokens, Some(1280));
        assert_eq!(usage.input_tokens, None);
        assert_eq!(usage.output_tokens, None);

        // A result with no token count → no usage (nothing to report).
        assert!(parse_bob_line(r#"{"type":"result","status":"success","stats":{"tool_calls":2}}"#)
            .is_empty());
    }

    #[test]
    fn incomplete_edit_is_ignored() {
        // Missing `end` → not a valid edit.
        let parsed = parse_bob_line(r#"{"filePath":"a.md","start":3,"replacement":"X"}"#);
        assert!(parsed.edits.is_empty());
    }

    #[test]
    fn normalize_stdout_text_event() {
        let events = normalize_bob_event(ProcessEvent::Stdout {
            run_id: "r1".to_owned(),
            line: r#"{"type":"message","role":"assistant","content":"hi"}"#.to_owned(),
        });
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            RunEvent::Text { run_id, delta } if run_id == "r1" && delta == "hi"
        ));
    }

    #[test]
    fn normalize_bob_tool_events() {
        let start = normalize_bob_event(ProcessEvent::Stdout {
            run_id: "r1".to_owned(),
            line: r#"{"type":"tool_use","tool_id":"t1","tool_name":"write_file"}"#.to_owned(),
        });
        assert!(matches!(
            start.as_slice(),
            [RunEvent::ToolStart { tool_call_id, title, .. }]
                if tool_call_id == "t1" && title == "write_file"
        ));
        let end = normalize_bob_event(ProcessEvent::Stdout {
            run_id: "r1".to_owned(),
            line: r#"{"type":"tool_result","tool_id":"t1","status":"success"}"#.to_owned(),
        });
        assert!(matches!(
            end.as_slice(),
            [RunEvent::ToolEnd { tool_call_id, ok, .. }] if tool_call_id == "t1" && *ok
        ));
    }

    #[test]
    fn attempt_completion_becomes_answer_text() {
        // bob's final answer is delivered via the attempt_completion tool,
        // not plain message content — surface it as text, not a card.
        let parsed = parse_bob_line(
            r#"{"type":"tool_use","tool_id":"tool-2","tool_name":"attempt_completion","parameters":{"result":"The answer is 42."}}"#,
        );
        assert_eq!(parsed.text.as_deref(), Some("The answer is 42."));
        assert!(parsed.tool_start.is_none());
    }

    #[test]
    fn using_tool_echo_is_dropped() {
        // bob narrates each tool call inline as `[using tool …]` right before
        // the structured tool_use; that echo is redundant with the ToolStart
        // card and must not appear in the message text.
        let mut parser = BobStreamParser::default();
        let msg =
            |c: &str| serde_json::json!({"type":"message","role":"assistant","content":c}).to_string();
        assert!(parser.parse_line(&msg("[using tool write_to_file: notes/x.md]")).text.is_none());
        // Surrounding narration still flows — to the thinking trace (only the
        // attempt_completion result is the visible answer).
        assert_eq!(parser.parse_line(&msg("All set.")).thinking.as_deref(), Some("All set."));
    }

    #[test]
    fn chunked_using_tool_echo_is_dropped_across_deltas() {
        // The echo's closing `]` can arrive in a later delta; suppression
        // carries across lines until it does.
        let mut parser = BobStreamParser::default();
        let msg =
            |c: &str| serde_json::json!({"type":"message","role":"assistant","content":c}).to_string();
        assert!(parser.parse_line(&msg("[using tool read_file: a/very")).text.is_none());
        assert!(parser.parse_line(&msg("/long/path.md")).text.is_none());
        assert!(parser.parse_line(&msg("]")).text.is_none());
        // Suppression ended; narration resumes — into the thinking trace.
        assert_eq!(parser.parse_line(&msg("ok")).thinking.as_deref(), Some("ok"));
    }

    #[test]
    fn message_prose_routes_to_thinking_only_attempt_completion_is_text() {
        // Neither bob's <thinking> reasoning NOR its plain preamble prose (both
        // ride inside assistant `message`s) is the answer — bob answers via the
        // attempt_completion tool. So the persistent parser folds ALL message
        // content into `thinking`, and only the attempt_completion text becomes
        // the visible message. (The <thinking> tags arrive as their own deltas;
        // state carries across them.)
        let mut parser = BobStreamParser::default();
        let msg = |content: &str| {
            serde_json::json!({ "type": "message", "role": "assistant", "content": content, "delta": true })
                .to_string()
        };
        // Opening tag (its own delta): the content after <thinking> is reasoning.
        let open = parser.parse_line(&msg("<thinking>\n"));
        assert_eq!(open.thinking.as_deref(), Some("\n"));
        assert!(open.text.is_none());
        // Mid-block chunk → thinking (state carried across deltas).
        let mid = parser.parse_line(&msg("the user wants X"));
        assert_eq!(mid.thinking.as_deref(), Some("the user wants X"));
        assert!(mid.text.is_none());
        // Close tag + trailing PROSE in one delta: the prose is narration, not
        // the answer → it also folds into thinking, never text.
        let close = parser.parse_line(&msg("</thinking>I'll update the file."));
        assert_eq!(close.thinking.as_deref(), Some("I'll update the file."));
        assert!(close.text.is_none());
        // Further plain message prose → thinking too.
        let more = parser.parse_line(&msg(" Working on it."));
        assert_eq!(more.thinking.as_deref(), Some(" Working on it."));
        assert!(more.text.is_none());
        // Only the attempt_completion result is the visible answer.
        let answer = parser.parse_line(
            r#"{"type":"tool_use","tool_id":"t","tool_name":"attempt_completion","parameters":{"result":"Done."}}"#,
        );
        assert_eq!(answer.text.as_deref(), Some("Done."));
        assert!(answer.thinking.is_none());
    }

    #[test]
    fn grounded_against_real_bob_capture() {
        // Verbatim shapes captured from `bob 1.0.4 -o stream-json`
        // (timestamps trimmed) — locks the parser to bob's real format.
        let mut parser = BobStreamParser::default();
        // init → Session (id + model), not empty.
        let session = parser
            .parse_line(r#"{"type":"init","session_id":"s","model":"premium"}"#)
            .session
            .expect("session");
        assert_eq!(session.session_id.as_deref(), Some("s"));
        assert_eq!(session.model.as_deref(), Some("premium"));
        // Echoed user prompt → skipped.
        assert!(parser
            .parse_line(r#"{"type":"message","role":"user","content":"list files"}"#)
            .is_empty());
        // Assistant reasoning wrapped in <thinking> → thinking, not text.
        assert_eq!(
            parser
                .parse_line(
                    r#"{"type":"message","role":"assistant","content":"<thinking>\n","delta":true}"#
                )
                .thinking
                .as_deref(),
            Some("\n")
        );
        let _ = parser.parse_line(
            r#"{"type":"message","role":"assistant","content":"</thinking>\n","delta":true}"#,
        );
        // Real tool_use (list_files) → ToolStart, parameters lifted as input.
        let start = parser
            .parse_line(r#"{"type":"tool_use","tool_name":"list_files","tool_id":"tool-1","parameters":{"dir_path":"/x/docs"}}"#)
            .tool_start
            .expect("tool_start");
        assert_eq!(start.tool_call_id, "tool-1");
        assert_eq!(start.name, "list_files");
        assert_eq!(start.input.as_deref(), Some(r#"{"dir_path":"/x/docs"}"#));
        // Real tool_result → ToolEnd(ok), output lifted.
        let end = parser
            .parse_line(r#"{"type":"tool_result","tool_id":"tool-1","status":"success","output":"Listed 11 item(s)."}"#)
            .tool_end
            .expect("tool_end");
        assert!(end.ok);
        assert_eq!(end.output.as_deref(), Some("Listed 11 item(s)."));
        // attempt_completion → the answer text.
        let answer = parser.parse_line(
            r#"{"type":"tool_use","tool_id":"tool-2","tool_name":"attempt_completion","parameters":{"result":"The docs directory contains 10 files."}}"#,
        );
        assert_eq!(answer.text.as_deref(), Some("The docs directory contains 10 files."));
        assert!(answer.tool_start.is_none());
        // result → ignored.
        assert!(parser
            .parse_line(r#"{"type":"result","status":"success","stats":{"tool_calls":2}}"#)
            .is_empty());
    }

    #[test]
    fn normalize_passes_through_lifecycle_events() {
        assert!(matches!(
            normalize_bob_event(ProcessEvent::Started { run_id: "r".into() }).as_slice(),
            [RunEvent::Started { .. }]
        ));
        assert!(matches!(
            normalize_bob_event(ProcessEvent::Exited {
                run_id: "r".into(),
                exit_code: Some(0),
                cancelled: false
            })
            .as_slice(),
            [RunEvent::Exited { exit_code: Some(0), cancelled: false, .. }]
        ));
    }
}
