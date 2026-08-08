//! Codex's `codex exec --json` parser — its JSONL wire format → the neutral
//! [`crate::RunEvent`] vocabulary.
//!
//! Two layers: the stateless [`parse_codex_line`] decodes one line into a
//! [`ParsedLine`] (tool cards, session, usage), and the stateful
//! [`CodexStreamParser`] wraps it per-run to resolve codex's
//! preamble-vs-answer ambiguity and drop its stderr noise.
//!
//! Wire format reference (verified against the official docs,
//! <https://developers.openai.com/codex/noninteractive>): `--json` emits one
//! JSON object per line. The assistant's reply is an `item.completed` event
//! whose `item.type == "agent_message"` with the full text in `item.text` —
//! Codex sends the whole message at once, not token deltas. Command
//! executions arrive as `command_execution` items; `thread.started` carries
//! the session (thread id) and `turn.completed` the token usage.

use serde_json::{Map, Value};

use crate::events::run_events_from_parsed;
use crate::{
    ParsedLine, ProcessEvent, RunEvent, SessionInfo, ToolCallEnd, ToolCallStart, ToolKind,
    UsageInfo,
};

/// A stable tool *identifier* for a codex item — the card's `name`, which the
/// consumer humanizes (see Compose's `toolLabels`). Returning an identifier
/// here rather than a display phrase keeps codex consistent with the other
/// adapters (bob's `read_file`, …) and keeps the raw command — which lives in
/// `input` — out of the `name`, so a live status never echoes a shell command.
/// `None` for non-tool items.
fn codex_tool_kind(item: &Map<String, Value>) -> Option<&'static str> {
    match item.get("type").and_then(Value::as_str)? {
        "command_execution" => Some("command_execution"),
        "file_change" => Some("file_change"),
        "web_search" => Some("web_search"),
        "mcp_tool_call" => Some("mcp_tool_call"),
        _ => None,
    }
}

/// The neutral [`ToolKind`] for a codex tool identifier (the `name` produced
/// by [`codex_tool_kind`]): codex edits files via `file_change`, runs shells
/// via `command_execution`, searches via `web_search`.
fn codex_tool_kind_class(identifier: &str) -> ToolKind {
    match identifier {
        "file_change" => ToolKind::Edit,
        "command_execution" => ToolKind::Execute,
        "web_search" => ToolKind::Search,
        _ => ToolKind::Other, // mcp_tool_call + any future identifier
    }
}

/// A human-readable fallback label for a codex tool item, used only when it
/// carries no `id` to key a card (rare — codex 0.125.0 always sends one). The
/// normal path emits [`codex_tool_kind`] as the `name` and lets the consumer
/// phrase it; this stays command-free so even the fallback can't leak a shell
/// command into the status line.
fn codex_tool_label(item: &Map<String, Value>) -> Option<String> {
    Some(
        match item.get("type").and_then(Value::as_str)? {
            "command_execution" => "Running a command",
            "file_change" => "Editing files",
            "web_search" => "Searching the web",
            "mcp_tool_call" => "Running a tool",
            _ => return None,
        }
        .to_owned(),
    )
}

/// The tool call's input, lifted inline. Only `command_execution`
/// carries a literal we can ground against (`command`); other item
/// types stream/structure their args differently, so leave them `None`
/// rather than guess.
fn codex_tool_input(item: &Map<String, Value>) -> Option<String> {
    if item.get("type").and_then(Value::as_str) == Some("command_execution") {
        return item
            .get("command")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_owned);
    }
    None
}

/// The tool call's output, lifted inline. `command_execution` reports
/// `aggregated_output`; other item types are left `None`.
fn codex_tool_output(item: &Map<String, Value>) -> Option<String> {
    if item.get("type").and_then(Value::as_str) == Some("command_execution") {
        return item
            .get("aggregated_output")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_owned);
    }
    None
}

/// Did a codex tool item succeed? `command_execution` reports
/// `exit_code` (0 = ok); otherwise fall back to `status` (anything but
/// an explicit failure is treated as ok, since not every tool type
/// carries an exit code).
fn codex_tool_ok(item: &Map<String, Value>) -> bool {
    if let Some(code) = item.get("exit_code").and_then(Value::as_i64) {
        return code == 0;
    }
    !matches!(
        item.get("status").and_then(Value::as_str),
        Some("failed") | Some("error")
    )
}

/// Stateful per-run wrapper over [`parse_codex_line`] that resolves codex's
/// preamble-vs-answer ambiguity and drops its stderr noise. One per run.
///
/// Codex emits several *complete* `agent_message` items in a turn: short
/// preambles before tool calls ("I'll read the file first") and a final
/// answer. Nothing on the item distinguishes them — the only signal is
/// position: the last `agent_message` before `turn.completed` is the answer;
/// every earlier one is a preamble. So we hold the latest `agent_message`
/// and classify it by what follows — another item means it was a preamble
/// (→ [`RunEvent::Activity`], transient narration), `turn.completed` (or
/// stream end) means it was the answer (→ [`RunEvent::Text`]). Without this
/// the preambles concatenate onto the answer in the bubble.
///
/// It also drops codex's stderr: in `--json` mode codex writes only tracing
/// logs there ("Reading additional input…", internal `ERROR codex_core::…`
/// lines) and reports real failures as stdout `error` items — so stderr is
/// pure noise, not status.
#[derive(Debug, Default)]
pub struct CodexStreamParser {
    /// The most recent `agent_message` text, not yet known to be a preamble
    /// (→ Activity) or the final answer (→ Text).
    pending_message: Option<String>,
}

impl CodexStreamParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Normalize one raw process event, applying the agent_message state
    /// machine to stdout and dropping stderr noise.
    pub fn on_process_event(&mut self, event: ProcessEvent) -> Vec<RunEvent> {
        match event {
            // codex's stderr is tracing noise in `--json` mode; real errors
            // arrive as stdout `error` items. Don't surface it as status.
            ProcessEvent::Stderr { .. } => Vec::new(),
            ProcessEvent::Started { run_id } => vec![RunEvent::Started { run_id }],
            ProcessEvent::Error { run_id, message } => {
                // Flush a held message as the answer before the terminal error.
                let mut out = self.take_pending_as_answer(&run_id);
                out.push(RunEvent::Error { run_id, message });
                out
            }
            ProcessEvent::Exited {
                run_id,
                exit_code,
                cancelled,
            } => {
                // Defensive: a turn normally ends with `turn.completed` (which
                // flushes the answer); if the stream ended without it, don't
                // lose a held final message.
                let mut out = self.take_pending_as_answer(&run_id);
                out.push(RunEvent::Exited {
                    run_id,
                    exit_code,
                    cancelled,
                });
                out
            }
            ProcessEvent::Stdout { run_id, line } => self.on_stdout(&run_id, &line),
            // `ProcessEvent` is #[non_exhaustive]; ignore any future variant.
            _ => Vec::new(),
        }
    }

    fn on_stdout(&mut self, run_id: &str, line: &str) -> Vec<RunEvent> {
        let value = serde_json::from_str::<Value>(line.trim()).ok();
        let typ = value
            .as_ref()
            .and_then(Value::as_object)
            .and_then(|o| o.get("type"))
            .and_then(Value::as_str);

        // A new assistant message arrived: whatever we held is now known to
        // be a preamble (it was superseded). Hold the new one.
        if let Some(text) = value.as_ref().and_then(codex_agent_message_text) {
            let out = self.take_pending_as_preamble(run_id);
            if !text.is_empty() {
                self.pending_message = Some(text);
            }
            return out;
        }

        // Any other line: a held message is a preamble — unless the turn just
        // ended, when it's the answer.
        let mut out = if typ == Some("turn.completed") {
            self.take_pending_as_answer(run_id)
        } else {
            self.take_pending_as_preamble(run_id)
        };
        // Non-`agent_message` lines still decode normally (tool cards,
        // session, usage, error).
        out.extend(run_events_from_parsed(run_id, parse_codex_line(line)));
        out
    }

    /// Emit a held message as transient narration (it was a preamble).
    fn take_pending_as_preamble(&mut self, run_id: &str) -> Vec<RunEvent> {
        match self.pending_message.take() {
            Some(text) if !text.is_empty() => vec![RunEvent::Activity {
                run_id: run_id.to_owned(),
                message: text,
            }],
            _ => Vec::new(),
        }
    }

    /// Emit a held message as the answer (the final assistant message).
    fn take_pending_as_answer(&mut self, run_id: &str) -> Vec<RunEvent> {
        match self.pending_message.take() {
            Some(text) if !text.is_empty() => vec![RunEvent::Text {
                run_id: run_id.to_owned(),
                delta: text,
            }],
            _ => Vec::new(),
        }
    }
}

/// The text of an `agent_message` `item.completed` line, else `None`. Returns
/// `Some("")` for an empty/absent `text` so the caller still treats the line
/// as a (superseding) message.
fn codex_agent_message_text(value: &Value) -> Option<String> {
    let obj = value.as_object()?;
    if obj.get("type").and_then(Value::as_str) != Some("item.completed") {
        return None;
    }
    let item = obj.get("item").and_then(Value::as_object)?;
    if item.get("type").and_then(Value::as_str) != Some("agent_message") {
        return None;
    }
    Some(
        item.get("text")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
    )
}

/// Parse one line of `codex exec --json` JSONL into the shared
/// [`ParsedLine`]. Assistant text is the full `agent_message` on
/// `item.completed`; tool items (`command_execution`, `file_change`,
/// `web_search`, `mcp_tool_call`) become structured tool cards
/// (`ToolStart`/`ToolEnd`). Codex edits files directly via tools
/// (reflected on disk by the file watcher), so it never emits
/// suggested-edit previews — `edits` stays empty.
pub fn parse_codex_line(line: &str) -> ParsedLine {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return ParsedLine::default();
    }
    let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
        return ParsedLine::default();
    };
    let Some(obj) = value.as_object() else {
        return ParsedLine::default();
    };

    match obj.get("type").and_then(Value::as_str) {
        Some("item.completed") => {
            let Some(item) = obj.get("item").and_then(Value::as_object) else {
                return ParsedLine::default();
            };
            // The assistant's reply: full text in one shot.
            if item.get("type").and_then(Value::as_str) == Some("agent_message") {
                if let Some(text) = item.get("text").and_then(Value::as_str) {
                    if !text.is_empty() {
                        return ParsedLine {
                            text: Some(text.to_owned()),
                            ..ParsedLine::default()
                        };
                    }
                }
            }
            // A tool item finished. Grounded in codex 0.125.0's
            // `--json` schema: command_execution / web_search /
            // file_change / mcp_tool_call items arrive on
            // `item.completed` carrying `id` + `status` (+ `exit_code`
            // for commands). It does NOT emit an `item.started` for
            // these, so emit BOTH start and end from this one event —
            // that's what makes a card appear. The frontend dedups a
            // repeated start by id, so a future codex that *does* send
            // `item.started` still renders correctly.
            if let Some(kind) = codex_tool_kind(item) {
                return match item.get("id").and_then(Value::as_str) {
                    Some(id) => {
                        let id = id.to_owned();
                        ParsedLine {
                            tool_start: Some(ToolCallStart {
                                tool_call_id: id.clone(),
                                name: kind.to_owned(),
                                input: codex_tool_input(item),
                                tool_kind: codex_tool_kind_class(kind),
                            }),
                            tool_end: Some(ToolCallEnd {
                                tool_call_id: id,
                                ok: codex_tool_ok(item),
                                output: codex_tool_output(item),
                            }),
                            ..ParsedLine::default()
                        }
                    }
                    None => ParsedLine {
                        activity: codex_tool_label(item),
                        ..ParsedLine::default()
                    },
                };
            }
            ParsedLine::default()
        }
        Some("item.started") => {
            let Some(item) = obj.get("item").and_then(Value::as_object) else {
                return ParsedLine::default();
            };
            // A tool announced before it finishes → a "running" card; the
            // matching `item.completed` flips it to done/error. If the
            // item has no `id` to key the card, degrade to a plain
            // activity line rather than drop it.
            if let Some(kind) = codex_tool_kind(item) {
                return match item.get("id").and_then(Value::as_str) {
                    Some(id) => ParsedLine {
                        tool_start: Some(ToolCallStart {
                            tool_call_id: id.to_owned(),
                            name: kind.to_owned(),
                            input: codex_tool_input(item),
                            tool_kind: codex_tool_kind_class(kind),
                        }),
                        ..ParsedLine::default()
                    },
                    None => ParsedLine {
                        activity: codex_tool_label(item),
                        ..ParsedLine::default()
                    },
                };
            }
            ParsedLine::default()
        }
        // A terminal in-band failure codex reports on stdout. It is NOT
        // transient narration — downgrading it to `activity` (as before) meant
        // a failed turn surfaced no error at all. Map it to `error` so it
        // becomes a `RunEvent::Error` the consumer can render.
        Some("error") => {
            let message = obj
                .get("message")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .unwrap_or("Codex error");
            ParsedLine {
                error: Some(truncate(message, 240)),
                ..ParsedLine::default()
            }
        }
        // thread.started → session. Codex gives a `thread_id` (its session)
        // but no model in this event, so `model` stays None.
        Some("thread.started") => ParsedLine {
            session: Some(SessionInfo {
                session_id: obj
                    .get("thread_id")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned),
                model: None,
            }),
            ..ParsedLine::default()
        },
        // turn.completed → token usage (codex reports input/output tokens).
        Some("turn.completed") => {
            let usage = obj.get("usage").and_then(Value::as_object);
            let input_tokens = usage.and_then(|u| u.get("input_tokens")).and_then(Value::as_u64);
            let output_tokens = usage.and_then(|u| u.get("output_tokens")).and_then(Value::as_u64);
            if input_tokens.is_none() && output_tokens.is_none() {
                return ParsedLine::default();
            }
            // Codex reports cache reads as `cached_input_tokens`; it has no
            // separate cache-write counter (read-through cache), so
            // `cache_write_tokens` stays None. Absent → None, unchanged.
            let cache_read_tokens = usage
                .and_then(|u| u.get("cached_input_tokens"))
                .and_then(Value::as_u64);
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
                    cache_write_tokens: None,
                }),
                ..ParsedLine::default()
            }
        }
        // turn.failed → a terminal in-band failure (quota mid-turn, context
        // overflow, model error). codex nests the reason at `error.message`
        // (verified against codex-rs `exec/src/exec_events.rs`); surface it as
        // a real error rather than silently ignoring it.
        Some("turn.failed") => {
            let message = obj
                .get("error")
                .and_then(Value::as_object)
                .and_then(|e| e.get("message"))
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .unwrap_or("Codex turn failed");
            ParsedLine {
                error: Some(truncate(message, 240)),
                ..ParsedLine::default()
            }
        }
        // turn.started / item.updated: lifecycle / partials — ignored.
        _ => ParsedLine::default(),
    }
}

fn truncate(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_message_completed_becomes_text() {
        let line = serde_json::json!({
            "type": "item.completed",
            "item": { "id": "item_3", "type": "agent_message", "text": "Repo has docs and sdk." }
        })
        .to_string();
        let parsed = parse_codex_line(&line);
        assert_eq!(parsed.text.as_deref(), Some("Repo has docs and sdk."));
        assert!(parsed.edits.is_empty());
        assert!(parsed.activity.is_none());
    }

    // Tool-card tests grounded in codex-cli 0.125.0's `--json` schema:
    // tool items (command_execution / web_search / file_change /
    // mcp_tool_call) arrive on `item.completed` carrying `id`, `status`
    // (in_progress|completed|failed) and, for commands, `exit_code`.
    // Example (verbatim from the documented schema):
    //   {"type":"item.completed","item":{"id":"item_2","type":
    //    "command_execution","command":"bash -lc false",
    //    "aggregated_output":"","exit_code":1,"status":"failed"}}

    #[test]
    fn command_execution_completed_becomes_finished_tool_card() {
        let line = serde_json::json!({
            "type": "item.completed",
            "item": {
                "id": "item_2",
                "type": "command_execution",
                "command": "bash -lc 'echo hi'",
                "aggregated_output": "hi\n",
                "exit_code": 0,
                "status": "completed"
            }
        })
        .to_string();
        let parsed = parse_codex_line(&line);
        let start = parsed.tool_start.expect("tool_start");
        let end = parsed.tool_end.expect("tool_end");
        assert_eq!(start.tool_call_id, "item_2");
        assert_eq!(end.tool_call_id, "item_2");
        // `name` is the tool *identifier* (the UI humanizes it); the command
        // rides in `input`, never in the name — so it can't leak into a
        // status line.
        assert_eq!(start.name, "command_execution");
        assert_eq!(start.input.as_deref(), Some("bash -lc 'echo hi'"));
        assert_eq!(end.output.as_deref(), Some("hi\n"));
        assert!(end.ok, "exit_code 0 → ok");
        assert!(parsed.activity.is_none());
        assert!(parsed.text.is_none());
    }

    #[test]
    fn command_execution_nonzero_exit_is_error_card() {
        let line = r#"{"type":"item.completed","item":{"id":"item_2","type":"command_execution","command":"bash -lc false","aggregated_output":"","exit_code":1,"status":"failed"}}"#;
        let end = parse_codex_line(line).tool_end.expect("tool_end");
        assert!(!end.ok, "exit_code 1 / status failed → error");
    }

    #[test]
    fn web_search_completed_becomes_tool_card() {
        let line = serde_json::json!({
            "type": "item.completed",
            "item": { "id": "item_5", "type": "web_search", "status": "completed" }
        })
        .to_string();
        let parsed = parse_codex_line(&line);
        assert_eq!(parsed.tool_start.expect("start").name, "web_search");
        assert!(parsed.tool_end.expect("end").ok, "no exit_code, status completed → ok");
    }

    #[test]
    fn started_tool_with_id_becomes_running_card() {
        let line = serde_json::json!({
            "type": "item.started",
            "item": { "id": "item_1", "type": "command_execution", "command": "bash -lc ls", "status": "in_progress" }
        })
        .to_string();
        let parsed = parse_codex_line(&line);
        let start = parsed.tool_start.expect("start");
        assert_eq!(start.name, "command_execution");
        assert_eq!(start.input.as_deref(), Some("bash -lc ls")); // input known at start
        assert!(parsed.tool_end.is_none(), "started → running (no end yet)");
        assert!(parsed.activity.is_none());
    }

    #[test]
    fn tool_without_id_degrades_to_activity() {
        // Defensive: an item lacking `id` can't key a card, so it falls
        // back to a plain activity line rather than vanishing.
        let line = serde_json::json!({
            "type": "item.completed",
            "item": { "type": "command_execution", "command": "ls -la", "exit_code": 0 }
        })
        .to_string();
        let parsed = parse_codex_line(&line);
        // No id to key a card → a command-free human fallback (never the
        // raw command).
        assert_eq!(parsed.activity.as_deref(), Some("Running a command"));
        assert!(parsed.tool_start.is_none() && parsed.tool_end.is_none());
    }

    #[test]
    fn thread_started_yields_session_and_turn_completed_yields_usage() {
        // thread.started → Session (thread id; codex reports no model here).
        let session = parse_codex_line(r#"{"type":"thread.started","thread_id":"abc"}"#)
            .session
            .expect("session");
        assert_eq!(session.session_id.as_deref(), Some("abc"));
        assert_eq!(session.model, None);

        // turn.completed → Usage.
        let usage =
            parse_codex_line(r#"{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":40}}"#)
                .usage
                .expect("usage");
        assert_eq!(usage.input_tokens, Some(100));
        assert_eq!(usage.output_tokens, Some(40));
        assert_eq!(usage.total_tokens, Some(140));

        // turn.started remains pure lifecycle.
        assert!(parse_codex_line(r#"{"type":"turn.started"}"#).is_empty());
    }

    #[test]
    fn error_event_becomes_error_not_activity() {
        // A codex `error` line is a real failure, not transient narration: it
        // must decode to `error` (→ RunEvent::Error), never to `activity`.
        let line = r#"{"type":"error","message":"rate limited"}"#;
        let parsed = parse_codex_line(line);
        assert_eq!(parsed.error.as_deref(), Some("rate limited"));
        assert!(parsed.activity.is_none());
    }

    #[test]
    fn turn_failed_becomes_error() {
        // codex reports a mid-turn failure as `turn.failed` with the reason
        // nested at `error.message`. Previously ignored (→ no answer AND no
        // error); now it surfaces as `error` → RunEvent::Error.
        let line = r#"{"type":"turn.failed","error":{"message":"context window exceeded"}}"#;
        let parsed = parse_codex_line(line);
        assert_eq!(parsed.error.as_deref(), Some("context window exceeded"));
        assert!(parsed.activity.is_none() && parsed.text.is_none());

        // Defensive: a turn.failed without a usable message still surfaces an
        // error (never silence).
        let bare = parse_codex_line(r#"{"type":"turn.failed"}"#);
        assert_eq!(bare.error.as_deref(), Some("Codex turn failed"));
    }

    #[test]
    fn non_json_is_ignored() {
        assert!(parse_codex_line("plain text").text.is_none());
    }

    // --- CodexStreamParser: preamble-vs-answer + stderr drop ----------------

    fn stdout(p: &mut CodexStreamParser, line: &str) -> Vec<RunEvent> {
        p.on_process_event(ProcessEvent::Stdout {
            run_id: "r".to_owned(),
            line: line.to_owned(),
        })
    }

    #[test]
    fn codex_preambles_are_narration_and_only_final_message_is_the_answer() {
        // The grounded multi-message turn (codex-cli 0.125.0): two preambles
        // before tool calls, then the final answer — nothing on the items
        // distinguishes them, only position does.
        let mut p = CodexStreamParser::new();
        let mut events = Vec::new();
        for line in [
            r#"{"type":"thread.started","thread_id":"t"}"#,
            r#"{"type":"item.completed","item":{"id":"m1","type":"agent_message","text":"I’m going to read a.txt first."}}"#,
            r#"{"type":"item.completed","item":{"id":"c1","type":"command_execution","command":"cat a.txt","aggregated_output":"alpha\n","exit_code":0,"status":"completed"}}"#,
            r#"{"type":"item.completed","item":{"id":"m2","type":"agent_message","text":"I’m going to read b.txt next."}}"#,
            r#"{"type":"item.completed","item":{"id":"c2","type":"command_execution","command":"cat b.txt","aggregated_output":"one\n","exit_code":0,"status":"completed"}}"#,
            r#"{"type":"item.completed","item":{"id":"m3","type":"agent_message","text":"a.txt has more lines."}}"#,
            r#"{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5}}"#,
        ] {
            events.extend(stdout(&mut p, line));
        }

        // Exactly one answer (Text) — the FINAL message; preambles are not in it.
        let texts: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                RunEvent::Text { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["a.txt has more lines."]);

        // The two preambles surface as transient Activity (narration), in order.
        let activity: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                RunEvent::Activity { message, .. } => Some(message.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            activity,
            vec![
                "I’m going to read a.txt first.",
                "I’m going to read b.txt next."
            ]
        );

        // Tool cards, session, and usage still flow through unchanged.
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, RunEvent::ToolStart { .. }))
                .count(),
            2
        );
        assert!(events.iter().any(|e| matches!(e, RunEvent::Session { .. })));
        assert!(events.iter().any(|e| matches!(e, RunEvent::Usage { .. })));
    }

    #[test]
    fn codex_single_message_turn_is_the_answer() {
        // No preamble: one agent_message → the answer, no spurious narration.
        let mut p = CodexStreamParser::new();
        let mut events = Vec::new();
        events.extend(stdout(
            &mut p,
            r#"{"type":"item.completed","item":{"id":"m1","type":"agent_message","text":"Done."}}"#,
        ));
        events.extend(stdout(
            &mut p,
            r#"{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}"#,
        ));
        let texts: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                RunEvent::Text { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["Done."]);
        assert!(!events.iter().any(|e| matches!(e, RunEvent::Activity { .. })));
    }

    #[test]
    fn codex_stderr_is_dropped_as_noise() {
        let mut p = CodexStreamParser::new();
        let out = p.on_process_event(ProcessEvent::Stderr {
            run_id: "r".to_owned(),
            line: "2026-05-31T05:20:28Z ERROR codex_core::memories::phase2::job: failed to claim job"
                .to_owned(),
        });
        assert!(out.is_empty(), "codex stderr is tracing noise → dropped, got {out:?}");
    }

    #[test]
    fn codex_turn_failed_surfaces_as_error_through_stream_parser() {
        // End-to-end on the production codex path (CodexStreamParser → not
        // normalize_process_event): a `turn.failed` stdout line must yield a
        // RunEvent::Error so the failure isn't silently swallowed.
        let mut p = CodexStreamParser::new();
        let out = stdout(
            &mut p,
            r#"{"type":"turn.failed","error":{"message":"quota exceeded"}}"#,
        );
        assert!(
            out.iter().any(
                |e| matches!(e, RunEvent::Error { message, .. } if message == "quota exceeded")
            ),
            "turn.failed must surface as RunEvent::Error, got {out:?}"
        );
    }

    #[test]
    fn codex_error_line_surfaces_as_error_through_stream_parser() {
        // The other in-band failure shape: a standalone `error` line.
        let mut p = CodexStreamParser::new();
        let out = stdout(&mut p, r#"{"type":"error","message":"rate limited"}"#);
        assert!(
            out.iter().any(
                |e| matches!(e, RunEvent::Error { message, .. } if message == "rate limited")
            ),
            "error line must surface as RunEvent::Error, got {out:?}"
        );
        // And it's a real error, not transient narration.
        assert!(!out.iter().any(|e| matches!(e, RunEvent::Activity { .. })));
    }

    #[test]
    fn codex_held_answer_is_flushed_if_stream_ends_without_turn_completed() {
        // Defensive: final message then the process exits with no
        // `turn.completed` — the answer must not be lost.
        let mut p = CodexStreamParser::new();
        let _ = stdout(
            &mut p,
            r#"{"type":"item.completed","item":{"id":"m1","type":"agent_message","text":"Final."}}"#,
        );
        let out = p.on_process_event(ProcessEvent::Exited {
            run_id: "r".to_owned(),
            exit_code: Some(0),
            cancelled: false,
        });
        assert!(
            matches!(out.first(), Some(RunEvent::Text { delta, .. }) if delta == "Final."),
            "held answer flushed as Text before Exited, got {out:?}"
        );
        assert!(matches!(out.last(), Some(RunEvent::Exited { .. })));
    }
}
