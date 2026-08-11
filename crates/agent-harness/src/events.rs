//! Normalized run events — the one shape the UI consumes regardless
//! of which harness produced them.
//!
//! Every adapter (bob's stream-json, Claude Code's stream-json,
//! Codex's format, a raw-API agent loop) parses its own wire format
//! into these variants *on the Rust side*. The front-end then learns
//! exactly one event vocabulary and never grows a per-harness
//! parser. This is the keystone of the harness abstraction: the cost
//! of adding a harness is "write a parser into `RunEvent`," not
//! "teach the UI another format."
//!
//! Suggested edits carry only the *raw* edit (path + byte range +
//! replacement). Turning those into previewable drafts needs the
//! workspace file content and the coordinate mapper, which live in
//! the consuming app layer, so that step stays there — this module's
//! job is just to lift the edit out of the harness's bespoke wire
//! format.

use serde::Serialize;

use cli_stream::Event;

/// A UTF-8 byte range into a document. Mirrors the persisted
/// `ByteOffset` discipline (see `docs/editor-guide.md`): positions
/// crossing the harness boundary are bytes, never code units.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ByteRange {
    pub start: u64,
    pub end: u64,
}

/// A raw suggested edit emitted by a harness. The app layer prepares
/// these into previewable drafts; this is the transport shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SuggestedEdit {
    pub file_path: String,
    pub range: ByteRange,
    pub replacement: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

/// A neutral, cross-harness classification of what a tool call *does* — so a
/// consumer can route by behaviour (a read → a context pill, an edit → a
/// file-op card) without re-encoding each harness's native tool vocabulary
/// (bob's `read_file`, Claude's `Read`, codex's `file_change`). The raw
/// `name` is kept alongside for display/phrasing; `tool_kind` is for
/// behaviour. Named `tool_kind` (not `kind`) so it never collides with the
/// `#[serde(tag = "kind")]` event discriminator on [`RunEvent`].
///
/// The values mirror ACP's tool-call `kind`
/// (`read`/`edit`/`delete`/`move`/`search`/`execute`/`fetch`/`other`), so a
/// [`RunEvent::ToolStart`] maps onto an ACP `tool_call` without a translation
/// table. The one divergence: our `Write` (create/overwrite a whole file) has
/// no ACP counterpart — ACP folds whole-file writes into `edit` — so an
/// ACP bridge maps `Write` → `edit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub enum ToolKind {
    /// Read or inspect a file's contents. (ACP `read`.)
    Read,
    /// Create or overwrite a whole file. (No ACP kind — bridges to `edit`.)
    Write,
    /// Modify part of an existing file. (ACP `edit`.)
    Edit,
    /// Delete a file. (ACP `delete`.)
    Delete,
    /// Move or rename a file. (ACP `move`.)
    Move,
    /// Search or list files / the web. (ACP `search`.)
    Search,
    /// Run a shell command or external process. (ACP `execute`.)
    Execute,
    /// Fetch a URL / remote resource. (ACP `fetch`.)
    Fetch,
    /// Anything else (MCP calls, task spawns, completion signals, …). (ACP `other`.)
    Other,
}

/// A file/path a tool call touches — the neutral mirror of ACP's
/// `ToolCallLocation`. Lets the app show the call's subject (and distinguish,
/// say, listing a directory from reading a file) and offer follow-along
/// navigation. Populated by the ACP adapter (passthrough) and openai-compatible's
/// own tools; empty for the CLI adapters that don't report paths structurally.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolLocation {
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
}

/// A tool call beginning — its id + name, so the UI can render a
/// state-ful card (running → done/✗) keyed by `tool_call_id`. `input`
/// carries the call's arguments when the harness delivers them inline
/// at the start (bob's `parameters`, codex's `command`); it is `None`
/// when the harness streams them incrementally (Claude's
/// `input_json_delta`), so the card stays correct either way. `tool_kind`
/// is the neutral behaviour class (see [`ToolKind`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallStart {
    pub tool_call_id: String,
    pub name: String,
    pub input: Option<String>,
    pub tool_kind: ToolKind,
}

/// A tool call finishing — matched to its start by `tool_call_id`.
/// `output` carries the tool's result when the harness reports it
/// inline at completion (bob's `tool_result.output`, codex's
/// `aggregated_output`, Claude's `tool_result.content`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallEnd {
    pub tool_call_id: String,
    pub ok: bool,
    pub output: Option<String>,
}

/// The normalized event stream. `#[serde(tag = "kind")]` +
/// camelCase mirrors the existing `Event` wire contract the TS
/// store already reads (`event.kind`, `event.runId`, …), so the
/// front-end consumes one shape regardless of which harness produced it.
// Not `Eq`: `Usage.cost_usd` is an `f64` (only `PartialEq`). `==`/`assert_eq!`
// still work; `RunEvent` just can't be a `HashSet`/`HashMap` key.
#[derive(Debug, Clone, PartialEq, Serialize)]
// `rename_all` camelCases the variant tags ("suggestedEdits"); serde
// does NOT cascade that to struct-variant fields, so `rename_all_fields`
// is required to get `runId` / `exitCode` on the wire rather than the
// snake_case Rust idents.
#[serde(tag = "kind", rename_all = "camelCase", rename_all_fields = "camelCase")]
// New event kinds (a richer Usage, a new lifecycle signal, …) can be added
// without breaking consumers — they must carry a `_` arm. Adding `Session` /
// `Usage` earlier was a breaking change precisely because this was missing.
#[non_exhaustive]
pub enum RunEvent {
    /// First event, before any output. UI shows "thinking…". Fired the
    /// instant the process spawns — *before* the CLI reports its
    /// session/model, which arrive separately as [`RunEvent::Session`].
    Started { run_id: String },
    /// The agent session is established — its id and the model in use.
    /// Distinct from `Started` because it arrives a beat later, in the
    /// CLI's first output line (bob's `init`, Claude's `system/init`,
    /// codex's `thread.started`); keeping `Started` instant matters for
    /// the "thinking…" feedback. Either field may be absent when the CLI
    /// doesn't report it (e.g. codex gives a thread id but no model).
    /// Constructible out-of-tree: any `Harness` (in-tree, or a third-party
    /// crate like `openai-compatible`) mints this directly, so it is *not*
    /// variant-`#[non_exhaustive]` — sealing it would break the open-producer
    /// contract (see the note on [`RunEvent::Exited`]).
    Session {
        run_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        session_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        model: Option<String>,
    },
    /// A chunk of assistant text. Appended to the active message.
    Text { run_id: String, delta: String },
    /// A chunk of model reasoning ("thinking"), rendered distinctly from
    /// `Text` so the UI can show reasoning without mixing it into the
    /// answer (e.g. Claude's `thinking_delta`).
    Thinking { run_id: String, delta: String },
    /// A tool call started — render a state-ful card keyed by id. Mirrors ACP's
    /// `ToolCall`: `title` + `kind` + `locations` (files it touches) + the raw
    /// arguments (`raw_input`, omitted when streamed separately, e.g. Claude).
    ToolStart {
        run_id: String,
        tool_call_id: String,
        title: String,
        // ACP calls this `kind`; our wire reserves `kind` for the event tag, so
        // it stays `tool_kind` (→ `toolKind` on the wire).
        tool_kind: ToolKind,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        locations: Vec<ToolLocation>,
        #[serde(skip_serializing_if = "Option::is_none")]
        raw_input: Option<String>,
    },
    /// A tool call finished (matched to its start by id). Mirrors ACP's tool
    /// result: `content` (human-readable, flattened to text) + `raw_output`
    /// (structured JSON) + the `locations` it touched. `ok` reduces ACP's
    /// terminal `Completed`/`Failed` status to a flag.
    ToolEnd {
        run_id: String,
        tool_call_id: String,
        ok: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        raw_output: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        locations: Vec<ToolLocation>,
    },
    /// One or more proposed edits. The app prepares + previews them.
    SuggestedEdits {
        run_id: String,
        edits: Vec<SuggestedEdit>,
    },
    /// A human-readable status line (tool call, file touch, edit
    /// count). Replaces the message's transient activity text.
    Activity { run_id: String, message: String },
    /// Token accounting for the run, emitted near its end (from the
    /// CLI's `result` / `turn.completed`). Neutral tokens only —
    /// harness-specific costs/credits (bob's coins) are NOT here; a
    /// consumer that wants them reads the harness's own output. Any
    /// field may be absent when the CLI doesn't break usage down.
    ///
    /// `cache_read_tokens` / `cache_write_tokens` are the prompt-cache
    /// counters reported *separately* from `input_tokens` (Claude's
    /// `cache_read_input_tokens` / `cache_creation_input_tokens`) — not folded
    /// into `input_tokens`, and omitted from the wire when the CLI doesn't
    /// report caching.
    Usage {
        run_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        input_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        output_tokens: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        total_tokens: Option<u64>,
        /// Prompt-cache tokens served from cache this run (~0.1x input cost).
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_read_tokens: Option<u64>,
        /// Prompt-cache tokens written to cache this run (~1.25x input cost).
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_write_tokens: Option<u64>,
        /// Estimated cost in USD for this run, when the adapter knows per-token
        /// rates (`openai-compatible` via `with_model_cost`); `None` otherwise.
        #[serde(skip_serializing_if = "Option::is_none")]
        cost_usd: Option<f64>,
    },
    /// The agent is asking the user one or more multiple-choice questions
    /// (Claude's `AskUserQuestion`, Codex's `tool/requestUserInput`). The host
    /// renders the options as selectable chips; the user's pick is sent back as
    /// their **next message** on the existing chat path (which resumes the
    /// session), so the agent continues with the answer in hand. Carrying the
    /// questions as a neutral event keeps the harness-specific tool shape in the
    /// adapter — the host never name-checks `AskUserQuestion` (cf. `ToolKind`).
    AskQuestion {
        run_id: String,
        /// Identifies this question instance (the harness's tool-call id), so
        /// the host can tie the answer + clear the chips for the right one.
        request_id: String,
        questions: Vec<Question>,
    },
    /// The agent's current task plan / todo list (Claude's `TodoWrite`,
    /// Codex's plan items), replacing any prior plan for this run. The host
    /// renders a checklist without knowing the harness's native plan tool.
    /// The neutral plan vocabulary adapters map onto — the `acp` adapter from
    /// ACP `plan`, and `openai-compatible` from its `todowrite` tool.
    Plan { run_id: String, entries: Vec<PlanEntry> },
    /// A live update to the session's display metadata — its title and/or
    /// last-updated time — emitted mid-run when the agent (re)names the
    /// conversation. Maps field-for-field onto ACP `session_info_update`
    /// (`title` + `updatedAt`); lets a sessions list show a meaningful title
    /// before the run ends. Both fields optional (a partial update).
    SessionInfoUpdate {
        run_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        /// ISO-8601 timestamp of the update, when the harness reports one
        /// (ACP `updatedAt`). Omitted from the wire when absent.
        #[serde(skip_serializing_if = "Option::is_none")]
        updated_at: Option<String>,
    },
    /// Command / IO / parse failure. Terminal — followed by `Exited`.
    Error { run_id: String, message: String },
    /// The run finished. Sent exactly once. Like every `RunEvent` variant it is
    /// constructible out-of-tree: harnesses live in their own crates (the
    /// `Registry` is open — `openai-compatible`, plus `examples/custom_harness.rs`),
    /// so no *produced* variant is variant-`#[non_exhaustive]` — that would
    /// close the producer door. The enum itself stays `#[non_exhaustive]`, which
    /// protects *consumers* (a new variant just needs a `_` arm) without
    /// blocking construction of the existing ones.
    Exited {
        run_id: String,
        exit_code: Option<i32>,
        cancelled: bool,
    },
}

/// Session identity decoded from a harness's init line → `RunEvent::Session`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SessionInfo {
    pub session_id: Option<String>,
    pub model: Option<String>,
}

/// Token accounting decoded from a harness's result line → `RunEvent::Usage`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct UsageInfo {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    /// Prompt-cache tokens read this run (Claude's `cache_read_input_tokens`).
    pub cache_read_tokens: Option<u64>,
    /// Prompt-cache tokens written (Claude's `cache_creation_input_tokens`).
    pub cache_write_tokens: Option<u64>,
}

/// One multiple-choice question carried by [`RunEvent::AskQuestion`]. The
/// neutral shape every adapter maps its harness's question tool onto. Wire-out
/// only (Serialize), like the rest of [`RunEvent`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Question {
    /// Short label for the question (Claude's `header`); optional.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub header: Option<String>,
    /// The question text shown to the user.
    pub prompt: String,
    pub options: Vec<QuestionOption>,
    /// Whether more than one option may be selected.
    pub multi_select: bool,
    /// Whether a free-text ("Other") answer is allowed alongside the options.
    pub allow_free_text: bool,
}

/// One selectable option of a [`Question`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QuestionOption {
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// One entry in a [`RunEvent::Plan`] — a single task the agent is tracking.
/// Wire-out only (Serialize), like the rest of [`RunEvent`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanEntry {
    /// The step text (the todo content).
    pub content: String,
    pub status: PlanEntryStatus,
    /// Relative importance when the harness ranks steps; omitted otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<PlanEntryPriority>,
}

/// Lifecycle of a [`PlanEntry`]. `#[non_exhaustive]` — a harness may report
/// states beyond these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub enum PlanEntryStatus {
    Pending,
    InProgress,
    Completed,
    /// The step was abandoned (OpenCode's `cancelled` todo status).
    Cancelled,
}

/// Relative priority of a [`PlanEntry`]. `#[non_exhaustive]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
#[non_exhaustive]
pub enum PlanEntryPriority {
    Low,
    Medium,
    High,
}

/// What a single harness output line decoded to. A line can yield
/// text *and* edits at once, so this is not one-event-per-line.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ParsedLine {
    pub text: Option<String>,
    /// Model reasoning chunk → `RunEvent::Thinking`. Kept separate from
    /// `text` so the UI can render it distinctly.
    pub thinking: Option<String>,
    /// Session identity (id + model) → `RunEvent::Session`.
    pub session: Option<SessionInfo>,
    /// A tool call began → `RunEvent::ToolStart`.
    pub tool_start: Option<ToolCallStart>,
    /// A tool call finished → `RunEvent::ToolEnd`.
    pub tool_end: Option<ToolCallEnd>,
    pub edits: Vec<SuggestedEdit>,
    /// Token accounting → `RunEvent::Usage`.
    pub usage: Option<UsageInfo>,
    pub activity: Option<String>,
    /// An in-band failure the harness reported on its *stdout* (codex's
    /// `turn.failed` / `error` lines) → `RunEvent::Error`. Terminal. Kept
    /// distinct from `activity` so a real failure (quota mid-turn, context
    /// overflow, model error) surfaces as an error instead of being downgraded
    /// to transient narration — otherwise a failed turn yields no answer *and*
    /// no error, looking like the harness silently did nothing.
    pub error: Option<String>,
    /// A harness asked the user a multiple-choice question (Claude's
    /// `AskUserQuestion`) → `RunEvent::AskQuestion`. The tuple is `(request_id,
    /// questions)`: the tool-call id the host echoes when tying the answer and
    /// clearing the chips, plus the parsed questions. The host renders the
    /// options as chips; the answer returns as the user's next message.
    pub ask_question: Option<(String, Vec<Question>)>,
    /// The agent's current plan / todo list → `RunEvent::Plan`. No built-in
    /// parser fills this yet; it's wired for adapters that will.
    pub plan: Option<Vec<PlanEntry>>,
    /// A live session-title update → `RunEvent::SessionInfoUpdate`.
    pub title: Option<String>,
}

impl ParsedLine {
    /// True when a line decoded to no actionable content. A useful
    /// predicate for adapters + their tests; the normalize skeleton
    /// relies instead on the natural no-op of pushing zero events.
    pub fn is_empty(&self) -> bool {
        self.text.is_none()
            && self.thinking.is_none()
            && self.session.is_none()
            && self.tool_start.is_none()
            && self.tool_end.is_none()
            && self.edits.is_empty()
            && self.usage.is_none()
            && self.activity.is_none()
            && self.error.is_none()
            && self.ask_question.is_none()
            && self.plan.is_none()
            && self.title.is_none()
    }
}

/// Translate one raw process event into zero or more normalized
/// [`RunEvent`]s, using `parse_line` to decode the harness's stdout
/// wire format. Lifecycle events (Started / Exited / Error) and
/// stderr are harness-neutral and handled here; only the stdout
/// parsing differs per harness — so every process-backed adapter
/// shares this skeleton and supplies just its own line parser.
pub fn normalize_process_event(
    event: Event,
    mut parse_line: impl FnMut(&str) -> ParsedLine,
) -> Vec<RunEvent> {
    match event {
        Event::Started { run_id } => vec![RunEvent::Started { run_id }],
        Event::Exited {
            run_id,
            exit_code,
            cancelled,
        } => vec![RunEvent::Exited {
            run_id,
            exit_code,
            cancelled,
        }],
        Event::Error { run_id, message } => vec![RunEvent::Error { run_id, message }],
        Event::Stderr { run_id, line } => {
            // stderr is warnings/progress; surface as activity,
            // truncated like the TS store did (240 chars).
            let message = truncate(&line, 240);
            if message.is_empty() {
                vec![]
            } else {
                vec![RunEvent::Activity { run_id, message }]
            }
        }
        Event::Stdout { run_id, line } => run_events_from_parsed(&run_id, parse_line(&line)),
        // `Event` is #[non_exhaustive]; a future variant yields no
        // events until an adapter learns to handle it.
        _ => Vec::new(),
    }
}

/// Expand a decoded [`ParsedLine`] into its [`RunEvent`]s for `run_id`, in a
/// stable order: session (the run's init) → session-info/title → text →
/// thinking → tool start/end → edits → plan → usage (end of turn) → activity
/// → error (a terminal in-band failure, emitted last so any text/usage on the
/// same line lands before it) → ask-question.
///
/// Used by [`normalize_process_event`] and by adapters that wrap the line
/// parser in their own per-run state (e.g. codex's preamble-vs-answer state
/// machine, which decides *where* a message goes but still relies on this
/// for everything else) — so the `ParsedLine` → `RunEvent` mapping lives in
/// exactly one place.
///
/// Public so an **out-of-tree** harness can build a stateful parser the same
/// way: decide your own routing per line, then call this to expand a
/// `ParsedLine` into events with the canonical ordering — instead of
/// hand-rolling (and drifting from) the mapping. See `examples/custom_harness.rs`.
pub fn run_events_from_parsed(run_id: &str, parsed: ParsedLine) -> Vec<RunEvent> {
    let mut out = Vec::new();
    if let Some(session) = parsed.session {
        out.push(RunEvent::Session {
            run_id: run_id.to_owned(),
            session_id: session.session_id,
            model: session.model,
        });
    }
    if let Some(title) = parsed.title {
        out.push(RunEvent::SessionInfoUpdate {
            run_id: run_id.to_owned(),
            title: Some(title),
            // No parser supplies a timestamp yet; an adapter that has one
            // constructs SessionInfoUpdate directly.
            updated_at: None,
        });
    }
    if let Some(text) = parsed.text {
        out.push(RunEvent::Text {
            run_id: run_id.to_owned(),
            delta: text,
        });
    }
    if let Some(thinking) = parsed.thinking {
        out.push(RunEvent::Thinking {
            run_id: run_id.to_owned(),
            delta: thinking,
        });
    }
    if let Some(start) = parsed.tool_start {
        out.push(RunEvent::ToolStart {
            run_id: run_id.to_owned(),
            tool_call_id: start.tool_call_id,
            title: start.name,
            tool_kind: start.tool_kind,
            // CLI adapters don't report structured paths; ACP + openai-compatible do.
            locations: Vec::new(),
            raw_input: start.input,
        });
    }
    if let Some(end) = parsed.tool_end {
        out.push(RunEvent::ToolEnd {
            run_id: run_id.to_owned(),
            tool_call_id: end.tool_call_id,
            ok: end.ok,
            content: end.output,
            raw_output: None,
            locations: Vec::new(),
        });
    }
    if !parsed.edits.is_empty() {
        out.push(RunEvent::SuggestedEdits {
            run_id: run_id.to_owned(),
            edits: parsed.edits,
        });
    }
    if let Some(entries) = parsed.plan {
        out.push(RunEvent::Plan {
            run_id: run_id.to_owned(),
            entries,
        });
    }
    if let Some(usage) = parsed.usage {
        out.push(RunEvent::Usage {
            run_id: run_id.to_owned(),
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
            cache_read_tokens: usage.cache_read_tokens,
            cache_write_tokens: usage.cache_write_tokens,
            // CLI adapters don't compute cost; openai-compatible does (with rates).
            cost_usd: None,
        });
    }
    if let Some(activity) = parsed.activity {
        out.push(RunEvent::Activity {
            run_id: run_id.to_owned(),
            message: activity,
        });
    }
    // A harness reported an in-band failure on its stdout. This is the one
    // place a parsed line can become `RunEvent::Error` (the only other source
    // is a process-level `Event::Error`), so an in-band failure from any
    // harness — not just a spawn/IO failure — reaches the consumer.
    if let Some(message) = parsed.error {
        out.push(RunEvent::Error {
            run_id: run_id.to_owned(),
            message,
        });
    }
    if let Some((request_id, questions)) = parsed.ask_question {
        out.push(RunEvent::AskQuestion {
            run_id: run_id.to_owned(),
            request_id,
            questions,
        });
    }
    out
}

/// Take the first `max_chars` characters (not bytes) of `s`. Bounds the
/// stderr activity line without splitting a multi-byte char.
fn truncate(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A line parser that yields nothing — exercises the neutral
    /// skeleton without any harness-specific decoding.
    fn empty_parser(_: &str) -> ParsedLine {
        ParsedLine::default()
    }

    #[test]
    fn normalize_passes_through_lifecycle_events() {
        assert!(matches!(
            normalize_process_event(Event::Started { run_id: "r".into() }, empty_parser)
                .as_slice(),
            [RunEvent::Started { .. }]
        ));
        assert!(matches!(
            normalize_process_event(
                Event::Exited {
                    run_id: "r".into(),
                    exit_code: Some(0),
                    cancelled: false
                },
                empty_parser
            )
            .as_slice(),
            [RunEvent::Exited { exit_code: Some(0), cancelled: false, .. }]
        ));
    }

    #[test]
    fn stderr_becomes_truncated_activity() {
        let long = "x".repeat(500);
        let events = normalize_process_event(
            Event::Stderr {
                run_id: "r1".into(),
                line: long,
            },
            empty_parser,
        );
        match events.as_slice() {
            [RunEvent::Activity { run_id, message }] => {
                assert_eq!(run_id, "r1");
                assert_eq!(message.chars().count(), 240);
            }
            other => panic!("expected one Activity, got {other:?}"),
        }
        // Empty stderr line → no event.
        assert!(normalize_process_event(
            Event::Stderr {
                run_id: "r1".into(),
                line: String::new(),
            },
            empty_parser,
        )
        .is_empty());
    }

    #[test]
    fn thinking_normalizes_and_serializes() {
        let events = normalize_process_event(
            Event::Stdout {
                run_id: "r1".to_owned(),
                line: "ignored".to_owned(),
            },
            |_| ParsedLine {
                thinking: Some("pondering".to_owned()),
                ..ParsedLine::default()
            },
        );
        assert!(matches!(
            events.as_slice(),
            [RunEvent::Thinking { run_id, delta }] if run_id == "r1" && delta == "pondering"
        ));
        let json = serde_json::to_value(RunEvent::Thinking {
            run_id: "r1".to_owned(),
            delta: "d".to_owned(),
        })
        .unwrap();
        assert_eq!(json["kind"], "thinking");
        assert_eq!(json["runId"], "r1");
        assert_eq!(json["delta"], "d");
    }

    #[test]
    fn run_event_serializes_with_kind_and_camelcase() {
        let json = serde_json::to_value(RunEvent::Exited {
            run_id: "r1".to_owned(),
            exit_code: Some(2),
            cancelled: true,
        })
        .unwrap();
        assert_eq!(json["kind"], "exited");
        assert_eq!(json["runId"], "r1");
        assert_eq!(json["exitCode"], 2);
        assert_eq!(json["cancelled"], true);
    }

    #[test]
    fn ask_question_serializes_with_camelcase_and_skips_empty_description() {
        let json = serde_json::to_value(RunEvent::AskQuestion {
            run_id: "r1".to_owned(),
            request_id: "q-7".to_owned(),
            questions: vec![Question {
                header: Some("Scope".to_owned()),
                prompt: "Which files?".to_owned(),
                options: vec![
                    QuestionOption { label: "All".to_owned(), description: None },
                    QuestionOption {
                        label: "Changed only".to_owned(),
                        description: Some("Just the diff".to_owned()),
                    },
                ],
                multi_select: true,
                allow_free_text: false,
            }],
        })
        .unwrap();
        assert_eq!(json["kind"], "askQuestion");
        assert_eq!(json["runId"], "r1");
        assert_eq!(json["requestId"], "q-7");
        let q = &json["questions"][0];
        assert_eq!(q["header"], "Scope");
        assert_eq!(q["prompt"], "Which files?");
        assert_eq!(q["multiSelect"], true);
        assert_eq!(q["allowFreeText"], false);
        assert_eq!(q["options"][0]["label"], "All");
        // A `None` description is omitted from the wire (skip_serializing_if).
        assert!(q["options"][0].get("description").is_none());
        assert_eq!(q["options"][1]["description"], "Just the diff");
    }

    #[test]
    fn session_normalizes_and_serializes() {
        let events = normalize_process_event(
            Event::Stdout {
                run_id: "r1".to_owned(),
                line: "ignored".to_owned(),
            },
            |_| ParsedLine {
                session: Some(SessionInfo {
                    session_id: Some("sess-1".to_owned()),
                    model: Some("opus".to_owned()),
                }),
                ..ParsedLine::default()
            },
        );
        assert!(matches!(
            events.as_slice(),
            [RunEvent::Session { run_id, session_id, model }]
                if run_id == "r1"
                    && session_id.as_deref() == Some("sess-1")
                    && model.as_deref() == Some("opus")
        ));
        let json = serde_json::to_value(RunEvent::Session {
            run_id: "r1".to_owned(),
            session_id: Some("sess-1".to_owned()),
            model: None,
        })
        .unwrap();
        assert_eq!(json["kind"], "session");
        assert_eq!(json["sessionId"], "sess-1");
        // model omitted from the wire when None (backward-compatible).
        assert!(json.get("model").is_none());
    }

    #[test]
    fn usage_normalizes_and_serializes() {
        let events = normalize_process_event(
            Event::Stdout {
                run_id: "r1".to_owned(),
                line: "ignored".to_owned(),
            },
            |_| ParsedLine {
                usage: Some(UsageInfo {
                    input_tokens: Some(10),
                    output_tokens: Some(20),
                    total_tokens: Some(30),
                    cache_read_tokens: Some(8),
                    cache_write_tokens: Some(2),
                }),
                ..ParsedLine::default()
            },
        );
        assert!(matches!(
            events.as_slice(),
            [RunEvent::Usage {
                run_id,
                input_tokens: Some(10),
                output_tokens: Some(20),
                total_tokens: Some(30),
                cache_read_tokens: Some(8),
                cache_write_tokens: Some(2),
                cost_usd: None,
            }] if run_id == "r1"
        ));
        // Cache counters ride on the wire as camelCase; omitted when None.
        let json = serde_json::to_value(RunEvent::Usage {
            run_id: "r1".to_owned(),
            input_tokens: Some(10),
            output_tokens: None,
            total_tokens: Some(30),
            cache_read_tokens: Some(8),
            cache_write_tokens: None,
            cost_usd: None,
        })
        .unwrap();
        assert_eq!(json["kind"], "usage");
        assert_eq!(json["inputTokens"], 10);
        assert_eq!(json["totalTokens"], 30);
        assert_eq!(json["cacheReadTokens"], 8);
        assert!(json.get("outputTokens").is_none()); // omitted when None
        assert!(json.get("cacheWriteTokens").is_none()); // omitted when None
    }

    #[test]
    fn plan_normalizes_and_serializes() {
        let events = normalize_process_event(
            Event::Stdout {
                run_id: "r1".to_owned(),
                line: "ignored".to_owned(),
            },
            |_| ParsedLine {
                plan: Some(vec![
                    PlanEntry {
                        content: "Write the parser".to_owned(),
                        status: PlanEntryStatus::InProgress,
                        priority: Some(PlanEntryPriority::High),
                    },
                    PlanEntry {
                        content: "Add tests".to_owned(),
                        status: PlanEntryStatus::Pending,
                        priority: None,
                    },
                ]),
                ..ParsedLine::default()
            },
        );
        assert!(matches!(
            events.as_slice(),
            [RunEvent::Plan { run_id, entries }] if run_id == "r1" && entries.len() == 2
        ));
        let json = serde_json::to_value(RunEvent::Plan {
            run_id: "r1".to_owned(),
            entries: vec![PlanEntry {
                content: "Add tests".to_owned(),
                status: PlanEntryStatus::Pending,
                priority: None,
            }],
        })
        .unwrap();
        assert_eq!(json["kind"], "plan");
        assert_eq!(json["entries"][0]["content"], "Add tests");
        assert_eq!(json["entries"][0]["status"], "pending");
        // priority omitted when None.
        assert!(json["entries"][0].get("priority").is_none());
    }

    #[test]
    fn session_info_update_serializes() {
        let json = serde_json::to_value(RunEvent::SessionInfoUpdate {
            run_id: "r1".to_owned(),
            title: Some("Refactor the parser".to_owned()),
            updated_at: Some("2026-06-16T12:00:00Z".to_owned()),
        })
        .unwrap();
        assert_eq!(json["kind"], "sessionInfoUpdate");
        assert_eq!(json["runId"], "r1");
        assert_eq!(json["title"], "Refactor the parser");
        // Maps onto ACP `session_info_update.updatedAt`.
        assert_eq!(json["updatedAt"], "2026-06-16T12:00:00Z");
    }

    #[test]
    fn tool_io_is_carried_and_omitted_when_absent() {
        // input on ToolStart, output on ToolEnd — distinct events, distinct moments.
        let start = normalize_process_event(
            Event::Stdout {
                run_id: "r1".to_owned(),
                line: "ignored".to_owned(),
            },
            |_| ParsedLine {
                tool_start: Some(ToolCallStart {
                    tool_call_id: "t1".to_owned(),
                    name: "ls".to_owned(),
                    input: Some("{\"dir\":\"/x\"}".to_owned()),
                    tool_kind: ToolKind::Other,
                }),
                ..ParsedLine::default()
            },
        );
        assert!(matches!(
            start.as_slice(),
            [RunEvent::ToolStart { raw_input: Some(i), .. }] if i == "{\"dir\":\"/x\"}"
        ));
        // A ToolStart with no input omits the field on the wire (byte-identical
        // to the pre-enrichment shape).
        let json = serde_json::to_value(RunEvent::ToolStart {
            run_id: "r1".to_owned(),
            tool_call_id: "t1".to_owned(),
            title: "ls".to_owned(),
            tool_kind: ToolKind::Execute,
            locations: Vec::new(),
            raw_input: None,
        })
        .unwrap();
        assert_eq!(json["kind"], "toolStart");
        // The neutral class rides alongside as `toolKind` — distinct wire key
        // from the `kind` event discriminator (no collision).
        assert_eq!(json["toolKind"], "execute");
        assert_eq!(json["toolCallId"], "t1");
        assert!(json.get("rawInput").is_none());

        let json = serde_json::to_value(RunEvent::ToolEnd {
            run_id: "r1".to_owned(),
            tool_call_id: "t1".to_owned(),
            ok: true,
            content: Some("done".to_owned()),
            raw_output: None,
            locations: Vec::new(),
        })
        .unwrap();
        assert_eq!(json["kind"], "toolEnd");
        assert_eq!(json["content"], "done");
    }

    #[test]
    fn parsed_error_normalizes_to_run_event_error() {
        // An in-band failure decoded onto a ParsedLine surfaces as
        // RunEvent::Error — the path codex's `turn.failed` / `error` lines now
        // take, so a failed turn no longer yields neither answer nor error.
        let events = normalize_process_event(
            Event::Stdout {
                run_id: "r1".to_owned(),
                line: "ignored".to_owned(),
            },
            |_| ParsedLine {
                error: Some("rate limited".to_owned()),
                ..ParsedLine::default()
            },
        );
        assert!(matches!(
            events.as_slice(),
            [RunEvent::Error { run_id, message }] if run_id == "r1" && message == "rate limited"
        ));
        // An error-only line is not empty (is_empty stays honest).
        assert!(!ParsedLine {
            error: Some("x".to_owned()),
            ..ParsedLine::default()
        }
        .is_empty());
    }

    #[test]
    fn suggested_edits_event_serializes_camelcase() {
        let json = serde_json::to_value(RunEvent::SuggestedEdits {
            run_id: "r1".to_owned(),
            edits: vec![SuggestedEdit {
                file_path: "a.md".to_owned(),
                range: ByteRange { start: 1, end: 2 },
                replacement: "x".to_owned(),
                title: None,
            }],
        })
        .unwrap();
        assert_eq!(json["kind"], "suggestedEdits");
        assert_eq!(json["edits"][0]["filePath"], "a.md");
        assert_eq!(json["edits"][0]["range"]["start"], 1);
        // title omitted when None
        assert!(json["edits"][0].get("title").is_none());
    }
}
