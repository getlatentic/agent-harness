//! The synchronous agent loop for the direct-model adapter, run on the worker
//! thread `OpenHarness::run` spawns.
//!
//! It POSTs the conversation to the chat endpoint, streams the assistant text
//! out as [`RunEvent`]s, dispatches any tool calls the model makes to the
//! built-in [`super::tools`], feeds the results back, and loops until the
//! model stops calling tools (or the turn cap / cancel fires). Non-streaming
//! for now: each `/v1/chat/completions` returns the whole message, so text is
//! emitted as one [`RunEvent::Text`] per turn (token deltas are a follow-up).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::{HarnessError, RunCallback, RunControl, RunEvent, RunMode};

use super::instructions;
use super::ollama;
use super::session::{self, FileStore};
use super::skills;
use super::tools;
use super::wire::{self, ChatMessage};

/// Default cap on agent loop iterations when the host doesn't set one — a
/// backstop against a model that never stops calling tools.
const DEFAULT_MAX_TURNS: u32 = 25;

/// Cancel handle for an in-flight direct-model run. Cooperative: the loop
/// checks the flag at each turn boundary.
pub(crate) struct OpenAiRun {
    cancel: Arc<AtomicBool>,
}

impl OpenAiRun {
    pub(crate) fn new(cancel: Arc<AtomicBool>) -> Self {
        Self { cancel }
    }
}

impl RunControl for OpenAiRun {
    fn cancel(&self) -> Result<(), HarnessError> {
        self.cancel.store(true, Ordering::SeqCst);
        Ok(())
    }
    fn was_cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

/// Everything the loop needs, assembled by `run()` from the [`crate::RunRequest`].
pub(crate) struct LoopConfig {
    pub run_id: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub prompt: String,
    pub cwd: PathBuf,
    pub mode: RunMode,
    pub max_turns: u32,
    /// The session to resume — its stored transcript is replayed before the new
    /// prompt; `None` starts a fresh session.
    pub resume: Option<String>,
    /// Where to persist the session, when the harness has one configured;
    /// `None` runs ephemerally (no disk writes, and resume is unavailable).
    pub store: Option<FileStore>,
    /// The model's context-window size in tokens, when known — enables
    /// compaction near the limit; `None` disables it.
    pub context_tokens: Option<u64>,
    /// When set, talk to Ollama's native `/api/chat` with this `num_ctx` instead
    /// of the OpenAI `/v1` endpoint (which ignores it and loads every model at
    /// 4096, truncating the prompt). Always equals `context_tokens` for Ollama;
    /// `None` for OpenAI-compatible providers, which use `/v1`.
    pub ollama_num_ctx: Option<u64>,
    /// Named subagents the `task` tool can spawn via `subagent_type`.
    pub agents: Vec<(String, crate::openai_compatible::AgentDef)>,
    /// MCP servers to launch over stdio and expose their tools to the model.
    pub mcp_servers: Vec<crate::openai_compatible::McpServer>,
    /// JSON Schema the model's final answer must conform to (structured output);
    /// `None` → free-form text.
    pub output_schema: Option<Value>,
    /// Per-token pricing for the run's model, for cost estimation on
    /// `RunEvent::Usage`; `None` → no cost emitted.
    pub model_cost: Option<crate::openai_compatible::ModelCost>,
    /// Image data URIs to attach to the first user message (multimodal input);
    /// empty for a text-only run.
    pub image_data_uris: Vec<String>,
    /// Pre-execution permission rules gating tool calls.
    pub permissions: Vec<crate::openai_compatible::PermissionRule>,
    /// Host callback consulted for `Permission::Ask` decisions.
    pub permission_prompt: Option<crate::openai_compatible::PermissionPrompt>,
    /// Inline reasoning tag to lift out of streamed output into `Thinking`
    /// (e.g. `Some("think")` for `<think>…</think>`); `None` disables it.
    pub reasoning_tag: Option<String>,
    /// The host's per-harness "custom instructions", appended to the system
    /// prompt as a final section; `None`/blank adds nothing.
    pub extra_instructions: Option<String>,
}

impl LoopConfig {
    pub(crate) fn max_turns_or_default(max_turns: Option<u32>) -> u32 {
        max_turns.filter(|&n| n > 0).unwrap_or(DEFAULT_MAX_TURNS)
    }
}

/// The base system prompt. Regenerated each run (it is *not* part of the
/// persisted transcript), so it can grow — Stage D appends the available-skills
/// catalog here.
const SYSTEM_PROMPT: &str = "You are a careful AI assistant working in the \
    user's files. Do exactly what the user asks — no more, no less — and \
    follow their instructions precisely.\n\
    \n\
    Match the request to the right action:\n\
    - A question, summary, explanation, review, or analysis is a READ-ONLY \
    task: read what you need, then answer directly in your reply. Do NOT \
    create, edit, or overwrite any file for these.\n\
    - Only use a write or edit tool when the user clearly asks you to create \
    or change a file. Then make the smallest change that satisfies the \
    request and keep the user's existing content and style.\n\
    - If the request is ambiguous, ask one brief clarifying question instead \
    of guessing or editing.\n\
    \n\
    Tools (paths are relative to the working directory): `read` to inspect a \
    file; `glob`, `grep`, and `list` to find files and content; `edit` for a \
    targeted change to an existing file; `write` to create or fully replace \
    one; `bash` for builds, tests, and git.\n\
    \n\
    To see what files exist or to find one, call `list` or `glob` first — \
    never guess file names or their contents from memory.\n\
    \n\
    If a write or edit is refused because the run is read-only, do NOT retry \
    it. Tell the user the run is read-only and that they can turn on editing, \
    then answer their request without changing files.\n\
    \n\
    When the task is done, reply with a short, clear final message and make \
    no further tool calls.";

/// Appended as the last message on every read-only (Ask) turn. Small local
/// models attend most to the end of the prompt and honor the system-message
/// hierarchy poorly, so the per-turn constraint is repeated here in the user
/// channel. The mutator tools are also withheld in Ask mode, so this is the
/// prompt half of the same read-only guarantee, not the only line of defense.
const READ_ONLY_REMINDER: &str = "Reminder: this is a read-only request. Do not \
    create, edit, or overwrite any file. Read what you need, then reply with your \
    answer directly.";

/// Build the STABLE part of the system prompt: base instructions, then project
/// instruction files (AGENTS.md / CLAUDE.md), then the available-skills catalog.
/// The volatile environment (working directory) is appended LAST by the caller
/// via [`environment_block`], so everything here stays a byte-identical,
/// cache-friendly prefix across runs in the same workspace. `cwd` only locates
/// the instruction / skill files.
fn build_system_prompt(base: &str, cwd: &Path, skills: &[skills::Skill]) -> String {
    let mut prompt = base.to_owned();
    if let Some(text) = instructions::gather(cwd) {
        prompt.push_str("\n\n# Project instructions\n");
        prompt.push_str(&text);
    }
    if let Some(catalog) = skills::catalog(skills) {
        prompt.push_str(&catalog);
    }
    prompt
}

/// The VARIABLE tail of the system prompt: the working directory, appended
/// after the fixed + user-configurable sections so everything above stays a
/// cacheable prefix. Deliberately minimal — no per-run file list (the model
/// discovers files with `list`/`glob`), so this block is identical for every
/// run in a given workspace.
fn environment_block(cwd: &Path) -> String {
    format!("\n\n# Environment\nWorking directory: {}", cwd.display())
}

/// A catalog of the registered subagents for the `task` tool's `subagent_type`,
/// or `None` if none are registered.
fn agent_catalog(agents: &[(String, crate::openai_compatible::AgentDef)]) -> Option<String> {
    if agents.is_empty() {
        return None;
    }
    let mut out = String::from(
        "\n\n## Subagent types\nPass one as the `task` tool's `subagent_type` (omit for the default coding agent):\n",
    );
    for (name, def) in agents {
        out.push_str(&format!("- `{name}` — {}\n", def.description));
    }
    Some(out)
}

/// Wrap a structured-output JSON Schema as an OpenAI `response_format` value, or
/// `None` when no schema is configured.
fn build_response_format(schema: Option<&Value>) -> Option<Value> {
    schema.map(|schema| {
        serde_json::json!({ "type": "json_schema", "json_schema": { "name": "response", "strict": true, "schema": schema } })
    })
}

/// A resolved session: its id and the transcript to replay before the new
/// prompt (empty for a fresh session, the stored history for a resume).
struct ResolvedSession {
    id: String,
    history: Vec<ChatMessage>,
}

/// Drive the loop to completion, emitting the normalized event stream through
/// `on_event`. Always ends with exactly one [`RunEvent::Exited`]. When the
/// harness has a session store, the conversation is persisted under a session id
/// (freshly minted, or the one named by `cfg.resume`) so a later run resumes it.
pub(crate) fn drive(cfg: LoopConfig, cancel: Arc<AtomicBool>, on_event: RunCallback) {
    let rid = cfg.run_id.as_str();
    (*on_event)(RunEvent::Started { run_id: rid.to_owned() });

    let session = match resolve_session(&cfg, &on_event) {
        Ok(s) => s,
        Err(()) => return, // resolve_session emitted Error + Exited already
    };
    (*on_event)(RunEvent::Session {
        run_id: rid.to_owned(),
        session_id: Some(session.id.clone()),
        model: Some(cfg.model.clone()),
    });

    // Connect any configured MCP servers (best-effort) and build the run's tool
    // set: built-ins + MCP tools, offered + dispatched as one set.
    let (mcp_tools, mcp_status) = tools::mcp::connect_all(&cfg.mcp_servers, &cfg.cwd);
    for message in mcp_status {
        (*on_event)(RunEvent::Activity { run_id: rid.to_owned(), message });
    }
    let toolset = tools::ToolSet::new(mcp_tools, cfg.permissions.clone(), cfg.permission_prompt.clone());
    let tool_defs = toolset.defs(cfg.mode, &cfg.model, false);
    // Structured-output schema (if set) as an OpenAI `response_format`, applied
    // each turn so the final answer conforms; tool-call turns carry null content
    // and stay unconstrained.
    let response_format = build_response_format(cfg.output_schema.as_ref());
    // Skills discovered from the cwd: their name+description catalog is appended
    // to the (regenerated, non-persisted) system prompt, and the model loads a
    // skill's body on demand via the `skill` tool.
    let skills = skills::discover(&cfg.cwd);
    let mut system_prompt = build_system_prompt(SYSTEM_PROMPT, &cfg.cwd, &skills);
    if let Some(catalog) = agent_catalog(&cfg.agents) {
        system_prompt.push_str(&catalog);
    }
    if let Some(extra) = cfg.extra_instructions.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        system_prompt.push_str("\n\n# Additional instructions\n");
        system_prompt.push_str(extra);
    }
    // VARIABLE section LAST: the working directory changes per workspace, so
    // appending it after the fixed + user-configurable sections keeps the whole
    // prefix above it byte-identical run-to-run (a cache-friendly KV prefix).
    system_prompt.push_str(&environment_block(&cfg.cwd));
    // The FULL transcript (no system prompt — that's regenerated each run) is
    // what we persist: never lossy. The request sent to the model is a *windowed
    // view* of it ([`window`]), so compaction can summarize old turns without
    // discarding them from disk.
    let mut transcript = session.history;
    transcript.push(ChatMessage::user(cfg.prompt.clone()));
    persist(&cfg, &session.id, &transcript, &on_event, rid);

    // A `task` call spawns a subagent through this runner: the parent's
    // connection config, running a child session under this one.
    let runner =
        Subagent { cfg: &cfg, parent_session_id: &session.id, skills: &skills, on_event: &on_event, toolset: &toolset };
    // One-shot model access for `summarize`'s map-reduce, over the same config.
    let model = Model { cfg: &cfg };

    for turn in 0..cfg.max_turns {
        if cancel.load(Ordering::SeqCst) {
            touch(&cfg, &session.id);
            (*on_event)(RunEvent::Exited { run_id: rid.to_owned(), exit_code: None, cancelled: true });
            return;
        }

        // Summarize old turns into a marker if the windowed request would near
        // the limit (the full transcript on disk is untouched), then build the
        // windowed view to send.
        compact_if_needed(&cfg, &mut transcript, &system_prompt, &on_event, rid);
        let mut sent = window(&system_prompt, &transcript);
        if cfg.mode == RunMode::Ask {
            sent.push(ChatMessage::user(READ_ONLY_REMINDER));
        }
        if turn + 1 == cfg.max_turns {
            // Final step — nudge the model to answer rather than call more tools.
            sent.push(ChatMessage::user(
                "This is your final step — do not call any more tools; give your final answer now.",
            ));
        }

        // Stream the turn: text deltas surface as `RunEvent::Text` as they
        // arrive; the assembled message (with tool calls) drives the dispatch.
        let extras = wire::RequestExtras {
            response_format: response_format.as_ref(),
            image_data_uris: &cfg.image_data_uris,
            reasoning_tag: cfg.reasoning_tag.as_deref(),
        };
        // Ollama gets its native `/api/chat` so `num_ctx` actually applies; every
        // other provider speaks the OpenAI `/v1` shape.
        let streamed = match cfg.ollama_num_ctx {
            Some(num_ctx) => ollama::post_chat_stream(
                &cfg.base_url,
                &cfg.model,
                &sent,
                &tool_defs,
                num_ctx,
                extras,
                |fragment| emit_fragment(&on_event, rid, fragment),
            ),
            None => wire::post_chat_stream(
                &cfg.base_url,
                cfg.api_key.as_deref(),
                &cfg.model,
                &sent,
                &tool_defs,
                extras,
                |fragment| emit_fragment(&on_event, rid, fragment),
            ),
        };
        let (msg, usage) = match streamed {
            Ok(pair) => pair,
            Err(message) => return finish_error(&on_event, rid, message),
        };

        // Clone the calls out before moving the assistant turn into history
        // (the history must keep the tool_calls so the model sees its own
        // request alongside our results).
        let calls = msg.tool_calls.clone();
        transcript.push(msg);

        if calls.is_empty() {
            persist(&cfg, &session.id, &transcript, &on_event, rid);
            touch(&cfg, &session.id);
            emit_usage(&on_event, rid, usage, cfg.model_cost);
            (*on_event)(RunEvent::Exited { run_id: rid.to_owned(), exit_code: Some(0), cancelled: false });
            return;
        }

        let mut stop_requested = false;
        for call in &calls {
            let args: Value = serde_json::from_str(&call.function.arguments).unwrap_or(Value::Null);
            let locations = tool_locations(&args);
            (*on_event)(RunEvent::ToolStart {
                run_id: rid.to_owned(),
                tool_call_id: call.id.clone(),
                title: call.function.name.clone(),
                tool_kind: toolset.kind(&call.function.name),
                locations: locations.clone(),
                raw_input: Some(call.function.arguments.clone()),
            });
            let ctx = tools::ToolCtx {
                cwd: cfg.cwd.as_path(),
                mode: cfg.mode,
                cancel: cancel.as_ref(),
                run_id: rid,
                call_id: &call.id,
                skills: &skills,
                subagent: Some(&runner),
                model: Some(&model),
            };
            let outcome = toolset.execute(&call.function.name, &args, &ctx);
            // Side events the tool produced (todowrite → Plan, question →
            // AskQuestion) ride between this call's ToolStart and ToolEnd.
            for ev in outcome.events {
                (*on_event)(ev);
            }
            (*on_event)(RunEvent::ToolEnd {
                run_id: rid.to_owned(),
                tool_call_id: call.id.clone(),
                ok: outcome.ok,
                content: Some(outcome.output.clone()),
                raw_output: None,
                locations,
            });
            stop_requested |= outcome.stop;
            transcript.push(ChatMessage::tool_result(call.id.clone(), outcome.output));
        }
        // Persist the assistant turn + its tool results, so a resume (or a crash
        // mid-run) keeps the progress made this turn.
        persist(&cfg, &session.id, &transcript, &on_event, rid);
        // A tool asked to end the run (e.g. `question`, which awaits the user's
        // answer — it arrives as the next prompt on resume).
        if stop_requested {
            touch(&cfg, &session.id);
            (*on_event)(RunEvent::Exited { run_id: rid.to_owned(), exit_code: Some(0), cancelled: false });
            return;
        }
    }

    // Ran out of turns with the model still calling tools.
    touch(&cfg, &session.id);
    (*on_event)(RunEvent::Activity {
        run_id: rid.to_owned(),
        message: format!("Reached the {}-turn limit.", cfg.max_turns),
    });
    (*on_event)(RunEvent::Exited { run_id: rid.to_owned(), exit_code: Some(0), cancelled: false });
}

/// Resolve the run's session: resume a stored transcript, or mint a fresh
/// session (creating its record when a store is configured). A resume that
/// can't be satisfied emits Error + Exited and returns `Err`.
fn resolve_session(cfg: &LoopConfig, on_event: &RunCallback) -> Result<ResolvedSession, ()> {
    let rid = cfg.run_id.as_str();
    if let Some(resume_id) = &cfg.resume {
        let Some(store) = &cfg.store else {
            finish_error(
                on_event,
                rid,
                "resume requested but this harness has no session store (set with_session_dir)".to_owned(),
            );
            return Err(());
        };
        return match store.load_messages(resume_id) {
            Ok(history) => Ok(ResolvedSession { id: resume_id.clone(), history }),
            Err(e) => {
                finish_error(on_event, rid, format!("cannot resume session {resume_id}: {e}"));
                Err(())
            }
        };
    }

    let id = session::new_session_id();
    if let Some(store) = &cfg.store {
        let now = session::now_millis();
        let record = session::SessionRecord {
            id: id.clone(),
            title: Some(session::title_from_prompt(&cfg.prompt)),
            model: Some(cfg.model.clone()),
            cwd: cfg.cwd.to_str().map(str::to_owned),
            parent_id: None, // a top-level session
            created_at: now,
            updated_at: now,
        };
        if let Err(e) = store.put_record(&record) {
            (*on_event)(RunEvent::Activity { run_id: rid.to_owned(), message: format!("session not persisted: {e}") });
        }
    }
    Ok(ResolvedSession { id, history: Vec::new() })
}

/// Persist the full transcript (no system prompt — it's regenerated each run) —
/// best-effort; a write failure surfaces as Activity but never aborts a useful
/// run. Never lossy: compaction inserts a summary marker, it doesn't drop
/// messages, so the stored transcript stays the complete history.
fn persist(cfg: &LoopConfig, session_id: &str, transcript: &[ChatMessage], on_event: &RunCallback, rid: &str) {
    if let Some(store) = &cfg.store {
        if let Err(e) = store.save_messages(session_id, transcript) {
            (*on_event)(RunEvent::Activity { run_id: rid.to_owned(), message: format!("transcript not saved: {e}") });
        }
    }
}

/// Advance the session record's `updated_at` — best-effort, silent (metadata).
fn touch(cfg: &LoopConfig, session_id: &str) {
    if let Some(store) = &cfg.store {
        let _ = store.touch(session_id, session::now_millis());
    }
}

/// The summarization instruction for compaction (OpenCode's structured-brief
/// template).
const SUMMARY_PROMPT: &str = "Summarize the conversation so far into a concise but \
complete brief, so work can continue without the full history. Use these sections:\n\
## Goal\n## Constraints & preferences\n## Progress (done / in progress / blocked)\n\
## Key decisions\n## Next steps\n## Critical context (files, commands, identifiers)\n\n\
Rules:\n\
- Keep every section even if empty — write \"(none)\" rather than dropping it.\n\
- Use terse bullet points, not prose paragraphs.\n\
- Preserve file paths, commands, error strings, and identifiers verbatim — never paraphrase them.\n\n\
Conversation:";

/// Internal role for a compaction marker stored in the transcript (its content
/// is the summary). Never sent to the model as-is — [`window`] renders the
/// latest one as a user message and hides everything before it.
const COMPACTION_ROLE: &str = "compaction";

/// The request view sent to the model: the (regenerated) system prompt, then —
/// if the transcript has a compaction marker — the latest summary as a user
/// message followed by the messages after it, else the whole transcript. The
/// persisted transcript keeps the full history; this just derives the window.
fn window(system_prompt: &str, transcript: &[ChatMessage]) -> Vec<ChatMessage> {
    let mut out = vec![ChatMessage::system(system_prompt)];
    match transcript.iter().rposition(|m| m.role == COMPACTION_ROLE) {
        Some(marker) => {
            let summary = transcript[marker].content.clone().unwrap_or_default();
            out.push(ChatMessage::user(format!("[Summary of the earlier conversation]\n{summary}")));
            out.extend(transcript[marker + 1..].iter().cloned());
        }
        None => out.extend(transcript.iter().cloned()),
    }
    out
}

/// Rough token count for a message slice — OpenCode's `len / 4` heuristic over
/// the content + tool-call payloads.
fn estimate_tokens(messages: &[ChatMessage]) -> usize {
    let chars: usize = messages
        .iter()
        .map(|m| {
            m.content.as_deref().map_or(0, str::len)
                + m.tool_calls.iter().map(|c| c.function.name.len() + c.function.arguments.len()).sum::<usize>()
        })
        .sum();
    chars / 4
}

/// The transcript index where the verbatim tail should start when compacting:
/// the earliest user-message boundary at or after `min` whose tail
/// (`transcript[idx..]`) fits `preserve` tokens, always keeping at least the
/// last user turn. Boundaries are user messages so no tool call is split from
/// its result. `None` if there's no user message at/after `min`.
fn tail_boundary(transcript: &[ChatMessage], preserve: usize, min: usize) -> Option<usize> {
    let users: Vec<usize> = transcript
        .iter()
        .enumerate()
        .skip(min)
        .filter(|(_, m)| m.role == "user")
        .map(|(i, _)| i)
        .collect();
    let mut chosen = *users.last()?;
    for &idx in users.iter().rev() {
        if estimate_tokens(&transcript[idx..]) <= preserve {
            chosen = idx;
        } else {
            break;
        }
    }
    Some(chosen)
}

/// Flatten head messages into a labeled transcript for the summarizer, with tool
/// outputs truncated (OpenCode caps tool output at ~2000 chars in the
/// summarization input).
fn flatten_for_summary(messages: &[ChatMessage]) -> String {
    const TOOL_OUTPUT_MAX: usize = 2000;
    let mut out = String::new();
    for m in messages {
        let label = match m.role.as_str() {
            "user" => "User",
            "assistant" => "Assistant",
            "tool" => "Tool result",
            other => other,
        };
        if let Some(content) = &m.content {
            let body = if m.role == "tool" && content.chars().count() > TOOL_OUTPUT_MAX {
                let head: String = content.chars().take(TOOL_OUTPUT_MAX).collect();
                format!("{head}… [truncated]")
            } else {
                content.clone()
            };
            out.push_str(&format!("[{label}]: {body}\n"));
        }
        for call in &m.tool_calls {
            out.push_str(&format!("[Assistant tool call]: {}({})\n", call.function.name, call.function.arguments));
        }
    }
    out
}

/// Route a streamed fragment to the host as the matching `RunEvent`. A free
/// function (not a captured closure) so the same handler can be handed to either
/// endpoint's streamer without tripping the closure's higher-ranked lifetime.
fn emit_fragment(on_event: &RunCallback, rid: &str, fragment: wire::Fragment) {
    match fragment {
        wire::Fragment::Text(t) => (*on_event)(RunEvent::Text { run_id: rid.to_owned(), delta: t.to_owned() }),
        wire::Fragment::Reasoning(r) => (*on_event)(RunEvent::Thinking { run_id: rid.to_owned(), delta: r.to_owned() }),
    }
}

/// One non-streaming completion returning just the assistant message, routed to
/// the right endpoint (native Ollama when `ollama_num_ctx` is set, else OpenAI
/// `/v1`). Used where streaming isn't needed: compaction summaries and subagent
/// turns. `model` may differ from `cfg.model` (subagents can override it).
fn chat_once(cfg: &LoopConfig, model: &str, messages: &[ChatMessage], tools: &[Value]) -> Result<ChatMessage, String> {
    match cfg.ollama_num_ctx {
        Some(num_ctx) => {
            let (msg, _usage) =
                ollama::post_chat_stream(&cfg.base_url, model, messages, tools, num_ctx, wire::RequestExtras::default(), |_| {})?;
            Ok(msg)
        }
        None => {
            let resp = wire::post_chat(&cfg.base_url, cfg.api_key.as_deref(), model, messages, tools)?;
            resp.choices
                .into_iter()
                .next()
                .map(|choice| choice.message)
                .ok_or_else(|| "the endpoint returned no choices".to_owned())
        }
    }
}

/// When the windowed request would near the model's context limit, summarize the
/// older turns into a `compaction` marker inserted into the transcript — the
/// model then sees `[system, summary, recent tail]` (via [`window`]) while the
/// full history stays on disk (non-lossy, OpenCode's approach). No-op unless a
/// context-window size is configured; best-effort (a failed summary is skipped).
fn compact_if_needed(
    cfg: &LoopConfig,
    transcript: &mut Vec<ChatMessage>,
    system_prompt: &str,
    on_event: &RunCallback,
    rid: &str,
) {
    let Some(limit) = cfg.context_tokens.map(|n| n as usize) else {
        return;
    };
    // Reserve half the window (capped) rather than a quarter: on a ~4K local
    // window a quarter is only ~1K headroom, so the request brushes the limit
    // before compaction fires and the model truncates mid-prompt. Half gives a
    // small window real slack; the cap keeps large windows from over-reserving.
    let reserve = (limit / 2).min(20_000);
    if estimate_tokens(&window(system_prompt, transcript)) <= limit.saturating_sub(reserve) {
        return;
    }
    let preserve = (limit / 4).clamp(2_000, 8_000);
    // Summarize only the turns after the last marker; keep the recent tail verbatim.
    let min = transcript.iter().rposition(|m| m.role == COMPACTION_ROLE).map_or(0, |m| m + 1);
    let Some(boundary) = tail_boundary(transcript, preserve, min) else {
        return;
    };
    if boundary <= min {
        return; // no new turns beyond the last marker to summarize
    }
    // Summarize the windowed head up to the boundary (folds in any prior summary).
    let head = window(system_prompt, &transcript[..boundary]);
    let flattened = flatten_for_summary(&head[1..]); // drop the system prompt
    let request = vec![ChatMessage::user(format!("{SUMMARY_PROMPT}\n\n{flattened}"))];
    let summary = chat_once(cfg, &cfg.model, &request, &[]).ok().and_then(|m| m.content);
    let Some(summary) = summary.filter(|s| !s.trim().is_empty()) else {
        return;
    };
    transcript.insert(
        boundary,
        ChatMessage { role: COMPACTION_ROLE.to_owned(), content: Some(summary), tool_calls: Vec::new(), tool_call_id: None },
    );
    (*on_event)(RunEvent::Activity {
        run_id: rid.to_owned(),
        message: format!("compacted the conversation (summarized {boundary} earlier message(s))"),
    });
}

/// Spawns subagents for the `task` tool: a child run with the parent's
/// connection config, persisted (when a store is set) as a child session.
struct Subagent<'a> {
    cfg: &'a LoopConfig,
    parent_session_id: &'a str,
    skills: &'a [skills::Skill],
    on_event: &'a RunCallback,
    toolset: &'a tools::ToolSet,
}

impl tools::SubagentRunner for Subagent<'_> {
    fn run(&self, subagent_type: Option<&str>, prompt: &str, cancel: &AtomicBool) -> Result<String, String> {
        run_subagent(self, subagent_type, prompt, cancel)
    }
}

/// One-shot model access for the `summarize` tool's map-reduce: a single
/// tool-less [`chat_once`] over the run's connection config, no agent loop. Held
/// by [`tools::ToolCtx::model`].
struct Model<'a> {
    cfg: &'a LoopConfig,
}

impl tools::ModelClient for Model<'_> {
    fn complete(&self, system: Option<&str>, user: &str, cancel: &AtomicBool) -> Result<String, String> {
        if cancel.load(Ordering::SeqCst) {
            return Err("cancelled".to_owned());
        }
        let mut messages = Vec::with_capacity(2);
        if let Some(s) = system {
            messages.push(ChatMessage::system(s));
        }
        messages.push(ChatMessage::user(user));
        let msg = chat_once(self.cfg, &self.cfg.model, &messages, &[])?;
        Ok(msg.content.unwrap_or_default())
    }
}

/// Run a child agent on `prompt` to completion and return its final text. It
/// uses the parent's endpoint/model/cwd/mode and the subagent tool set (no
/// `task`/`question`); with a store configured it's persisted as a child
/// session (`parent_id` set). Quiet — its internal steps aren't emitted to the
/// parent's stream; only the final result returns.
fn run_subagent(
    ctx: &Subagent,
    subagent_type: Option<&str>,
    prompt: &str,
    cancel: &AtomicBool,
) -> Result<String, String> {
    let parent = ctx.cfg;
    let parent_session_id = ctx.parent_session_id;
    let skills = ctx.skills;
    let on_event = ctx.on_event;
    let toolset = ctx.toolset;
    // Resolve the requested subagent type to its role prompt + model override.
    let agent = match subagent_type {
        Some(t) => match parent.agents.iter().find(|(name, _)| name == t) {
            Some((_, def)) => Some(def),
            None => {
                let available: Vec<&str> = parent.agents.iter().map(|(n, _)| n.as_str()).collect();
                let list = if available.is_empty() {
                    "(none registered)".to_owned()
                } else {
                    available.join(", ")
                };
                return Err(format!("unknown subagent_type `{t}` — available: {list}"));
            }
        },
        None => None,
    };
    let base = agent.and_then(|a| a.system_prompt.as_deref()).unwrap_or(SYSTEM_PROMPT);
    let model = agent.and_then(|a| a.model.as_deref()).unwrap_or(parent.model.as_str());

    let child_id = session::new_session_id();
    if let Some(store) = &parent.store {
        let now = session::now_millis();
        let _ = store.put_record(&session::SessionRecord {
            id: child_id.clone(),
            title: Some(session::title_from_prompt(prompt)),
            model: Some(model.to_owned()),
            cwd: parent.cwd.to_str().map(str::to_owned),
            parent_id: Some(parent_session_id.to_owned()),
            created_at: now,
            updated_at: now,
        });
    }

    let mut system_prompt = build_system_prompt(base, &parent.cwd, skills);
    system_prompt.push_str(&environment_block(&parent.cwd));
    let tool_defs = toolset.defs(parent.mode, model, true);
    let model_client = Model { cfg: parent };
    let mut transcript = vec![ChatMessage::user(prompt.to_owned())];
    let mut final_text = String::new();

    for _turn in 0..parent.max_turns {
        if cancel.load(Ordering::SeqCst) {
            return Err("cancelled".to_owned());
        }
        compact_if_needed(parent, &mut transcript, &system_prompt, on_event, &parent.run_id);
        let sent = window(&system_prompt, &transcript);
        let msg = chat_once(parent, model, &sent, &tool_defs)?;
        if let Some(text) = msg.content.as_deref().filter(|t| !t.is_empty()) {
            final_text = text.to_owned();
        }
        let calls = msg.tool_calls.clone();
        transcript.push(msg);
        if calls.is_empty() {
            if let Some(store) = &parent.store {
                let _ = store.save_messages(&child_id, &transcript);
                let _ = store.touch(&child_id, session::now_millis());
            }
            return Ok(if final_text.is_empty() { "(the subagent produced no text)".to_owned() } else { final_text });
        }
        for call in &calls {
            // Surface the subagent's progress on the parent's stream (using the
            // parent run id so host routing is unaffected) — not fully quiet,
            // but without leaking the child's own ToolStart/Plan events.
            (*on_event)(RunEvent::Activity {
                run_id: parent.run_id.clone(),
                message: format!("subagent → {}", call.function.name),
            });
            let args: Value = serde_json::from_str(&call.function.arguments).unwrap_or(Value::Null);
            let ctx = tools::ToolCtx {
                cwd: parent.cwd.as_path(),
                mode: parent.mode,
                cancel,
                run_id: &child_id,
                call_id: &call.id,
                skills,
                subagent: None, // no nesting
                model: Some(&model_client),
            };
            let outcome = toolset.execute(&call.function.name, &args, &ctx);
            transcript.push(ChatMessage::tool_result(call.id.clone(), outcome.output));
        }
        if let Some(store) = &parent.store {
            let _ = store.save_messages(&child_id, &transcript);
        }
    }
    Err("the subagent reached its turn limit".to_owned())
}

/// Emit a terminal in-band error followed by `Exited` — the loop's single
/// failure exit, mirroring how the parsers surface a `turn.failed`.
/// File path(s) a tool call touches, for `ToolStart`/`ToolEnd.locations` — taken
/// from the common `path` argument (read/write/edit/list/glob/grep); empty for
/// tools without one (bash, webfetch, …).
fn tool_locations(args: &Value) -> Vec<crate::ToolLocation> {
    args.get("path")
        .and_then(Value::as_str)
        .map(|p| vec![crate::ToolLocation { path: p.to_owned(), line: None }])
        .unwrap_or_default()
}

fn finish_error(on_event: &RunCallback, run_id: &str, message: String) {
    (*on_event)(RunEvent::Error { run_id: run_id.to_owned(), message });
    (*on_event)(RunEvent::Exited { run_id: run_id.to_owned(), exit_code: Some(1), cancelled: false });
}

/// Map the OpenAI `usage` block onto the neutral [`RunEvent::Usage`], including
/// prompt-cache counters when the provider reports them (auto-caching providers
/// like OpenAI/DeepSeek, or Anthropic-compatible endpoints).
fn emit_usage(on_event: &RunCallback, run_id: &str, usage: Option<wire::Usage>, model_cost: Option<crate::openai_compatible::ModelCost>) {
    if let Some(u) = usage {
        let cost_usd = model_cost.map(|c| compute_cost(&u, c));
        (*on_event)(RunEvent::Usage {
            run_id: run_id.to_owned(),
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            cache_read_tokens: u.cache_read(),
            cache_write_tokens: u.cache_write(),
            cost_usd,
        });
    }
}

/// Estimate run cost in USD from token usage + per-token rates: cached prompt
/// tokens are billed at the cache-read rate (falling back to the input rate),
/// the rest of the prompt at the input rate, and completion at the output rate.
fn compute_cost(u: &wire::Usage, cost: crate::openai_compatible::ModelCost) -> f64 {
    let prompt = u.prompt_tokens.unwrap_or(0);
    let cache_read = u.cache_read().unwrap_or(0);
    let completion = u.completion_tokens.unwrap_or(0);
    let non_cached_input = prompt.saturating_sub(cache_read);
    let cache_rate = cost.cache_read_per_mtok.unwrap_or(cost.input_per_mtok);
    (non_cached_input as f64 * cost.input_per_mtok
        + cache_read as f64 * cache_rate
        + completion as f64 * cost.output_per_mtok)
        / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::Mutex;

    #[test]
    fn system_prompt_joins_cleanly_and_states_the_rules() {
        // The `\`-continued literal must read as clean prose (no mashed words,
        // no leaked indentation) and carry the behavioural rules that keep a
        // weak model from editing on a read-only ask.
        assert!(SYSTEM_PROMPT.contains("no more, no less"));
        assert!(SYSTEM_PROMPT.contains("READ-ONLY"));
        assert!(SYSTEM_PROMPT.contains("Only use a write or edit tool when the user clearly asks"));
        assert!(SYSTEM_PROMPT.contains("do NOT retry"));
        assert!(SYSTEM_PROMPT.contains("\n\n"), "paragraph breaks survive as newlines");
        assert!(!SYSTEM_PROMPT.contains("  "), "no double spaces / leaked indentation");
    }

    #[test]
    fn environment_block_is_a_minimal_trailing_cwd_section() {
        let block = environment_block(Path::new("/work/space"));
        assert!(block.starts_with("\n\n# Environment"), "its own trailing section");
        assert!(block.contains("Working directory: /work/space"));
        // No per-run file list — the variable block must stay byte-identical
        // run-to-run in a workspace so the cacheable prefix above it never moves.
        assert!(!block.contains("- "), "no file listing in the variable block");
    }

    #[test]
    fn base_prompt_directs_tool_use_for_file_discovery() {
        // The dropped manifest's job is now the model's: discover via tools, not
        // memory. The fixed prompt must say so (a weak model otherwise guesses).
        assert!(SYSTEM_PROMPT.contains("call `list` or `glob` first"));
        assert!(SYSTEM_PROMPT.contains("never guess file names"));
    }

    /// A callback that records every event into a shared vec.
    fn capturing() -> (RunCallback, Arc<Mutex<Vec<RunEvent>>>) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&events);
        let cb: RunCallback = Arc::new(move |ev| sink.lock().unwrap().push(ev));
        (cb, events)
    }

    fn cfg(prompt: &str, resume: Option<String>, store: Option<FileStore>) -> LoopConfig {
        LoopConfig {
            run_id: "t".into(),
            base_url: "http://unused".into(),
            api_key: None,
            model: "m".into(),
            prompt: prompt.into(),
            cwd: PathBuf::from("/tmp"),
            mode: RunMode::Edit,
            max_turns: 1,
            resume,
            store,
            context_tokens: None,
            ollama_num_ctx: None,
            agents: Vec::new(),
            mcp_servers: Vec::new(),
            output_schema: None,
            model_cost: None,
            image_data_uris: Vec::new(),
            permissions: Vec::new(),
            permission_prompt: None,
            reasoning_tag: Some("think".to_owned()),
            extra_instructions: None,
        }
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hl-run-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn new_session_mints_id_and_persists_a_record() {
        let dir = scratch("new");
        let store = FileStore::new(&dir);
        let (cb, _) = capturing();
        let s = resolve_session(&cfg("Do the thing", None, Some(store.clone())), &cb).unwrap();
        assert!(s.id.starts_with("ses_"));
        assert!(s.history.is_empty());
        let rec = store.get_record(&s.id).unwrap().expect("record persisted");
        assert_eq!(rec.title.as_deref(), Some("Do the thing"));
        assert_eq!(rec.model.as_deref(), Some("m"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resume_replays_the_stored_transcript() {
        let dir = scratch("resume");
        let store = FileStore::new(&dir);
        store.save_messages("ses_x", &[ChatMessage::user("earlier")]).unwrap();
        let (cb, _) = capturing();
        let s = resolve_session(&cfg("again", Some("ses_x".into()), Some(store)), &cb).unwrap();
        assert_eq!(s.id, "ses_x");
        assert_eq!(s.history.len(), 1);
        assert_eq!(s.history[0].content.as_deref(), Some("earlier"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resume_without_a_store_errors() {
        let (cb, events) = capturing();
        let r = resolve_session(&cfg("x", Some("ses_x".into()), None), &cb);
        assert!(r.is_err());
        let evs = events.lock().unwrap();
        assert!(evs.iter().any(|e| matches!(e, RunEvent::Error { .. })), "emits Error");
        assert!(evs.iter().any(|e| matches!(e, RunEvent::Exited { .. })), "and Exited");
    }

    #[test]
    fn new_session_without_a_store_is_ephemeral() {
        let (cb, _) = capturing();
        let s = resolve_session(&cfg("x", None, None), &cb).unwrap();
        assert!(s.id.starts_with("ses_"));
        assert!(s.history.is_empty());
    }

    #[test]
    fn agent_catalog_lists_registered_agents() {
        assert!(agent_catalog(&[]).is_none(), "no agents → no catalog");
        let agents = vec![(
            "reviewer".to_owned(),
            crate::openai_compatible::AgentDef { description: "reviews code".to_owned(), system_prompt: None, model: None },
        )];
        let cat = agent_catalog(&agents).expect("catalog");
        assert!(cat.contains("reviewer") && cat.contains("reviews code") && cat.contains("subagent_type"));
    }

    #[test]
    fn compute_cost_prices_cached_and_uncached_tokens() {
        // 1000 prompt (200 cached) + 500 completion, at $3/$15/$0.30 per Mtok.
        let usage: wire::Usage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
            "prompt_tokens_details": { "cached_tokens": 200 }
        }))
        .unwrap();
        let cost = crate::openai_compatible::ModelCost { input_per_mtok: 3.0, output_per_mtok: 15.0, cache_read_per_mtok: Some(0.3) };
        // 800*3 + 200*0.3 + 500*15 = 2400 + 60 + 7500 = 9960 micro-USD.
        assert!((compute_cost(&usage, cost) - 0.009_96).abs() < 1e-9);
        // No cache rate → cached tokens billed at the input rate.
        let flat = crate::openai_compatible::ModelCost { input_per_mtok: 3.0, output_per_mtok: 15.0, cache_read_per_mtok: None };
        assert!((compute_cost(&usage, flat) - 0.010_5).abs() < 1e-9);
    }

    #[test]
    fn response_format_wraps_schema_or_none() {
        assert!(build_response_format(None).is_none());
        let schema = serde_json::json!({ "type": "object", "properties": { "answer": { "type": "string" } } });
        let rf = build_response_format(Some(&schema)).expect("some");
        assert_eq!(rf["type"], "json_schema");
        assert_eq!(rf["json_schema"]["strict"], true);
        assert_eq!(rf["json_schema"]["schema"], schema);
    }

    #[test]
    fn unknown_subagent_type_errors_before_any_call() {
        // The type is validated up front, so this never touches the network.
        let (cb, _) = capturing();
        let mut c = cfg("x", None, None);
        c.agents = vec![(
            "reviewer".to_owned(),
            crate::openai_compatible::AgentDef { description: "reviews".to_owned(), system_prompt: None, model: None },
        )];
        let cancel = AtomicBool::new(false);
        let toolset = tools::ToolSet::builtin();
        let runner =
            Subagent { cfg: &c, parent_session_id: "ses_parent", skills: &[], on_event: &cb, toolset: &toolset };
        let err = run_subagent(&runner, Some("nope"), "do it", &cancel).unwrap_err();
        assert!(err.contains("unknown subagent_type") && err.contains("reviewer"), "got: {err}");
    }

    #[test]
    fn tail_boundary_and_window() {
        // The transcript has NO system prompt (it's regenerated); window() adds it.
        let transcript = vec![ChatMessage::user("first older turn"), ChatMessage::user("the most recent turn")];
        assert!(estimate_tokens(&transcript) > 0);
        assert_eq!(tail_boundary(&transcript, 100_000, 0), Some(0), "huge budget summarizes nothing, tail = all");
        assert_eq!(tail_boundary(&transcript, 1, 0), Some(1), "tiny budget keeps only the latest turn");

        // No marker → window is system + the whole transcript.
        let sent = window("SYS", &transcript);
        assert_eq!(sent[0].role, "system");
        assert_eq!(sent.len(), 3);

        // A compaction marker → system + summary + the messages after it (the
        // pre-marker history is on disk but hidden from the request).
        let compacted = vec![
            ChatMessage::user("old q"),
            ChatMessage { role: COMPACTION_ROLE.to_owned(), content: Some("the summary".into()), tool_calls: vec![], tool_call_id: None },
            ChatMessage::user("recent q"),
        ];
        let sent = window("SYS", &compacted);
        assert_eq!(sent.len(), 3, "system + summary + recent; 'old q' hidden");
        assert_eq!(sent[0].role, "system");
        assert!(sent[1].content.as_deref().unwrap().contains("the summary"));
        assert_eq!(sent[2].content.as_deref(), Some("recent q"));
    }

    #[test]
    fn flatten_labels_and_truncates_tool_output() {
        let msgs = vec![ChatMessage::user("hi"), ChatMessage::tool_result("c1", "x".repeat(5000))];
        let f = flatten_for_summary(&msgs);
        assert!(f.contains("[User]: hi"));
        assert!(f.contains("[Tool result]:") && f.contains("[truncated]"));
        assert!(f.len() < 5000, "tool output truncated in the summary input");
    }
}
