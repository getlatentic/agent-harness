//! The built-in tool surface — the tools the model is offered and which this
//! runtime executes itself (the CLI adapters get their tools from the CLI;
//! here we own them).
//!
//! Structure mirrors OpenCode's (MIT): each tool is a self-describing unit
//! (`id` + `description` + `parameters` schema + `execute`), collected in a
//! [`registry`] the loop dispatches through — no central match to grow. Tools
//! live in submodules by concern: [`file`] (read/write/edit), [`search`]
//! (glob/grep), [`shell`] (bash).
//!
//! Two deviations from OpenCode, by design:
//! * **Read-only withholds, then refuses.** Mutating tools (`write`/`edit`/
//!   `apply_patch`/`bash`/`task`) are withheld from the model in a read-only
//!   ([`RunMode::Ask`]) run, because a small local model that can *see* them
//!   tends to call them even when only asked to summarize, then loops on the
//!   refusal. OpenCode instead offers everything and denies at execution; for
//!   weak models, not offering is the stronger lever. A mutating call is still
//!   refused at `execute` as a backstop. Review of applied edits stays in the
//!   host, as for the CLI adapters.
//! * **Descriptions are inline** `&str`, not sibling `.txt` files.
//!
//! Paths are **relative to the run's `cwd`**; traversal outside it is refused
//! ([`safe_join`]) — a guardrail, not the security boundary (the host's
//! snapshot/clone review is that).

use std::path::{Component, Path, PathBuf};
use std::sync::atomic::AtomicBool;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::{json, Value};

use crate::{RunEvent, RunMode, ToolKind};

mod fetch;
mod file;
pub(crate) mod mcp;
mod patch;
mod question;
mod search;
mod shell;
mod skill;
mod summarize;
mod task;
mod todo;
mod websearch;

/// Per-line output cap shared by `read` and `grep` (OpenCode's limit).
const MAX_LINE_CHARS: usize = 2000;
/// Cap on `glob`/`grep` results (OpenCode's limit), with a truncation note.
const SEARCH_LIMIT: usize = 100;
/// Caps on a tool's returned output (OpenCode's `truncate` limits) — output past
/// either is trimmed to the head before it reaches the model + transcript, so a
/// chatty command can't blow the context window. Tools whose full output is
/// intentional context (e.g. `skill`) opt out via [`Tool::truncates_output`].
const MAX_OUTPUT_LINES: usize = 2000;
const MAX_OUTPUT_BYTES: usize = 50 * 1024;

/// A single built-in tool: its schema (offered to the model) + its execution.
/// Mirrors OpenCode's `Tool.define` shape, adapted to our `RunMode`/host-review
/// model. Implementors are zero-sized unit structs held in the [`registry`].
pub(crate) trait Tool: Send + Sync {
    /// Stable id the model calls.
    fn id(&self) -> &str;
    /// One-line description offered to the model.
    fn description(&self) -> &str;
    /// JSON Schema for the parameters object.
    fn parameters(&self) -> Value;
    /// Neutral behaviour class, for `RunEvent::ToolStart` routing.
    fn kind(&self) -> ToolKind;
    /// Whether the tool mutates state. Mutating tools are withheld from the model
    /// in a read-only ([`RunMode::Ask`]) run by [`ToolSet::defs`], and refused at
    /// `execute` as a backstop if one is somehow called anyway.
    fn mutating(&self) -> bool;
    /// Whether the tool is offered to the model, before [`ToolSet::defs`] applies
    /// the read-only filter (which withholds mutating tools in `Ask`). Default:
    /// yes; only the apply_patch ⇄ edit/write swap overrides this (model-dependent).
    fn offered(&self, _mode: RunMode, _model: &str) -> bool {
        true
    }
    /// Whether the tool is available to a subagent (a `task` child). Default
    /// yes; `task` (no nesting) and `question` (no user to answer) opt out.
    fn in_subagent(&self) -> bool {
        true
    }
    /// Whether the tool's output is subject to the [`MAX_OUTPUT_LINES`] /
    /// [`MAX_OUTPUT_BYTES`] cap. Default yes; tools whose full output is
    /// intentional context the model must see in full (`skill`) opt out.
    fn truncates_output(&self) -> bool {
        true
    }
    /// Which end of an over-cap output to keep. Default [`Keep::Head`] (the
    /// start, where most tools' useful output is); `bash` overrides to keep the
    /// tail too, because a command's exit/error lines land at the end and a small
    /// model needs exactly those.
    fn keep_output(&self) -> Keep {
        Keep::Head
    }
    /// The "subject" of a call for permission-rule pattern matching — the command
    /// for `bash`, the path for file tools; `None` if the tool isn't pattern-gated
    /// (a whole-tool rule still applies). Default `None`.
    fn permission_subject(&self, _args: &Value) -> Option<String> {
        None
    }
    /// Execute the call. `args` is the parsed parameters object.
    fn execute(&self, args: &Value, ctx: &ToolCtx) -> ToolOutcome;
}

/// Every built-in tool, boxed so a [`ToolSet`] can hold MCP-provided tools
/// (owned at runtime) alongside them. Add one here and it's offered + dispatched
/// with no other wiring (OpenCode's `registry.ts` builtin list).
fn builtins() -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(file::Read),
        Box::new(search::Glob),
        Box::new(search::Grep),
        Box::new(search::List),
        Box::new(fetch::WebFetch),
        Box::new(todo::TodoWrite),
        Box::new(question::QuestionTool),
        Box::new(skill::LoadSkill),
        Box::new(summarize::Summarize),
        Box::new(websearch::WebSearch),
        Box::new(file::Write),
        Box::new(file::Edit),
        Box::new(shell::Bash),
        Box::new(patch::ApplyPatch),
        Box::new(task::Task),
    ]
}

/// Whether [`ToolSet::defs`] builds the tool list for the main agent or a `task`
/// subagent — a subagent's set drops the opt-out tools (`task`, `question`), so
/// it can neither spawn its own children nor stop to ask the user.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AgentContext {
    Main,
    Subagent,
}

/// The tools a run offers + dispatches: the built-ins plus any from connected
/// MCP servers. Built once per run and shared with subagents, so a dynamic tool
/// source (MCP) slots in beside the static built-ins behind one type — no
/// central match to grow.
pub(crate) struct ToolSet {
    tools: Vec<Box<dyn Tool>>,
    permissions: Vec<crate::openai_compatible::PermissionRule>,
    permission_prompt: Option<crate::openai_compatible::PermissionPrompt>,
}

impl ToolSet {
    /// The built-in tools only (no MCP, no permission rules) — a test convenience;
    /// production builds the set via [`ToolSet::new`].
    #[cfg(test)]
    pub(crate) fn builtin() -> Self {
        Self {
            tools: builtins(),
            permissions: Vec::new(),
            permission_prompt: None,
        }
    }

    /// The built-ins plus MCP-provided tools, gated by `permissions` (with
    /// `permission_prompt` deciding any `Ask` rules). MCP tools are namespaced
    /// (`server_tool`), so they never shadow a built-in.
    pub(crate) fn new(
        mcp: Vec<Box<dyn Tool>>,
        permissions: Vec<crate::openai_compatible::PermissionRule>,
        permission_prompt: Option<crate::openai_compatible::PermissionPrompt>,
    ) -> Self {
        let mut tools = builtins();
        tools.extend(mcp);
        Self {
            tools,
            permissions,
            permission_prompt,
        }
    }

    /// The OpenAI `tools` array offered to the model: the read-only tools always,
    /// the mutating tools only in [`RunMode::Edit`] — they're withheld from a
    /// read-only `Ask` run, because a small local model that can *see* `write` /
    /// `edit` tends to call them even when asked only to summarize, then loops on
    /// the refusal. Withholding is the evidenced fix; `execute` still refuses a
    /// mutating call as a backstop against a hallucinated name. Also drops the
    /// model's apply_patch ⇄ edit/write swap and, in a subagent, the opt-outs
    /// (`task`, `question`).
    pub(crate) fn defs(&self, mode: RunMode, model: &str, context: AgentContext) -> Vec<Value> {
        let subagent = matches!(context, AgentContext::Subagent);
        self.tools
            .iter()
            .filter(|t| t.offered(mode, model))
            .filter(|t| !subagent || t.in_subagent())
            .filter(|t| !(t.mutating() && matches!(mode, RunMode::Ask)))
            .map(|t| {
                json!({
                    "type": "function",
                    "function": { "name": t.id(), "description": t.description(), "parameters": t.parameters() }
                })
            })
            .collect()
    }

    /// The neutral [`ToolKind`] for a tool name, so the host routes the card
    /// (read → context pill, write/edit → file-op, bash → command) without a name
    /// table. Unknown → [`ToolKind::Other`].
    pub(crate) fn kind(&self, name: &str) -> ToolKind {
        self.tools
            .iter()
            .find(|t| t.id() == name)
            .map_or(ToolKind::Other, |t| t.kind())
    }

    /// Execute one tool call. `ctx.mode` is re-checked so a model that
    /// hallucinates a mutating tool in `Ask` mode is refused even though it
    /// wasn't offered. Output past the caps is truncated (+ spilled) here.
    pub(crate) fn execute(&self, name: &str, args: &Value, ctx: &ToolCtx) -> ToolOutcome {
        let Some(tool) = self.tools.iter().find(|t| t.id() == name) else {
            return ToolOutcome::err(format!("unknown tool `{name}`"));
        };
        if tool.mutating() && !matches!(ctx.mode, RunMode::Edit) {
            return ToolOutcome::err(format!(
                "`{name}` is disabled in read-only mode — this run can't change files. \
                 Answer the user directly; do not retry. Tell them to turn on editing if a \
                 change is needed."
            ));
        }
        if let Some(reason) = self.denied(tool.as_ref(), args) {
            return ToolOutcome::err(reason);
        }
        let mut outcome = tool.execute(args, ctx);
        if tool.truncates_output() {
            outcome.output = truncate_output(outcome.output, tool.keep_output());
        }
        outcome
    }

    /// Evaluate the permission rules against a call: the first rule whose tool +
    /// pattern match decides. `Some(reason)` = denied; `None` = allowed (no rule
    /// matched, or an explicit allow rule did).
    fn denied(&self, tool: &dyn Tool, args: &Value) -> Option<String> {
        if self.permissions.is_empty() {
            return None;
        }
        let subject = tool.permission_subject(args);
        for rule in &self.permissions {
            let tool_matches = rule.tool.as_deref().map_or(true, |t| t == tool.id());
            let pattern_matches = match (&rule.pattern, &subject) {
                (None, _) => true,
                (Some(p), Some(s)) => s.contains(p.as_str()),
                (Some(_), None) => false,
            };
            if tool_matches && pattern_matches {
                return match rule.effect {
                    crate::openai_compatible::Permission::Allow => None,
                    crate::openai_compatible::Permission::Deny => Some(format!(
                        "`{}` denied by a permission rule{}",
                        tool.id(),
                        rule.pattern
                            .as_deref()
                            .map(|p| format!(" (matched `{p}`)"))
                            .unwrap_or_default()
                    )),
                    // Defer to the host's prompt (blocking); absent prompt → deny.
                    crate::openai_compatible::Permission::Ask => {
                        let request = crate::openai_compatible::PermissionRequest {
                            tool: tool.id().to_owned(),
                            subject: subject.clone(),
                        };
                        if self
                            .permission_prompt
                            .as_ref()
                            .is_some_and(|prompt| prompt(&request))
                        {
                            None
                        } else {
                            Some(format!(
                                "`{}` denied (permission prompt declined or unset)",
                                tool.id()
                            ))
                        }
                    }
                };
            }
        }
        None
    }
}

/// Runs a child agent (a `task`) to completion and returns its final text.
/// Implemented by the run loop; the `task` tool invokes it through
/// [`ToolCtx::subagent`]. Object-safe, so `ToolCtx` can hold `&dyn`.
pub(crate) trait SubagentRunner {
    fn run(
        &self,
        subagent_type: Option<&str>,
        prompt: &str,
        cancel: &AtomicBool,
    ) -> Result<String, String>;
}

/// One stateless model completion (no tools, no agent loop) — the model access a
/// tool needs to drive its own multi-call routine. Implemented by the run loop
/// over the run's connection config; the `summarize` tool reaches it through
/// [`ToolCtx::model`] to run its map-reduce. Object-safe, so `ToolCtx` holds
/// `&dyn`. Distinct from [`SubagentRunner`], which spawns a whole tool-using
/// child agent; this is a single prompt→text call.
pub(crate) trait ModelClient {
    fn complete(
        &self,
        system: Option<&str>,
        user: &str,
        cancel: &AtomicBool,
    ) -> Result<String, String>;
}

/// Per-run context for executing a tool call: where the run operates, what it
/// may do, and a cooperative cancel flag (so a long `bash` can be interrupted).
pub(crate) struct ToolCtx<'a> {
    pub cwd: &'a Path,
    pub mode: RunMode,
    pub cancel: &'a AtomicBool,
    /// The run this call belongs to — stamped on any side event a tool emits.
    pub run_id: &'a str,
    /// The current tool call's id — used by `question` as the `AskQuestion`
    /// request id the host echoes when returning the answer.
    pub call_id: &'a str,
    /// Skills discovered for this run — `skill` looks its argument up here.
    pub skills: &'a [crate::openai_compatible::skills::Skill],
    /// Spawns a subagent for the `task` tool, when this run allows it. `None`
    /// inside a subagent (no nesting) and in contexts without a runner.
    pub subagent: Option<&'a dyn SubagentRunner>,
    /// One-shot model access for tools that drive their own model loop
    /// (`summarize`'s map-reduce). `None` in contexts without a reachable model.
    pub model: Option<&'a dyn ModelClient>,
}

/// Outcome of executing one tool call: `ok` maps to `RunEvent::ToolEnd.ok`,
/// `output` is fed back to the model (and surfaced on the card).
pub(crate) struct ToolOutcome {
    pub ok: bool,
    pub output: String,
    /// When set, the loop ends the run after this turn instead of continuing —
    /// for a tool that must wait for the user (`question`, whose answer arrives
    /// as the next prompt on resume). Most tools leave it false.
    pub stop: bool,
    /// Side events the tool produced (`todowrite` → Plan, `question` →
    /// AskQuestion) for the loop to emit. Empty for most tools.
    pub events: Vec<RunEvent>,
}

impl ToolOutcome {
    fn ok(output: impl Into<String>) -> Self {
        Self {
            ok: true,
            output: output.into(),
            stop: false,
            events: Vec::new(),
        }
    }
    fn err(output: impl Into<String>) -> Self {
        Self {
            ok: false,
            output: output.into(),
            stop: false,
            events: Vec::new(),
        }
    }
    /// A successful result that also ends the run after this turn.
    fn stop(output: impl Into<String>) -> Self {
        Self {
            ok: true,
            output: output.into(),
            stop: true,
            events: Vec::new(),
        }
    }
    /// Attach side events for the loop to emit.
    fn with_events(mut self, events: Vec<RunEvent>) -> Self {
        self.events = events;
        self
    }
}

/// Which end of an over-cap output to keep when truncating.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Keep {
    /// Keep the start — most tools' useful output is at the top.
    Head,
    /// Keep the end — for a tool whose diagnostic lands last (`bash`'s exit/error
    /// lines), so a small model isn't shown only the leading noise.
    #[allow(dead_code)] // a direction tools may request; bash uses HeadAndTail
    Tail,
    /// Keep both ends (split the budget), with a middle marker for the gap — the
    /// `bash` case: the command and its trailing error/exit are both preserved.
    HeadAndTail,
}

/// A single end of an output — which one [`take_end`] keeps.
#[derive(Clone, Copy, PartialEq, Eq)]
enum End {
    Head,
    Tail,
}

/// Cap a tool's output when it exceeds the output caps, spilling the full text to
/// a scratch file so the omitted span stays reachable via `read` rather than
/// forcing a narrower re-run. `keep` chooses which end(s) to retain (the start
/// for most tools, both ends for `bash` so its trailing error survives).
fn truncate_output(output: String, keep: Keep) -> String {
    let total_lines = output.lines().count();
    if total_lines <= MAX_OUTPUT_LINES && output.len() <= MAX_OUTPUT_BYTES {
        return output;
    }
    // Spill once up front so every branch can reference the full-output path.
    let spilled = spill_output(&output);
    let lines: Vec<&str> = output.lines().collect();
    match keep {
        Keep::Head => take_end(
            &lines,
            total_lines,
            MAX_OUTPUT_LINES,
            MAX_OUTPUT_BYTES,
            End::Head,
            &spilled,
        ),
        Keep::Tail => take_end(
            &lines,
            total_lines,
            MAX_OUTPUT_LINES,
            MAX_OUTPUT_BYTES,
            End::Tail,
            &spilled,
        ),
        Keep::HeadAndTail => take_both_ends(&lines, total_lines, &spilled),
    }
}

/// Keep one `end` of `lines` within the line/byte caps, then append a note
/// pointing at the spill file (or advising a narrower command when the spill
/// failed).
fn take_end(
    lines: &[&str],
    total_lines: usize,
    max_lines: usize,
    max_bytes: usize,
    end: End,
    spilled: &Option<String>,
) -> String {
    let mut chosen: Vec<&str> = Vec::new();
    let mut bytes = 0usize;
    let ordered: Box<dyn Iterator<Item = &&str>> = match end {
        End::Tail => Box::new(lines.iter().rev()),
        End::Head => Box::new(lines.iter()),
    };
    for line in ordered {
        if chosen.len() >= max_lines || bytes + line.len() + 1 > max_bytes {
            break;
        }
        bytes += line.len() + 1;
        chosen.push(line);
    }
    if end == End::Tail {
        chosen.reverse();
    }
    let omitted = total_lines.saturating_sub(chosen.len());
    let mut body = chosen.join("\n");
    body.push('\n');
    let note = truncation_note(omitted, spilled);
    match end {
        End::Tail => format!("{note}\n{body}"),
        End::Head => format!("{body}{note}"),
    }
}

/// Keep both ends of `lines` (half the line/byte budget each) with a middle
/// marker for the omitted span — `bash`'s shape, so the command's leading output
/// and its trailing exit/error are both preserved.
fn take_both_ends(lines: &[&str], total_lines: usize, spilled: &Option<String>) -> String {
    let half_lines = MAX_OUTPUT_LINES / 2;
    let half_bytes = MAX_OUTPUT_BYTES / 2;
    let head = collect_within(lines.iter(), half_lines, half_bytes);
    // The tail skips lines already in the head, so the two halves never overlap
    // on a small input that fits within `2 * half` lines.
    let tail_pool = &lines[head.len()..];
    let tail = collect_within(tail_pool.iter().rev(), half_lines, half_bytes);
    let omitted = total_lines.saturating_sub(head.len() + tail.len());
    let mut out = head.join("\n");
    out.push('\n');
    out.push_str(&middle_marker(omitted, spilled));
    out.push('\n');
    for line in tail.iter().rev() {
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Take lines from `iter` until the line or byte budget is hit, preserving the
/// iterator's order in the returned vec.
fn collect_within<'a>(
    iter: impl Iterator<Item = &'a &'a str>,
    max_lines: usize,
    max_bytes: usize,
) -> Vec<&'a str> {
    let mut chosen = Vec::new();
    let mut bytes = 0usize;
    for line in iter {
        if chosen.len() >= max_lines || bytes + line.len() + 1 > max_bytes {
            break;
        }
        bytes += line.len() + 1;
        chosen.push(*line);
    }
    chosen
}

/// The trailing note for a head/tail truncation: point at the spill file when it
/// was written, else advise narrowing.
fn truncation_note(omitted: usize, spilled: &Option<String>) -> String {
    match spilled {
        Some(path) => format!(
            "[output truncated — {omitted} more line(s) omitted. Full output saved to {path} — \
             read it with the `read` tool (offset/limit), or narrow the command.]"
        ),
        None => format!(
            "[output truncated — {omitted} more line(s) omitted; narrow the command, or use \
             read/grep with offset/limit on the relevant file.]"
        ),
    }
}

/// The middle marker between a kept head and tail (the head+tail case).
fn middle_marker(omitted: usize, spilled: &Option<String>) -> String {
    match spilled {
        Some(path) => format!(
            "[… {omitted} line(s) omitted from the middle. Full output saved to {path} — \
             read it with the `read` tool (offset/limit) for the gap. …]"
        ),
        None => format!(
            "[… {omitted} line(s) omitted from the middle; narrow the command for the gap. …]"
        ),
    }
}

/// Write a tool's full output to a scratch file and return its absolute path, so
/// the model can read the part beyond the truncation cap (OpenCode spills to its
/// data dir; we use a temp dir). `None` on any I/O failure.
fn spill_output(output: &str) -> Option<String> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir()
        .join("openai-compatible")
        .join("tool-output");
    std::fs::create_dir_all(&dir).ok()?;
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = dir.join(format!("{nanos:x}-{n:x}.txt"));
    std::fs::write(&path, output).ok()?;
    Some(path.to_string_lossy().into_owned())
}

// --- helpers shared across the tool modules ---

/// JSON Schema for a tool's typed parameters, normalized for the chat tools
/// API: schemars generates a draft-2020-12 schema from the type (the single
/// source of truth — the Rust equivalent of Zod/Pydantic), and we drop the
/// `$schema`/`title` metadata the model doesn't need, leaving the
/// `{type, properties, required}` object. Field descriptions come from each
/// struct field's doc comment.
fn schema_for<T: JsonSchema>() -> Value {
    let schema = schemars::generate::SchemaSettings::draft2020_12()
        .into_generator()
        .into_root_schema_for::<T>();
    let mut value = serde_json::to_value(schema).unwrap_or_else(|_| json!({ "type": "object" }));
    if let Some(obj) = value.as_object_mut() {
        obj.remove("$schema");
        obj.remove("title");
    }
    value
}

/// Deserialize a tool call's arguments into its typed params, turning a decode
/// failure into a tool error fed back to the model (rather than killing the run).
fn parse_args<T: DeserializeOwned>(args: &Value) -> Result<T, ToolOutcome> {
    serde_json::from_value(args.clone())
        .map_err(|e| ToolOutcome::err(format!("invalid arguments: {e}")))
}

/// OpenCode's model gate: gpt-5-class OpenAI models are offered `apply_patch`
/// instead of `edit`/`write` (substring match on the model id — it contains
/// `gpt-` but not `oss` or `gpt-4`).
fn uses_apply_patch(model: &str) -> bool {
    model.contains("gpt-") && !model.contains("oss") && !model.contains("gpt-4")
}

/// Join `rel` under `cwd`, refusing absolute paths and any `..` traversal so a
/// tool call can't reach outside the run's directory. Lexical (no filesystem
/// touch), so it works for not-yet-created files.
fn safe_join(cwd: &Path, rel: &str) -> Option<PathBuf> {
    let mut out = cwd.to_path_buf();
    for comp in Path::new(rel).components() {
        match comp {
            Component::Normal(seg) => out.push(seg),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    Some(out)
}

/// Replace the single window of `content` lines matching `old`'s lines (after
/// trimming each line's surrounding whitespace) with `new` — the
/// whitespace-tolerant fallback shared by `edit` and `apply_patch`. Returns the
/// updated content, or `None` if there's no match or it's ambiguous.
fn replace_line_trimmed(content: &str, old: &str, new: &str) -> Option<String> {
    let (begin, end) = find_line_trimmed_unique(content, old)?;
    let mut updated = String::with_capacity(content.len() + new.len());
    updated.push_str(&content[..begin]);
    updated.push_str(new);
    if content[begin..end].ends_with('\n') && !new.ends_with('\n') {
        updated.push('\n'); // keep the following line from merging onto the replacement
    }
    updated.push_str(&content[end..]);
    Some(updated)
}

/// Byte range of the single window of `content` lines that equals `old`'s lines
/// after trimming each line's surrounding whitespace. `None` if absent or
/// ambiguous (more than one match → refuse).
fn find_line_trimmed_unique(content: &str, old: &str) -> Option<(usize, usize)> {
    let old_lines: Vec<&str> = old.lines().collect();
    if old_lines.is_empty() {
        return None;
    }
    // `split_inclusive` keeps each line's trailing '\n', so byte offsets are exact.
    let lines: Vec<&str> = content.split_inclusive('\n').collect();
    let window = old_lines.len();
    if window > lines.len() {
        return None;
    }
    let mut offsets = Vec::with_capacity(lines.len() + 1);
    let mut acc = 0;
    offsets.push(0);
    for l in &lines {
        acc += l.len();
        offsets.push(acc);
    }
    let mut found: Option<(usize, usize)> = None;
    for start in 0..=(lines.len() - window) {
        let matches = old_lines
            .iter()
            .enumerate()
            .all(|(i, ol)| lines[start + i].trim_end_matches('\n').trim() == ol.trim());
        if matches {
            if found.is_some() {
                return None; // ambiguous
            }
            found = Some((offsets[start], offsets[start + window]));
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run a tool through the public dispatch with a fresh (un-cancelled) context.
    fn run(name: &str, args: Value, cwd: &Path, mode: RunMode) -> ToolOutcome {
        let cancel = AtomicBool::new(false);
        ToolSet::builtin().execute(
            name,
            &args,
            &ToolCtx {
                cwd,
                mode,
                cancel: &cancel,
                run_id: "t",
                call_id: "c",
                skills: &[],
                subagent: None,
                model: None,
            },
        )
    }

    fn names(mode: RunMode, model: &str) -> Vec<String> {
        ToolSet::builtin()
            .defs(mode, model, AgentContext::Main)
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap().to_owned())
            .collect()
    }

    /// A real directory for tests that only need *some* valid cwd. `/tmp` is
    /// not a path on Windows, where a bogus cwd fails the spawn outright
    /// ("The directory name is invalid").
    fn any_cwd() -> &'static Path {
        static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
        DIR.get_or_init(|| {
            let d = std::env::temp_dir().join("hl-tools-cwd");
            let _ = std::fs::create_dir_all(&d);
            d
        })
    }

    /// Sleep ~5s in the platform's shell — Windows `cmd` has no `sleep`.
    fn sleep_5() -> &'static str {
        if cfg!(windows) {
            "ping -n 6 127.0.0.1 > nul"
        } else {
            "sleep 5"
        }
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hl-tools-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn mutating_tools_withheld_in_ask_offered_in_edit() {
        // A local (non-gpt) model, so edit/write aren't swapped for apply_patch.
        let ask = names(RunMode::Ask, "qwen2.5-coder");
        let edit = names(RunMode::Edit, "qwen2.5-coder");
        // Read-only tools are offered in both modes.
        for t in [
            "read",
            "glob",
            "grep",
            "list",
            "webfetch",
            "todowrite",
            "question",
            "skill",
            "summarize",
        ] {
            assert!(ask.contains(&t.to_string()), "{t} offered in Ask");
            assert!(edit.contains(&t.to_string()), "{t} offered in Edit");
        }
        // Mutating tools are withheld from a read-only run, offered in Edit — the
        // schema half of the read-only guarantee (`execute` is the backstop).
        for t in ["write", "edit", "bash", "task"] {
            assert!(!ask.contains(&t.to_string()), "{t} withheld in Ask");
            assert!(edit.contains(&t.to_string()), "{t} offered in Edit");
        }
    }

    #[test]
    fn mutating_tools_refused_in_ask_mode() {
        for name in ["write", "edit", "bash"] {
            let out = run(name, json!({}), any_cwd(), RunMode::Ask);
            assert!(!out.ok, "{name} should be refused in Ask");
            assert!(out.output.contains("read-only"));
        }
    }

    #[test]
    fn list_shows_entries_with_dirs_marked() {
        let dir = scratch("list");
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("a.txt"), "x").unwrap();
        std::fs::write(dir.join("b.rs"), "y").unwrap();
        let out = run("list", json!({}), &dir, RunMode::Ask);
        assert!(out.ok, "{}", out.output);
        assert!(
            out.output.contains("sub/"),
            "directory marked with /: {}",
            out.output
        );
        assert!(out.output.contains("a.txt") && out.output.contains("b.rs"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unknown_tool_is_an_error() {
        let out = run("nope", json!({}), any_cwd(), RunMode::Edit);
        assert!(!out.ok);
        assert!(out.output.contains("unknown tool"));
    }

    #[test]
    fn permission_rule_denies_matching_calls() {
        let tools = ToolSet::new(
            vec![],
            vec![crate::openai_compatible::PermissionRule::deny_matching(
                "bash", "rm -rf",
            )],
            None,
        );
        let cancel = AtomicBool::new(false);
        let ctx = ToolCtx {
            cwd: any_cwd(),
            mode: RunMode::Edit,
            cancel: &cancel,
            run_id: "t",
            call_id: "c",
            skills: &[],
            subagent: None,
            model: None,
        };
        // The matching command is refused *before* executing (rm never runs).
        let denied = tools.execute("bash", &json!({ "command": "rm -rf /tmp/x" }), &ctx);
        assert!(
            !denied.ok && denied.output.contains("denied"),
            "{}",
            denied.output
        );
        // A non-matching command is allowed through and runs.
        let ok = tools.execute("bash", &json!({ "command": "echo hi" }), &ctx);
        assert!(ok.ok, "{}", ok.output);
    }

    #[test]
    fn permission_ask_consults_the_prompt() {
        let cancel = AtomicBool::new(false);
        let ctx = ToolCtx {
            cwd: any_cwd(),
            mode: RunMode::Edit,
            cancel: &cancel,
            run_id: "t",
            call_id: "c",
            skills: &[],
            subagent: None,
            model: None,
        };
        // Prompt denies any command mentioning "secret", allows the rest.
        let prompt: crate::openai_compatible::PermissionPrompt =
            std::sync::Arc::new(|r: &crate::openai_compatible::PermissionRequest| {
                !r.subject.as_deref().unwrap_or("").contains("secret")
            });
        let tools = ToolSet::new(
            vec![],
            vec![crate::openai_compatible::PermissionRule::ask("bash")],
            Some(prompt),
        );
        assert!(
            !tools
                .execute("bash", &json!({ "command": "cat secret.txt" }), &ctx)
                .ok,
            "secret → denied"
        );
        assert!(
            tools
                .execute("bash", &json!({ "command": "echo hi" }), &ctx)
                .ok,
            "echo → allowed"
        );
        // An `Ask` rule with no prompt set denies (safe default).
        let no_prompt = ToolSet::new(
            vec![],
            vec![crate::openai_compatible::PermissionRule::ask("bash")],
            None,
        );
        assert!(
            !no_prompt
                .execute("bash", &json!({ "command": "echo hi" }), &ctx)
                .ok,
            "no prompt → denied"
        );
    }

    #[test]
    fn tool_kind_maps_names_to_behaviour() {
        let ts = ToolSet::builtin();
        assert!(matches!(ts.kind("read"), ToolKind::Read));
        assert!(matches!(ts.kind("glob"), ToolKind::Search));
        assert!(matches!(ts.kind("write"), ToolKind::Write));
        assert!(matches!(ts.kind("edit"), ToolKind::Edit));
        assert!(matches!(ts.kind("bash"), ToolKind::Execute));
        assert!(matches!(ts.kind("mystery"), ToolKind::Other));
    }

    #[test]
    fn schemas_are_typed_and_normalized() {
        let defs = ToolSet::builtin().defs(RunMode::Edit, "qwen2.5-coder", AgentContext::Main);
        let read = defs
            .iter()
            .find(|d| d["function"]["name"] == "read")
            .unwrap();
        let params = &read["function"]["parameters"];
        // schemars metadata is stripped — just the schema the model needs.
        assert!(params.get("$schema").is_none());
        assert!(params.get("title").is_none());
        assert_eq!(params["type"], "object");
        // Typed fields are present; `path` (String) is required, `offset`
        // (Option) is not — required-ness derives from the type, not by hand.
        assert!(params["properties"]["path"].is_object());
        assert!(params["properties"]["offset"].is_object());
        let required = params["required"].as_array().unwrap();
        assert!(required.iter().any(|r| r == "path"), "path is required");
        assert!(
            !required.iter().any(|r| r == "offset"),
            "Option field is not required"
        );
        // Field descriptions come from the struct's doc comments.
        let desc = params["properties"]["path"]["description"]
            .as_str()
            .unwrap_or("");
        assert!(
            desc.contains("relative"),
            "doc-comment description carried through: {desc:?}"
        );
    }

    #[test]
    fn safe_join_refuses_traversal_and_absolute() {
        let cwd = Path::new("/work");
        assert_eq!(
            safe_join(cwd, "a/b.txt"),
            Some(PathBuf::from("/work/a/b.txt"))
        );
        assert_eq!(
            safe_join(cwd, "./a.txt"),
            Some(PathBuf::from("/work/a.txt"))
        );
        assert_eq!(safe_join(cwd, "../escape"), None);
        assert_eq!(safe_join(cwd, "a/../../escape"), None);
        assert_eq!(safe_join(cwd, "/etc/passwd"), None);
    }

    #[test]
    fn read_is_line_numbered_with_offset_and_limit() {
        let dir = scratch("read");
        std::fs::write(dir.join("f.txt"), "alpha\nbravo\ncharlie\ndelta\n").unwrap();
        let out = run(
            "read",
            json!({ "path": "f.txt", "offset": 2, "limit": 2 }),
            &dir,
            RunMode::Ask,
        );
        assert!(out.ok);
        assert!(out.output.contains("2: bravo"));
        assert!(out.output.contains("3: charlie"));
        assert!(!out.output.contains("1: alpha"));
        assert!(out.output.contains("more lines"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn edit_enforces_uniqueness_and_replace_all() {
        let dir = scratch("edit");
        let f = dir.join("c.txt");
        std::fs::write(&f, "x = 1\nx = 1\n").unwrap();
        let amb = run(
            "edit",
            json!({ "path": "c.txt", "old_string": "x = 1", "new_string": "x = 2" }),
            &dir,
            RunMode::Edit,
        );
        assert!(!amb.ok);
        assert!(amb.output.contains("not unique"));
        let all = run(
            "edit",
            json!({ "path": "c.txt", "old_string": "x = 1", "new_string": "x = 2", "replace_all": true }),
            &dir,
            RunMode::Edit,
        );
        assert!(all.ok, "{}", all.output);
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "x = 2\nx = 2\n");
        let miss = run(
            "edit",
            json!({ "path": "c.txt", "old_string": "nope", "new_string": "y" }),
            &dir,
            RunMode::Edit,
        );
        assert!(!miss.ok);
        assert!(miss.output.contains("not found"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn edit_falls_back_to_whitespace_tolerant_match() {
        let dir = scratch("edit-fuzzy");
        let f = dir.join("d.rs");
        std::fs::write(&f, "fn main() {\n    let x = 1;\n}\n").unwrap();
        let out = run(
            "edit",
            json!({ "path": "d.rs", "old_string": "let x = 1; ", "new_string": "    let x = 2;" }),
            &dir,
            RunMode::Edit,
        );
        assert!(out.ok, "{}", out.output);
        assert!(out.output.contains("whitespace-tolerant"));
        assert_eq!(
            std::fs::read_to_string(&f).unwrap(),
            "fn main() {\n    let x = 2;\n}\n"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bash_runs_reports_exit_and_times_out() {
        use std::time::{Duration, Instant};
        let dir = scratch("bash");
        let ok = run(
            "bash",
            json!({ "command": "echo hi" }),
            &dir,
            RunMode::Edit,
        );
        assert!(ok.ok, "{}", ok.output);
        // Trimmed: `cmd` ends lines with CRLF, `sh` with LF.
        assert_eq!(ok.output.trim(), "hi");
        let bad = run("bash", json!({ "command": "exit 3" }), &dir, RunMode::Edit);
        assert!(!bad.ok);
        assert!(bad.output.contains("exit 3"));
        let started = Instant::now();
        let slow = run(
            "bash",
            json!({ "command": sleep_5(), "timeout": 150 }),
            &dir,
            RunMode::Edit,
        );
        assert!(!slow.ok);
        assert!(slow.output.contains("timed out"));
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "should not have waited for the full sleep"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bash_honors_cancel() {
        let dir = scratch("bash-cancel");
        let cancel = AtomicBool::new(true); // already cancelled
        let out = ToolSet::builtin().execute(
            "bash",
            &json!({ "command": sleep_5() }),
            &ToolCtx {
                cwd: &dir,
                mode: RunMode::Edit,
                cancel: &cancel,
                run_id: "t",
                call_id: "c",
                skills: &[],
                subagent: None,
                model: None,
            },
        );
        assert!(!out.ok);
        assert!(out.output.contains("cancelled"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn glob_matches_gitignore_aware() {
        let dir = scratch("glob");
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("src/a.rs"), "").unwrap();
        std::fs::write(dir.join("src/b.rs"), "").unwrap();
        std::fs::write(dir.join("README.md"), "").unwrap();
        let out = run("glob", json!({ "pattern": "**/*.rs" }), &dir, RunMode::Ask);
        assert!(out.ok, "{}", out.output);
        // Windows reports `src\\a.rs`; compare on normalised separators.
        let listed = out.output.replace('\\', "/");
        assert!(listed.contains("src/a.rs"));
        assert!(listed.contains("src/b.rs"));
        assert!(!listed.contains("README.md"));
        let none = run("glob", json!({ "pattern": "*.zzz" }), &dir, RunMode::Ask);
        assert!(none.ok);
        assert!(none.output.contains("no files matched"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn grep_matches_with_include_and_invalid_regex() {
        let dir = scratch("grep");
        std::fs::write(dir.join("a.rs"), "fn foo() {}\nfn bar() {}\n").unwrap();
        std::fs::write(dir.join("b.txt"), "fn foo() {}\n").unwrap();
        let out = run(
            "grep",
            json!({ "pattern": "fn \\w+", "include": "*.rs" }),
            &dir,
            RunMode::Ask,
        );
        assert!(out.ok, "{}", out.output);
        assert!(out.output.contains("a.rs:1: fn foo"));
        assert!(out.output.contains("a.rs:2: fn bar"));
        assert!(!out.output.contains("b.txt")); // excluded by include
        let none = run("grep", json!({ "pattern": "zzzznope" }), &dir, RunMode::Ask);
        assert!(none.ok);
        assert!(none.output.contains("no matches"));
        let bad = run(
            "grep",
            json!({ "pattern": "[unclosed" }),
            &dir,
            RunMode::Ask,
        );
        assert!(!bad.ok);
        assert!(bad.output.contains("invalid regex"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn todowrite_produces_a_plan_event() {
        let out = run(
            "todowrite",
            json!({ "todos": [
                { "content": "first", "status": "in_progress", "priority": "high" },
                { "content": "second", "status": "pending" }
            ]}),
            any_cwd(),
            RunMode::Edit,
        );
        assert!(out.ok, "{}", out.output);
        let plan = out
            .events
            .iter()
            .find_map(|e| match e {
                RunEvent::Plan { entries, .. } => Some(entries),
                _ => None,
            })
            .expect("a Plan event");
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].content, "first");
        assert!(matches!(plan[0].status, crate::PlanEntryStatus::InProgress));
    }

    #[test]
    fn question_asks_and_stops_the_run() {
        let out = run(
            "question",
            json!({ "questions": [
                { "header": "Pick", "question": "Which option?",
                  "options": [ { "label": "A", "description": "first" }, { "label": "B" } ] }
            ]}),
            any_cwd(),
            RunMode::Ask,
        );
        assert!(out.ok, "{}", out.output);
        assert!(out.stop, "question ends the run to await the answer");
        assert!(
            out.events
                .iter()
                .any(|e| matches!(e, RunEvent::AskQuestion { .. })),
            "emits AskQuestion"
        );
    }

    #[test]
    fn webfetch_rejects_a_non_http_url() {
        let out = run(
            "webfetch",
            json!({ "url": "ftp://example.com/x" }),
            any_cwd(),
            RunMode::Ask,
        );
        assert!(!out.ok);
        assert!(out.output.contains("valid http"));
    }

    #[test]
    fn apply_patch_swaps_with_edit_write_for_gpt5() {
        let gpt5 = names(RunMode::Edit, "gpt-5");
        assert!(
            gpt5.contains(&"apply_patch".to_string()),
            "gpt-5 gets apply_patch"
        );
        assert!(!gpt5.contains(&"edit".to_string()), "edit hidden for gpt-5");
        assert!(
            !gpt5.contains(&"write".to_string()),
            "write hidden for gpt-5"
        );
        // gpt-4 and `oss` are excluded from the swap; local models too.
        for m in ["gpt-4o", "qwen2.5-coder", "gpt-oss"] {
            let offered = names(RunMode::Edit, m);
            assert!(
                !offered.contains(&"apply_patch".to_string()),
                "{m} does not get apply_patch"
            );
            assert!(offered.contains(&"edit".to_string()), "{m} keeps edit");
        }
    }

    #[test]
    fn apply_patch_adds_updates_and_deletes() {
        let dir = scratch("patch");
        std::fs::write(dir.join("keep.txt"), "alpha\nbeta\ngamma\n").unwrap();
        std::fs::write(dir.join("gone.txt"), "x\n").unwrap();
        // Built by join (not a `\`-continued literal) so context-line leading
        // spaces survive.
        let patch = [
            "*** Begin Patch",
            "*** Add File: new.txt",
            "+hello",
            "+world",
            "*** Update File: keep.txt",
            "@@",
            " alpha",
            "-beta",
            "+BETA",
            " gamma",
            "*** Delete File: gone.txt",
            "*** End Patch",
        ]
        .join("\n");
        let out = run(
            "apply_patch",
            json!({ "patchText": patch }),
            &dir,
            RunMode::Edit,
        );
        assert!(out.ok, "{}", out.output);
        assert_eq!(
            std::fs::read_to_string(dir.join("new.txt")).unwrap(),
            "hello\nworld\n"
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("keep.txt")).unwrap(),
            "alpha\nBETA\ngamma\n"
        );
        assert!(!dir.join("gone.txt").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn skill_loads_a_body_by_name() {
        use crate::openai_compatible::skills::Skill;
        let cancel = AtomicBool::new(false);
        let skills = vec![Skill {
            name: "deploy".into(),
            description: Some("how to deploy".into()),
            body: "Run the deploy script.".into(),
        }];
        let ctx = ToolCtx {
            cwd: any_cwd(),
            mode: RunMode::Ask,
            cancel: &cancel,
            run_id: "t",
            call_id: "c",
            skills: &skills,
            subagent: None,
            model: None,
        };
        let out = ToolSet::builtin().execute("skill", &json!({ "name": "deploy" }), &ctx);
        assert!(out.ok, "{}", out.output);
        assert!(out.output.contains("Run the deploy script."));
        let miss = ToolSet::builtin().execute("skill", &json!({ "name": "nope" }), &ctx);
        assert!(!miss.ok);
        assert!(
            miss.output.contains("deploy"),
            "lists available: {}",
            miss.output
        );
    }

    #[test]
    fn subagent_toolset_excludes_task_and_question() {
        // The full set offers task + question…
        let full = names(RunMode::Edit, "qwen2.5-coder");
        assert!(full.contains(&"task".to_string()) && full.contains(&"question".to_string()));
        // …but a subagent's set omits them (no nesting, no user to answer)
        // while keeping the file/shell tools.
        let sub: Vec<String> = ToolSet::builtin()
            .defs(RunMode::Edit, "qwen2.5-coder", AgentContext::Subagent)
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap().to_owned())
            .collect();
        assert!(!sub.contains(&"task".to_string()), "no nested task");
        assert!(
            !sub.contains(&"question".to_string()),
            "no question in a subagent"
        );
        assert!(
            sub.contains(&"read".to_string()) && sub.contains(&"edit".to_string()),
            "keeps file tools"
        );
    }

    #[test]
    fn task_runs_a_subagent_via_the_runner() {
        // A fake runner proves `task` wires its prompt through to a result.
        struct Echo;
        impl SubagentRunner for Echo {
            fn run(
                &self,
                _subagent_type: Option<&str>,
                prompt: &str,
                _cancel: &AtomicBool,
            ) -> Result<String, String> {
                Ok(format!("did: {prompt}"))
            }
        }
        let cancel = AtomicBool::new(false);
        let echo = Echo;
        let ctx = ToolCtx {
            cwd: any_cwd(),
            mode: RunMode::Edit,
            cancel: &cancel,
            run_id: "t",
            call_id: "c",
            skills: &[],
            subagent: Some(&echo),
            model: None,
        };
        let out = ToolSet::builtin().execute(
            "task",
            &json!({ "description": "do it", "prompt": "the work" }),
            &ctx,
        );
        assert!(out.ok, "{}", out.output);
        assert!(out.output.contains("did: the work"));

        // Without a runner (e.g. inside a subagent), task refuses.
        let ctx2 = ToolCtx {
            subagent: None,
            ..ctx
        };
        let no = ToolSet::builtin().execute(
            "task",
            &json!({ "description": "x", "prompt": "y" }),
            &ctx2,
        );
        assert!(!no.ok);
        assert!(no.output.contains("not available"));
    }

    #[test]
    fn apply_patch_update_tolerates_off_context_whitespace() {
        let dir = scratch("patch-fuzzy");
        // File indents the line; the patch's `-` context has a trailing space
        // and no indent, so the exact match fails and the fuzzy path applies it.
        std::fs::write(dir.join("f.rs"), "fn main() {\n    let x = 1;\n}\n").unwrap();
        let patch = [
            "*** Begin Patch",
            "*** Update File: f.rs",
            "@@",
            "-let x = 1; ",
            "+    let x = 2;",
            "*** End Patch",
        ]
        .join("\n");
        let out = run(
            "apply_patch",
            json!({ "patchText": patch }),
            &dir,
            RunMode::Edit,
        );
        assert!(out.ok, "{}", out.output);
        assert_eq!(
            std::fs::read_to_string(dir.join("f.rs")).unwrap(),
            "fn main() {\n    let x = 2;\n}\n"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn long_tool_output_is_truncated_to_the_head() {
        // Short output is untouched (regardless of direction).
        assert_eq!(
            truncate_output("a\nb\nc".to_string(), Keep::Head),
            "a\nb\nc"
        );
        // Past the line cap → trimmed to the head + a note.
        let big: String = (0..5000).map(|i| format!("line {i}\n")).collect();
        let out = truncate_output(big, Keep::Head);
        assert!(out.contains("truncated"), "has a truncation note");
        assert!(
            out.lines().count() <= MAX_OUTPUT_LINES + 2,
            "trimmed near the line cap"
        );
        assert!(out.starts_with("line 0\n"), "keeps the head");
        assert!(!out.contains("line 4999"), "drops the tail");
    }

    #[test]
    fn tail_truncation_keeps_the_end() {
        // The diagnostic at the very end survives; the leading noise is dropped.
        let big: String = (0..5000).map(|i| format!("line {i}\n")).collect();
        let out = truncate_output(big, Keep::Tail);
        assert!(out.contains("truncated"), "has a truncation note");
        assert!(
            out.lines().count() <= MAX_OUTPUT_LINES + 2,
            "trimmed near the line cap"
        );
        assert!(
            out.trim_end().ends_with("line 4999"),
            "keeps the tail: {:?}",
            &out[out.len().saturating_sub(40)..]
        );
        assert!(!out.contains("line 0\n"), "drops the head");
    }

    #[test]
    fn head_and_tail_truncation_keeps_both_ends() {
        // bash's shape: a chatty command whose error lands on the last line. Both
        // the leading output and the trailing diagnostic must be preserved.
        let mut big: String = (0..5000).map(|i| format!("info line {i}\n")).collect();
        big.push_str("error: command failed\n(exit 1)\n");
        let out = truncate_output(big, Keep::HeadAndTail);
        assert!(out.contains("info line 0"), "keeps the head");
        assert!(
            out.contains("error: command failed") && out.contains("(exit 1)"),
            "keeps the trailing diagnostic: {:?}",
            &out[out.len().saturating_sub(60)..]
        );
        assert!(
            out.contains("omitted from the middle"),
            "marks the gap in the middle"
        );
        assert!(!out.contains("info line 2500"), "the middle is dropped");
        assert!(
            out.lines().count() <= MAX_OUTPUT_LINES + 3,
            "within the line budget (+ marker)"
        );
    }

    #[test]
    fn bash_truncation_keeps_the_trailing_error() {
        // End-to-end through dispatch: a real over-cap `bash` run keeps its tail
        // (where the exit line lives) — the head-only cap would have lost it.
        let dir = scratch("bash-trunc");
        let cmd = "for i in $(seq 1 5000); do echo info line $i; done; echo 'BOOM the real error'; exit 7";
        let out = run("bash", json!({ "command": cmd }), &dir, RunMode::Edit);
        assert!(!out.ok, "non-zero exit is an error");
        assert!(
            out.output.contains("BOOM the real error"),
            "the trailing error survives truncation"
        );
        assert!(out.output.contains("exit 7"), "the exit framing survives");
        assert!(
            out.output.contains("omitted from the middle"),
            "middle elided, ends kept"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_accepts_absolute_paths() {
        let dir = scratch("read-abs");
        let f = dir.join("abs.txt");
        std::fs::write(&f, "absolute content\n").unwrap();
        // cwd is unrelated to the file — read still finds it by absolute path.
        let out = run(
            "read",
            json!({ "path": f.to_str().unwrap() }),
            Path::new("/nonexistent-cwd"),
            RunMode::Ask,
        );
        assert!(out.ok, "{}", out.output);
        assert!(out.output.contains("absolute content"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_caps_on_total_bytes_within_the_line_limit() {
        let dir = scratch("read-bytes");
        // Many lines, each under the per-line char cap (2000) but cumulatively
        // megabytes — the case the line cap (2000 lines) alone wouldn't catch.
        let line = "x".repeat(1500);
        let content = (0..1000).map(|_| format!("{line}\n")).collect::<String>();
        let f = dir.join("wide.txt");
        std::fs::write(&f, &content).unwrap();
        let out = run("read", json!({ "path": "wide.txt" }), &dir, RunMode::Ask);
        assert!(out.ok, "{}", out.output);
        // Trimmed to roughly the byte cap (+ one line + footer), far below the
        // ~1.5 MB the line cap alone would have let through.
        assert!(
            out.output.len() <= MAX_OUTPUT_BYTES + 2 * 1024,
            "stays near the byte cap: {} bytes",
            out.output.len()
        );
        assert!(
            out.output.contains("output capped at"),
            "byte-cap footer: {}",
            &out.output[out.output.len().saturating_sub(120)..]
        );
        assert!(
            out.output.contains("offset="),
            "tells the model how to page on"
        );
        // A single line that alone exceeds the budget still makes progress (it's
        // char-truncated, then emitted — the loop never stalls empty).
        std::fs::write(&f, format!("{}\n", "y".repeat(60 * 1024))).unwrap();
        let one = run("read", json!({ "path": "wide.txt" }), &dir, RunMode::Ask);
        assert!(
            one.ok && one.output.starts_with("1: yyy"),
            "emits the one huge line: {}",
            &one.output[..40.min(one.output.len())]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
