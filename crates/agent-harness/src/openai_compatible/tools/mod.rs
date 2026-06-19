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
//! * **Mode gating is at the call, not the schema.** Every tool is offered in
//!   every mode (the model sees the full surface); a mutating call in a
//!   read-only ([`RunMode::Ask`]) run is refused at `execute`, with a message
//!   telling the model to answer instead. Review of applied edits stays in the
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
    /// Whether the tool mutates state — a mutating call is refused at `execute`
    /// in a read-only ([`RunMode::Ask`]) run.
    fn mutating(&self) -> bool;
    /// Whether the tool is offered to the model. Default: always — the model
    /// sees the full tool surface in every mode, and a mutating call in a
    /// read-only run is refused at `execute` (not hidden here). Only the
    /// apply_patch ⇄ edit/write swap overrides this (model-dependent).
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
        Box::new(websearch::WebSearch),
        Box::new(file::Write),
        Box::new(file::Edit),
        Box::new(shell::Bash),
        Box::new(patch::ApplyPatch),
        Box::new(task::Task),
    ]
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
        Self { tools: builtins(), permissions: Vec::new(), permission_prompt: None }
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
        Self { tools, permissions, permission_prompt }
    }

    /// The OpenAI `tools` array offered to the model: every tool in every mode
    /// (a mutating call is gated at `execute`, not hidden here), minus the
    /// model's apply_patch ⇄ edit/write swap and, in a subagent, the opt-outs
    /// (`task`, `question`).
    pub(crate) fn defs(&self, mode: RunMode, model: &str, subagent: bool) -> Vec<Value> {
        self.tools
            .iter()
            .filter(|t| t.offered(mode, model) && (!subagent || t.in_subagent()))
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
        self.tools.iter().find(|t| t.id() == name).map_or(ToolKind::Other, |t| t.kind())
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
            outcome.output = truncate_output(outcome.output);
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
                        rule.pattern.as_deref().map(|p| format!(" (matched `{p}`)")).unwrap_or_default()
                    )),
                    // Defer to the host's prompt (blocking); absent prompt → deny.
                    crate::openai_compatible::Permission::Ask => {
                        let request =
                            crate::openai_compatible::PermissionRequest { tool: tool.id().to_owned(), subject: subject.clone() };
                        if self.permission_prompt.as_ref().is_some_and(|prompt| prompt(&request)) {
                            None
                        } else {
                            Some(format!("`{}` denied (permission prompt declined or unset)", tool.id()))
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
        Self { ok: true, output: output.into(), stop: false, events: Vec::new() }
    }
    fn err(output: impl Into<String>) -> Self {
        Self { ok: false, output: output.into(), stop: false, events: Vec::new() }
    }
    /// A successful result that also ends the run after this turn.
    fn stop(output: impl Into<String>) -> Self {
        Self { ok: true, output: output.into(), stop: true, events: Vec::new() }
    }
    /// Attach side events for the loop to emit.
    fn with_events(mut self, events: Vec<RunEvent>) -> Self {
        self.events = events;
        self
    }
}

/// Trim a tool's output to its head when it exceeds the output caps, with a
/// note. We don't spill the remainder to a file (OpenCode does) because our file
/// tools are sandboxed to the cwd and couldn't read it back — the note tells the
/// model to narrow its command instead.
fn truncate_output(output: String) -> String {
    let total_lines = output.lines().count();
    if total_lines <= MAX_OUTPUT_LINES && output.len() <= MAX_OUTPUT_BYTES {
        return output;
    }
    let mut kept = String::new();
    let mut kept_lines = 0usize;
    for line in output.lines() {
        if kept_lines >= MAX_OUTPUT_LINES || kept.len() + line.len() + 1 > MAX_OUTPUT_BYTES {
            break;
        }
        kept.push_str(line);
        kept.push('\n');
        kept_lines += 1;
    }
    let omitted = total_lines.saturating_sub(kept_lines);
    // Spill the full output to a file the model can fetch with `read` (OpenCode's
    // approach). Best-effort — if the spill fails, just advise narrowing.
    match spill_output(&output) {
        Some(path) => kept.push_str(&format!(
            "[output truncated — {omitted} more line(s) omitted. Full output saved to {path} — \
             read it with the `read` tool (offset/limit), or narrow the command.]"
        )),
        None => kept.push_str(&format!(
            "[output truncated — {omitted} more line(s) omitted; narrow the command, or use \
             read/grep with offset/limit on the relevant file.]"
        )),
    }
    kept
}

/// Write a tool's full output to a scratch file and return its absolute path, so
/// the model can read the part beyond the truncation cap (OpenCode spills to its
/// data dir; we use a temp dir). `None` on any I/O failure.
fn spill_output(output: &str) -> Option<String> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join("openai-compatible").join("tool-output");
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
    serde_json::from_value(args.clone()).map_err(|e| ToolOutcome::err(format!("invalid arguments: {e}")))
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
        ToolSet::builtin().execute(name, &args, &ToolCtx { cwd, mode, cancel: &cancel, run_id: "t", call_id: "c", skills: &[], subagent: None })
    }

    fn names(mode: RunMode, model: &str) -> Vec<String> {
        ToolSet::builtin()
            .defs(mode, model, false)
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap().to_owned())
            .collect()
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hl-tools-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn all_tools_are_offered_in_every_mode() {
        // A local (non-gpt) model, so edit/write are offered (no apply_patch swap).
        let ask = names(RunMode::Ask, "qwen2.5-coder");
        let edit = names(RunMode::Edit, "qwen2.5-coder");
        // Every tool — read-only AND mutating — is offered in both modes; a
        // mutating CALL in Ask is refused at execute (see
        // `mutating_tools_refused_in_ask_mode`), not withheld from the schema.
        for t in [
            "read", "glob", "grep", "list", "webfetch", "todowrite", "question", "skill", "write",
            "edit", "bash", "task",
        ] {
            assert!(ask.contains(&t.to_string()), "{t} offered in Ask");
            assert!(edit.contains(&t.to_string()), "{t} offered in Edit");
        }
    }

    #[test]
    fn mutating_tools_refused_in_ask_mode() {
        for name in ["write", "edit", "bash"] {
            let out = run(name, json!({}), Path::new("/tmp"), RunMode::Ask);
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
        assert!(out.output.contains("sub/"), "directory marked with /: {}", out.output);
        assert!(out.output.contains("a.txt") && out.output.contains("b.rs"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unknown_tool_is_an_error() {
        let out = run("nope", json!({}), Path::new("/tmp"), RunMode::Edit);
        assert!(!out.ok);
        assert!(out.output.contains("unknown tool"));
    }

    #[test]
    fn permission_rule_denies_matching_calls() {
        let tools = ToolSet::new(vec![], vec![crate::openai_compatible::PermissionRule::deny_matching("bash", "rm -rf")], None);
        let cancel = AtomicBool::new(false);
        let ctx = ToolCtx {
            cwd: Path::new("/tmp"),
            mode: RunMode::Edit,
            cancel: &cancel,
            run_id: "t",
            call_id: "c",
            skills: &[],
            subagent: None,
        };
        // The matching command is refused *before* executing (rm never runs).
        let denied = tools.execute("bash", &json!({ "command": "rm -rf /tmp/x" }), &ctx);
        assert!(!denied.ok && denied.output.contains("denied"), "{}", denied.output);
        // A non-matching command is allowed through and runs.
        let ok = tools.execute("bash", &json!({ "command": "echo hi" }), &ctx);
        assert!(ok.ok, "{}", ok.output);
    }

    #[test]
    fn permission_ask_consults_the_prompt() {
        let cancel = AtomicBool::new(false);
        let ctx = ToolCtx {
            cwd: Path::new("/tmp"),
            mode: RunMode::Edit,
            cancel: &cancel,
            run_id: "t",
            call_id: "c",
            skills: &[],
            subagent: None,
        };
        // Prompt denies any command mentioning "secret", allows the rest.
        let prompt: crate::openai_compatible::PermissionPrompt =
            std::sync::Arc::new(|r: &crate::openai_compatible::PermissionRequest| !r.subject.as_deref().unwrap_or("").contains("secret"));
        let tools = ToolSet::new(vec![], vec![crate::openai_compatible::PermissionRule::ask("bash")], Some(prompt));
        assert!(!tools.execute("bash", &json!({ "command": "cat secret.txt" }), &ctx).ok, "secret → denied");
        assert!(tools.execute("bash", &json!({ "command": "echo hi" }), &ctx).ok, "echo → allowed");
        // An `Ask` rule with no prompt set denies (safe default).
        let no_prompt = ToolSet::new(vec![], vec![crate::openai_compatible::PermissionRule::ask("bash")], None);
        assert!(!no_prompt.execute("bash", &json!({ "command": "echo hi" }), &ctx).ok, "no prompt → denied");
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
        let defs = ToolSet::builtin().defs(RunMode::Edit, "qwen2.5-coder", false);
        let read = defs.iter().find(|d| d["function"]["name"] == "read").unwrap();
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
        assert!(!required.iter().any(|r| r == "offset"), "Option field is not required");
        // Field descriptions come from the struct's doc comments.
        let desc = params["properties"]["path"]["description"].as_str().unwrap_or("");
        assert!(desc.contains("relative"), "doc-comment description carried through: {desc:?}");
    }

    #[test]
    fn safe_join_refuses_traversal_and_absolute() {
        let cwd = Path::new("/work");
        assert_eq!(safe_join(cwd, "a/b.txt"), Some(PathBuf::from("/work/a/b.txt")));
        assert_eq!(safe_join(cwd, "./a.txt"), Some(PathBuf::from("/work/a.txt")));
        assert_eq!(safe_join(cwd, "../escape"), None);
        assert_eq!(safe_join(cwd, "a/../../escape"), None);
        assert_eq!(safe_join(cwd, "/etc/passwd"), None);
    }

    #[test]
    fn read_is_line_numbered_with_offset_and_limit() {
        let dir = scratch("read");
        std::fs::write(dir.join("f.txt"), "alpha\nbravo\ncharlie\ndelta\n").unwrap();
        let out = run("read", json!({ "path": "f.txt", "offset": 2, "limit": 2 }), &dir, RunMode::Ask);
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
        let amb = run("edit", json!({ "path": "c.txt", "old_string": "x = 1", "new_string": "x = 2" }), &dir, RunMode::Edit);
        assert!(!amb.ok);
        assert!(amb.output.contains("not unique"));
        let all = run("edit", json!({ "path": "c.txt", "old_string": "x = 1", "new_string": "x = 2", "replace_all": true }), &dir, RunMode::Edit);
        assert!(all.ok, "{}", all.output);
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "x = 2\nx = 2\n");
        let miss = run("edit", json!({ "path": "c.txt", "old_string": "nope", "new_string": "y" }), &dir, RunMode::Edit);
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
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "fn main() {\n    let x = 2;\n}\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bash_runs_reports_exit_and_times_out() {
        use std::time::{Duration, Instant};
        let dir = scratch("bash");
        let ok = run("bash", json!({ "command": "printf hi" }), &dir, RunMode::Edit);
        assert!(ok.ok, "{}", ok.output);
        assert_eq!(ok.output, "hi");
        let bad = run("bash", json!({ "command": "exit 3" }), &dir, RunMode::Edit);
        assert!(!bad.ok);
        assert!(bad.output.contains("exit 3"));
        let started = Instant::now();
        let slow = run("bash", json!({ "command": "sleep 5", "timeout": 150 }), &dir, RunMode::Edit);
        assert!(!slow.ok);
        assert!(slow.output.contains("timed out"));
        assert!(started.elapsed() < Duration::from_secs(2), "should not have waited for the full sleep");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bash_honors_cancel() {
        let dir = scratch("bash-cancel");
        let cancel = AtomicBool::new(true); // already cancelled
        let out = ToolSet::builtin().execute("bash", &json!({ "command": "sleep 5" }), &ToolCtx { cwd: &dir, mode: RunMode::Edit, cancel: &cancel, run_id: "t", call_id: "c", skills: &[], subagent: None });
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
        assert!(out.output.contains("src/a.rs"));
        assert!(out.output.contains("src/b.rs"));
        assert!(!out.output.contains("README.md"));
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
        let out = run("grep", json!({ "pattern": "fn \\w+", "include": "*.rs" }), &dir, RunMode::Ask);
        assert!(out.ok, "{}", out.output);
        assert!(out.output.contains("a.rs:1: fn foo"));
        assert!(out.output.contains("a.rs:2: fn bar"));
        assert!(!out.output.contains("b.txt")); // excluded by include
        let none = run("grep", json!({ "pattern": "zzzznope" }), &dir, RunMode::Ask);
        assert!(none.ok);
        assert!(none.output.contains("no matches"));
        let bad = run("grep", json!({ "pattern": "[unclosed" }), &dir, RunMode::Ask);
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
            Path::new("/tmp"),
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
            Path::new("/tmp"),
            RunMode::Ask,
        );
        assert!(out.ok, "{}", out.output);
        assert!(out.stop, "question ends the run to await the answer");
        assert!(
            out.events.iter().any(|e| matches!(e, RunEvent::AskQuestion { .. })),
            "emits AskQuestion"
        );
    }

    #[test]
    fn webfetch_rejects_a_non_http_url() {
        let out = run("webfetch", json!({ "url": "ftp://example.com/x" }), Path::new("/tmp"), RunMode::Ask);
        assert!(!out.ok);
        assert!(out.output.contains("valid http"));
    }

    #[test]
    fn apply_patch_swaps_with_edit_write_for_gpt5() {
        let gpt5 = names(RunMode::Edit, "gpt-5");
        assert!(gpt5.contains(&"apply_patch".to_string()), "gpt-5 gets apply_patch");
        assert!(!gpt5.contains(&"edit".to_string()), "edit hidden for gpt-5");
        assert!(!gpt5.contains(&"write".to_string()), "write hidden for gpt-5");
        // gpt-4 and `oss` are excluded from the swap; local models too.
        for m in ["gpt-4o", "qwen2.5-coder", "gpt-oss"] {
            let offered = names(RunMode::Edit, m);
            assert!(!offered.contains(&"apply_patch".to_string()), "{m} does not get apply_patch");
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
        let out = run("apply_patch", json!({ "patchText": patch }), &dir, RunMode::Edit);
        assert!(out.ok, "{}", out.output);
        assert_eq!(std::fs::read_to_string(dir.join("new.txt")).unwrap(), "hello\nworld\n");
        assert_eq!(std::fs::read_to_string(dir.join("keep.txt")).unwrap(), "alpha\nBETA\ngamma\n");
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
        let ctx =
            ToolCtx { cwd: Path::new("/tmp"), mode: RunMode::Ask, cancel: &cancel, run_id: "t", call_id: "c", skills: &skills, subagent: None };
        let out = ToolSet::builtin().execute("skill", &json!({ "name": "deploy" }), &ctx);
        assert!(out.ok, "{}", out.output);
        assert!(out.output.contains("Run the deploy script."));
        let miss = ToolSet::builtin().execute("skill", &json!({ "name": "nope" }), &ctx);
        assert!(!miss.ok);
        assert!(miss.output.contains("deploy"), "lists available: {}", miss.output);
    }

    #[test]
    fn subagent_toolset_excludes_task_and_question() {
        // The full set offers task + question…
        let full = names(RunMode::Edit, "qwen2.5-coder");
        assert!(full.contains(&"task".to_string()) && full.contains(&"question".to_string()));
        // …but a subagent's set omits them (no nesting, no user to answer)
        // while keeping the file/shell tools.
        let sub: Vec<String> = ToolSet::builtin()
            .defs(RunMode::Edit, "qwen2.5-coder", true)
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap().to_owned())
            .collect();
        assert!(!sub.contains(&"task".to_string()), "no nested task");
        assert!(!sub.contains(&"question".to_string()), "no question in a subagent");
        assert!(sub.contains(&"read".to_string()) && sub.contains(&"edit".to_string()), "keeps file tools");
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
            cwd: Path::new("/tmp"),
            mode: RunMode::Edit,
            cancel: &cancel,
            run_id: "t",
            call_id: "c",
            skills: &[],
            subagent: Some(&echo),
        };
        let out = ToolSet::builtin().execute("task", &json!({ "description": "do it", "prompt": "the work" }), &ctx);
        assert!(out.ok, "{}", out.output);
        assert!(out.output.contains("did: the work"));

        // Without a runner (e.g. inside a subagent), task refuses.
        let ctx2 = ToolCtx { subagent: None, ..ctx };
        let no = ToolSet::builtin().execute("task", &json!({ "description": "x", "prompt": "y" }), &ctx2);
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
        let out = run("apply_patch", json!({ "patchText": patch }), &dir, RunMode::Edit);
        assert!(out.ok, "{}", out.output);
        assert_eq!(std::fs::read_to_string(dir.join("f.rs")).unwrap(), "fn main() {\n    let x = 2;\n}\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn long_tool_output_is_truncated_to_the_head() {
        // Short output is untouched.
        assert_eq!(truncate_output("a\nb\nc".to_string()), "a\nb\nc");
        // Past the line cap → trimmed to the head + a note.
        let big: String = (0..5000).map(|i| format!("line {i}\n")).collect();
        let out = truncate_output(big);
        assert!(out.contains("truncated"), "has a truncation note");
        assert!(out.lines().count() <= MAX_OUTPUT_LINES + 2, "trimmed near the line cap");
        assert!(out.starts_with("line 0\n"), "keeps the head");
    }

    #[test]
    fn read_accepts_absolute_paths() {
        let dir = scratch("read-abs");
        let f = dir.join("abs.txt");
        std::fs::write(&f, "absolute content\n").unwrap();
        // cwd is unrelated to the file — read still finds it by absolute path.
        let out = run("read", json!({ "path": f.to_str().unwrap() }), Path::new("/nonexistent-cwd"), RunMode::Ask);
        assert!(out.ok, "{}", out.output);
        assert!(out.output.contains("absolute content"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
