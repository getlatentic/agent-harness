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
pub(crate) mod discovery;
pub(crate) mod host;
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

/// One tool's OpenAI function definition.
fn tool_def(tool: &dyn Tool) -> Value {
    json!({
        "type": "function",
        "function": { "name": tool.id(), "description": tool.description(), "parameters": tool.parameters() }
    })
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
/// The name of the search tool offered when tools are deferred.
pub(crate) const TOOL_SEARCH: &str = "tool_search";

/// Matches returned per tool search. Enough to choose from, few enough that the
/// reply stays smaller than the schemas it replaced.
const TOOL_SEARCH_LIMIT: usize = 8;

/// MCP schema bytes past which MCP tools move behind [`TOOL_SEARCH`].
///
/// Under this, deferral costs more than it saves: the search tool's own schema
/// plus a round trip to find what was already in the prompt. Claude Code makes
/// the same call with a character threshold behind `ENABLE_TOOL_SEARCH=auto:N`.
const DEFER_MCP_ABOVE_BYTES: usize = 4_096;

/// How many registered names an unknown-tool error lists. Enough to choose
/// from; short enough that a large surface does not paste itself into the
/// transcript on every miss.
const NAMES_IN_UNKNOWN_TOOL_ERROR: usize = 12;

/// How many deferred near-misses it suggests.
const NEAR_MISSES_SUGGESTED: usize = 3;

pub(crate) struct ToolSet {
    tools: Vec<Box<dyn Tool>>,
    permissions: Vec<crate::openai_compatible::PermissionRule>,
    permission_prompt: Option<crate::openai_compatible::PermissionPrompt>,
    /// Ids registered and callable but kept out of the initial tool list. Empty
    /// unless the MCP surface is large enough to be worth hiding.
    deferred: std::collections::HashSet<String>,
    /// Ranked lookup over the deferred tools.
    index: discovery::Index,
}

impl ToolSet {
    /// The built-in tools only (no MCP, no permission rules) — a test convenience;
    /// production builds the set via [`ToolSet::new`].
    #[cfg(test)]
    pub(crate) fn builtin() -> Self {
        Self {
            deferred: std::collections::HashSet::new(),
            index: discovery::Index::build(Vec::new()),
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
        disabled: &[String],
    ) -> Self {
        let mut tools = builtins();
        // Only MCP tools are candidates for deferral: the built-ins are a fixed
        // handful that a coding task needs, where an MCP surface is open-ended
        // and mostly irrelevant to any given request. Identified by id, not by
        // position — the retain below can remove builtins, which would shift
        // any positional split.
        let mcp_ids: Vec<String> = mcp
            .iter()
            .map(|t| t.id().to_owned())
            .filter(|id| !disabled.iter().any(|name| name == id))
            .collect();
        tools.extend(mcp);
        // Withheld at construction, not refused at call time. A tool the host
        // disabled should never reach the model at all: an advertised-then-
        // refused tool still costs its schema in every request, and still
        // invites the model to try.
        tools.retain(|t| !disabled.iter().any(|name| name == t.id()));
        let mcp_bytes: usize = tools
            .iter()
            .filter(|t| mcp_ids.iter().any(|id| id == t.id()))
            .map(|t| t.description().len() + t.parameters().to_string().len())
            .sum();

        let (deferred, index) = if mcp_bytes > DEFER_MCP_ABOVE_BYTES {
            let entries = tools
                .iter()
                .filter(|t| mcp_ids.iter().any(|id| id == t.id()))
                .map(|t| discovery::Entry {
                    id: t.id().to_owned(),
                    text: format!("{} {}", t.id(), t.description()),
                })
                .collect();
            (mcp_ids.into_iter().collect(), discovery::Index::build(entries))
        } else {
            (std::collections::HashSet::new(), discovery::Index::build(Vec::new()))
        };

        Self {
            tools,
            permissions,
            permission_prompt,
            deferred,
            index,
        }
    }

    /// Every built-in tool's id, so a host can offer the choice rather than
    /// hardcoding a list that drifts as tools are added.
    pub fn builtin_tool_names() -> Vec<String> {
        builtins().iter().map(|t| t.id().to_owned()).collect()
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
        let mut defs: Vec<Value> = self
            .tools
            .iter()
            .filter(|t| t.offered(mode, model))
            .filter(|t| !subagent || t.in_subagent())
            .filter(|t| !(t.mutating() && matches!(mode, RunMode::Ask)))
            // Deferred tools stay registered and callable; they are simply not
            // advertised. `tool_search` is how the model reaches them.
            .filter(|t| !self.deferred.contains(t.id()))
            .map(|t| tool_def(t.as_ref()))
            .collect();
        if !self.deferred.is_empty() {
            defs.push(self.search_def());
        }
        defs
    }

    /// The `tool_search` schema, naming how many tools are behind it so the
    /// model can judge whether searching is worth a turn.
    fn search_def(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": TOOL_SEARCH,
                "description": format!(
                    "Find tools that are available but not listed here — {} of them, from connected \
                     integrations. Describe the task in a few words (\"file a github issue\", \"query \
                     the database\") and the matching tools are returned with their full schemas, ready \
                     to call. Search before concluding something cannot be done.",
                    self.deferred.len()
                ),
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": { "type": "string", "description": "What you are trying to do." }
                    }
                }
            }
        })
    }

    /// Answer a `tool_search` call with the schemas of the best matches.
    fn search_tools(&self, args: &Value) -> ToolOutcome {
        let Some(query) = args.get("query").and_then(Value::as_str) else {
            return ToolOutcome::err(format!("{TOOL_SEARCH}: `query` is required"));
        };
        let matched = self.index.search(query, TOOL_SEARCH_LIMIT);
        if matched.is_empty() {
            return ToolOutcome::ok(format!(
                "No available tool matches \"{query}\". Do not search again for the same thing."
            ));
        }
        let defs: Vec<Value> = matched
            .iter()
            .filter_map(|id| self.tools.iter().find(|t| t.id() == *id))
            .map(|t| tool_def(t.as_ref()))
            .collect();
        ToolOutcome::ok(
            serde_json::to_string_pretty(&defs)
                .unwrap_or_else(|_| "could not encode the matching tools".to_owned()),
        )
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

    /// What to say when a model calls a name nothing registered.
    ///
    /// `unknown tool `x`` on its own gives the model nothing to act on, and a
    /// small one answers by calling `x` again. Observed: a host whose prompt
    /// named a tool the server did not mount spent eight of its fifteen turns
    /// retrying that one name, then ended the run with no answer at all. The
    /// read-only refusal below has always said what to do instead; this now
    /// does too — the offered names, and the closest deferred ones, which is
    /// what makes the next turn a recovery rather than a repeat.
    fn no_such_tool(&self, name: &str) -> String {
        let offered: Vec<&str> = self
            .tools
            .iter()
            .map(|t| t.id())
            .filter(|id| !self.deferred.contains(*id))
            .take(NAMES_IN_UNKNOWN_TOOL_ERROR)
            .collect();
        // A name is its own best query: `plane_create_proposal` finds
        // `plane_propose_fact` on the words it shares.
        let near = self.index.search(&name.replace('_', " "), NEAR_MISSES_SUGGESTED);
        let mut msg = format!("unknown tool `{name}` — nothing by that name is registered.");
        // With nothing to point at, "call one of those instead" names nothing
        // and is the same dead end `unknown tool x` was on its own: a model
        // given no way forward answers by retrying the name. This is the
        // `ToolAccess::None` case, where the honest instruction is to stop
        // reaching and answer.
        if near.is_empty() && offered.is_empty() {
            msg.push_str(" This run has no tools at all. Answer from the prompt; do not retry.");
            return msg;
        }
        if !near.is_empty() {
            msg.push_str(&format!(" Closest available: {}.", near.join(", ")));
        }
        if !offered.is_empty() {
            msg.push_str(&format!(" Offered here: {}.", offered.join(", ")));
        }
        msg.push_str(" Call one of those instead; do not retry this name.");
        msg
    }

    /// Execute one tool call. `ctx.mode` is re-checked so a model that
    /// hallucinates a mutating tool in `Ask` mode is refused even though it
    /// wasn't offered. Output past the caps is truncated (+ spilled) here.
    pub(crate) fn execute(&self, name: &str, args: &Value, ctx: &ToolCtx) -> ToolOutcome {
        if name == TOOL_SEARCH {
            return self.search_tools(args);
        }
        let Some(tool) = self.tools.iter().find(|t| t.id() == name) else {
            return ToolOutcome::err(self.no_such_tool(name));
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

    #[test]
    fn every_tool_is_offered_by_default() {
        // Opt-out, not opt-in: a host that says nothing gets the full set.
        let all = ToolSet::new(vec![], vec![], None, &[]);
        let names = tool_names(&all.defs(RunMode::Edit, "qwen", AgentContext::Main));
        assert!(names.contains(&"bash".to_string()));
        assert!(names.contains(&"read".to_string()));
    }

    #[test]
    fn a_disabled_tool_is_never_offered() {
        // The difference from PermissionRule::deny, which advertises the tool
        // and refuses the call: this one never reaches the model, so it costs
        // no schema and cannot be attempted.
        let set = ToolSet::new(vec![], vec![], None, &["bash".to_owned()]);
        let names = tool_names(&set.defs(RunMode::Edit, "qwen", AgentContext::Main));
        assert!(!names.contains(&"bash".to_string()));
        // Everything else survives — disabling one tool is not a mode switch.
        assert!(names.contains(&"read".to_string()) && names.contains(&"edit".to_string()));
    }

    /// A refusal has to leave the model somewhere to go. Naming alternatives
    /// is what makes the next turn a recovery — but under `ToolAccess::None`
    /// there are none, and telling a model to "call one of those" with nothing
    /// named is the dead end that cost one host eight of fifteen turns on a
    /// single wrong name.
    #[test]
    fn a_run_with_no_tools_at_all_is_told_to_answer_not_to_pick_one() {
        let all_withheld = ToolSet::builtin_tool_names();
        let set = ToolSet::new(vec![], vec![], None, &all_withheld);
        let message = set.no_such_tool("list");

        assert!(
            !message.contains("Call one of those"),
            "nothing was named, so there is no `those`: {message}"
        );
        assert!(message.contains("no tools at all"), "say so plainly: {message}");
        assert!(message.contains("do not retry"), "and still discourage the retry: {message}");
    }

    #[test]
    fn a_disabled_tool_is_also_refused_if_the_model_guesses_it() {
        // Withholding the schema is not the whole defence: a model can still
        // name a tool it was never shown.
        let set = ToolSet::new(vec![], vec![], None, &["bash".to_owned()]);
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
        assert!(!set.execute("bash", &json!({ "command": "echo hi" }), &ctx).ok);
    }

    /// Tool names in the list the model is actually shown.
    fn offered_names(set: &ToolSet) -> Vec<String> {
        set.defs(RunMode::Ask, "test-model", AgentContext::Main)
            .iter()
            .filter_map(|d| d["function"]["name"].as_str().map(str::to_owned))
            .collect()
    }

    /// A ToolCtx for a search or a direct call in these tests.
    fn search_ctx(cancel: &AtomicBool) -> ToolCtx<'_> {
        ToolCtx {
            cwd: any_cwd(),
            mode: RunMode::Ask,
            cancel,
            run_id: "t",
            call_id: "c",
            skills: &[],
            subagent: None,
            model: None,
        }
    }

    /// A stand-in MCP tool with a schema big enough to matter.
    struct FakeMcp {
        id: String,
        description: String,
    }
    impl Tool for FakeMcp {
        fn id(&self) -> &str {
            &self.id
        }
        fn description(&self) -> &str {
            &self.description
        }
        fn parameters(&self) -> Value {
            json!({ "type": "object", "properties": { "padding": { "type": "string" } } })
        }
        fn kind(&self) -> ToolKind {
            ToolKind::Other
        }
        fn mutating(&self) -> bool {
            false
        }
        fn execute(&self, _args: &Value, _ctx: &ToolCtx) -> ToolOutcome {
            ToolOutcome::ok("called")
        }
    }

    fn many_mcp_tools() -> Vec<Box<dyn Tool>> {
        let subjects = ["github issue", "slack message", "database query", "calendar event", "email draft"];
        (0..20)
            .map(|i| {
                let subject = subjects[i % subjects.len()];
                Box::new(FakeMcp {
                    id: format!("mcp_tool_{i:02}"),
                    description: format!(
                        "Work with a {subject}. {}",
                        "This description is long enough to make the surface worth deferring. ".repeat(3)
                    ),
                }) as Box<dyn Tool>
            })
            .collect()
    }

    #[test]
    fn a_small_mcp_surface_stays_in_the_prompt() {
        // Below the threshold, deferral costs more than it saves: the search
        // tool's own schema plus a round trip to find what was already there.
        let one = vec![Box::new(FakeMcp { id: "solo".into(), description: "does one thing".into() })
            as Box<dyn Tool>];
        let set = ToolSet::new(one, Vec::new(), None, &[]);
        let names = offered_names(&set);
        assert!(names.contains(&"solo".to_owned()), "a small surface is listed: {names:?}");
        assert!(!names.contains(&TOOL_SEARCH.to_owned()), "and needs no search tool");
    }

    #[test]
    fn a_large_mcp_surface_is_deferred_behind_one_search_tool() {
        let set = ToolSet::new(many_mcp_tools(), Vec::new(), None, &[]);
        let names = offered_names(&set);

        assert!(names.contains(&TOOL_SEARCH.to_owned()), "the search tool is offered: {names:?}");
        assert!(!names.iter().any(|n| n.starts_with("mcp_tool_")), "no MCP schema rides along: {names:?}");
        // The built-ins are not candidates — a coding task needs them.
        assert!(names.contains(&"read".to_owned()) && names.contains(&"list".to_owned()));
    }

    #[test]
    fn disabled_builtins_do_not_break_mcp_deferral() {
        // mcp_ids was once derived positionally (`skip(builtin_count)`) from a
        // list the disabled-tool retain had already shortened: a host disabling
        // N builtins skipped past N MCP tools, so a large surface silently
        // stopped deferring. MCP tools are identified by id, not position.
        let disabled: Vec<String> = ["bash", "write", "edit", "task", "webfetch"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let set = ToolSet::new(many_mcp_tools(), Vec::new(), None, &disabled);
        let names = offered_names(&set);
        assert!(names.contains(&TOOL_SEARCH.to_owned()), "deferral still triggers: {names:?}");
        assert!(!names.iter().any(|n| n.starts_with("mcp_tool_")), "every MCP schema stays deferred: {names:?}");

        // And the deferred tools are still reachable through the search.
        let found = set.execute(
            TOOL_SEARCH,
            &json!({ "query": "file a github issue" }),
            &search_ctx(&AtomicBool::new(false)),
        );
        assert!(found.ok, "{}", found.output);
        let defs: Vec<Value> = serde_json::from_str(&found.output).expect("a JSON array of tool defs");
        assert!(defs.iter().any(|d| d["function"]["name"].as_str().is_some_and(|n| n.starts_with("mcp_tool_"))));
    }

    #[test]
    fn a_disabled_mcp_tool_is_neither_offered_nor_indexed() {
        // The id filter has to apply to MCP tools too: a disabled MCP tool must
        // not be discoverable through the search index it was removed from.
        let disabled = ["mcp_tool_00".to_owned(), "bash".to_owned()];
        let set = ToolSet::new(many_mcp_tools(), Vec::new(), None, &disabled);
        assert!(!set.deferred.contains("mcp_tool_00"), "a disabled tool is not deferred, it is gone");
        let found = set.execute(
            TOOL_SEARCH,
            &json!({ "query": "github issue" }),
            &search_ctx(&AtomicBool::new(false)),
        );
        assert!(
            !found.output.contains("mcp_tool_00"),
            "a disabled MCP tool must not resurface via search: {}",
            found.output
        );
    }

    #[test]
    fn searching_returns_callable_schemas_for_what_matches() {
        let set = ToolSet::new(many_mcp_tools(), Vec::new(), None, &[]);

        let found = set.execute(TOOL_SEARCH, &json!({ "query": "file a github issue" }), &search_ctx(&AtomicBool::new(false)));
        assert!(found.ok, "search should succeed: {}", found.output);
        let defs: Vec<Value> = serde_json::from_str(&found.output).expect("a JSON array of tool defs");
        assert!(!defs.is_empty(), "a matching query returns tools");
        assert!(
            defs.iter().all(|d| d["function"]["parameters"].is_object()),
            "each result carries the schema needed to call it"
        );

        // And a deferred tool remains callable even though it was never listed.
        let name = defs[0]["function"]["name"].as_str().unwrap().to_owned();
        let called = set.execute(&name, &json!({}), &search_ctx(&AtomicBool::new(false)));
        assert!(called.ok, "deferred does not mean unavailable: {}", called.output);
    }

    #[test]
    fn a_search_matching_nothing_says_so_rather_than_guessing() {
        let set = ToolSet::new(many_mcp_tools(), Vec::new(), None, &[]);
        let out = set.execute(TOOL_SEARCH, &json!({ "query": "photosynthesis" }), &search_ctx(&AtomicBool::new(false)));
        assert!(out.ok);
        assert!(out.output.contains("No available tool matches"), "got {}", out.output);
        assert!(out.output.contains("Do not search again"), "a repeat search would burn turns");
    }

    #[test]
    fn disabling_shrinks_the_prompt() {
        // Tool schemas are sent on every request, so withholding one is also a
        // per-turn token saving — the reason to prefer it over a runtime deny.
        let all = ToolSet::new(vec![], vec![], None, &[]);
        let fewer = ToolSet::new(vec![], vec![], None, &["bash".to_owned()]);
        let full = serde_json::to_string(&all.defs(RunMode::Edit, "qwen", AgentContext::Main)).unwrap();
        let cut = serde_json::to_string(&fewer.defs(RunMode::Edit, "qwen", AgentContext::Main)).unwrap();
        assert!(cut.len() < full.len());
    }

    #[test]
    fn the_host_can_enumerate_what_it_may_disable() {
        // So a host builds its UI from the crate's list rather than hardcoding
        // names that drift as tools are added.
        let names = ToolSet::builtin_tool_names();
        assert!(names.iter().any(|n| n == "bash"));
        assert!(names.iter().any(|n| n == "read"));
    }

    /// Tool ids out of an OpenAI `tools` array.
    fn tool_names(defs: &[Value]) -> Vec<String> {
        defs.iter()
            .filter_map(|d| d["function"]["name"].as_str().map(str::to_owned))
            .collect()
    }


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

    /// The failure this was written for: a host's prompt named
    /// `plane_create_proposal`; the server mounts `plane_propose_fact`. The
    /// model called the name it was given, was told only that it was unknown,
    /// and called it again — eight times across one run.
    #[test]
    fn a_near_miss_on_a_deferred_tool_is_named_in_the_error() {
        let ids = ["plane_propose_fact", "plane_search_saved_items"];
        let entries: Vec<discovery::Entry> = ids
            .iter()
            .map(|id| discovery::Entry {
                id: (*id).to_string(),
                text: format!("{} propose a fact about a saved item", id.replace('_', " ")),
            })
            .collect();
        let set = ToolSet {
            tools: builtins(),
            permissions: Vec::new(),
            permission_prompt: None,
            deferred: ids.iter().map(|id| (*id).to_string()).collect(),
            index: discovery::Index::build(entries),
        };
        let msg = set.no_such_tool("plane_create_proposal");
        assert!(
            msg.contains("plane_propose_fact"),
            "the tool it meant is suggested: {msg}"
        );
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
    fn unknown_tool_is_an_error_that_says_what_to_call_instead() {
        let out = run("nope", json!({}), any_cwd(), RunMode::Edit);
        assert!(!out.ok);
        assert!(out.output.contains("unknown tool"));
        // The bare error made a small model retry the same wrong name. It has
        // to carry the way out: real names, and an instruction not to repeat.
        assert!(out.output.contains("read"), "names what is offered: {}", out.output);
        assert!(
            out.output.contains("do not retry this name"),
            "says not to repeat it: {}",
            out.output
        );
    }

    #[test]
    fn permission_rule_denies_matching_calls() {
        let tools = ToolSet::new(
            vec![],
            vec![crate::openai_compatible::PermissionRule::deny_matching(
                "bash", "rm -rf",
            )],
            None,
            &[],
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
            &[],
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
            &[],
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
    fn a_zero_timeout_falls_back_to_the_default_rather_than_killing_instantly() {
        // `timeout.filter(|&t| t > 0)` treats 0 as "unset". Relaxing that to
        // `>= 0` accepts it, and a zero-millisecond budget kills every command
        // the moment it starts — a model that passes 0 gets a tool that never
        // works, with a timeout message rather than an argument error.
        let dir = scratch("bash-zero-timeout");
        let out = run("bash", json!({ "command": "echo hi", "timeout": 0 }), &dir, RunMode::Edit);
        assert!(out.ok, "a zero timeout must mean the default, got: {}", out.output);
        assert_eq!(out.output.trim(), "hi");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // The joining logic is platform-independent, but producing "stdout with no
    // trailing newline" is not: the tool runs `sh -c` on unix and `cmd /C` on
    // Windows, where `;` is not a separator and there is no `printf`. Asserting
    // the seam needs a shell that can emit exactly these two streams.
    #[cfg(unix)]
    #[test]
    fn stdout_and_stderr_are_joined_without_running_together() {
        // Three mutants lived in this one condition: `&&` to `||`, and either
        // `!` deleted. Each produces output the model reads as one stream —
        // a missing separator glues the last line of stdout to the [stderr]
        // header, and a spurious one puts a blank line before it.
        let dir = scratch("bash-streams");

        // stdout with no trailing newline, then stderr: exactly one newline.
        let both = run(
            "bash",
            json!({ "command": "printf out; printf err >&2" }),
            &dir,
            RunMode::Edit,
        );
        assert!(both.ok, "{}", both.output);
        assert!(both.output.contains("out\n[stderr]"), "one separator: {:?}", both.output);

        // stderr alone: no leading blank line, because there is no stdout to
        // separate it from.
        let only_err = run("bash", json!({ "command": "printf err >&2" }), &dir, RunMode::Edit);
        assert!(only_err.ok, "{}", only_err.output);
        assert!(
            only_err.output.starts_with("[stderr]"),
            "nothing to separate from: {:?}",
            only_err.output
        );

        // stdout alone: the marker never appears.
        let only_out = run("bash", json!({ "command": "printf out" }), &dir, RunMode::Edit);
        assert!(!only_out.output.contains("[stderr]"), "{:?}", only_out.output);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_bash_schema_tells_the_model_how_to_call_it() {
        // An empty schema is worse than a missing tool: the model is invited to
        // call something it cannot form arguments for.
        let set = ToolSet::builtin();
        let defs = set.defs(RunMode::Edit, "test-model", AgentContext::Main);
        let bash = defs
            .iter()
            .find(|d| d["function"]["name"] == "bash")
            .expect("bash is offered in Edit mode");

        assert!(
            bash["function"]["parameters"]["properties"]["command"].is_object(),
            "the schema must declare `command`: {}",
            bash["function"]["parameters"]
        );
        let description = bash["function"]["description"].as_str().unwrap_or_default();
        assert!(
            description.len() > 20 && description.to_lowercase().contains("command"),
            "a tool with no description is one the model cannot choose deliberately: {description:?}"
        );
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
    fn the_output_caps_are_the_sizes_they_claim_to_be() {
        // A cap is only a cap at a specific size. `50 * 1024` becoming
        // `50 + 1024` is a 1 KB ceiling that truncates almost every real
        // command's output, and nothing else would fail.
        assert_eq!(MAX_OUTPUT_BYTES, 50 * 1024);
        assert_eq!(MAX_OUTPUT_LINES, 2000);
    }

    #[test]
    fn a_tool_without_a_subject_offers_none_for_permission_matching() {
        // The trait's default. A rule with a `pattern` matches on the subject,
        // so a default of `Some("")` or any string would make pattern rules
        // start matching tools that have no subject to match against.
        struct Subjectless;
        impl Tool for Subjectless {
            fn id(&self) -> &str {
                "subjectless"
            }
            fn description(&self) -> &str {
                "has no subject"
            }
            fn parameters(&self) -> Value {
                json!({ "type": "object" })
            }
            fn kind(&self) -> ToolKind {
                ToolKind::Other
            }
            fn mutating(&self) -> bool {
                false
            }
            fn execute(&self, _args: &Value, _ctx: &ToolCtx) -> ToolOutcome {
                ToolOutcome::ok("done")
            }
        }
        assert_eq!(Subjectless.permission_subject(&json!({})), None);

        // And the consequence: a pattern rule cannot match it.
        let set = ToolSet::new(
            vec![Box::new(Subjectless)],
            vec![crate::PermissionRule::deny_matching("subjectless", "anything")],
            None,
            &[],
        );
        let out = set.execute("subjectless", &json!({}), &search_ctx(&AtomicBool::new(false)));
        assert!(out.ok, "a pattern rule must not match a tool with no subject: {}", out.output);
    }

    #[test]
    fn both_ends_keeps_a_half_from_each_and_counts_what_it_dropped() {
        // `MAX_OUTPUT_LINES / 2` becoming `* 2` stops the halves being halves,
        // and the omitted count is what tells the model how much it is missing —
        // a wrong number there is a confident lie.
        let lines: Vec<String> = (0..5_000).map(|i| format!("line{i}")).collect();
        let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
        let out = take_both_ends(&refs, refs.len(), &None);

        assert!(out.starts_with("line0\n"), "the head survives: {:?}", &out[..40]);
        assert!(out.trim_end().ends_with("line4999"), "the tail survives");

        let kept = out.lines().filter(|l| l.starts_with("line")).count();
        assert!(kept <= MAX_OUTPUT_LINES, "kept {kept} of a {MAX_OUTPUT_LINES} cap");

        let omitted: usize = out
            .lines()
            .find(|l| l.contains("omitted"))
            .and_then(|l| l.split_whitespace().find_map(|w| w.replace(',', "").parse().ok()))
            .expect("the marker states how many lines were dropped");
        assert_eq!(omitted, refs.len() - kept, "the count must match what was actually dropped");
    }

    #[test]
    fn both_ends_splits_the_line_cap_in_half() {
        // Short lines, so the LINE cap binds rather than the byte cap — which is
        // what makes `MAX_OUTPUT_LINES / 2` observable. Turned into `* 2` the
        // halves stop being halves and four times as much comes back.
        let lines: Vec<&str> = vec!["x"; 5_000];
        let out = take_both_ends(&lines, lines.len(), &None);
        let kept = out.lines().filter(|l| *l == "x").count();
        assert_eq!(
            kept, MAX_OUTPUT_LINES,
            "half from each end is the whole cap, no more: got {kept}"
        );
    }

    #[test]
    fn take_end_stops_at_the_byte_budget_not_one_line_past_it() {
        // The same exact-boundary case `collect_within` has, through the public
        // path: each 3-byte line costs 4 with its newline, so 8 fits two and 7
        // fits one. `>` relaxed to `>=`, or the `+ 1` dropped, moves that line.
        let lines = ["aaa", "bbb", "ccc"];
        for (budget, expected) in [(8usize, 2usize), (7, 1), (4, 1), (3, 0)] {
            let body = take_end(&lines, lines.len(), 100, budget, End::Head, &None);
            let kept = body.lines().filter(|l| lines.contains(l)).count();
            assert_eq!(kept, expected, "budget {budget} should keep {expected}: {body:?}");
        }
    }

    #[test]
    fn both_ends_splits_the_byte_cap_in_half_too() {
        // The companion to the line-cap test, and it needs LONG lines: with
        // short ones the line cap binds first and the byte half is never
        // exercised, so doubling it changes nothing observable.
        let long = "y".repeat(200);
        let lines: Vec<&str> = vec![long.as_str(); 5_000];
        let out = take_both_ends(&lines, lines.len(), &None);

        let kept: usize = out.lines().filter(|l| *l == long).map(|l| l.len() + 1).sum();
        assert!(
            kept <= MAX_OUTPUT_BYTES,
            "half of the byte cap from each end is the whole cap, no more: {kept} > {MAX_OUTPUT_BYTES}"
        );
        assert!(kept > MAX_OUTPUT_BYTES / 2, "and it should actually fill it: {kept}");
    }

    #[test]
    fn spilled_output_is_written_somewhere_the_model_can_read_it() {
        // The spill file is the whole point of truncating: the model is told
        // where the rest went. Returning None, or a path to nothing, silently
        // turns a truncation into a loss.
        let body = "the full output\n".repeat(100);
        let path = spill_output(&body).expect("a spill path");

        let written = std::fs::read_to_string(&path).expect("the file the path names");
        assert_eq!(written, body, "the spill holds the untruncated output");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn collect_within_fills_the_budget_and_stops_exactly_there() {
        // The existing truncation tests assert what the output *says*. Nothing
        // asserted the bound itself, which is the only reason the cap exists:
        // an oversized tool result is what blows the model's context.
        //
        // Each line here costs its length plus one for the newline, so a budget
        // of 8 fits exactly two 3-byte lines and cannot fit a third.
        let lines = ["aaa", "bbb", "ccc"];
        assert_eq!(collect_within(lines.iter(), 100, 8), vec!["aaa", "bbb"]);

        // One byte less and only one line fits — the boundary where `>` and
        // `>=` diverge, and where a mis-signed `+` shows up.
        assert_eq!(collect_within(lines.iter(), 100, 7), vec!["aaa"]);

        // A budget below even one line yields nothing rather than one line over.
        assert!(collect_within(lines.iter(), 100, 2).is_empty());

        // The line cap binds independently of the byte cap.
        assert_eq!(collect_within(lines.iter(), 2, 10_000), vec!["aaa", "bbb"]);
        assert!(collect_within(lines.iter(), 0, 10_000).is_empty());
    }

    #[test]
    fn a_truncated_body_never_exceeds_the_byte_budget() {
        // The property, over a range of budgets rather than one example: the
        // lines kept must fit, whichever end they are taken from.
        let lines: Vec<&str> = ["alpha", "beta", "gamma", "delta", "epsilon"].to_vec();
        for max_bytes in [1usize, 5, 6, 11, 17, 40, 1_000] {
            for end in [End::Head, End::Tail] {
                let body = take_end(&lines, lines.len(), 100, max_bytes, end, &None);
                // Isolate the kept lines from the truncation note, which is
                // deliberately outside the budget — it explains the cut.
                let kept: usize = body
                    .lines()
                    .filter(|l| lines.contains(l))
                    .map(|l| l.len() + 1)
                    .sum();
                assert!(
                    kept <= max_bytes,
                    "{end:?} kept {kept} bytes against a {max_bytes} budget: {body:?}"
                );
            }
        }
    }

    #[test]
    fn find_line_trimmed_unique_locates_the_line_and_refuses_an_ambiguous_one() {
        // This is how `edit` finds what to replace. An off-by-one here edits the
        // wrong line of a user's file, which no other test would notice.
        let content = "first\n  target  \nthird\n";
        let (start, end) = find_line_trimmed_unique(content, "target").expect("a unique match");
        assert_eq!(
            &content[start..end], "  target  \n",
            "the span covers the line's own spacing AND its newline, so a replacement \
             substitutes the whole line rather than splicing into it"
        );

        // Two candidates must be refused rather than guessed between.
        let ambiguous = "dup\nother\ndup\n";
        assert!(find_line_trimmed_unique(ambiguous, "dup").is_none());

        // No candidate at all.
        assert!(find_line_trimmed_unique(content, "absent").is_none());

        // A multi-line match, which is where the window arithmetic lives: the
        // span must cover exactly the matched lines and no neighbour.
        let multi = "keep\nfirst\nsecond\ntrailing\n";
        let (start, end) = find_line_trimmed_unique(multi, "first\nsecond").expect("a two-line match");
        assert_eq!(&multi[start..end], "first\nsecond\n", "exactly the window, not one line either side");

        // A pattern longer than the file cannot match — the guard that stops
        // the window arithmetic running off the end.
        assert!(find_line_trimmed_unique("only\n", "only\nmore\n").is_none());

        // A pattern spanning the WHOLE file must still match. This is the exact
        // boundary of that guard: `window > len` allows it, `window >= len`
        // rejects it, and rejecting it means `edit` cannot replace the entire
        // contents of a short file.
        let whole = "alpha\nbeta\n";
        let (start, end) = find_line_trimmed_unique(whole, "alpha\nbeta").expect("a whole-file match");
        assert_eq!(&whole[start..end], whole);
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
        // Spelled per shell: `$(seq)` is sh-only, `for /L` is cmd-only.
        let cmd = if cfg!(windows) {
            // Parenthesised: in cmd an unbracketed `&` binds inside the `do`
            // clause, so the error and exit would run on the first iteration.
            "(for /L %i in (1,1,5000) do @echo info line %i) & echo BOOM the real error & exit 7"
        } else {
            "for i in $(seq 1 5000); do echo info line $i; done; echo 'BOOM the real error'; exit 7"
        };
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
