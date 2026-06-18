//! `task` — delegate a self-contained subtask to a fresh subagent. The subagent
//! runs the same agent loop autonomously (with the file/shell/search tools, but
//! no `task`/`question`) and its final text is returned as this tool's result.
//! Ported from OpenCode's task tool (MIT). Mutating (a subagent can edit files)
//! → Edit mode; not offered to subagents (no nesting).

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::ToolKind;

use super::{parse_args, schema_for, Tool, ToolCtx, ToolOutcome};

#[derive(Deserialize, JsonSchema)]
struct TaskArgs {
    /// A short (3-5 word) description of the subtask, for display.
    description: String,
    /// The full, self-contained instructions for the subagent to carry out.
    prompt: String,
    /// Which registered subagent to use — see the "Subagent types" list in your
    /// system prompt. Omit for the default coding agent.
    #[serde(default)]
    subagent_type: Option<String>,
}

pub(super) struct Task;
impl Tool for Task {
    fn id(&self) -> &str {
        "task"
    }
    fn description(&self) -> &str {
        "Delegate a self-contained subtask to a fresh subagent that runs \
         autonomously with the file/shell/search tools and returns its final \
         result. Use it to keep a focused multi-step subtask off the main thread."
    }
    fn parameters(&self) -> Value {
        schema_for::<TaskArgs>()
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Other
    }
    fn mutating(&self) -> bool {
        true
    }
    fn in_subagent(&self) -> bool {
        false
    }
    fn execute(&self, args: &Value, ctx: &ToolCtx) -> ToolOutcome {
        let a: TaskArgs = match parse_args(args) {
            Ok(a) => a,
            Err(o) => return o,
        };
        let Some(runner) = ctx.subagent else {
            return ToolOutcome::err("task: subagents are not available in this context");
        };
        match runner.run(a.subagent_type.as_deref(), &a.prompt, ctx.cancel) {
            Ok(text) => ToolOutcome::ok(text),
            Err(e) => ToolOutcome::err(format!("task `{}` failed: {e}", a.description)),
        }
    }
}
