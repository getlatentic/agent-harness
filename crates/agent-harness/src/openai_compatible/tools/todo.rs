//! `todowrite` — the agent's task list. Each call replaces the list and emits a
//! [`RunEvent::Plan`] (the host renders a checklist). Ported from OpenCode's
//! todowrite (MIT). Read-only w.r.t. the filesystem → offered in both modes.

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::{PlanEntry, PlanEntryPriority, PlanEntryStatus, RunEvent, ToolKind};

use super::{parse_args, schema_for, Tool, ToolCtx, ToolOutcome};

#[derive(Deserialize, JsonSchema)]
struct TodoArgs {
    /// The full todo list (replaces any previous list for this session).
    todos: Vec<TodoItem>,
}

#[derive(Deserialize, JsonSchema)]
struct TodoItem {
    /// Brief description of the task.
    content: String,
    /// Task state: one of "pending", "in_progress", "completed", "cancelled"
    /// (keep exactly one "in_progress" at a time).
    status: String,
    /// Relative importance: one of "high", "medium", "low".
    priority: Option<String>,
}

pub(super) struct TodoWrite;
impl Tool for TodoWrite {
    fn id(&self) -> &str {
        "todowrite"
    }
    fn description(&self) -> &str {
        "Update the task list for the current session. Use it to plan \
         multi-step work and track progress; keep exactly one item in_progress \
         at a time. Replaces the previous list."
    }
    fn parameters(&self) -> Value {
        schema_for::<TodoArgs>()
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Other
    }
    fn mutating(&self) -> bool {
        false
    }
    fn execute(&self, args: &Value, ctx: &ToolCtx) -> ToolOutcome {
        let a: TodoArgs = match parse_args(args) {
            Ok(a) => a,
            Err(o) => return o,
        };
        let entries: Vec<PlanEntry> = a
            .todos
            .iter()
            .map(|t| PlanEntry {
                content: t.content.clone(),
                status: parse_status(&t.status),
                priority: t.priority.as_deref().map(parse_priority),
            })
            .collect();
        let n = entries.len();
        let plan = RunEvent::Plan { run_id: ctx.run_id.to_owned(), entries };
        ToolOutcome::ok(format!("Updated the task list ({n} item{}).", if n == 1 { "" } else { "s" }))
            .with_events(vec![plan])
    }
}

fn parse_status(s: &str) -> PlanEntryStatus {
    match s {
        "in_progress" => PlanEntryStatus::InProgress,
        "completed" => PlanEntryStatus::Completed,
        "cancelled" => PlanEntryStatus::Cancelled,
        _ => PlanEntryStatus::Pending,
    }
}

fn parse_priority(s: &str) -> PlanEntryPriority {
    match s {
        "high" => PlanEntryPriority::High,
        "low" => PlanEntryPriority::Low,
        _ => PlanEntryPriority::Medium,
    }
}
