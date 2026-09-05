//! Adapts the host's [`ToolServer`]s into this runtime's [`Tool`]s, so a
//! function the host program implements is offered and dispatched through the
//! same set as the built-ins and MCP tools — one loop, no special case.

use std::sync::Arc;

use serde_json::Value;

use crate::{HostTool, ToolKind, ToolServer};

use super::{Tool, ToolCtx, ToolOutcome};

/// One host tool, named the way an MCP tool from the same server would be
/// (`<server>_<tool>`), so a host can move a tool between an in-process server
/// and an external one without the model seeing a different name.
pub(crate) struct HostToolAdapter {
    server: ToolServer,
    tool: Arc<dyn HostTool>,
    id: String,
}

impl Tool for HostToolAdapter {
    fn id(&self) -> &str {
        &self.id
    }
    fn description(&self) -> &str {
        self.tool.description()
    }
    fn parameters(&self) -> Value {
        self.tool.input_schema()
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Other
    }
    fn mutating(&self) -> bool {
        // The host's own declaration, trusted the way an MCP server's
        // `readOnlyHint` is: a host that says read-only gets the tool in `Ask`.
        !self.tool.read_only()
    }
    fn execute(&self, args: &Value, _ctx: &ToolCtx) -> ToolOutcome {
        let arguments = if args.is_object() { args.clone() } else { Value::Object(Default::default()) };
        match self.server.call(self.tool.name(), arguments) {
            Ok(text) => ToolOutcome::ok(if text.trim().is_empty() { "(no content)".to_owned() } else { text }),
            Err(text) => ToolOutcome::err(text),
        }
    }
}

/// Every tool on every server, as [`Tool`]s for the run's set.
pub(crate) fn tools(servers: &[ToolServer]) -> Vec<Box<dyn Tool>> {
    servers
        .iter()
        .flat_map(|server| {
            server.tools().iter().map(move |tool| {
                Box::new(HostToolAdapter {
                    server: server.clone(),
                    tool: Arc::clone(tool),
                    id: format!("{}_{}", server.name(), tool.name()),
                }) as Box<dyn Tool>
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai_compatible::tools::ToolSet;
    use crate::{FnTool, RunMode};
    use serde_json::json;
    use std::sync::atomic::AtomicBool;

    fn shop() -> ToolServer {
        ToolServer::new("shop")
            .with_tool(
                FnTool::new("stock", "units on hand", json!({"type":"object"}), |_| Ok("3 units".into())).as_read_only(),
            )
            .with_tool(FnTool::new("restock", "orders more", json!({"type":"object"}), |_| Err("closed".into())))
    }

    fn offered(set: &ToolSet, mode: RunMode) -> Vec<String> {
        set.defs(mode, "any-model", crate::openai_compatible::tools::AgentContext::Main)
            .iter()
            .map(|d| d["function"]["name"].as_str().unwrap().to_owned())
            .collect()
    }

    #[test]
    fn host_tools_join_the_set_under_the_mcp_style_name_and_respect_mode() {
        let set = ToolSet::new(tools(&[shop()]), Vec::new(), None, &[]);
        let ask = offered(&set, RunMode::Ask);
        assert!(ask.contains(&"shop_stock".to_owned()), "read-only offered in Ask: {ask:?}");
        assert!(!ask.contains(&"shop_restock".to_owned()), "mutating withheld in Ask: {ask:?}");
        let edit = offered(&set, RunMode::Edit);
        assert!(edit.contains(&"shop_restock".to_owned()), "mutating offered in Edit: {edit:?}");
    }

    #[test]
    fn a_host_tool_is_dispatched_like_any_other_and_its_error_is_a_failed_call() {
        let set = ToolSet::new(tools(&[shop()]), Vec::new(), None, &[]);
        let cancel = AtomicBool::new(false);
        let ctx = ToolCtx {
            cwd: std::path::Path::new("."),
            mode: RunMode::Edit,
            cancel: &cancel,
            run_id: "r",
            call_id: "c",
            skills: &[],
            subagent: None,
            model: None,
        };
        let ok = set.execute("shop_stock", &json!({}), &ctx);
        assert!(ok.ok && ok.output == "3 units", "{}", ok.output);
        let failed = set.execute("shop_restock", &json!({}), &ctx);
        assert!(!failed.ok && failed.output == "closed", "{}", failed.output);
    }
}
