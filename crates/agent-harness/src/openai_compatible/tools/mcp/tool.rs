//! Adapts one MCP server tool into a openai-compatible [`Tool`], so MCP tools are
//! offered to the model and dispatched through the very same set as the
//! built-ins — no separate code path in the loop.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::ToolKind;

use super::client::{McpClient, McpResource, McpToolDef};
use crate::openai_compatible::tools::{Tool, ToolCtx, ToolOutcome};

/// A single tool exposed by an MCP server.
pub(crate) struct McpTool {
    /// Shared connection to the owning server (kept alive while any of its tools
    /// live; the last drop shuts the server down).
    client: Arc<McpClient>,
    /// Id offered to the model — `server_tool`, namespaced so two servers'
    /// identically-named tools don't collide.
    id: String,
    /// The server's own (un-namespaced) tool name, sent in `tools/call`.
    remote_name: String,
    description: String,
    schema: Value,
}

impl McpTool {
    pub(crate) fn new(client: Arc<McpClient>, server: &str, def: McpToolDef) -> Self {
        Self {
            client,
            id: format!("{server}_{}", def.name),
            remote_name: def.name,
            description: def.description,
            schema: def.input_schema,
        }
    }
}

impl Tool for McpTool {
    fn id(&self) -> &str {
        &self.id
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters(&self) -> Value {
        self.schema.clone()
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Other
    }
    fn mutating(&self) -> bool {
        // An MCP tool's effect is opaque, so treat it as mutating: offered only
        // in `RunMode::Edit`, never in read-only `Ask`. Conservative by design —
        // we won't expose an arbitrary external side effect in a read-only run.
        true
    }
    fn execute(&self, args: &Value, _ctx: &ToolCtx) -> ToolOutcome {
        // MCP arguments are an object; tolerate a missing/!object value as `{}`.
        let arguments = if args.is_object() { args.clone() } else { Value::Object(Default::default()) };
        match self.client.call(&self.remote_name, &arguments) {
            Ok(text) => ToolOutcome::ok(if text.trim().is_empty() { "(no content)".to_owned() } else { text }),
            Err(e) => ToolOutcome::err(e),
        }
    }
}

/// One per-server tool for reading that server's MCP **resources** by URI. Added
/// only when the server exposes resources; its description enumerates them (so
/// the model knows what URIs are available, like the skill catalog). Reading is
/// read-only, so it's offered in both modes.
pub(crate) struct McpResourceTool {
    client: Arc<McpClient>,
    id: String,
    description: String,
}

impl McpResourceTool {
    pub(crate) fn new(client: Arc<McpClient>, server: &str, resources: &[McpResource]) -> Self {
        let mut description =
            format!("Read a resource exposed by the `{server}` MCP server, by URI. Available resources:\n");
        for r in resources {
            let name = if r.name.is_empty() { String::new() } else { format!(" ({})", r.name) };
            let desc = if r.description.is_empty() { String::new() } else { format!(" — {}", r.description) };
            description.push_str(&format!("- `{}`{name}{desc}\n", r.uri));
        }
        Self { client, id: format!("{server}_read_resource"), description }
    }
}

impl Tool for McpResourceTool {
    fn id(&self) -> &str {
        &self.id
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "uri": { "type": "string", "description": "The resource URI to read (one listed in this tool's description)." } },
            "required": ["uri"]
        })
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }
    fn mutating(&self) -> bool {
        false // reading a resource is read-only
    }
    fn execute(&self, args: &Value, _ctx: &ToolCtx) -> ToolOutcome {
        let Some(uri) = args.get("uri").and_then(Value::as_str) else {
            return ToolOutcome::err("read_resource: a `uri` string is required");
        };
        match self.client.read_resource(uri) {
            Ok(text) => ToolOutcome::ok(if text.trim().is_empty() { "(empty resource)".to_owned() } else { text }),
            Err(e) => ToolOutcome::err(e),
        }
    }
}
