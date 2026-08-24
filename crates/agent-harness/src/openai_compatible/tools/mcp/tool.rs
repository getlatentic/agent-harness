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

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::client::ScriptedConnection;

    fn def(name: &str) -> McpToolDef {
        McpToolDef {
            name: name.to_owned(),
            description: format!("does {name}"),
            input_schema: json!({ "type": "object", "properties": { "q": { "type": "string" } } }),
        }
    }

    fn client(conn: ScriptedConnection) -> Arc<McpClient> {
        Arc::new(McpClient::over(Box::new(conn)))
    }

    /// Nothing here reaches the filesystem or a subagent — an MCP call goes out
    /// over the connection — so the context only has to exist.
    fn ctx(cancel: &std::sync::atomic::AtomicBool) -> ToolCtx<'_> {
        ToolCtx {
            cwd: std::path::Path::new("."),
            mode: crate::RunMode::Edit,
            cancel,
            run_id: "t",
            call_id: "c",
            skills: &[],
            subagent: None,
            model: None,
        }
    }

    #[test]
    fn a_tool_is_namespaced_by_its_server_but_called_by_its_own_name() {
        // Two servers may both offer `search`. The model needs distinct ids, and
        // the server needs the name it actually registered — sending the
        // namespaced one back would be an unknown tool.
        let tool = McpTool::new(client(ScriptedConnection::new()), "github", def("search"));
        assert_eq!(tool.id(), "github_search");
        assert_eq!(tool.remote_name, "search");
        assert_eq!(tool.description(), "does search");
        assert_eq!(tool.parameters()["properties"]["q"]["type"], "string", "the server's schema, verbatim");
    }

    #[test]
    fn an_mcp_tool_is_treated_as_mutating_because_its_effect_is_unknown() {
        // Withheld from read-only runs. An MCP server can do anything, and
        // guessing "probably safe" is how a read-only request posts a message.
        let tool = McpTool::new(client(ScriptedConnection::new()), "s", def("anything"));
        assert!(tool.mutating());
        assert_eq!(tool.kind(), ToolKind::Other);
    }

    #[test]
    fn a_call_sends_the_arguments_through_and_returns_the_flattened_text() {
        let conn = ScriptedConnection::new().on("tools/call", json!({ "content": [{ "type": "text", "text": "found it" }] }));
        let recorder = conn.clone();
        let tool = McpTool::new(client(conn), "github", def("search"));

        let outcome = tool.execute(&json!({ "q": "rust" }), &ctx(&Default::default()));
        assert_eq!(outcome.output, "found it");

        let (method, params) = recorder.asked().into_iter().next().expect("one call");
        assert_eq!(method, "tools/call");
        assert_eq!(params["name"], "search", "the un-namespaced name");
        assert_eq!(params["arguments"], json!({ "q": "rust" }));
    }

    #[test]
    fn a_non_object_argument_becomes_an_empty_object_rather_than_a_bad_request() {
        // Models send `null` or a bare string when a tool takes no arguments.
        // MCP requires an object, so this would be rejected by the server.
        let conn = ScriptedConnection::new().on("tools/call", json!({ "content": [] }));
        let recorder = conn.clone();
        let tool = McpTool::new(client(conn), "s", def("ping"));

        let outcome = tool.execute(&json!("not an object"), &ctx(&Default::default()));
        assert_eq!(recorder.asked()[0].1["arguments"], json!({}));
        assert_eq!(outcome.output, "(no content)", "an empty result is said out loud, not returned blank");
    }

    #[test]
    fn a_failing_call_surfaces_the_servers_reason() {
        let tool = McpTool::new(
            client(ScriptedConnection::new().failing("tools/call", "the server went away")),
            "s",
            def("search"),
        );
        let outcome = tool.execute(&json!({}), &ctx(&Default::default()));
        assert!(!outcome.ok, "a failure is not an answer");
        assert!(outcome.output.contains("went away"), "got {}", outcome.output);
    }

    fn resources() -> Vec<McpResource> {
        vec![
            McpResource { uri: "file:///a.txt".into(), name: "A".into(), description: "the first".into() },
            McpResource { uri: "file:///b.txt".into(), name: String::new(), description: String::new() },
        ]
    }

    #[test]
    fn the_resource_tool_lists_what_can_be_read_in_its_description() {
        // The model only learns which URIs exist from this text — the same
        // progressive disclosure as the skills catalog.
        let tool = McpResourceTool::new(client(ScriptedConnection::new()), "docs", &resources());
        assert_eq!(tool.id(), "docs_read_resource");
        assert!(tool.description().contains("`file:///a.txt` (A) — the first"));
        assert!(tool.description().contains("`file:///b.txt`"), "a nameless resource is still listed");
        assert!(!tool.mutating(), "reading is offered in read-only runs too");
        assert_eq!(tool.kind(), ToolKind::Read);
        assert_eq!(tool.parameters()["required"], json!(["uri"]));
    }

    #[test]
    fn reading_a_resource_requires_a_uri_and_reports_an_empty_one() {
        let conn = ScriptedConnection::new().on("resources/read", json!({ "contents": [{ "text": "  " }] }));
        let tool = McpResourceTool::new(client(conn), "docs", &resources());

        let missing = tool.execute(&json!({}), &ctx(&Default::default()));
        assert!(!missing.ok, "without a uri there is nothing to read");

        let empty = tool.execute(&json!({ "uri": "file:///a.txt" }), &ctx(&Default::default()));
        assert_eq!(empty.output, "(empty resource)", "distinguishable from a failure");
    }
}
