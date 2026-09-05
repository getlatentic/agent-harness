//! Tools the host program implements, offered to the agent from inside this
//! process.
//!
//! An agent's own tools reach the filesystem and the shell. What they cannot
//! reach is the program that started the run — its database, its typed model
//! of the problem, a function it already has. A [`ToolServer`] is how that
//! program hands the agent a function to call: a named group of [`HostTool`]s
//! presented to the agent as one MCP server, with no server process, no port,
//! and no second binary to ship.
//!
//! How the agent reaches it differs by adapter and is the adapter's business.
//! Claude Code is told about the server over its control protocol and sends
//! each MCP message back up the same pipe, where [`jsonrpc`] answers it. The
//! `openai-compatible` runtime owns its loop, so it dispatches to the tool
//! directly. Both offer the tool under the MCP naming the agent expects
//! (`mcp__<server>__<tool>` in Claude Code, `<server>_<tool>` here). An adapter
//! that cannot serve one exposes no way to attach one — see
//! [`Features::host_tools`](crate::Features::host_tools).
//!
//! ```no_run
//! use harness::{Claude, FnTool, Harness, RunRequest, ToolServer};
//! use serde_json::json;
//!
//! let inventory = ToolServer::new("shop").with_tool(FnTool::new(
//!     "stock_level",
//!     "How many units of a SKU are on hand.",
//!     json!({ "type": "object", "properties": { "sku": { "type": "string" } }, "required": ["sku"] }),
//!     |args| Ok(format!("{} units", 42 + args["sku"].as_str().unwrap_or("").len())),
//! ));
//! let claude = Claude::new().with_tool_server(inventory);
//! let (_handle, _events) = claude.run(RunRequest {
//!     prompt: "How many units of SKU A-7 do we have?".into(),
//!     ..Default::default()
//! })?;
//! # Ok::<(), harness::Error>(())
//! ```

use std::fmt;
use std::sync::Arc;

use serde_json::Value;

// The MCP-server half is spoken only to Claude Code; the public types above are
// in every build so a host can construct servers without knowing which adapter
// will serve them.
#[cfg(feature = "claude")]
pub(crate) mod jsonrpc;

/// One function the host program offers the agent.
///
/// Implement it on your own type, or build one from a closure with [`FnTool`].
/// `call` runs on a thread the adapter owns, so it may block; a panic inside it
/// is caught and reported to the agent as a failed call rather than ending the
/// run.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a tool the agent can be offered",
    label = "implement `HostTool` for it, or wrap a closure in `FnTool::new`",
    note = "a `HostTool` is a name, a description, a JSON Schema for its arguments, and a `call`"
)]
pub trait HostTool: Send + Sync {
    /// The name the agent calls it by. Unique within its [`ToolServer`].
    fn name(&self) -> &str;
    /// What it does and when to use it — this is what the model reads to decide.
    fn description(&self) -> &str;
    /// A JSON Schema object describing `arguments`. `{"type": "object"}` with
    /// no properties is a tool that takes nothing.
    fn input_schema(&self) -> Value;
    /// Whether a call changes nothing the agent could observe. Declared, not
    /// checked: a read-only tool is offered in [`RunMode::Ask`](crate::RunMode::Ask)
    /// runs by adapters that withhold mutating tools there, and carries the MCP
    /// `readOnlyHint` annotation.
    fn read_only(&self) -> bool {
        false
    }
    /// Run it. `Ok` is the text the agent sees as the result; `Err` is the text
    /// it sees as a failed call — still delivered, so the agent can recover.
    fn call(&self, arguments: Value) -> Result<String, String>;
}

/// A [`HostTool`] built from a closure.
pub struct FnTool {
    name: String,
    description: String,
    input_schema: Value,
    read_only: bool,
    call: Box<dyn Fn(Value) -> Result<String, String> + Send + Sync>,
}

impl FnTool {
    /// A tool named `name`, described to the model by `description`, taking
    /// arguments shaped by `input_schema`, implemented by `call`.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
        call: impl Fn(Value) -> Result<String, String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            read_only: false,
            call: Box::new(call),
        }
    }

    /// Declare the tool read-only — see [`HostTool::read_only`].
    pub fn as_read_only(mut self) -> Self {
        self.read_only = true;
        self
    }
}

impl HostTool for FnTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn input_schema(&self) -> Value {
        self.input_schema.clone()
    }
    fn read_only(&self) -> bool {
        self.read_only
    }
    fn call(&self, arguments: Value) -> Result<String, String> {
        (self.call)(arguments)
    }
}

impl fmt::Debug for FnTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FnTool")
            .field("name", &self.name)
            .field("read_only", &self.read_only)
            .finish_non_exhaustive()
    }
}

/// A named group of [`HostTool`]s, presented to the agent as one MCP server.
///
/// The name is the namespace the agent sees the tools under, so pick one that
/// reads well in a tool call: `shop`, not `my-app-tools-v2`.
#[derive(Clone)]
pub struct ToolServer {
    name: String,
    tools: Vec<Arc<dyn HostTool>>,
}

impl ToolServer {
    /// An empty server called `name`.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), tools: Vec::new() }
    }

    /// Add a tool. A later tool with the same name replaces the earlier one,
    /// so registration order is a precedence order rather than a source of
    /// duplicates the agent would have to pick between.
    pub fn with_tool(mut self, tool: impl HostTool + 'static) -> Self {
        let tool: Arc<dyn HostTool> = Arc::new(tool);
        self.tools.retain(|existing| existing.name() != tool.name());
        self.tools.push(tool);
        self
    }

    /// The server's name — the agent's namespace for its tools.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The tools, in registration order.
    pub fn tools(&self) -> &[Arc<dyn HostTool>] {
        &self.tools
    }

    /// The tool called `name`, if the server has one.
    #[cfg(any(feature = "claude", feature = "openai-compatible"))]
    pub(crate) fn tool(&self, name: &str) -> Option<&Arc<dyn HostTool>> {
        self.tools.iter().find(|tool| tool.name() == name)
    }
}

/// Call a host tool so that a panic in it is a failed call, not a dead thread.
///
/// The adapters run tools on threads whose only job is to write the result
/// back; a panic there would leave the agent waiting for a reply that never
/// comes. `catch_unwind` turns it into an error the agent can read instead.
#[cfg(any(feature = "claude", feature = "openai-compatible"))]
pub(crate) fn call_guarded(tool: &dyn HostTool, arguments: Value) -> Result<String, String> {
    let name = tool.name().to_owned();
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tool.call(arguments)))
        .unwrap_or_else(|payload| Err(format!("tool `{name}` panicked: {}", panic_text(payload.as_ref()))))
}

impl fmt::Debug for ToolServer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolServer")
            .field("name", &self.name)
            .field("tools", &self.tools.iter().map(|t| t.name()).collect::<Vec<_>>())
            .finish()
    }
}

/// The message a panic carried, when it was a string.
#[cfg(any(feature = "claude", feature = "openai-compatible"))]
fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .unwrap_or_else(|| "no message".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn echo(name: &str) -> FnTool {
        FnTool::new(name, "echoes", json!({"type": "object"}), |args| Ok(args.to_string()))
    }

    #[test]
    fn a_later_tool_with_the_same_name_replaces_the_earlier_one() {
        let server = ToolServer::new("s")
            .with_tool(FnTool::new("dup", "first", json!({}), |_| Ok("first".into())))
            .with_tool(FnTool::new("dup", "second", json!({}), |_| Ok("second".into())));
        assert_eq!(server.tools().len(), 1, "one tool, not two: {server:?}");
        assert_eq!(server.tools()[0].description(), "second");
    }

    #[cfg(any(feature = "claude", feature = "openai-compatible"))]
    #[test]
    fn a_panicking_tool_is_a_failed_call_not_a_crash() {
        let boom = FnTool::new("boom", "panics", json!({}), |_| -> Result<String, String> { panic!("host bug") });
        let err = call_guarded(&boom, json!({})).unwrap_err();
        assert!(err.contains("boom") && err.contains("host bug"), "{err}");
    }

    #[test]
    fn read_only_is_off_unless_declared() {
        assert!(!HostTool::read_only(&echo("t")));
        assert!(HostTool::read_only(&echo("t").as_read_only()));
    }
}
