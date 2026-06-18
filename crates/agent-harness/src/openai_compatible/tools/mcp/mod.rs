//! MCP (Model Context Protocol) tool source. Each configured server is launched
//! over stdio, handshaken, and its advertised tools are surfaced as
//! [`crate::openai_compatible::tools::Tool`]s — so external MCP tools sit beside the built-ins in
//! the same [`ToolSet`](crate::openai_compatible::tools::ToolSet), offered and dispatched
//! identically. Connection is best-effort: a server that fails to start or
//! handshake is skipped with a status line, never aborting the run.

use std::path::Path;
use std::sync::Arc;

use self::client::McpClient;
use self::tool::{McpResourceTool, McpTool};
use crate::openai_compatible::tools::Tool;

mod client;
mod http;
mod stdio;
mod tool;

/// An MCP server to expose to the model, over stdio (a launched process) or HTTP
/// (a remote endpoint). Registered on the harness via
/// [`crate::openai_compatible::OpenHarness::with_mcp_server`].
#[derive(Debug, Clone)]
pub struct McpServer {
    /// Short name used to namespace this server's tools (offered as `name_tool`).
    pub name: String,
    /// How to reach the server.
    pub transport: McpTransport,
}

/// How an [`McpServer`] is reached.
#[derive(Debug, Clone)]
pub enum McpTransport {
    /// Launch a local server process and speak over its stdin/stdout.
    Stdio {
        /// Executable to run (e.g. `npx`, `uvx`, or an absolute path).
        command: String,
        /// Arguments passed to the command.
        args: Vec<String>,
        /// Extra environment variables for the server process.
        env: Vec<(String, String)>,
    },
    /// Connect to a remote server over HTTP (the Streamable-HTTP JSON-RPC
    /// transport): each request is a POST whose reply is JSON or an SSE stream.
    Http {
        /// The server endpoint URL.
        url: String,
        /// Extra request headers (e.g. `Authorization: Bearer …`).
        headers: Vec<(String, String)>,
    },
}

impl McpServer {
    /// A local stdio server — `command` plus `args`, no extra environment.
    pub fn stdio(name: impl Into<String>, command: impl Into<String>, args: Vec<String>) -> Self {
        Self { name: name.into(), transport: McpTransport::Stdio { command: command.into(), args, env: Vec::new() } }
    }

    /// A remote HTTP server at `url`.
    pub fn http(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self { name: name.into(), transport: McpTransport::Http { url: url.into(), headers: Vec::new() } }
    }

    /// Add an environment variable for the server process (stdio only; a no-op
    /// for an HTTP server).
    #[must_use]
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        if let McpTransport::Stdio { env, .. } = &mut self.transport {
            env.push((key.into(), value.into()));
        }
        self
    }

    /// Add a request header (HTTP only; a no-op for a stdio server).
    #[must_use]
    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        if let McpTransport::Http { headers, .. } = &mut self.transport {
            headers.push((key.into(), value.into()));
        }
        self
    }
}

/// Connect to every configured server and collect their tools, along with a
/// human-readable status line per server (connected + tool count, or skipped +
/// reason) for the caller to surface. Servers that fail are simply omitted.
pub(crate) fn connect_all(servers: &[McpServer], cwd: &Path) -> (Vec<Box<dyn Tool>>, Vec<String>) {
    let mut tools: Vec<Box<dyn Tool>> = Vec::new();
    let mut status = Vec::new();
    for server in servers {
        match McpClient::connect(server, cwd) {
            Ok((client, defs)) => {
                let n = defs.len();
                let client = Arc::new(client);
                for def in defs {
                    tools.push(Box::new(McpTool::new(client.clone(), &server.name, def)));
                }
                // Resources are a bonus surface: a read-only `{server}_read_resource`
                // tool is added only when the server exposes any.
                let resources = client.list_resources();
                let r = resources.len();
                if !resources.is_empty() {
                    tools.push(Box::new(McpResourceTool::new(client.clone(), &server.name, &resources)));
                }
                let res_note = if r > 0 { format!(", {r} resource{}", plural(r)) } else { String::new() };
                status.push(format!("mcp: connected `{}` ({n} tool{}{res_note})", server.name, plural(n)));
            }
            Err(e) => status.push(format!("mcp: `{}` unavailable — {e}", server.name)),
        }
    }
    (tools, status)
}

fn plural(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

/// A prompt template advertised by an MCP server (`prompts/list`). A host
/// surfaces these (e.g. as slash-commands) and resolves one to messages with
/// [`crate::openai_compatible::OpenHarness::get_mcp_prompt`] to seed a run.
#[derive(Debug, Clone)]
pub struct McpPrompt {
    /// The registered server name this prompt came from.
    pub server: String,
    /// The prompt's name (pass to `get_mcp_prompt`).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Declared arguments the prompt accepts.
    pub arguments: Vec<McpPromptArg>,
}

/// One argument a [`McpPrompt`] accepts.
#[derive(Debug, Clone)]
pub struct McpPromptArg {
    /// Argument name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Whether the argument is required.
    pub required: bool,
}

/// One message of a resolved prompt (`prompts/get`), with its text flattened.
#[derive(Debug, Clone)]
pub struct PromptMessage {
    /// The message role (`user` / `assistant`).
    pub role: String,
    /// The message text.
    pub content: String,
}

/// Connect each server, list its prompt templates, and tag each with the server
/// name. Best-effort: a server that fails to connect or doesn't support prompts
/// is skipped. Spawns (and drops) the server processes.
pub(crate) fn list_prompts(servers: &[McpServer], cwd: &Path) -> Vec<McpPrompt> {
    let mut out = Vec::new();
    for server in servers {
        if let Ok((client, _tools)) = McpClient::connect(server, cwd) {
            for p in client.list_prompts() {
                out.push(McpPrompt {
                    server: server.name.clone(),
                    name: p.name,
                    description: p.description,
                    arguments: p.arguments,
                });
            }
        }
    }
    out
}

/// Resolve a prompt template (by server + name, with arguments) to its messages.
pub(crate) fn get_prompt(
    servers: &[McpServer],
    server: &str,
    name: &str,
    arguments: &[(String, String)],
    cwd: &Path,
) -> Result<Vec<PromptMessage>, String> {
    let cfg = servers.iter().find(|s| s.name == server).ok_or_else(|| format!("no MCP server named `{server}`"))?;
    let (client, _tools) = McpClient::connect(cfg, cwd)?;
    client.get_prompt(name, arguments)
}
