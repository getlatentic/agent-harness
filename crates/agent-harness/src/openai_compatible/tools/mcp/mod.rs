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

#[cfg(test)]
mod tests {
    use super::*;

    /// A server that speaks just enough of the protocol to be connected to,
    /// offering `n_tools` tools and `n_resources` resources.
    fn sh_server(name: &str, n_tools: usize, n_resources: usize) -> McpServer {
        let tools: Vec<String> = (0..n_tools)
            .map(|i| format!(r#"{{"name":"t{i}","description":"d","inputSchema":{{"type":"object"}}}}"#))
            .collect();
        let resources: Vec<String> =
            (0..n_resources).map(|i| format!(r#"{{"uri":"file:///r{i}","name":"R{i}"}}"#)).collect();
        let script = format!(
            r#"
            read _initialize
            printf '%s\n' '{{"jsonrpc":"2.0","id":1,"result":{{"protocolVersion":"2025-11-25"}}}}'
            read _initialized
            read _list
            printf '%s\n' '{{"jsonrpc":"2.0","id":2,"result":{{"tools":[{}]}}}}'
            read _reslist
            printf '%s\n' '{{"jsonrpc":"2.0","id":3,"result":{{"resources":[{}]}}}}'
            "#,
            tools.join(","),
            resources.join(",")
        );
        McpServer::stdio(name, "sh", vec!["-c".to_owned(), script])
    }

    #[test]
    fn a_transport_option_meant_for_the_other_transport_does_nothing() {
        // Both builders take anything, because a host assembling servers from
        // config should not have to branch. The quiet part is that the option
        // is dropped rather than misapplied — an `env` on an HTTP server is not
        // smuggled in as a header.
        let stdio = McpServer::stdio("s", "npx", vec!["-y".to_owned()])
            .env("TOKEN", "abc")
            .header("Authorization", "Bearer x");
        match &stdio.transport {
            McpTransport::Stdio { command, args, env } => {
                assert_eq!(command, "npx");
                assert_eq!(args, &["-y"]);
                assert_eq!(env, &[("TOKEN".to_owned(), "abc".to_owned())], "the env applies");
            }
            other => panic!("expected stdio, got {other:?}"),
        }

        let http = McpServer::http("h", "https://example.test/mcp")
            .header("Authorization", "Bearer x")
            .env("TOKEN", "abc");
        match &http.transport {
            McpTransport::Http { url, headers } => {
                assert_eq!(url, "https://example.test/mcp");
                assert_eq!(headers, &[("Authorization".to_owned(), "Bearer x".to_owned())], "the header applies");
            }
            other => panic!("expected http, got {other:?}"),
        }
    }

    #[test]
    fn a_server_that_will_not_start_is_reported_and_skipped() {
        // Best-effort is the whole contract here: one misconfigured server must
        // not take the run down with it, and the reason has to reach the user
        // or the tools simply appear to be missing.
        let servers = vec![McpServer::stdio("broken", "definitely-not-a-real-binary-xyz", vec![])];
        let (tools, status) = connect_all(&servers, Path::new("."));

        assert!(tools.is_empty());
        assert_eq!(status.len(), 1);
        assert!(status[0].contains("`broken` unavailable"), "got {:?}", status[0]);
        assert!(status[0].contains("spawning"), "with the reason: {:?}", status[0]);
    }

    #[test]
    fn a_connected_server_reports_what_it_actually_offered() {
        // The status line is the only place a user learns an MCP server came up
        // with nothing, which is otherwise indistinguishable from it working.
        let (tools, status) = connect_all(&[sh_server("many", 2, 1)], Path::new("."));
        assert_eq!(tools.len(), 3, "two tools plus the resource reader");
        assert_eq!(status, ["mcp: connected `many` (2 tools, 1 resource)"]);

        // Singular, and no resource note when there are none to read.
        let (tools, status) = connect_all(&[sh_server("one", 1, 0)], Path::new("."));
        assert_eq!(tools.len(), 1, "no resource tool is added for a server with no resources");
        assert_eq!(status, ["mcp: connected `one` (1 tool)"]);
    }

    #[test]
    fn asking_an_unregistered_server_for_a_prompt_says_which_name_was_wrong() {
        let servers = vec![McpServer::stdio("known", "sh", vec![])];
        let err = get_prompt(&servers, "unknown", "p", &[], Path::new(".")).unwrap_err();
        assert!(err.contains("no MCP server named `unknown`"), "got {err}");
    }
}
