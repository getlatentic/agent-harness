//! The transport-agnostic MCP client: the `initialize` handshake, `tools/list`
//! (with cursor pagination), and `tools/call` (flattening result content to
//! text). The wire is one of two [`McpConnection`] transports — [`stdio`] (a
//! launched process) or [`http`] (a remote endpoint) — which differ only in how
//! a single JSON-RPC request/notification is delivered. Grounded in the MCP
//! 2025-11-25 spec.
//!
//! [`stdio`]: super::stdio
//! [`http`]: super::http

use std::path::Path;

use serde_json::{json, Value};

use super::{McpPromptArg, McpServer, McpTransport, PromptMessage};

/// Protocol version we negotiate (the latest as of writing). A server speaking
/// an older version still responds; we don't gate on the echoed value.
pub(super) const PROTOCOL_VERSION: &str = "2025-11-25";

/// One way to deliver a JSON-RPC message to an MCP server. The lifecycle in
/// [`McpClient`] is written once against this; stdio and HTTP each implement it.
pub(crate) trait McpConnection: Send + Sync {
    /// Send a request and return its `result` (or an `Err` for a JSON-RPC error
    /// / transport failure).
    fn request(&self, method: &str, params: Value) -> Result<Value, String>;
    /// Send a fire-and-forget notification (no response expected).
    fn notify(&self, method: &str, params: Value) -> Result<(), String>;
}

/// One tool advertised by an MCP server (`tools/list` entry).
pub(crate) struct McpToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// One resource advertised by an MCP server (`resources/list` entry).
pub(crate) struct McpResource {
    pub uri: String,
    pub name: String,
    pub description: String,
}

/// A prompt template from `prompts/list` (the server name is added by the
/// caller, which knows it).
pub(crate) struct McpPromptDef {
    pub name: String,
    pub description: String,
    pub arguments: Vec<McpPromptArg>,
}

/// A connected MCP server: a transport plus the protocol lifecycle. Shared by
/// that server's [`McpTool`]s via `Arc`.
///
/// [`McpTool`]: super::tool::McpTool
pub(crate) struct McpClient {
    conn: Box<dyn McpConnection>,
}

impl McpClient {
    /// Open the transport for `server`, perform the handshake, and return the
    /// client together with the tools it advertises. `Err` on connect / handshake
    /// failure. `cwd` is the working directory for a launched stdio server.
    pub(crate) fn connect(server: &McpServer, cwd: &Path) -> Result<(McpClient, Vec<McpToolDef>), String> {
        let conn: Box<dyn McpConnection> = match &server.transport {
            McpTransport::Stdio { command, args, env } => {
                Box::new(super::stdio::StdioConnection::spawn(&server.name, command, args, env, cwd)?)
            }
            McpTransport::Http { url, headers } => Box::new(super::http::HttpConnection::new(&server.name, url, headers)),
        };
        Self::handshake(conn)
    }

    /// The protocol lifecycle, independent of how the bytes get there: announce
    /// ourselves, confirm, then ask what the server offers.
    ///
    /// Separate from [`Self::connect`] because opening a transport and speaking
    /// the protocol are different concerns — and because the sequence is worth
    /// exercising without a process on the other end.
    fn handshake(conn: Box<dyn McpConnection>) -> Result<(McpClient, Vec<McpToolDef>), String> {
        let client = McpClient { conn };
        client.initialize()?;
        let tools = client.list_tools()?;
        Ok((client, tools))
    }

    /// `initialize` request → `notifications/initialized` (the spec's handshake).
    fn initialize(&self) -> Result<(), String> {
        let params = json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": { "name": "openai-compatible", "version": env!("CARGO_PKG_VERSION") }
        });
        self.conn.request("initialize", params)?;
        self.conn.notify("notifications/initialized", json!({}))
    }

    /// `tools/list`, following `nextCursor` pagination to the end.
    fn list_tools(&self) -> Result<Vec<McpToolDef>, String> {
        let mut out = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let params = cursor.as_ref().map_or_else(|| json!({}), |c| json!({ "cursor": c }));
            let result = self.conn.request("tools/list", params)?;
            if let Some(arr) = result.get("tools").and_then(Value::as_array) {
                for t in arr {
                    let Some(name) = t.get("name").and_then(Value::as_str) else { continue };
                    out.push(McpToolDef {
                        name: name.to_owned(),
                        description: t.get("description").and_then(Value::as_str).unwrap_or_default().to_owned(),
                        input_schema: t.get("inputSchema").cloned().unwrap_or_else(|| json!({ "type": "object" })),
                    });
                }
            }
            match result.get("nextCursor").and_then(Value::as_str) {
                Some(c) => cursor = Some(c.to_owned()),
                None => return Ok(out),
            }
        }
    }

    /// `tools/call`, flattening the result's content blocks to text. `Err` when
    /// the result is flagged `isError` (the tool itself failed) or on transport
    /// failure.
    pub(crate) fn call(&self, name: &str, arguments: &Value) -> Result<String, String> {
        let result = self.conn.request("tools/call", json!({ "name": name, "arguments": arguments }))?;
        let text = match result.get("content").and_then(Value::as_array) {
            Some(blocks) => flatten_content(blocks),
            None => String::new(),
        };
        if result.get("isError").and_then(Value::as_bool).unwrap_or(false) {
            return Err(if text.is_empty() { "the tool reported an error".to_owned() } else { text });
        }
        Ok(text)
    }

    /// `resources/list` — the server's readable resources. A server without
    /// resource support answers "method not found"; we map that (and any error)
    /// to an empty list, so it's simply skipped — resources are a bonus, not
    /// required.
    pub(crate) fn list_resources(&self) -> Vec<McpResource> {
        let Ok(result) = self.conn.request("resources/list", json!({})) else {
            return Vec::new();
        };
        result
            .get("resources")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|r| {
                        let uri = r.get("uri").and_then(Value::as_str)?;
                        Some(McpResource {
                            uri: uri.to_owned(),
                            name: r.get("name").and_then(Value::as_str).unwrap_or(uri).to_owned(),
                            description: r.get("description").and_then(Value::as_str).unwrap_or_default().to_owned(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// `resources/read` — read a resource's text contents by uri.
    pub(crate) fn read_resource(&self, uri: &str) -> Result<String, String> {
        let result = self.conn.request("resources/read", json!({ "uri": uri }))?;
        Ok(match result.get("contents").and_then(Value::as_array) {
            Some(items) => flatten_resource_contents(items),
            None => String::new(),
        })
    }

    /// `prompts/list` — the server's prompt templates. Unsupported / error → empty.
    pub(crate) fn list_prompts(&self) -> Vec<McpPromptDef> {
        let Ok(result) = self.conn.request("prompts/list", json!({})) else {
            return Vec::new();
        };
        result
            .get("prompts")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(parse_prompt_def).collect())
            .unwrap_or_default()
    }

    /// `prompts/get` — resolve a prompt template to its messages (text flattened).
    pub(crate) fn get_prompt(&self, name: &str, arguments: &[(String, String)]) -> Result<Vec<PromptMessage>, String> {
        let args: serde_json::Map<String, Value> =
            arguments.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect();
        let result = self.conn.request("prompts/get", json!({ "name": name, "arguments": args }))?;
        Ok(result
            .get("messages")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        let role = m.get("role").and_then(Value::as_str)?;
                        Some(PromptMessage { role: role.to_owned(), content: prompt_content_text(m.get("content")) })
                    })
                    .collect()
            })
            .unwrap_or_default())
    }
}

/// Flatten MCP content blocks to text — text blocks verbatim, other block types
/// noted by kind so the model knows non-text content came back.
pub(super) fn flatten_content(blocks: &[Value]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b.get("type").and_then(Value::as_str) {
            Some("text") => b.get("text").and_then(Value::as_str).map(str::to_owned),
            Some(other) => Some(format!("[{other} content omitted]")),
            None => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Flatten `resources/read` content items to text (text items verbatim; a binary
/// blob noted by placeholder).
fn flatten_resource_contents(items: &[Value]) -> String {
    items
        .iter()
        .filter_map(|c| {
            c.get("text")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .or_else(|| c.get("blob").map(|_| "[binary resource omitted]".to_owned()))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parse one `prompts/list` entry into a [`McpPromptDef`].
fn parse_prompt_def(p: &Value) -> Option<McpPromptDef> {
    let name = p.get("name").and_then(Value::as_str)?;
    let arguments = p
        .get("arguments")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|arg| {
                    let n = arg.get("name").and_then(Value::as_str)?;
                    Some(McpPromptArg {
                        name: n.to_owned(),
                        description: arg.get("description").and_then(Value::as_str).unwrap_or_default().to_owned(),
                        required: arg.get("required").and_then(Value::as_bool).unwrap_or(false),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    Some(McpPromptDef {
        name: name.to_owned(),
        description: p.get("description").and_then(Value::as_str).unwrap_or_default().to_owned(),
        arguments,
    })
}

/// Extract text from a prompt message's `content` — a single `{type,text}` block,
/// an array of blocks, or a bare string.
fn prompt_content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::Array(arr)) => flatten_content(arr),
        Some(Value::Object(_)) => {
            content.and_then(|c| c.get("text")).and_then(Value::as_str).unwrap_or_default().to_owned()
        }
        Some(Value::String(s)) => s.clone(),
        _ => String::new(),
    }
}

/// Extract a JSON-RPC `result` for `id` from a response body that is either a
/// plain JSON object or an SSE (`data:`-prefixed) stream — the two shapes the
/// Streamable-HTTP transport returns. Shared by the HTTP connection.
pub(super) fn parse_rpc_result(text: &str, id: i64) -> Result<Value, String> {
    // Collect every JSON object in the body: the whole thing if it parses, else
    // each SSE `data:` line that parses.
    let mut messages: Vec<Value> = Vec::new();
    if let Ok(v) = serde_json::from_str::<Value>(text) {
        messages.push(v);
    } else {
        for data in text.lines().filter_map(|l| l.strip_prefix("data:").map(str::trim)) {
            if let Ok(v) = serde_json::from_str::<Value>(data) {
                messages.push(v);
            }
        }
    }
    // Prefer the response carrying our id; fall back to any result/error message.
    let response = messages
        .iter()
        .find(|m| m.get("id").and_then(Value::as_i64) == Some(id))
        .or_else(|| messages.iter().find(|m| m.get("result").is_some() || m.get("error").is_some()));
    match response {
        Some(m) if m.get("error").is_some() => Err(format!("{}", m["error"])),
        Some(m) => Ok(m.get("result").cloned().unwrap_or(Value::Null)),
        None => Err("no JSON-RPC response in the server reply".to_owned()),
    }
}

/// An [`McpConnection`] that answers from a script rather than a server.
///
/// The transport is the only part of MCP that needs a process; the lifecycle,
/// the pagination and the result handling are decisions about JSON. Those get
/// tested here, against replies queued per method, with the requests recorded
/// so a test can assert what was *asked* and not only what came back.
///
/// Cloneable, sharing one script and one log, so a test can keep a handle on
/// the recording after handing the connection to the client.
#[cfg(test)]
#[derive(Clone, Default)]
pub(super) struct ScriptedConnection {
    #[allow(clippy::type_complexity)]
    replies: std::sync::Arc<
        std::sync::Mutex<std::collections::HashMap<String, std::collections::VecDeque<Result<Value, String>>>>,
    >,
    asked: std::sync::Arc<std::sync::Mutex<Vec<(String, Value)>>>,
}

#[cfg(test)]
impl ScriptedConnection {
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Queue one reply for `method`. Called twice for the same method, the
    /// replies come back in order — which is how a paginated list is scripted.
    pub(super) fn on(self, method: &str, reply: Value) -> Self {
        self.queue(method, Ok(reply));
        self
    }

    /// Queue a failure — a JSON-RPC error or a dead transport, which the client
    /// cannot tell apart and should not need to.
    pub(super) fn failing(self, method: &str, error: &str) -> Self {
        self.queue(method, Err(error.to_owned()));
        self
    }

    fn queue(&self, method: &str, reply: Result<Value, String>) {
        self.replies.lock().unwrap().entry(method.to_owned()).or_default().push_back(reply);
    }

    /// Every request and notification, in the order it was sent.
    pub(super) fn asked(&self) -> Vec<(String, Value)> {
        self.asked.lock().unwrap().clone()
    }

    pub(super) fn methods(&self) -> Vec<String> {
        self.asked().into_iter().map(|(method, _)| method).collect()
    }
}

#[cfg(test)]
impl McpConnection for ScriptedConnection {
    fn request(&self, method: &str, params: Value) -> Result<Value, String> {
        self.asked.lock().unwrap().push((method.to_owned(), params));
        self.replies
            .lock()
            .unwrap()
            .get_mut(method)
            .and_then(std::collections::VecDeque::pop_front)
            .unwrap_or_else(|| Err(format!("nothing scripted for {method}")))
    }

    fn notify(&self, method: &str, params: Value) -> Result<(), String> {
        self.asked.lock().unwrap().push((method.to_owned(), params));
        Ok(())
    }
}

#[cfg(test)]
impl McpClient {
    /// A client over a given connection, skipping the handshake — for tests
    /// that are about a single call rather than the lifecycle.
    pub(super) fn over(conn: Box<dyn McpConnection>) -> Self {
        Self { conn }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_keeps_text_and_notes_other_blocks() {
        let blocks = vec![
            json!({ "type": "text", "text": "hello" }),
            json!({ "type": "image", "data": "..." }),
            json!({ "type": "text", "text": "world" }),
        ];
        assert_eq!(flatten_content(&blocks), "hello\n[image content omitted]\nworld");
    }

    #[test]
    fn parse_rpc_result_reads_json_sse_and_errors() {
        // Plain JSON response.
        let json = r#"{"jsonrpc":"2.0","id":2,"result":{"ok":true}}"#;
        assert_eq!(parse_rpc_result(json, 2).unwrap(), json!({ "ok": true }));
        // SSE-framed response (the matching id among several data lines).
        let sse = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":5,\"result\":{\"v\":1}}\n\n";
        assert_eq!(parse_rpc_result(sse, 5).unwrap(), json!({ "v": 1 }));
        // JSON-RPC error surfaces as Err.
        let err = r#"{"jsonrpc":"2.0","id":3,"error":{"code":-32601,"message":"nope"}}"#;
        assert!(parse_rpc_result(err, 3).unwrap_err().contains("nope"));
    }

    fn tools_page(tools: &[&str], next: Option<&str>) -> Value {
        let listed: Vec<Value> = tools
            .iter()
            .map(|name| json!({ "name": name, "description": format!("does {name}"), "inputSchema": { "type": "object" } }))
            .collect();
        match next {
            Some(cursor) => json!({ "tools": listed, "nextCursor": cursor }),
            None => json!({ "tools": listed }),
        }
    }

    #[test]
    fn the_handshake_announces_itself_before_asking_what_is_offered() {
        // A server may reject anything sent before `initialize`, and the
        // `initialized` notification is what tells it the client is ready.
        let conn = ScriptedConnection::new()
            .on("initialize", json!({ "protocolVersion": PROTOCOL_VERSION }))
            .on("tools/list", tools_page(&["search"], None));
        let recorder = conn.clone();

        let (_client, tools) = McpClient::handshake(Box::new(conn)).expect("handshake");

        assert_eq!(
            recorder.methods(),
            ["initialize", "notifications/initialized", "tools/list"],
            "in that order"
        );
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "search");
    }

    #[test]
    fn a_server_that_refuses_the_handshake_is_not_reported_as_having_no_tools() {
        // Failing open here would offer the model an empty toolset from a server
        // that never came up, and nothing downstream would know why.
        let conn = ScriptedConnection::new().failing("initialize", "protocol version not supported");
        let err = McpClient::handshake(Box::new(conn)).map(|_| ()).expect_err("must fail");
        assert!(err.contains("protocol version"), "got {err}");
    }

    #[test]
    fn a_paginated_tool_list_is_followed_to_the_end() {
        // `nextCursor` is the server saying "there are more". Stopping at the
        // first page silently hides tools; not passing the cursor back re-reads
        // page one forever.
        let conn = ScriptedConnection::new()
            .on("initialize", json!({}))
            .on("tools/list", tools_page(&["one", "two"], Some("page2")))
            .on("tools/list", tools_page(&["three"], None));
        let recorder = conn.clone();

        let (_client, tools) = McpClient::handshake(Box::new(conn)).expect("handshake");

        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["one", "two", "three"], "every page is collected");
        let cursors: Vec<Value> = recorder
            .asked()
            .into_iter()
            .filter(|(method, _)| method == "tools/list")
            .map(|(_, params)| params)
            .collect();
        assert_eq!(cursors[0], json!({}), "the first page asks for no cursor");
        assert_eq!(cursors[1], json!({ "cursor": "page2" }), "and the second sends the one it was given");
    }

    #[test]
    fn a_tool_that_reports_its_own_failure_is_an_error_not_an_answer() {
        // `isError` is the tool saying it failed while the transport succeeded.
        // Passing its text back as a result would have the model treat a failure
        // as the answer.
        let client = McpClient::over(Box::new(
            ScriptedConnection::new()
                .on("tools/call", json!({ "content": [{ "type": "text", "text": "no such file" }], "isError": true }))
                .on("tools/call", json!({ "content": [{ "type": "text", "text": "it worked" }] })),
        ));
        assert_eq!(client.call("read", &json!({})).unwrap_err(), "no such file");
        assert_eq!(client.call("read", &json!({})).unwrap(), "it worked");
    }

    #[test]
    fn optional_capabilities_are_skipped_rather_than_fatal() {
        // Resources and prompts are a bonus. A server without them answers
        // "method not found", and treating that as a failure would lose the
        // tools it *does* offer.
        let client = McpClient::over(Box::new(
            ScriptedConnection::new()
                .failing("resources/list", "method not found")
                .failing("prompts/list", "method not found"),
        ));
        assert!(client.list_resources().is_empty());
        assert!(client.list_prompts().is_empty());
    }

    #[test]
    fn a_resource_without_a_name_is_listed_under_its_uri() {
        let client = McpClient::over(Box::new(ScriptedConnection::new().on(
            "resources/list",
            json!({ "resources": [
                { "uri": "file:///a.txt", "name": "A", "description": "the first" },
                { "uri": "file:///b.txt" },
                { "name": "no uri, so not addressable" }
            ]}),
        )));
        let resources = client.list_resources();
        assert_eq!(resources.len(), 2, "an entry with no uri cannot be read, so it is dropped");
        assert_eq!(resources[0].name, "A");
        assert_eq!(resources[1].name, "file:///b.txt", "the uri stands in for a missing name");
    }

    #[test]
    fn connect_reports_a_spawn_failure() {
        let server = McpServer::stdio("missing", "definitely-not-a-real-binary-xyz", vec![]);
        match McpClient::connect(&server, Path::new(".")) {
            Err(e) => assert!(e.contains("spawning"), "got: {e}"),
            Ok(_) => panic!("expected a spawn failure"),
        }
    }

    /// End-to-end against a tiny `sh` server that speaks the protocol: it replies
    /// to `initialize` (id 1), `tools/list` (id 2), and one `tools/call` (id 3) —
    /// the ids our client assigns in order — verifying the full handshake.
    #[test]
    fn connect_handshakes_lists_and_calls() {
        let script = r#"
            read _initialize
            printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{},"serverInfo":{"name":"fake","version":"0"}}}'
            read _initialized
            read _list
            printf '%s\n' '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"echo","description":"echoes input","inputSchema":{"type":"object"}}]}}'
            read _call
            printf '%s\n' '{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"pong"}],"isError":false}}'
            read _reslist
            printf '%s\n' '{"jsonrpc":"2.0","id":4,"result":{"resources":[{"uri":"file:///doc","name":"Doc","description":"a doc"}]}}'
            read _resread
            printf '%s\n' '{"jsonrpc":"2.0","id":5,"result":{"contents":[{"uri":"file:///doc","text":"doc body"}]}}'
            read _promptslist
            printf '%s\n' '{"jsonrpc":"2.0","id":6,"result":{"prompts":[{"name":"greet","description":"greeting","arguments":[{"name":"who","required":true}]}]}}'
            read _promptsget
            printf '%s\n' '{"jsonrpc":"2.0","id":7,"result":{"messages":[{"role":"user","content":{"type":"text","text":"Hello there"}}]}}'
        "#;
        let server = McpServer::stdio("fake", "sh", vec!["-c".to_owned(), script.to_owned()]);
        let (client, tools) = McpClient::connect(&server, Path::new(".")).expect("handshake succeeds");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");
        assert_eq!(tools[0].description, "echoes input");
        assert_eq!(client.call("echo", &json!({ "x": 1 })).expect("call succeeds"), "pong");
        // resources/list then resources/read.
        let resources = client.list_resources();
        assert_eq!(resources.len(), 1);
        assert_eq!(resources[0].uri, "file:///doc");
        assert_eq!(client.read_resource("file:///doc").expect("read succeeds"), "doc body");
        // prompts/list then prompts/get.
        let prompts = client.list_prompts();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].name, "greet");
        assert!(prompts[0].arguments[0].required && prompts[0].arguments[0].name == "who");
        let msgs = client.get_prompt("greet", &[("who".to_owned(), "world".to_owned())]).expect("get prompt");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].content, "Hello there");
    }
}
