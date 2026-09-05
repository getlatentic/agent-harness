//! Claude Code's control protocol, used to serve the host's [`ToolServer`]s.
//!
//! With `--input-format stream-json` the CLI reads its input as newline JSON and
//! the pipe becomes two-way: the prompt goes down as a `user` message, and the
//! CLI sends `control_request`s back up for things only the host can answer.
//! One of those is `mcp_message` — the CLI acting as MCP client to a server the
//! host said it hosts. Each JSON-RPC message arrives in a request envelope, is
//! answered by [`jsonrpc::serve`](crate::host_tools::jsonrpc::serve), and goes
//! back down in a `control_response`. This is the Agent SDK's `type: "sdk"` MCP
//! server, spoken from Rust.
//!
//! Two facts about the CLI, measured against 2.1.241 rather than read: under
//! stream-json input the positional prompt is ignored — a run with one and a
//! closed stdin produces nothing — so the prompt has to travel as a message.
//! And the CLI does not exit on its own after the answer; it waits for the
//! next message, so stdin is closed once the `result` line has passed.
//!
//! An `initialize` control request naming the servers is what registers them.
//! `--mcp-config` with `type: "sdk"` is not needed, and is not sent.

use std::sync::mpsc::Receiver;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::host_tools::jsonrpc;
use crate::{normalize_process_event, Command, Error, Event, ProcessHandle, RunCallback, ToolServer};
use cli_stream::Stdin;

use super::parser::parse_claude_line;

/// Spawn the CLI with the control channel open, register `servers`, send the
/// prompt, and serve requests until the run ends.
///
/// The returned handle is the same one the pump thread writes through:
/// [`ProcessHandle`] is a shared handle, so the caller's `cancel` reaches the
/// child the pump is still reading from.
pub(super) fn start(
    command: Command,
    run_id: &str,
    prompt: String,
    servers: Vec<ToolServer>,
    on_event: RunCallback,
) -> Result<ProcessHandle, Error> {
    let (handle, events) = command.stdin(Stdin::Piped).start().map_err(Error::spawn)?;
    let names: Vec<&str> = servers.iter().map(ToolServer::name).collect();
    // The CLI answers the initialize before it reads the prompt, and both are
    // written before any output is read: the probe that established the
    // protocol did the same and the CLI took them in order.
    write(&handle, &json!({
        "type": "control_request",
        "request_id": format!("{run_id}-initialize"),
        "request": { "subtype": "initialize", "sdkMcpServers": names },
    }))?;
    write(&handle, &json!({
        "type": "user",
        "message": { "role": "user", "content": prompt },
        "parent_tool_use_id": null,
    }))?;
    let reader = handle.clone();
    let servers = Arc::new(servers);
    std::thread::spawn(move || relay(events, reader, servers, on_event));
    Ok(handle)
}

/// A write that failed before the run produced anything is the run failing to
/// start — the child is already gone.
fn write(handle: &ProcessHandle, message: &Value) -> Result<(), Error> {
    handle.write_line(&message.to_string()).map_err(Error::spawn)
}

/// What one stdout line is, from the control channel's point of view.
enum Line {
    /// The CLI asks the host something; answered off this thread.
    Request { id: String, request: Value },
    /// The answer to our own `initialize`, or a cancel for a request already
    /// answered. Neither reaches the consumer.
    Housekeeping,
    /// The final result. Forwarded like any line, then stdin is closed so the
    /// CLI exits instead of waiting for a next message.
    Result,
    /// An ordinary event line, parsed by the adapter's parser.
    Other,
}

fn classify(line: &str) -> Line {
    let Ok(value) = serde_json::from_str::<Value>(line) else { return Line::Other };
    match (value["type"].as_str(), value["request_id"].as_str()) {
        (Some("control_request"), Some(id)) => Line::Request { id: id.to_owned(), request: value["request"].clone() },
        (Some("control_response" | "control_cancel_request" | "keep_alive"), _) => Line::Housekeeping,
        (Some("result"), _) => Line::Result,
        _ => Line::Other,
    }
}

/// Read the child until it exits: answer control requests, forward everything
/// else as the adapter always has.
fn relay(events: Receiver<Event>, handle: ProcessHandle, servers: Arc<Vec<ToolServer>>, on_event: RunCallback) {
    for event in events {
        let line = match &event {
            Event::Stdout { line, .. } => classify(line),
            _ => Line::Other,
        };
        match line {
            Line::Request { id, request } => {
                // Off this thread: a host tool may take as long as it likes,
                // and the reader must keep draining the pipe meanwhile.
                let (handle, servers) = (handle.clone(), Arc::clone(&servers));
                std::thread::spawn(move || answer(&handle, &servers, id, &request));
            }
            Line::Housekeeping => {}
            Line::Result => {
                forward(event, &on_event);
                // Best-effort: a child that already exited has no stdin to
                // close, and its `Exited` is on its way regardless.
                let _ = handle.close_stdin();
            }
            Line::Other => forward(event, &on_event),
        }
    }
}

fn forward(event: Event, on_event: &RunCallback) {
    for normalized in normalize_process_event(event, parse_claude_line) {
        (*on_event)(normalized);
    }
}

/// Answer one control request. Anything this adapter does not implement is
/// declined with an error response rather than left unanswered, because an
/// unanswered request is a turn that never ends.
fn answer(handle: &ProcessHandle, servers: &[ToolServer], request_id: String, request: &Value) {
    let response = match request.get("subtype").and_then(Value::as_str) {
        Some("mcp_message") => mcp_message(servers, request),
        Some(other) => Err(format!("agent-harness does not handle the `{other}` control request")),
        None => Err("control request without a subtype".to_owned()),
    };
    let envelope = match response {
        Ok(payload) => json!({ "subtype": "success", "request_id": request_id, "response": payload }),
        Err(error) => json!({ "subtype": "error", "request_id": request_id, "error": error }),
    };
    // A failed write means the child is gone; its exit is already on the
    // event stream, and there is no one left to tell.
    let _ = handle.write_line(&json!({ "type": "control_response", "response": envelope }).to_string());
}

/// Route one MCP JSON-RPC message to the server it names.
fn mcp_message(servers: &[ToolServer], request: &Value) -> Result<Value, String> {
    let name = request.get("server_name").and_then(Value::as_str).unwrap_or_default();
    let Some(server) = servers.iter().find(|s| s.name() == name) else {
        return Err(format!("no host tool server named `{name}`"));
    };
    let message = request.get("message").cloned().unwrap_or(Value::Null);
    let reply = jsonrpc::serve(server, &message)
        // A notification has no reply; the SDK acknowledges it with this
        // empty result, and the CLI accepts exactly that.
        .unwrap_or_else(|| json!({ "jsonrpc": "2.0", "result": {}, "id": 0 }));
    Ok(json!({ "mcp_response": reply }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FnTool;

    fn line(v: Value) -> String {
        v.to_string()
    }

    #[test]
    fn lines_are_sorted_into_what_the_channel_needs_to_do_with_them() {
        let req = line(json!({"type":"control_request","request_id":"r1","request":{"subtype":"mcp_message"}}));
        assert!(matches!(classify(&req), Line::Request { id, .. } if id == "r1"));
        assert!(matches!(classify(&line(json!({"type":"control_response","response":{}}))), Line::Housekeeping));
        assert!(matches!(classify(&line(json!({"type":"control_cancel_request","request_id":"r1"}))), Line::Housekeeping));
        assert!(matches!(classify(&line(json!({"type":"result","is_error":false}))), Line::Result));
        assert!(matches!(classify(&line(json!({"type":"assistant","message":{}}))), Line::Other));
        assert!(matches!(classify("not json"), Line::Other));
        // A request nobody could answer — no id to answer it under — is left to
        // the parser like any other line rather than swallowed.
        assert!(matches!(classify(&line(json!({"type":"control_request","request":{}}))), Line::Other));
    }

    fn shop() -> Vec<ToolServer> {
        vec![ToolServer::new("shop").with_tool(FnTool::new("stock", "d", json!({"type":"object"}), |_| Ok("3".into())))]
    }

    #[test]
    fn an_mcp_request_is_answered_under_mcp_response_and_a_notification_gets_the_empty_ack() {
        let call = json!({"subtype":"mcp_message","server_name":"shop","message":{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"stock"}}});
        let reply = mcp_message(&shop(), &call).unwrap();
        assert_eq!(reply["mcp_response"]["id"], 5);
        assert_eq!(reply["mcp_response"]["result"]["content"][0]["text"], "3");

        let note = json!({"subtype":"mcp_message","server_name":"shop","message":{"jsonrpc":"2.0","method":"notifications/initialized"}});
        let ack = mcp_message(&shop(), &note).unwrap();
        assert_eq!(ack["mcp_response"], json!({"jsonrpc":"2.0","result":{},"id":0}));
    }

    #[test]
    fn a_request_for_an_unknown_server_or_subtype_is_declined_not_dropped() {
        let other = json!({"subtype":"mcp_message","server_name":"warehouse","message":{}});
        assert!(mcp_message(&shop(), &other).unwrap_err().contains("warehouse"));
    }
}
