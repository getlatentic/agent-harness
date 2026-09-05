//! The MCP server half of a [`ToolServer`]: answers the JSON-RPC messages an
//! MCP client sends, so an adapter that receives them (Claude Code's control
//! protocol) can hand them here and write back what comes out.
//!
//! Only what a tools-only server needs — `initialize`, `ping`, `tools/list`,
//! `tools/call`. Resources and prompts are declined by omission from the
//! advertised capabilities, so a well-behaved client never asks.

use serde_json::{json, Value};

use super::ToolServer;

/// The MCP protocol revision answered when the client names none.
const DEFAULT_PROTOCOL_VERSION: &str = "2024-11-05";

/// Answer one JSON-RPC message for `server`.
///
/// `None` for a notification (no `id`): there is nothing to send back, and the
/// adapter acknowledges on its own protocol. `Some` for a request — a result or
/// a JSON-RPC error, both carrying the request's `id`.
pub(crate) fn serve(server: &ToolServer, message: &Value) -> Option<Value> {
    let id = message.get("id").filter(|id| !id.is_null())?.clone();
    let Some(method) = message.get("method").and_then(Value::as_str) else {
        return Some(error(id, -32600, "invalid request: no method"));
    };
    let params = message.get("params").cloned().unwrap_or(Value::Null);
    Some(match method {
        "initialize" => result(id, initialize(server, &params)),
        "ping" => result(id, json!({})),
        "tools/list" => result(id, json!({ "tools": tool_list(server) })),
        "tools/call" => tools_call(server, id, &params),
        other => error(id, -32601, &format!("method not found: {other}")),
    })
}

fn initialize(server: &ToolServer, params: &Value) -> Value {
    // Echo the client's revision: the tool surface here is the same under every
    // revision so far, and a mismatch is what makes a client refuse the server.
    let version = params
        .get("protocolVersion")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_PROTOCOL_VERSION);
    json!({
        "protocolVersion": version,
        "capabilities": { "tools": {} },
        "serverInfo": { "name": server.name(), "version": env!("CARGO_PKG_VERSION") },
    })
}

fn tool_list(server: &ToolServer) -> Vec<Value> {
    server
        .tools()
        .iter()
        .map(|tool| {
            json!({
                "name": tool.name(),
                "description": tool.description(),
                "inputSchema": tool.input_schema(),
                "annotations": { "readOnlyHint": tool.read_only() },
            })
        })
        .collect()
}

fn tools_call(server: &ToolServer, id: Value, params: &Value) -> Value {
    let Some(name) = params.get("name").and_then(Value::as_str) else {
        return error(id, -32602, "invalid params: tools/call needs a tool name");
    };
    if server.tool(name).is_none() {
        // Unknown tool is a protocol error in MCP; a tool that ran and failed is
        // a result with `isError`, below, so the model can read the failure.
        return error(id, -32602, &format!("unknown tool: {name}"));
    }
    let arguments = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
    let (text, is_error) = match server.call(name, arguments) {
        Ok(text) => (text, false),
        Err(text) => (text, true),
    };
    result(id, json!({ "content": [{ "type": "text", "text": text }], "isError": is_error }))
}

fn result(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn error(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host_tools::FnTool;

    fn shop() -> ToolServer {
        ToolServer::new("shop")
            .with_tool(
                FnTool::new(
                    "stock",
                    "units on hand",
                    json!({"type": "object", "properties": {"sku": {"type": "string"}}}),
                    |args| Ok(format!("{} units", args["sku"].as_str().unwrap_or("?").len())),
                )
                .as_read_only(),
            )
            .with_tool(FnTool::new("restock", "orders more", json!({"type": "object"}), |_| {
                Err("supplier closed".to_owned())
            }))
    }

    fn request(id: impl Into<Value>, method: &str, params: Value) -> Value {
        json!({ "jsonrpc": "2.0", "id": id.into(), "method": method, "params": params })
    }

    #[test]
    fn initialize_echoes_the_clients_protocol_version_and_names_the_server() {
        let reply = serve(&shop(), &request(1, "initialize", json!({"protocolVersion": "2025-06-18"})))
            .expect("a request gets a reply");
        assert_eq!(reply["id"], 1);
        assert_eq!(reply["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(reply["result"]["serverInfo"]["name"], "shop");
        assert!(reply["result"]["capabilities"]["tools"].is_object());
    }

    #[test]
    fn a_notification_gets_no_reply() {
        let note = json!({ "jsonrpc": "2.0", "method": "notifications/initialized" });
        assert!(serve(&shop(), &note).is_none());
    }

    #[test]
    fn tools_list_carries_schema_and_read_only_hint() {
        let reply = serve(&shop(), &request("a", "tools/list", Value::Null)).unwrap();
        let tools = reply["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["name"], "stock");
        assert_eq!(tools[0]["inputSchema"]["properties"]["sku"]["type"], "string");
        assert_eq!(tools[0]["annotations"]["readOnlyHint"], true);
        assert_eq!(tools[1]["annotations"]["readOnlyHint"], false);
    }

    #[test]
    fn tools_call_returns_text_content_and_a_failed_call_is_flagged_not_errored() {
        let ok = serve(&shop(), &request(7, "tools/call", json!({"name": "stock", "arguments": {"sku": "A-7"}})))
            .unwrap();
        assert_eq!(ok["result"]["content"][0]["text"], "3 units");
        assert_eq!(ok["result"]["isError"], false);

        // The tool ran and said no: that is a result the model reads, not a
        // protocol error that ends the exchange.
        let failed = serve(&shop(), &request(8, "tools/call", json!({"name": "restock"}))).unwrap();
        assert_eq!(failed["result"]["isError"], true);
        assert_eq!(failed["result"]["content"][0]["text"], "supplier closed");
        assert!(failed.get("error").is_none());
    }

    #[test]
    fn an_unknown_tool_and_an_unknown_method_are_json_rpc_errors() {
        let tool = serve(&shop(), &request(9, "tools/call", json!({"name": "price"}))).unwrap();
        assert_eq!(tool["error"]["code"], -32602);
        let method = serve(&shop(), &request(10, "resources/list", Value::Null)).unwrap();
        assert_eq!(method["error"]["code"], -32601);
        let bare = serve(&shop(), &json!({ "jsonrpc": "2.0", "id": 11 })).unwrap();
        assert_eq!(bare["error"]["code"], -32600);
    }
}
