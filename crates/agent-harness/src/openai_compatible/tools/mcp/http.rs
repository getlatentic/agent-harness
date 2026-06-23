//! The HTTP MCP transport (MCP's Streamable-HTTP): each JSON-RPC message is an
//! independent POST to the server endpoint, whose reply is either a single JSON
//! object or an SSE (`data:`) stream. A `Mcp-Session-Id` handed back on
//! `initialize` is replayed on every later request for session continuity.

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use serde_json::{json, Value};

use super::client::{parse_rpc_result, McpConnection};

/// Per-request HTTP timeout.
const HTTP_TIMEOUT: Duration = Duration::from_secs(60);

/// A remote MCP connection over HTTP. Stateless between calls apart from the
/// negotiated session id.
pub(super) struct HttpConnection {
    server: String,
    url: String,
    headers: Vec<(String, String)>,
    /// `Mcp-Session-Id` from the server, replayed on subsequent requests.
    session: Mutex<Option<String>>,
    next_id: AtomicI64,
}

impl HttpConnection {
    pub(super) fn new(server: &str, url: &str, headers: &[(String, String)]) -> HttpConnection {
        HttpConnection {
            server: server.to_owned(),
            url: url.to_owned(),
            headers: headers.to_vec(),
            session: Mutex::new(None),
            next_id: AtomicI64::new(1),
        }
    }

    /// Build a POST carrying the JSON body, the configured headers, and the
    /// negotiated session id (if any).
    fn post(&self, body: &Value) -> Result<ureq::Response, String> {
        let mut req = ureq::post(&self.url)
            .timeout(HTTP_TIMEOUT)
            .set("Content-Type", "application/json")
            .set("Accept", "application/json, text/event-stream");
        for (k, v) in &self.headers {
            req = req.set(k, v);
        }
        if let Some(sid) = self.session.lock().ok().and_then(|s| s.clone()) {
            req = req.set("Mcp-Session-Id", &sid);
        }
        let method = body.get("method").and_then(Value::as_str).unwrap_or("request");
        let resp = req.send_json(body.clone()).map_err(|e| format!("{} {method}: request failed: {e}", self.server))?;
        // Capture a session id the server hands back (typically on initialize).
        if let Some(sid) = resp.header("Mcp-Session-Id") {
            if let Ok(mut slot) = self.session.lock() {
                *slot = Some(sid.to_owned());
            }
        }
        Ok(resp)
    }
}

impl McpConnection for HttpConnection {
    fn request(&self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let resp = self.post(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }))?;
        let text = resp.into_string().map_err(|e| format!("{} {method}: reading response: {e}", self.server))?;
        parse_rpc_result(&text, id).map_err(|e| format!("{} {method}: {e}", self.server))
    }

    fn notify(&self, method: &str, params: Value) -> Result<(), String> {
        // A notification has no id and expects no body (the server returns 202).
        self.post(&json!({ "jsonrpc": "2.0", "method": method, "params": params })).map(|_| ())
    }
}
