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
        if let Some(sid) = resp.header("Mcp-Session-Id")
            && let Ok(mut slot) = self.session.lock()
        {
            *slot = Some(sid.to_owned());
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// What one request looked like on the wire.
    struct Seen {
        headers: Vec<(String, String)>,
        body: Value,
    }

    impl Seen {
        fn header(&self, name: &str) -> Option<&str> {
            self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
        }
    }

    /// A server answering each POST with the next queued `(session_id, body)`.
    fn fake_mcp(replies: Vec<(Option<&'static str>, String)>) -> (String, Arc<Mutex<Vec<Seen>>>) {
        let server = tiny_http::Server::http("127.0.0.1:0").expect("bind ephemeral port");
        let url = format!("http://{}/mcp", server.server_addr());
        let seen: Arc<Mutex<Vec<Seen>>> = Arc::default();
        let log = Arc::clone(&seen);
        let mut queued = replies.into_iter();

        std::thread::spawn(move || {
            while let Ok(mut request) = server.recv() {
                let headers = request
                    .headers()
                    .iter()
                    .map(|h| (h.field.as_str().as_str().to_owned(), h.value.as_str().to_owned()))
                    .collect();
                let mut raw = String::new();
                let _ = std::io::Read::read_to_string(request.as_reader(), &mut raw);
                log.lock().unwrap().push(Seen { headers, body: serde_json::from_str(&raw).unwrap_or(Value::Null) });

                let (session, payload) = queued.next().unwrap_or((None, String::new()));
                let mut response = tiny_http::Response::from_string(payload);
                if let Some(sid) = session {
                    response.add_header(
                        tiny_http::Header::from_bytes(&b"Mcp-Session-Id"[..], sid.as_bytes()).unwrap(),
                    );
                }
                let _ = request.respond(response);
            }
        });
        (url, seen)
    }

    fn result(id: i64, value: Value) -> String {
        json!({ "jsonrpc": "2.0", "id": id, "result": value }).to_string()
    }

    #[test]
    fn a_session_the_server_opens_is_replayed_on_every_later_request() {
        // The whole reason this transport holds state. A server that hands back
        // `Mcp-Session-Id` expects it echoed; dropping it means each request
        // looks like a new client, and the handshake silently un-does itself.
        let (url, seen) = fake_mcp(vec![
            (Some("sess-42"), result(1, json!({ "protocolVersion": "2025-11-25" }))),
            (None, result(2, json!({ "tools": [] }))),
        ]);
        let conn = HttpConnection::new("remote", &url, &[]);

        conn.request("initialize", json!({})).expect("initialize");
        conn.request("tools/list", json!({})).expect("list");

        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 2);
        assert!(seen[0].header("Mcp-Session-Id").is_none(), "there is no session to send yet");
        assert_eq!(seen[1].header("Mcp-Session-Id"), Some("sess-42"), "and afterwards there is");
    }

    #[test]
    fn configured_headers_and_the_json_rpc_envelope_go_out_on_every_request() {
        let (url, seen) = fake_mcp(vec![(None, result(1, json!({})))]);
        let headers = vec![("Authorization".to_owned(), "Bearer t0ken".to_owned())];
        let conn = HttpConnection::new("remote", &url, &headers);

        conn.request("tools/list", json!({ "cursor": "p2" })).expect("request");

        let seen = seen.lock().unwrap();
        let sent = &seen[0];
        assert_eq!(sent.header("Authorization"), Some("Bearer t0ken"), "a remote server usually needs a key");
        assert_eq!(sent.header("Content-Type"), Some("application/json"));
        assert!(
            sent.header("Accept").is_some_and(|a| a.contains("text/event-stream")),
            "a reply may be streamed, and the server picks based on this"
        );
        assert_eq!(sent.body["jsonrpc"], "2.0");
        assert_eq!(sent.body["method"], "tools/list");
        assert_eq!(sent.body["params"]["cursor"], "p2");
        assert_eq!(sent.body["id"], 1, "ids start at one and the reply is matched against them");
    }

    #[test]
    fn each_request_gets_its_own_id_and_the_matching_reply() {
        // Ids identify which reply belongs to which call; reusing one would let
        // a stale response answer the wrong question.
        let (url, seen) = fake_mcp(vec![
            (None, result(1, json!({ "n": "first" }))),
            (None, result(2, json!({ "n": "second" }))),
        ]);
        let conn = HttpConnection::new("remote", &url, &[]);

        assert_eq!(conn.request("a", json!({})).unwrap()["n"], "first");
        assert_eq!(conn.request("b", json!({})).unwrap()["n"], "second");

        let seen = seen.lock().unwrap();
        assert_eq!(seen[0].body["id"], 1);
        assert_eq!(seen[1].body["id"], 2);
    }

    #[test]
    fn a_notification_carries_no_id_because_no_reply_is_coming() {
        let (url, seen) = fake_mcp(vec![(None, String::new())]);
        let conn = HttpConnection::new("remote", &url, &[]);

        conn.notify("notifications/initialized", json!({})).expect("notify");

        let seen = seen.lock().unwrap();
        assert_eq!(seen[0].body["method"], "notifications/initialized");
        assert!(seen[0].body.get("id").is_none(), "an id would make the server answer it");
    }

    #[test]
    fn a_streamed_reply_is_understood_as_well_as_a_plain_one() {
        // Streamable-HTTP lets the server answer either way for the same call,
        // so a client that only understands one of them works until it doesn't.
        let sse = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"ok\":true}}\n\n";
        let (url, _seen) = fake_mcp(vec![(None, sse.to_owned())]);
        let conn = HttpConnection::new("remote", &url, &[]);
        assert_eq!(conn.request("tools/list", json!({})).unwrap(), json!({ "ok": true }));
    }

    #[test]
    fn an_unreachable_server_names_itself_and_what_it_was_asked() {
        // The status line this ends up in says which server is unavailable, so
        // a user with three configured knows which one to look at.
        let conn = HttpConnection::new("remote", "http://127.0.0.1:1/mcp", &[]);
        let err = conn.request("initialize", json!({})).unwrap_err();
        assert!(err.contains("remote"), "got {err}");
        assert!(err.contains("initialize"), "got {err}");
    }
}
