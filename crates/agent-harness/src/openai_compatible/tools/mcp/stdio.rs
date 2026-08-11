//! The stdio MCP transport: launch the server process and exchange
//! newline-delimited JSON-RPC over its stdin/stdout (one message per line, no
//! embedded newlines). [`request`](StdioConnection::request) writes a line and
//! blocks for the response carrying its id, skipping any interleaved
//! notifications. The child is killed on `Drop`.
//!
//! Spawning, reading and cancelling are [`cli_stream`]'s job; this bridges its
//! pushed lines into a channel a blocking request can pull from, and adds the
//! id correlation the protocol needs.

use std::path::Path;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use serde_json::{json, Value};

use cli_stream::{ProcessEvent, ProcessHandle, Spawn};

use super::client::McpConnection;

/// How long to wait for one request's response before giving up — a hung server
/// fails that single call rather than wedging the whole run.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// A live stdio connection to one MCP server process.
pub(super) struct StdioConnection {
    /// Server name, for prefixing error messages.
    server: String,
    handle: ProcessHandle,
    /// Parsed messages from the server. Behind a mutex because a request reads
    /// until it sees *its* id, and two concurrent readers would steal each
    /// other's replies.
    inbox: Mutex<Receiver<Value>>,
    next_id: AtomicI64,
}

impl StdioConnection {
    /// Spawn `command` and start reading it. Does not handshake — that's the
    /// [`McpClient`](super::client::McpClient)'s job.
    pub(super) fn spawn(
        server: &str,
        command: &str,
        args: &[String],
        env: &[(String, String)],
        cwd: &Path,
    ) -> Result<StdioConnection, String> {
        let (tx, rx) = mpsc::channel();
        let spawn = Spawn::new(command, cwd, format!("mcp-{server}"))
            .args(args.to_vec())
            .env(env.to_vec())
            .writable();
        let handle = crate::node_cli::spawn_cli(spawn, move |event| {
            // Only stdout carries protocol; a server's stderr is its logging,
            // and a line that is not JSON is logging too.
            if let ProcessEvent::Stdout { line, .. } = event {
                if let Ok(message) = serde_json::from_str::<Value>(line.trim()) {
                    let _ = tx.send(message);
                }
            }
        })
        .map_err(|e| format!("spawning `{command}`: {e}"))?;

        Ok(StdioConnection {
            server: server.to_owned(),
            handle,
            inbox: Mutex::new(rx),
            next_id: AtomicI64::new(1),
        })
    }

    fn send(&self, message: &Value) -> Result<(), String> {
        let line = serde_json::to_string(message).map_err(|e| format!("{}: encoding: {e}", self.server))?;
        self.handle.write_line(&line).map_err(|e| format!("{}: write failed: {e}", self.server))
    }
}

impl McpConnection for StdioConnection {
    fn request(&self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let inbox = self.inbox.lock().map_err(|_| format!("{}: connection poisoned", self.server))?;
        self.send(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }))?;
        loop {
            match inbox.recv_timeout(REQUEST_TIMEOUT) {
                Ok(message) => {
                    if message.get("id").and_then(Value::as_i64) != Some(id) {
                        continue; // a notification or stale message — keep waiting
                    }
                    if let Some(err) = message.get("error") {
                        return Err(format!("{} {method}: {err}", self.server));
                    }
                    return Ok(message.get("result").cloned().unwrap_or(Value::Null));
                }
                Err(RecvTimeoutError::Timeout) => return Err(format!("{} {method}: timed out", self.server)),
                Err(RecvTimeoutError::Disconnected) => return Err(format!("{} {method}: server closed", self.server)),
            }
        }
    }

    fn notify(&self, method: &str, params: Value) -> Result<(), String> {
        self.send(&json!({ "jsonrpc": "2.0", "method": method, "params": params }))
    }
}

impl Drop for StdioConnection {
    fn drop(&mut self) {
        // The reader threads end when the child's pipes close, so cancelling is
        // the whole teardown.
        let _ = self.handle.cancel();
    }
}
