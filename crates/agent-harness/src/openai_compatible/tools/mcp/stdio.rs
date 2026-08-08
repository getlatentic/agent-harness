//! The stdio MCP transport: launch the server process and exchange
//! newline-delimited JSON-RPC over its stdin/stdout (one message per line, no
//! embedded newlines). A background reader thread parses lines into a channel,
//! and [`request`](StdioConnection::request) blocks for the response carrying
//! its id, skipping any interleaved notifications. The child is killed on `Drop`.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::Mutex;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use serde_json::{json, Value};

use super::client::McpConnection;

/// How long to wait for one request's response before giving up — a hung server
/// fails that single call rather than wedging the whole run.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// A live stdio connection to one MCP server process.
pub(super) struct StdioConnection {
    /// Server name, for prefixing error messages.
    server: String,
    inner: Mutex<Inner>,
}

struct Inner {
    child: Child,
    stdin: ChildStdin,
    /// Parsed messages from the server, fed by the reader thread.
    rx: Receiver<Value>,
    /// The reader thread, joined on `Drop` (it exits once stdout closes).
    reader: Option<JoinHandle<()>>,
    next_id: i64,
}

impl StdioConnection {
    /// Spawn `command` and wire up the reader thread. Does not handshake — that's
    /// the [`McpClient`](super::client::McpClient)'s job.
    pub(super) fn spawn(
        server: &str,
        command: &str,
        args: &[String],
        env: &[(String, String)],
        cwd: &Path,
    ) -> Result<StdioConnection, String> {
        let mut child = crate::hidden_command(command)
            .args(args)
            .envs(env.iter().map(|(k, v)| (k.clone(), v.clone())))
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("spawning `{command}`: {e}"))?;
        let stdin = child.stdin.take().ok_or("no stdin pipe")?;
        let stdout = child.stdout.take().ok_or("no stdout pipe")?;

        let (tx, rx) = mpsc::channel();
        let reader = thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if line.trim().is_empty() {
                    continue;
                }
                // Skip non-JSON lines (stray logging); stop if the receiver is
                // gone (the connection was dropped).
                if let Ok(v) = serde_json::from_str::<Value>(&line) {
                    if tx.send(v).is_err() {
                        break;
                    }
                }
            }
        });

        Ok(StdioConnection {
            server: server.to_owned(),
            inner: Mutex::new(Inner { child, stdin, rx, reader: Some(reader), next_id: 1 }),
        })
    }
}

impl McpConnection for StdioConnection {
    fn request(&self, method: &str, params: Value) -> Result<Value, String> {
        let mut inner = self.inner.lock().map_err(|_| format!("{}: connection poisoned", self.server))?;
        let id = inner.next_id;
        inner.next_id += 1;
        write_line(&mut inner.stdin, &json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }))
            .map_err(|e| format!("{} {method}: write failed: {e}", self.server))?;
        loop {
            match inner.rx.recv_timeout(REQUEST_TIMEOUT) {
                Ok(v) => {
                    if v.get("id").and_then(Value::as_i64) != Some(id) {
                        continue; // a notification or stale message — keep waiting
                    }
                    if let Some(err) = v.get("error") {
                        return Err(format!("{} {method}: {err}", self.server));
                    }
                    return Ok(v.get("result").cloned().unwrap_or(Value::Null));
                }
                Err(RecvTimeoutError::Timeout) => return Err(format!("{} {method}: timed out", self.server)),
                Err(RecvTimeoutError::Disconnected) => return Err(format!("{} {method}: server closed", self.server)),
            }
        }
    }

    fn notify(&self, method: &str, params: Value) -> Result<(), String> {
        let mut inner = self.inner.lock().map_err(|_| format!("{}: connection poisoned", self.server))?;
        write_line(&mut inner.stdin, &json!({ "jsonrpc": "2.0", "method": method, "params": params }))
            .map_err(|e| format!("{} {method}: write failed: {e}", self.server))
    }
}

impl Drop for StdioConnection {
    fn drop(&mut self) {
        if let Ok(mut inner) = self.inner.lock() {
            let _ = inner.child.kill();
            let _ = inner.child.wait();
            // Killing the child closes stdout, so the reader thread's `lines()`
            // ends and the thread returns — join it so it doesn't outlive us.
            if let Some(h) = inner.reader.take() {
                let _ = h.join();
            }
        }
    }
}

/// Write one JSON message as a single newline-terminated line (the stdio frame).
fn write_line(stdin: &mut ChildStdin, msg: &Value) -> std::io::Result<()> {
    let mut line = serde_json::to_string(msg)?;
    line.push('\n');
    stdin.write_all(line.as_bytes())?;
    stdin.flush()
}
