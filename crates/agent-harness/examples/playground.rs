//! A small local web playground for `openai-compatible`: run it, open the printed
//! URL, type a prompt, and watch the `RunEvent` stream — collapsible thinking,
//! tool cards (before → after), markdown output, usage/cost — live over
//! Server-Sent Events. Multi-turn via session resume; pick a model, browse for a
//! working folder, and toggle Ask/Edit in the page.
//!
//! ```text
//! cargo run --example playground      # needs a reachable model:
//!                                     #   ollama serve && ollama pull qwen2.5-coder
//! ```
//!
//! It drives [`OpenHarness::ollama`] and bridges `run_channel`'s
//! receiver straight to the browser. A run defaults to a throwaway scratch dir
//! (so Edit mode can't touch your files) and denies the most destructive shell
//! commands. This is a dev/test tool (it pulls `tiny_http`, a dev-dependency) —
//! not part of the library. The UI lives in `playground.html`.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use harness::{Harness, RunEvent, RunMode, RunRequest, RunTuning};
use harness::{OpenHarness, PermissionRule};
use tiny_http::{Header, Response, Server, StatusCode};

/// The page (HTML + CSS + JS), kept in its own file for readability.
const INDEX_HTML: &str = include_str!("playground.html");

fn main() {
    let addr = "127.0.0.1:8765";
    let server = Server::http(addr).unwrap_or_else(|e| panic!("bind {addr}: {e}"));
    println!("openai-compatible playground → http://{addr}");
    println!("(needs a reachable model — e.g. `ollama serve` + `ollama pull qwen2.5-coder`)");
    for request in server.incoming_requests() {
        std::thread::spawn(move || handle(request));
    }
}

fn handle(request: tiny_http::Request) {
    let url = request.url();
    if url == "/" {
        let resp = Response::from_string(INDEX_HTML).with_header(header("Content-Type", "text/html; charset=utf-8"));
        let _ = request.respond(resp);
    } else if url.starts_with("/run?") {
        stream_run(request);
    } else if url == "/dirs" || url.starts_with("/dirs?") {
        list_dirs(request);
    } else if url == "/models" || url.starts_with("/models?") {
        list_models(request);
    } else if url.starts_with("/file?") {
        read_file(request);
    } else {
        let _ = request.respond(Response::from_string("not found").with_status_code(404));
    }
}

/// Start a run from the query params and stream its `RunEvent`s to the browser
/// as SSE (`data: {json}\n\n` per event), ending after `Exited`.
fn stream_run(request: tiny_http::Request) {
    let params = parse_query(request.url().split_once('?').map_or("", |(_, q)| q));
    let get = |k: &str| params.get(k).filter(|s| !s.is_empty()).cloned();

    let req = RunRequest {
        run_id: "playground".to_owned(),
        prompt: get("prompt").unwrap_or_default(),
        // Default to a throwaway scratch dir so Edit mode can't touch real files.
        cwd: Some(get("cwd").map(PathBuf::from).unwrap_or_else(scratch_workspace)),
        mode: if get("mode").as_deref() == Some("ask") { RunMode::Ask } else { RunMode::Edit },
        tuning: RunTuning { model: get("model"), ..Default::default() },
        resume: get("session"),
        attachments: Vec::new(),
    };

    respond_stream(build_harness(&params).as_ref(), req, request);
}

/// Build the harness the query params select, shared by `/run` and `/models` so
/// the model list always matches what a run would actually use. Archetype B (an
/// ACP agent — opencode / Gemini, which owns its own tools) or archetype A (an
/// OpenAI-compatible endpoint we drive: Ollama by default, or any local server
/// via a `base` param — Llamabarn `http://localhost:2276`, LM Studio, MLX, …).
/// Archetype-A runs default to a scratch dir and deny the most destructive shell
/// commands (casual-test safety).
fn build_harness(params: &HashMap<String, String>) -> Box<dyn Harness> {
    let get = |k: &str| params.get(k).filter(|s| !s.is_empty()).cloned();
    match get("harness").as_deref() {
        Some("acp") => Box::new(harness::AcpHarness::opencode()),
        Some("acp-gemini") => Box::new(harness::AcpHarness::custom(harness::AcpHarnessConfig {
            id: "gemini".to_owned(),
            display_name: "Gemini".to_owned(),
            command: "gemini".to_owned(),
            args: vec!["--experimental-acp".to_owned()],
        })),
        _ => {
            let session_dir = std::env::temp_dir().join("openai-compatible-playground");
            let base = match get("base") {
                Some(url) => OpenHarness::custom(harness::OpenHarnessConfig {
                    id: "custom".to_owned(),
                    display_name: "Custom".to_owned(),
                    base_url: url,
                    ..Default::default()
                }),
                None => OpenHarness::ollama(),
            };
            Box::new(
                base.with_session_dir(session_dir)
                    .with_permission_rule(PermissionRule::deny_matching("bash", "rm -rf"))
                    .with_permission_rule(PermissionRule::deny_matching("bash", "sudo")),
            )
        }
    }
}

/// Run a request on any harness and stream its `RunEvent`s to the browser as SSE
/// (`data: {json}\n\n` per event), ending after `Exited`. Generic over the
/// harness, so archetype A (`OpenHarness`) and archetype B
/// (`AcpHarness`) share one path.
fn respond_stream(harness: &dyn Harness, req: RunRequest, request: tiny_http::Request) {
    match harness.run_channel(req) {
        Ok((_handle, rx)) => {
            let headers = vec![
                header("Content-Type", "text/event-stream"),
                header("Cache-Control", "no-cache"),
                header("Access-Control-Allow-Origin", "*"),
            ];
            // `data_length: None` → chunked transfer, which the browser's
            // EventSource consumes incrementally. `respond` blocks while the body
            // streams (so `_handle` outlives the run).
            let _ = request.respond(Response::new(StatusCode(200), headers, SseBody::new(rx), None, None));
        }
        Err(e) => {
            let _ = request.respond(Response::from_string(format!("run failed: {e}")).with_status_code(500));
        }
    }
}

/// `GET /models?harness=..&base=..` — the model list for the chosen harness, via
/// its `Harness::list_models()`: Ollama → `/api/tags` (installed models);
/// opencode → `opencode models`; a generic ACP agent has none. Replies
/// `{ models: [..] }`.
fn list_models(request: tiny_http::Request) {
    let params = parse_query(request.url().split_once('?').map_or("", |(_, q)| q));
    let models: Vec<String> = build_harness(&params)
        .list_models()
        .map(|ms| ms.into_iter().map(|m| m.value).collect())
        .unwrap_or_default();
    let body = serde_json::json!({ "models": models });
    let resp = Response::from_string(body.to_string()).with_header(header("Content-Type", "application/json"));
    let _ = request.respond(resp);
}

/// `GET /dirs?path=<abs>` — list a directory's sub-folders for the folder picker,
/// defaulting to `$HOME`. Replies `{ path, parent, dirs }` as JSON.
fn list_dirs(request: tiny_http::Request) {
    let params = parse_query(request.url().split_once('?').map_or("", |(_, q)| q));
    let path = params.get("path").filter(|s| !s.is_empty()).map(PathBuf::from).unwrap_or_else(home_dir);
    let mut dirs: Vec<String> = std::fs::read_dir(&path)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|name| !name.starts_with('.'))
        .collect();
    dirs.sort();
    let body = serde_json::json!({
        "path": path.to_string_lossy(),
        "parent": path.parent().map(|p| p.to_string_lossy().into_owned()),
        "dirs": dirs,
    });
    let resp = Response::from_string(body.to_string()).with_header(header("Content-Type", "application/json"));
    let _ = request.respond(resp);
}

/// `GET /file?path=<rel|abs>&cwd=<dir>` — read a file the agent created/edited so
/// the page can preview it (the "artifact" the chat links to). Scoped to the
/// run's cwd (same default as `/run`): the resolved path must stay inside it.
/// Replies `{ path, content }` or `{ path, error }`.
fn read_file(request: tiny_http::Request) {
    let params = parse_query(request.url().split_once('?').map_or("", |(_, q)| q));
    let get = |k: &str| params.get(k).filter(|s| !s.is_empty()).cloned();
    let cwd = get("cwd").map(PathBuf::from).unwrap_or_else(scratch_workspace);
    let path = get("path").unwrap_or_default();
    let body = match read_file_scoped(&cwd, &path) {
        Ok(content) => serde_json::json!({ "path": path, "content": content }),
        Err(error) => serde_json::json!({ "path": path, "error": error }),
    };
    let resp = Response::from_string(body.to_string()).with_header(header("Content-Type", "application/json"));
    let _ = request.respond(resp);
}

/// Read a file for preview, refusing anything outside `cwd` (the run dir). `path`
/// may be relative to `cwd` or absolute; either way the canonicalized result must
/// stay within `cwd`, so a crafted `..`/abs path can't read the rest of the disk.
/// Caps the returned text so a huge file can't flood the page.
fn read_file_scoped(cwd: &Path, path: &str) -> Result<String, String> {
    let candidate =
        if Path::new(path).is_absolute() { PathBuf::from(path) } else { cwd.join(path) };
    let canon = candidate.canonicalize().map_err(|e| format!("can't open {path}: {e}"))?;
    let root = cwd.canonicalize().map_err(|e| format!("bad working dir: {e}"))?;
    if !canon.starts_with(&root) {
        return Err("refusing to read outside the run directory".to_owned());
    }
    if canon.is_dir() {
        return Err("that path is a directory, not a file".to_owned());
    }
    let mut content = std::fs::read_to_string(&canon).map_err(|e| format!("can't read {path}: {e}"))?;
    const CAP: usize = 256 * 1024;
    if content.len() > CAP {
        content.truncate(CAP);
        content.push_str("\n\n… (truncated for preview)");
    }
    Ok(content)
}

/// A `Read` that turns the `RunEvent` receiver into an SSE byte stream: each read
/// blocks for the next event, formats it as one `data:` frame, and yields EOF
/// once `Exited` has been sent (or the sender drops).
struct SseBody {
    rx: Receiver<RunEvent>,
    buf: Vec<u8>,
    pos: usize,
    done: bool,
}

impl SseBody {
    fn new(rx: Receiver<RunEvent>) -> Self {
        Self { rx, buf: Vec::new(), pos: 0, done: false }
    }
}

impl Read for SseBody {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.buf.len() {
            if self.done {
                return Ok(0);
            }
            match self.rx.recv() {
                Ok(event) => {
                    self.done = matches!(event, RunEvent::Exited { .. });
                    let json = serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_owned());
                    self.buf = format!("data: {json}\n\n").into_bytes();
                    self.pos = 0;
                }
                Err(_) => return Ok(0), // sender dropped — run finished
            }
        }
        let n = (self.buf.len() - self.pos).min(out.len());
        out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

fn header(key: &str, value: &str) -> Header {
    Header::from_bytes(key.as_bytes(), value.as_bytes()).expect("valid header")
}

/// A throwaway working directory under the temp dir — the default, so a run in
/// Edit mode can't touch real files unless the user explicitly sets a path.
fn scratch_workspace() -> PathBuf {
    let dir = std::env::temp_dir().join("openai-compatible-playground").join("workspace");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/"))
}

/// Parse a URL query string into a map, percent-decoding values.
fn parse_query(query: &str) -> HashMap<String, String> {
    query
        .split('&')
        .filter_map(|kv| kv.split_once('='))
        .map(|(k, v)| (k.to_owned(), percent_decode(v)))
        .collect()
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                match std::str::from_utf8(&bytes[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok()) {
                    Some(byte) => {
                        out.push(byte);
                        i += 3;
                    }
                    None => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            byte => {
                out.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}
