//! Ollama's native HTTP API (`/api/chat`, `/api/tags`, `/api/show`).
//!
//! Ollama also speaks the OpenAI `/v1/chat/completions` shape ([`super::wire`]),
//! but that endpoint silently ignores `num_ctx` and loads every model at its
//! 4096-token default — which truncates our system prompt + tool schemas and
//! breaks tool calling. The native `/api/chat` endpoint accepts
//! `options.num_ctx`, so for local models we talk to it directly and load the
//! model's real context window. The native wire format differs from OpenAI's in
//! three ways this module bridges: the stream is newline-delimited JSON (not SSE
//! `data:` frames), tool-call `arguments` are a JSON *object* both ways (OpenAI
//! uses a string), and usage is reported as `prompt_eval_count`/`eval_count`.

use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::{ModelChoice, InstalledModel, PullProgress};

use super::wire::{ChatMessage, Fragment, FunctionCall, ThinkSplitter, ToolCall, Usage};

/// Keep a loaded model resident between turns so it isn't reloaded each request.
pub(super) const KEEP_ALIVE: &str = "5m";
/// Temperature for the agent loop: deterministic tool selection beats creativity
/// for a coding/notes assistant, and temp 0 is the documented recommendation for
/// reliable tool/structured calls on small local models.
pub(super) const TEMPERATURE: f64 = 0.0;

/// Translate our OpenAI-shaped [`ChatMessage`]s into native request messages.
/// The one shape difference that matters: tool-call `arguments` must be a JSON
/// *object* (native rejects the OpenAI string form), so the stored string is
/// parsed back. Inline images ride on the first user message's `images` array as
/// raw base64 (the part after the data-URI comma).
pub(super) fn to_native_messages(messages: &[ChatMessage], image_data_uris: &[String]) -> Vec<Value> {
    let mut out: Vec<Value> = messages
        .iter()
        .map(|m| {
            let mut obj = serde_json::Map::new();
            obj.insert("role".into(), json!(m.role));
            obj.insert("content".into(), json!(m.content.clone().unwrap_or_default()));
            if !m.tool_calls.is_empty() {
                let calls: Vec<Value> = m
                    .tool_calls
                    .iter()
                    .map(|tc| {
                        let args: Value = serde_json::from_str(&tc.function.arguments).unwrap_or_else(|_| json!({}));
                        json!({ "id": tc.id, "function": { "name": tc.function.name, "arguments": args } })
                    })
                    .collect();
                obj.insert("tool_calls".into(), Value::Array(calls));
            }
            if let Some(id) = &m.tool_call_id {
                obj.insert("tool_call_id".into(), json!(id));
            }
            Value::Object(obj)
        })
        .collect();
    attach_images(&mut out, image_data_uris);
    out
}

/// Attach inline images to the first user message as native `images` (base64).
fn attach_images(messages: &mut [Value], uris: &[String]) {
    if uris.is_empty() {
        return;
    }
    let images: Vec<Value> = uris.iter().filter_map(|u| u.split_once(',').map(|(_, b64)| json!(b64))).collect();
    if let Some(first_user) = messages.iter_mut().find(|m| m["role"] == "user").and_then(Value::as_object_mut) {
        first_user.insert("images".into(), Value::Array(images));
    }
}

/// Pull the bare JSON Schema out of an OpenAI `response_format` wrapper for
/// native `format`; `None` when the shape isn't a json_schema wrapper.
pub(super) fn native_format(response_format: &Value) -> Option<Value> {
    response_format.get("json_schema")?.get("schema").cloned()
}

/// Parse a native NDJSON chat stream into the assembled assistant message +
/// usage, invoking `on_delta` per fragment. Each line is a complete JSON object;
/// content streams incrementally, tool calls and usage arrive whole.
pub(super) fn drain_native_stream(
    lines: impl Iterator<Item = String>,
    reasoning_tag: Option<&str>,
    cancel: &AtomicBool,
    mut on_delta: impl FnMut(Fragment),
) -> Result<(ChatMessage, Option<Usage>), String> {
    let mut content = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut usage = None;
    let mut think = ThinkSplitter::new(reasoning_tag);
    for line in lines {
        // Stop-button responsiveness: a set flag ends the read NOW — dropping
        // the reader hangs up the connection, which tells the server to stop
        // generating (#115; mirrors drain_pull_stream).
        if cancel.load(Ordering::SeqCst) {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let chunk: NativeChunk = match serde_json::from_str(line) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if let Some(error) = chunk.error {
            return Err(error);
        }
        if let Some(message) = chunk.message {
            if let Some(text) = message.content.filter(|t| !t.is_empty()) {
                // Models that inline reasoning as `<think>…</think>` route it to
                // the reasoning stream; only visible text folds into the message.
                content.push_str(&think.feed(&text, &mut on_delta));
            }
            // Thinking models that expose reasoning in a dedicated field.
            if let Some(thinking) = message.thinking.filter(|t| !t.is_empty()) {
                on_delta(Fragment::Reasoning(&thinking));
            }
            for tc in message.tool_calls {
                // OpenAI encodes call arguments as a string; we store them that
                // way, so re-serialize the native object form.
                let arguments = serde_json::to_string(&tc.function.arguments).unwrap_or_else(|_| "{}".to_owned());
                let id = tc.id.filter(|s| !s.is_empty()).unwrap_or_else(|| format!("call_{}", tool_calls.len()));
                tool_calls.push(ToolCall {
                    id,
                    kind: "function".to_owned(),
                    function: FunctionCall { name: tc.function.name, arguments },
                });
            }
        }
        if chunk.done {
            usage = native_usage(chunk.prompt_eval_count, chunk.eval_count);
        }
    }
    content.push_str(&think.finish(&mut on_delta));
    let message = ChatMessage {
        role: "assistant".to_owned(),
        content: (!content.is_empty()).then_some(content),
        tool_calls,
        tool_call_id: None,
    };
    Ok((message, usage))
}

/// Map native token counters onto the OpenAI [`Usage`] shape (no cache counters).
fn native_usage(prompt: Option<u64>, eval: Option<u64>) -> Option<Usage> {
    if prompt.is_none() && eval.is_none() {
        return None;
    }
    Some(Usage {
        prompt_tokens: prompt,
        completion_tokens: eval,
        total_tokens: Some(prompt.unwrap_or(0) + eval.unwrap_or(0)),
        prompt_tokens_details: None,
        cache_read_input_tokens: None,
        cache_creation_input_tokens: None,
    })
}

#[derive(Deserialize)]
struct NativeChunk {
    #[serde(default)]
    message: Option<NativeMessage>,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    #[serde(default)]
    eval_count: Option<u64>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize)]
struct NativeMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    tool_calls: Vec<NativeToolCall>,
}

#[derive(Deserialize)]
struct NativeToolCall {
    #[serde(default)]
    id: Option<String>,
    function: NativeFunctionCall,
}

#[derive(Deserialize)]
struct NativeFunctionCall {
    name: String,
    #[serde(default)]
    arguments: Value,
}

/// GET `{base}/api/tags` — Ollama's installed-model list — mapping each model
/// name to a [`ModelChoice`] for the picker. (Ollama lists models under
/// `/api/tags`, not under `/v1`.)
pub(crate) fn list_tags(base: &str) -> Result<Vec<ModelChoice>, String> {
    let url = format!("{base}/api/tags");
    let resp = ureq::get(&url)
        .timeout(Duration::from_secs(5))
        .call()
        .map_err(|e| format!("model list from {url} failed: {e}"))?;
    let body: Value = resp.into_json().map_err(|e| format!("decoding model list from {url}: {e}"))?;
    let models = body
        .get("models")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("name").and_then(Value::as_str))
                .map(|name| ModelChoice { value: name.to_owned(), label: name.to_owned() })
                .collect()
        })
        .unwrap_or_default();
    Ok(models)
}

/// POST `{base}/api/show` once and read both facts it carries about a model:
/// its context window and its parameter count. Best-effort — any failure
/// yields `(None, None)`, and each half is independently optional.
///
/// One call for both because they answer different questions and the loop needs
/// both: the window decides what a run can *afford* to send, the parameter
/// count how much the model can be *trusted* to do with it.
pub(crate) fn model_facts(base: &str, model: &str) -> (Option<u64>, Option<f64>) {
    let url = format!("{base}/api/show");
    let Ok(resp) = ureq::post(&url).timeout(Duration::from_secs(10)).send_json(json!({ "model": model }))
    else {
        return (None, None);
    };
    let Ok(body) = resp.into_json::<Value>() else { return (None, None) };
    (context_length_from_show(&body), parameters_from_show(&body))
}

/// A model's parameter count in billions, from `/api/show`. Prefers the exact
/// `general.parameter_count`, falling back to the human `details.parameter_size`
/// (`"7.6B"`, `"800M"`) that older Ollama builds report instead.
fn parameters_from_show(body: &Value) -> Option<f64> {
    let exact = body
        .get("model_info")
        .and_then(|info| info.get("general.parameter_count"))
        .and_then(Value::as_u64);
    if let Some(count) = exact {
        return Some(count as f64 / 1e9);
    }
    let text = body.get("details")?.get("parameter_size")?.as_str()?.trim();
    let (digits, scale) = match text.chars().last() {
        Some('B' | 'b') => (&text[..text.len() - 1], 1.0),
        Some('M' | 'm') => (&text[..text.len() - 1], 0.001),
        _ => (text, 1.0),
    };
    digits.trim().parse::<f64>().ok().map(|n| n * scale)
}

/// Read a model's context window from an `/api/show` body: the value lives in
/// `model_info` under an architecture-keyed `<arch>.context_length` (e.g.
/// `qwen2.context_length`), so we scan for that suffix.
fn context_length_from_show(body: &Value) -> Option<u64> {
    body.get("model_info")
        .and_then(Value::as_object)?
        .iter()
        .find(|(k, _)| k.ends_with(".context_length"))
        .and_then(|(_, v)| v.as_u64())
}

/// GET `{base}/api/tags` parsed for the manager: every installed model with its
/// on-disk size and `details`. ([`list_tags`] returns the picker's leaner
/// name-only shape from the same endpoint.)
pub(crate) fn list_installed(base: &str) -> Result<Vec<InstalledModel>, String> {
    let url = format!("{base}/api/tags");
    let resp = ureq::get(&url)
        .timeout(Duration::from_secs(5))
        .call()
        .map_err(|e| format!("model list from {url} failed: {e}"))?;
    let body: Value = resp.into_json().map_err(|e| format!("decoding model list from {url}: {e}"))?;
    Ok(body.get("models").and_then(Value::as_array).map(|arr| arr.iter().map(installed_from_tag).collect()).unwrap_or_default())
}

/// Map one `/api/tags` entry onto an [`InstalledModel`]; missing fields default
/// (a tag with no `details` still lists, just without parameter/quant labels).
fn installed_from_tag(model: &Value) -> InstalledModel {
    let details = model.get("details");
    let detail_str = |key: &str| details.and_then(|d| d.get(key)).and_then(Value::as_str).map(str::to_owned);
    InstalledModel {
        name: model.get("name").and_then(Value::as_str).unwrap_or_default().to_owned(),
        size: model.get("size").and_then(Value::as_u64).unwrap_or(0),
        parameter_size: detail_str("parameter_size"),
        quantization_level: detail_str("quantization_level"),
    }
}

/// POST `{base}/api/pull` and stream the download, invoking `on_progress` per
/// NDJSON line. Returns `Ok(())` once a `"success"` line arrives; an `{"error"}`
/// line or non-2xx response is an `Err`. `cancel` is polled between lines —
/// flipping it drops the connection, which tells Ollama to stop the pull.
pub(crate) fn pull(base: &str, model: &str, cancel: &AtomicBool, mut on_progress: impl FnMut(PullProgress)) -> Result<(), String> {
    let url = format!("{base}/api/pull");
    let body = json!({ "model": model, "stream": true });
    let resp = ureq::post(&url).send_json(body).map_err(|e| format!("pull request to {url} failed: {e}"))?;
    let reader = BufReader::new(resp.into_reader());
    drain_pull_stream(reader.lines().map_while(Result::ok), cancel, &mut on_progress)
}

/// Parse a `/api/pull` NDJSON stream: emit a [`PullProgress`] per line, stop on
/// `"success"`, surface an `{"error"}` line, and abort early when `cancel` is
/// set (dropping the reader hangs up the connection).
fn drain_pull_stream(lines: impl Iterator<Item = String>, cancel: &AtomicBool, on_progress: &mut impl FnMut(PullProgress)) -> Result<(), String> {
    let mut saw_success = false;
    for line in lines {
        if cancel.load(Ordering::SeqCst) {
            return Err("Download cancelled.".to_owned());
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let chunk: PullChunk = match serde_json::from_str(line) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if let Some(error) = chunk.error {
            return Err(error);
        }
        let status = chunk.status.unwrap_or_default();
        saw_success = saw_success || status == "success";
        on_progress(PullProgress { status, digest: chunk.digest, total: chunk.total, completed: chunk.completed });
    }
    if saw_success {
        Ok(())
    } else {
        // The stream ended without a `success` line — the connection dropped
        // mid-pull (server died, network cut) rather than completing.
        Err("Download ended before completing.".to_owned())
    }
}

/// DELETE `{base}/api/delete` to remove an installed model. A 404 (model already
/// absent) is treated as success — the end state the caller wanted.
pub(crate) fn delete(base: &str, model: &str) -> Result<(), String> {
    let url = format!("{base}/api/delete");
    match ureq::delete(&url).send_json(json!({ "model": model })) {
        Ok(_) => Ok(()),
        Err(ureq::Error::Status(404, _)) => Ok(()),
        Err(e) => Err(format!("deleting {model} failed: {e}")),
    }
}

#[derive(Deserialize)]
struct PullChunk {
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    digest: Option<String>,
    #[serde(default)]
    total: Option<u64>,
    #[serde(default)]
    completed: Option<u64>,
    #[serde(default)]
    error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drains_native_stream_with_content_tools_and_usage() {
        let lines = vec![
            r#"{"message":{"role":"assistant","content":"2+2 is "},"done":false}"#.to_string(),
            r#"{"message":{"role":"assistant","content":"4","tool_calls":[{"id":"c1","function":{"name":"calc","arguments":{"expr":"2+2"}}}]},"done":false}"#.to_string(),
            r#"{"message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":10,"eval_count":5}"#.to_string(),
        ];
        let mut text = String::new();
        let (msg, usage) = drain_native_stream(lines.into_iter(), Some("think"), &AtomicBool::new(false), |f| {
            if let Fragment::Text(t) = f {
                text.push_str(t);
            }
        })
        .unwrap();
        assert_eq!(text, "2+2 is 4");
        assert_eq!(msg.content.as_deref(), Some("2+2 is 4"));
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].function.name, "calc");
        // Native arguments object is re-encoded as the OpenAI string form.
        assert_eq!(msg.tool_calls[0].function.arguments, r#"{"expr":"2+2"}"#);
        let usage = usage.unwrap();
        assert_eq!(usage.prompt_tokens, Some(10));
        assert_eq!(usage.completion_tokens, Some(5));
        assert_eq!(usage.total_tokens, Some(15));
    }

    #[test]
    fn native_stream_routes_inline_think_tags_to_reasoning() {
        let lines = vec![
            r#"{"message":{"role":"assistant","content":"Hi <think>plan</think>answer"},"done":true}"#.to_string(),
        ];
        let (mut text, mut reasoning) = (String::new(), String::new());
        let (msg, _) = drain_native_stream(lines.into_iter(), Some("think"), &AtomicBool::new(false), |f| match f {
            Fragment::Text(t) => text.push_str(t),
            Fragment::Reasoning(r) => reasoning.push_str(r),
        })
        .unwrap();
        assert_eq!(reasoning, "plan");
        assert_eq!(msg.content.as_deref(), Some("Hi answer"));
    }

    #[test]
    fn native_stream_surfaces_error_line() {
        let lines = vec![r#"{"error":"model not found"}"#.to_string()];
        let err = drain_native_stream(lines.into_iter(), None, &AtomicBool::new(false), |_| {}).unwrap_err();
        assert_eq!(err, "model not found");
    }

    #[test]
    fn to_native_messages_objectifies_tool_call_arguments() {
        let messages = vec![ChatMessage {
            role: "assistant".to_owned(),
            content: None,
            tool_calls: vec![ToolCall {
                id: "c1".to_owned(),
                function: FunctionCall { name: "calc".to_owned(), arguments: r#"{"expr":"2+2"}"#.to_owned() },
            }],
            tool_call_id: None,
        }];
        let native = to_native_messages(&messages, &[]);
        let args = &native[0]["tool_calls"][0]["function"]["arguments"];
        assert!(args.is_object(), "arguments must be an object for native Ollama, got {args}");
        assert_eq!(args["expr"], "2+2");
    }

    #[test]
    fn attaches_images_to_first_user_message() {
        let mut messages = vec![json!({ "role": "user", "content": "look" })];
        attach_images(&mut messages, &["data:image/png;base64,QUJD".to_owned()]);
        assert_eq!(messages[0]["images"][0], "QUJD");
    }

    #[test]
    fn native_format_unwraps_json_schema() {
        let rf = json!({ "type": "json_schema", "json_schema": { "name": "r", "schema": { "type": "object" } } });
        assert_eq!(native_format(&rf), Some(json!({ "type": "object" })));
        assert_eq!(native_format(&json!({ "type": "text" })), None);
    }

    #[test]
    fn reads_context_length_from_model_info() {
        let body = json!({
            "model_info": {
                "general.architecture": "qwen2",
                "qwen2.context_length": 32768,
                "qwen2.embedding_length": 3584
            }
        });
        assert_eq!(context_length_from_show(&body), Some(32768));
        assert_eq!(context_length_from_show(&json!({ "model_info": {} })), None);
        assert_eq!(context_length_from_show(&json!({})), None);
    }

    #[test]
    fn maps_tag_entry_to_installed_model() {
        let tag = json!({
            "name": "llama3.2:3b",
            "size": 2_019_393_189u64,
            "details": { "parameter_size": "3.2B", "quantization_level": "Q4_K_M" }
        });
        let model = installed_from_tag(&tag);
        assert_eq!(model.name, "llama3.2:3b");
        assert_eq!(model.size, 2_019_393_189);
        assert_eq!(model.parameter_size.as_deref(), Some("3.2B"));
        assert_eq!(model.quantization_level.as_deref(), Some("Q4_K_M"));
        // A tag without details still lists, sans the optional labels.
        let bare = installed_from_tag(&json!({ "name": "x:1b", "size": 10 }));
        assert_eq!(bare.parameter_size, None);
        assert_eq!(bare.quantization_level, None);
    }

    #[test]
    fn drains_pull_stream_to_success() {
        let lines = vec![
            r#"{"status":"pulling manifest"}"#.to_string(),
            r#"{"status":"pulling sha256:a","digest":"sha256:a","total":100,"completed":40}"#.to_string(),
            r#"{"status":"pulling sha256:a","digest":"sha256:a","total":100,"completed":100}"#.to_string(),
            r#"{"status":"success"}"#.to_string(),
        ];
        let cancel = AtomicBool::new(false);
        let mut seen: Vec<PullProgress> = Vec::new();
        drain_pull_stream(lines.into_iter(), &cancel, &mut |p| seen.push(p)).unwrap();
        assert_eq!(seen.len(), 4);
        assert_eq!(seen[0].status, "pulling manifest");
        assert_eq!(seen.last().unwrap().status, "success");
    }

    #[test]
    fn pull_stream_surfaces_error_line() {
        let lines = vec![r#"{"error":"file does not exist"}"#.to_string()];
        let cancel = AtomicBool::new(false);
        let err = drain_pull_stream(lines.into_iter(), &cancel, &mut |_| {}).unwrap_err();
        assert_eq!(err, "file does not exist");
    }

    #[test]
    fn pull_stream_without_success_is_an_error() {
        // A dropped connection: progress arrives but no `success` line.
        let lines = vec![r#"{"status":"pulling sha256:a","digest":"sha256:a","total":100,"completed":40}"#.to_string()];
        let cancel = AtomicBool::new(false);
        let err = drain_pull_stream(lines.into_iter(), &cancel, &mut |_| {}).unwrap_err();
        assert!(err.contains("before completing"), "got: {err}");
    }

    #[test]
    fn pull_stream_aborts_when_cancelled() {
        let lines = vec![
            r#"{"status":"pulling sha256:a","digest":"sha256:a","total":100,"completed":40}"#.to_string(),
            r#"{"status":"success"}"#.to_string(),
        ];
        let cancel = AtomicBool::new(true);
        let err = drain_pull_stream(lines.into_iter(), &cancel, &mut |_| {}).unwrap_err();
        assert_eq!(err, "Download cancelled.");
    }

    #[test]
    fn drain_native_stream_stops_within_one_chunk_of_cancel() {
        // #115 — same contract as the OpenAI-shape drain.
        let cancel = AtomicBool::new(false);
        let lines: Vec<String> = (0..100)
            .map(|i| format!(r#"{{"message":{{"role":"assistant","content":"c{i}"}},"done":false}}"#))
            .collect();
        let mut seen = 0;
        let out = drain_native_stream(
            lines.into_iter().inspect(|_| {
                seen += 1;
                cancel.store(true, Ordering::SeqCst);
            }),
            None,
            &cancel,
            |_| {},
        );
        let (msg, _) = out.expect("early stop is not an error");
        assert_eq!(seen, 1, "the poll after the first pull saw the flag");
        assert!(msg.content.as_deref().unwrap_or("").is_empty());
    }
}
