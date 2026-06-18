//! The OpenAI-compatible chat wire format + the blocking HTTP calls.
//!
//! Works against any endpoint that speaks the OpenAI
//! `/v1/chat/completions` shape — Ollama (`http://localhost:11434/v1`),
//! OpenRouter, vLLM, LM Studio — plus Ollama's native `/api/tags` for
//! model discovery. HTTP is blocking (`ureq`), driven from the worker
//! thread `run()` spawns; errors come back as `String` and the loop turns
//! them into a [`crate::RunEvent::Error`].

use std::io::{BufRead, BufReader};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::HarnessModel;

/// One chat message, in either direction (request history or response).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ChatMessage {
    pub role: String,
    /// Absent on an assistant turn that only calls tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool calls the assistant requested this turn.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// Set on a `role:"tool"` result message, matching the call's id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: Some(content.into()), tool_calls: Vec::new(), tool_call_id: None }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: Some(content.into()), tool_calls: Vec::new(), tool_call_id: None }
    }
    /// A tool result fed back to the model, keyed to the call it answers.
    pub fn tool_result(tool_call_id: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: Some(output.into()),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// A tool call the assistant emitted (OpenAI shape).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ToolCall {
    #[serde(default)]
    pub id: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct FunctionCall {
    pub name: String,
    /// Arguments as a JSON *string* (OpenAI encodes them this way) — parse
    /// with `serde_json` before use.
    #[serde(default)]
    pub arguments: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatResponse {
    #[serde(default)]
    pub choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Choice {
    pub message: ChatMessage,
}

/// Token usage in the OpenAI shape, plus prompt-cache counters when the provider
/// reports them: `prompt_tokens_details.cached_tokens` (OpenAI/DeepSeek
/// auto-caching) or `cache_read_input_tokens`/`cache_creation_input_tokens`
/// (Anthropic-compatible). Mapped onto `RunEvent::Usage` by the loop.
#[derive(Debug, Deserialize)]
pub(crate) struct Usage {
    #[serde(default)]
    pub prompt_tokens: Option<u64>,
    #[serde(default)]
    pub completion_tokens: Option<u64>,
    #[serde(default)]
    pub total_tokens: Option<u64>,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,
}

impl Usage {
    /// Prompt tokens served from cache this turn, across provider shapes.
    pub(crate) fn cache_read(&self) -> Option<u64> {
        self.cache_read_input_tokens
            .or_else(|| self.prompt_tokens_details.as_ref().and_then(|d| d.cached_tokens))
    }
    /// Prompt tokens written to cache this turn (Anthropic-style).
    pub(crate) fn cache_write(&self) -> Option<u64> {
        self.cache_creation_input_tokens
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

/// Max retry attempts for a transient chat-request failure.
const MAX_RETRIES: u32 = 3;

/// Send a request, retrying transient failures (HTTP 429 / 5xx, connection
/// errors) with exponential backoff that honors `Retry-After` — OpenCode's retry
/// policy. A 4xx (bad request, auth, context-overflow) is terminal.
fn send_with_retry(url: &str, make: impl Fn() -> Result<ureq::Response, Box<ureq::Error>>) -> Result<ureq::Response, String> {
    let mut attempt = 0u32;
    loop {
        match make() {
            Ok(resp) => return Ok(resp),
            Err(e) if attempt < MAX_RETRIES && is_retryable(&e) => {
                let backoff = retry_after(&e).unwrap_or_else(|| Duration::from_millis(1000 * 2u64.pow(attempt)));
                std::thread::sleep(backoff);
                attempt += 1;
            }
            Err(e) => return Err(format!("chat request to {url} failed: {e}")),
        }
    }
}

fn is_retryable(e: &ureq::Error) -> bool {
    match e {
        ureq::Error::Status(code, _) => status_is_retryable(*code),
        ureq::Error::Transport(_) => true, // connection-level failure
    }
}

fn status_is_retryable(code: u16) -> bool {
    code == 429 || (500..=599).contains(&code)
}

fn retry_after(e: &ureq::Error) -> Option<Duration> {
    match e {
        ureq::Error::Status(_, resp) => {
            resp.header("Retry-After").and_then(|v| v.parse::<u64>().ok()).map(Duration::from_secs)
        }
        ureq::Error::Transport(_) => None,
    }
}

/// POST `{base}/v1/chat/completions` (blocking). `base` carries no trailing
/// slash. `tools` is the OpenAI `tools` array (built in [`super::tools`]);
/// when empty it's omitted.
pub(crate) fn post_chat(
    base: &str,
    api_key: Option<&str>,
    model: &str,
    messages: &[ChatMessage],
    tools: &[Value],
) -> Result<ChatResponse, String> {
    let url = format!("{base}/v1/chat/completions");
    let mut body = json!({
        "model": model,
        "messages": messages,
        "stream": false,
    });
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools.to_vec());
    }
    let resp = send_with_retry(&url, || {
        let mut req = ureq::post(&url);
        if let Some(key) = api_key {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        req.send_json(body.clone()).map_err(Box::new)
    })?;
    resp.into_json::<ChatResponse>()
        .map_err(|e| format!("decoding chat response from {url}: {e}"))
}

/// A streamed fragment handed to the caller as it arrives: assistant text, or
/// model reasoning (which the host renders distinctly from the answer).
pub(crate) enum Fragment<'a> {
    Text(&'a str),
    Reasoning(&'a str),
}

/// Stream `{base}/v1/chat/completions` (SSE). Calls `on_delta` for each text /
/// reasoning fragment as it arrives, accumulates the full assistant message
/// (content + tool calls) and any usage, and returns them. Blocking — driven on
/// the worker thread, like [`post_chat`].
pub(crate) fn post_chat_stream(
    base: &str,
    api_key: Option<&str>,
    model: &str,
    messages: &[ChatMessage],
    tools: &[Value],
    extras: RequestExtras,
    on_delta: impl FnMut(Fragment),
) -> Result<(ChatMessage, Option<Usage>), String> {
    let url = format!("{base}/v1/chat/completions");
    let mut body = json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "stream_options": { "include_usage": true },
    });
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools.to_vec());
    }
    if let Some(rf) = extras.response_format {
        body["response_format"] = rf.clone();
    }
    if !extras.image_data_uris.is_empty() {
        attach_images(&mut body, extras.image_data_uris);
    }
    let resp = send_with_retry(&url, || {
        let mut req = ureq::post(&url);
        if let Some(key) = api_key {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        req.send_json(body.clone()).map_err(Box::new)
    })?;
    let reader = BufReader::new(resp.into_reader());
    Ok(drain_stream(reader.lines().map_while(Result::ok), extras.reasoning_tag, on_delta))
}

/// Optional request shaping beyond messages + tools, bundled so `post_chat_stream`
/// stays within its argument budget.
#[derive(Default)]
pub(crate) struct RequestExtras<'a> {
    /// OpenAI `response_format` for structured output, if any.
    pub response_format: Option<&'a Value>,
    /// Image data URIs to attach to the first user message (multimodal input).
    pub image_data_uris: &'a [String],
    /// Inline reasoning tag to lift out of the stream (e.g. `Some("think")` for
    /// `<think>…</think>`); `None` disables extraction. See [`ThinkSplitter`].
    pub reasoning_tag: Option<&'a str>,
}

/// Rewrite the first user message's content into a multimodal parts array — the
/// original text plus one `image_url` part per data URI (the OpenAI vision shape).
fn attach_images(body: &mut Value, uris: &[String]) {
    let Some(messages) = body["messages"].as_array_mut() else { return };
    let Some(first_user) = messages.iter_mut().find(|m| m["role"] == "user") else { return };
    let text = first_user["content"].as_str().unwrap_or_default().to_owned();
    let mut parts = vec![json!({ "type": "text", "text": text })];
    for uri in uris {
        parts.push(json!({ "type": "image_url", "image_url": { "url": uri } }));
    }
    first_user["content"] = Value::Array(parts);
}

/// Build a `data:` URI from raw bytes + MIME type, for inline image input.
pub(crate) fn image_data_uri(mime: &str, data: &[u8]) -> String {
    format!("data:{mime};base64,{}", base64_encode(data))
}

/// Standard base64 (RFC 4648, padded). Small enough to inline rather than pull a
/// dependency for one call site.
fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        let n = ((chunk[0] as u32) << 16) | ((b1 as u32) << 8) | (b2 as u32);
        out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 { ALPHABET[((n >> 6) & 63) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { ALPHABET[(n & 63) as usize] as char } else { '=' });
    }
    out
}

/// Parse an OpenAI SSE chat stream into the assembled assistant message + usage,
/// invoking `on_delta` per text fragment. Split out from the HTTP so it's unit-
/// testable without a live endpoint.
fn drain_stream(
    lines: impl Iterator<Item = String>,
    reasoning_tag: Option<&str>,
    mut on_delta: impl FnMut(Fragment),
) -> (ChatMessage, Option<Usage>) {
    let mut content = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut usage = None;
    let mut think = ThinkSplitter::new(reasoning_tag);
    for line in lines {
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let chunk: StreamChunk = match serde_json::from_str(data) {
            Ok(c) => c,
            Err(_) => continue, // skip keep-alives / non-JSON lines
        };
        if chunk.usage.is_some() {
            usage = chunk.usage;
        }
        if let Some(choice) = chunk.choices.into_iter().next() {
            if let Some(text) = choice.delta.content {
                if !text.is_empty() {
                    // Models that inline reasoning as `<think>…</think>` in the
                    // content get it routed to the reasoning stream; only the
                    // visible text folds into the message.
                    content.push_str(&think.feed(&text, &mut on_delta));
                }
            }
            // Other reasoning models stream it in a dedicated field
            // (`reasoning_content` on DeepSeek, `reasoning` on OpenRouter);
            // surface it distinctly but don't fold it into the message.
            if let Some(reasoning) = choice.delta.reasoning_content.or(choice.delta.reasoning) {
                if !reasoning.is_empty() {
                    on_delta(Fragment::Reasoning(&reasoning));
                }
            }
            for delta in choice.delta.tool_calls {
                accumulate_tool_call(&mut tool_calls, delta);
            }
        }
    }
    content.push_str(&think.finish(&mut on_delta));
    let message = ChatMessage {
        role: "assistant".to_owned(),
        content: (!content.is_empty()).then_some(content),
        tool_calls,
        tool_call_id: None,
    };
    (message, usage)
}

/// Splits streamed content into visible-output and `<think>…</think>` reasoning
/// runs — the inline form some models use (DeepSeek-R1, Qwen3, …) instead of a
/// dedicated `reasoning_content` field. State is carried across chunks because a
/// tag can straddle two SSE deltas. Reasoning is surfaced via
/// [`Fragment::Reasoning`]; only visible text is returned, to fold into the
/// assistant message (the thinking is shown, not stored as the answer).
#[derive(Default)]
struct ThinkSplitter {
    /// `<tag>` / `</tag>` to match; empty when inactive.
    open: String,
    close: String,
    /// Whether extraction is on (off → content passes straight through as text).
    active: bool,
    in_think: bool,
    /// A trailing run that might be the start of a split tag, held until the next
    /// chunk confirms or denies it.
    carry: String,
}

impl ThinkSplitter {
    /// `Some("think")` extracts `<think>…</think>`; `None` (or empty) passes
    /// content through unchanged. The tag is configurable because the convention
    /// is model-specific (DeepSeek-R1/Qwen use `think`; others differ, and some
    /// expose reasoning via a field instead — handled separately).
    fn new(tag: Option<&str>) -> Self {
        match tag {
            Some(t) if !t.is_empty() => {
                Self { open: format!("<{t}>"), close: format!("</{t}>"), active: true, ..Self::default() }
            }
            _ => Self::default(),
        }
    }

    fn feed(&mut self, piece: &str, on: &mut impl FnMut(Fragment)) -> String {
        if !self.active {
            on(Fragment::Text(piece));
            return piece.to_owned();
        }
        let mut buf = std::mem::take(&mut self.carry);
        buf.push_str(piece);
        let mut visible = String::new();
        loop {
            // Cloned so `self` stays free to mutate `in_think`/`carry` below.
            let tag = if self.in_think { self.close.clone() } else { self.open.clone() };
            if let Some(i) = buf.find(&tag) {
                let before = &buf[..i];
                if !before.is_empty() {
                    if self.in_think {
                        on(Fragment::Reasoning(before));
                    } else {
                        on(Fragment::Text(before));
                        visible.push_str(before);
                    }
                }
                buf.replace_range(..i + tag.len(), "");
                self.in_think = !self.in_think;
            } else {
                // No complete tag: emit all but a possible partial-tag suffix.
                let cut = buf.len() - partial_tag_suffix_len(&buf, &tag);
                if cut > 0 {
                    let run = &buf[..cut];
                    if self.in_think {
                        on(Fragment::Reasoning(run));
                    } else {
                        on(Fragment::Text(run));
                        visible.push_str(run);
                    }
                }
                self.carry = buf[cut..].to_owned();
                break;
            }
        }
        visible
    }

    /// Flush any carried text at stream end (a dangling partial tag is shown as-is).
    fn finish(&mut self, on: &mut impl FnMut(Fragment)) -> String {
        let rest = std::mem::take(&mut self.carry);
        if rest.is_empty() {
            return String::new();
        }
        if self.in_think {
            on(Fragment::Reasoning(&rest));
            String::new()
        } else {
            on(Fragment::Text(&rest));
            rest
        }
    }
}

/// Length of the longest suffix of `buf` that is a prefix of `tag` — the bytes to
/// hold back in case the tag is being split across chunks. (Tags are ASCII, so a
/// matched suffix begins on a char boundary.)
fn partial_tag_suffix_len(buf: &str, tag: &str) -> usize {
    let b = buf.as_bytes();
    let max = tag.len().min(b.len());
    (1..=max).rev().find(|&n| tag.as_bytes().starts_with(&b[b.len() - n..])).unwrap_or(0)
}

/// Merge a streamed tool-call delta into the accumulating list by `index`: the
/// first fragment carries the id + function name, later fragments append
/// argument text.
fn accumulate_tool_call(calls: &mut Vec<ToolCall>, delta: DeltaToolCall) {
    while calls.len() <= delta.index {
        calls.push(ToolCall { id: String::new(), function: FunctionCall { name: String::new(), arguments: String::new() } });
    }
    let call = &mut calls[delta.index];
    if let Some(id) = delta.id.filter(|s| !s.is_empty()) {
        call.id = id;
    }
    if let Some(function) = delta.function {
        if let Some(name) = function.name.filter(|s| !s.is_empty()) {
            call.function.name = name;
        }
        if let Some(args) = function.arguments {
            call.function.arguments.push_str(&args);
        }
    }
}

#[derive(Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: Delta,
}

#[derive(Deserialize, Default)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Vec<DeltaToolCall>,
}

#[derive(Deserialize)]
struct DeltaToolCall {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<DeltaFunction>,
}

#[derive(Deserialize)]
struct DeltaFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// GET `{base}/api/tags` — Ollama's installed-model list — mapping each
/// model name to a [`HarnessModel`] for the picker. Ollama lists live under
/// `/api/tags`, *not* under `/v1`.
pub(crate) fn list_ollama_tags(base: &str) -> Result<Vec<HarnessModel>, String> {
    let url = format!("{base}/api/tags");
    let resp = ureq::get(&url)
        .call()
        .map_err(|e| format!("model list from {url} failed: {e}"))?;
    let body: Value = resp
        .into_json()
        .map_err(|e| format!("decoding model list from {url}: {e}"))?;
    let models = body
        .get("models")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("name").and_then(Value::as_str))
                .map(|name| HarnessModel { value: name.to_owned(), label: name.to_owned() })
                .collect()
        })
        .unwrap_or_default();
    Ok(models)
}

/// POST `{base}/api/show` — Ollama's per-model metadata — and read the model's
/// context-window size, so compaction can auto-configure for local models
/// without the host hardcoding it. Best-effort: any failure yields `None`
/// (compaction simply stays off, as it would without a configured limit).
pub(crate) fn ollama_context_length(base: &str, model: &str) -> Option<u64> {
    let url = format!("{base}/api/show");
    let resp = ureq::post(&url).timeout(Duration::from_secs(10)).send_json(json!({ "model": model })).ok()?;
    let body: Value = resp.into_json().ok()?;
    context_length_from_show(&body)
}

/// Read a model's context window from an Ollama `/api/show` body: the value
/// lives in `model_info` under an architecture-keyed `<arch>.context_length`
/// (e.g. `qwen2.context_length`), so we scan for that suffix.
fn context_length_from_show(body: &Value) -> Option<u64> {
    body.get("model_info")
        .and_then(Value::as_object)?
        .iter()
        .find(|(k, _)| k.ends_with(".context_length"))
        .and_then(|(_, v)| v.as_u64())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drain_stream_routes_inline_think_tags_to_reasoning() {
        // `<think>` straddles chunk boundaries; only visible text folds into the message.
        let lines = vec![
            r#"data: {"choices":[{"delta":{"content":"Hi <thi"}}]}"#.to_string(),
            r#"data: {"choices":[{"delta":{"content":"nk>secret pl"}}]}"#.to_string(),
            r#"data: {"choices":[{"delta":{"content":"an</think> answer"}}]}"#.to_string(),
            "data: [DONE]".to_string(),
        ];
        let (mut text, mut reasoning) = (String::new(), String::new());
        let (msg, _) = drain_stream(lines.into_iter(), Some("think"), |f| match f {
            Fragment::Text(t) => text.push_str(t),
            Fragment::Reasoning(r) => reasoning.push_str(r),
        });
        assert_eq!(reasoning, "secret plan");
        assert_eq!(text, "Hi  answer");
        assert_eq!(msg.content.as_deref(), Some("Hi  answer"));
    }

    #[test]
    fn drain_stream_passthrough_when_reasoning_disabled() {
        // With no tag configured, `<think>` is left in the visible output.
        let lines = vec![
            r#"data: {"choices":[{"delta":{"content":"<think>x</think>hi"}}]}"#.to_string(),
            "data: [DONE]".to_string(),
        ];
        let mut text = String::new();
        let (msg, _) = drain_stream(lines.into_iter(), None, |f| {
            if let Fragment::Text(t) = f {
                text.push_str(t);
            }
        });
        assert_eq!(text, "<think>x</think>hi");
        assert_eq!(msg.content.as_deref(), Some("<think>x</think>hi"));
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
    fn base64_and_image_data_uri() {
        // RFC 4648 test vectors, including 1- and 2-byte padding.
        assert_eq!(base64_encode(b"Man"), "TWFu");
        assert_eq!(base64_encode(b"Ma"), "TWE=");
        assert_eq!(base64_encode(b"M"), "TQ==");
        assert_eq!(image_data_uri("image/png", b"M"), "data:image/png;base64,TQ==");
    }

    #[test]
    fn attach_images_rewrites_first_user_message() {
        let mut body = json!({ "messages": [
            { "role": "system", "content": "sys" },
            { "role": "user", "content": "look at this" },
        ]});
        attach_images(&mut body, &["data:image/png;base64,TQ==".to_owned()]);
        let user = &body["messages"][1]["content"];
        assert_eq!(user[0], json!({ "type": "text", "text": "look at this" }));
        assert_eq!(user[1]["type"], "image_url");
        assert_eq!(user[1]["image_url"]["url"], "data:image/png;base64,TQ==");
        assert_eq!(body["messages"][0]["content"], "sys", "system message untouched");
    }

    #[test]
    fn drain_stream_assembles_text_tool_calls_and_usage() {
        let lines = vec![
            r#"data: {"choices":[{"delta":{"reasoning_content":"think"}}]}"#.to_string(),
            r#"data: {"choices":[{"delta":{"content":"Hel"}}]}"#.to_string(),
            r#"data: {"choices":[{"delta":{"content":"lo"}}]}"#.to_string(),
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read","arguments":"{\"pa"}}]}}]}"#.to_string(),
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"x\"}"}}]}}]}"#.to_string(),
            r#"data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}"#.to_string(),
            "data: [DONE]".to_string(),
        ];
        let mut text = String::new();
        let mut reasoning = String::new();
        let (msg, usage) = drain_stream(lines.into_iter(), Some("think"), |f| match f {
            Fragment::Text(t) => text.push_str(t),
            Fragment::Reasoning(r) => reasoning.push_str(r),
        });
        assert_eq!(reasoning, "think", "reasoning surfaced separately");
        assert_eq!(text, "Hello", "text deltas streamed live");
        assert_eq!(msg.content.as_deref(), Some("Hello"));
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].id, "c1");
        assert_eq!(msg.tool_calls[0].function.name, "read");
        assert_eq!(msg.tool_calls[0].function.arguments, r#"{"path":"x"}"#);
        assert_eq!(usage.unwrap().total_tokens, Some(8));
    }

    #[test]
    fn drain_stream_ignores_keepalives_and_tolerates_no_done() {
        let lines = vec![
            ": keep-alive".to_string(),
            String::new(),
            r#"data: {"choices":[{"delta":{"content":"hi"}}]}"#.to_string(),
        ];
        let (msg, usage) = drain_stream(lines.into_iter(), Some("think"), |_| {});
        assert_eq!(msg.content.as_deref(), Some("hi"));
        assert!(usage.is_none());
    }

    #[test]
    fn usage_reads_cache_tokens_across_shapes() {
        let openai: Usage =
            serde_json::from_str(r#"{"prompt_tokens":10,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":7}}"#).unwrap();
        assert_eq!(openai.cache_read(), Some(7));
        assert_eq!(openai.cache_write(), None);
        let anthropic: Usage =
            serde_json::from_str(r#"{"cache_read_input_tokens":3,"cache_creation_input_tokens":2}"#).unwrap();
        assert_eq!(anthropic.cache_read(), Some(3));
        assert_eq!(anthropic.cache_write(), Some(2));
    }

    #[test]
    fn retryable_status_classification() {
        assert!(status_is_retryable(429), "rate limit retries");
        assert!(status_is_retryable(500) && status_is_retryable(503), "5xx retries");
        assert!(!status_is_retryable(400) && !status_is_retryable(401) && !status_is_retryable(404), "4xx is terminal");
    }
}
