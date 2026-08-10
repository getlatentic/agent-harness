//! The OpenAI-compatible chat wire format, and the HTTP pieces shared with the
//! native Ollama path.
//!
//! The message and usage types here are the crate's internal currency: both
//! dialects in [`super::chat`] speak them, and [`super::ollama`] translates
//! to and from its own shape at the edge. This module also owns the parts
//! neither dialect should reimplement — the retry policy, the SSE drain, the
//! cache breakpoints, and [`ThinkSplitter`].
//!
//! HTTP is blocking (`ureq`), driven from the worker thread `run()` spawns;
//! errors come back as `String` and the loop turns them into a
//! [`crate::RunEvent::Error`].

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

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
/// policy. A 4xx (bad request, auth, context-overflow) is terminal. Shared with
/// the native Ollama path ([`super::ollama`]).
pub(super) fn send_with_retry(url: &str, make: impl Fn() -> Result<ureq::Response, Box<ureq::Error>>) -> Result<ureq::Response, String> {
    let mut attempt = 0u32;
    loop {
        match make() {
            Ok(resp) => return Ok(resp),
            Err(e) if attempt < MAX_RETRIES && is_retryable(&e) => {
                let backoff = retry_after(&e).unwrap_or_else(|| Duration::from_millis(1000 * 2u64.pow(attempt)));
                std::thread::sleep(backoff);
                attempt += 1;
            }
            Err(e) => return Err(describe_failure(url, e)),
        }
    }
}

/// Mark the cacheable prefix with Anthropic-style `cache_control` breakpoints.
///
/// Anthropic caches only what a request explicitly marks, unlike OpenAI and
/// DeepSeek which cache a matching prefix implicitly. Reached through an
/// OpenAI-compatible gateway — OpenRouter forwards the field — an unmarked
/// request re-charges the whole system prompt and tool block at full input
/// price on every turn of a conversation.
///
/// Three breakpoints, because a cache breakpoint covers everything *before* it
/// and nothing after:
///
/// 1. the last tool, covering the whole schema block;
/// 2. the system message;
/// 3. the last message of settled history.
///
/// The first two are the fixed prefix. The third is what makes a conversation
/// cheap: without it every turn re-sends the entire transcript at full price,
/// and the transcript is the part that grows. Placing it on the newest settled
/// message means the turn that just completed is cached for the next one.
///
/// Anthropic allows four; the fourth is left unspent rather than guessed at.
///
/// Carrying the field forces a message's content from a bare string into a
/// one-element text part — the only shape that has somewhere to put it.
pub(super) fn mark_cache_breakpoints(body: &mut Value) {
    let breakpoint = json!({ "type": "ephemeral" });

    if let Some(last) = body.get_mut("tools").and_then(Value::as_array_mut).and_then(|tools| tools.last_mut()) {
        last["cache_control"] = breakpoint.clone();
    }

    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else { return };
    if let Some(system) = messages.iter_mut().find(|m| m["role"] == "system") {
        mark_message(system, &breakpoint);
    }
    // The final message is the prompt just added, which by definition has never
    // been sent before — a breakpoint there would cache nothing and burn a slot.
    // The one before it ends the history both this turn and the next will share.
    if messages.len() >= 3 {
        let settled = messages.len() - 2;
        if messages[settled]["role"] != "system" {
            mark_message(&mut messages[settled], &breakpoint);
        }
    }
}

/// Attach `cache_control` to a message, promoting string content to a text part.
/// Leaves a message alone when its content is absent — a tool-call-only
/// assistant turn has no text to hang it on.
fn mark_message(message: &mut Value, breakpoint: &Value) {
    if let Some(text) = message["content"].as_str().map(str::to_owned) {
        message["content"] = json!([{ "type": "text", "text": text, "cache_control": breakpoint }]);
    }
}

/// Maximum characters of a provider's error body to quote.
const MAX_BODY_SNIPPET: usize = 500;

/// A failed request, carrying the provider's own explanation when it sent one.
///
/// `ureq`'s `Display` for a status error stops at the code, and the body is
/// where a provider names the field it rejected — which model id is unknown,
/// which tool schema it would not accept, why the key was refused. Discarding
/// it leaves a bare "status code 400" that could mean anything.
fn describe_failure(url: &str, error: Box<ureq::Error>) -> String {
    let ureq::Error::Status(code, response) = *error else {
        return format!("chat request to {url} failed: {error}");
    };
    let body = response.into_string().unwrap_or_default();
    let body = body.trim();
    if body.is_empty() {
        return format!("chat request to {url} failed: status {code}");
    }
    let mut snippet: String = body.chars().take(MAX_BODY_SNIPPET).collect();
    if body.chars().nth(MAX_BODY_SNIPPET).is_some() {
        snippet.push('…');
    }
    format!("chat request to {url} failed: status {code}: {snippet}")
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

/// A streamed fragment handed to the caller as it arrives: assistant text, or
/// model reasoning (which the host renders distinctly from the answer).
pub(crate) enum Fragment<'a> {
    Text(&'a str),
    Reasoning(&'a str),
}

/// Rewrite the first user message's content into a multimodal parts array — the
/// original text plus one `image_url` part per data URI (the OpenAI vision shape).
pub(super) fn attach_images(body: &mut Value, uris: &[String]) {
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
pub(super) fn drain_stream(
    lines: impl Iterator<Item = String>,
    reasoning_tag: Option<&str>,
    cancel: &AtomicBool,
    mut on_delta: impl FnMut(Fragment),
) -> (ChatMessage, Option<Usage>) {
    let mut content = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut usage = None;
    let mut think = ThinkSplitter::new(reasoning_tag);
    for line in lines {
        // Stop-button responsiveness (#115): end the read now; dropping the
        // reader hangs up, telling the server to stop generating.
        if cancel.load(Ordering::SeqCst) {
            break;
        }
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
pub(super) struct ThinkSplitter {
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
    pub(super) fn new(tag: Option<&str>) -> Self {
        match tag {
            Some(t) if !t.is_empty() => {
                Self { open: format!("<{t}>"), close: format!("</{t}>"), active: true, ..Self::default() }
            }
            _ => Self::default(),
        }
    }

    pub(super) fn feed(&mut self, piece: &str, on: &mut impl FnMut(Fragment)) -> String {
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
    pub(super) fn finish(&mut self, on: &mut impl FnMut(Fragment)) -> String {
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
        let (msg, _) = drain_stream(lines.into_iter(), Some("think"), &AtomicBool::new(false), |f| match f {
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
        let (msg, _) = drain_stream(lines.into_iter(), None, &AtomicBool::new(false), |f| {
            if let Fragment::Text(t) = f {
                text.push_str(t);
            }
        });
        assert_eq!(text, "<think>x</think>hi");
        assert_eq!(msg.content.as_deref(), Some("<think>x</think>hi"));
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
        let (msg, usage) = drain_stream(lines.into_iter(), Some("think"), &AtomicBool::new(false), |f| match f {
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
        let (msg, usage) = drain_stream(lines.into_iter(), Some("think"), &AtomicBool::new(false), |_| {});
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

    #[test]
    fn drain_stream_stops_within_one_chunk_of_cancel() {
        // #115: the Stop button sets this flag; the drain must quit reading
        // immediately instead of riding out the whole generation.
        let cancel = AtomicBool::new(false);
        let lines: Vec<String> = (0..100)
            .map(|i| format!(r#"data: {{"choices":[{{"delta":{{"content":"c{i}"}}}}]}}"#))
            .collect();
        let mut seen = 0;
        let (msg, _) = drain_stream(
            lines.into_iter().inspect(|_| {
                seen += 1;
                cancel.store(true, Ordering::SeqCst);
            }),
            None,
            &cancel,
            |_| {},
        );
        // The flag is polled before each pulled line is processed: exactly one
        // pull happens, its chunk is discarded, and the read ends.
        assert_eq!(seen, 1, "the poll after the first pull saw the flag");
        assert!(msg.content.as_deref().unwrap_or("").is_empty());
    }

    fn status_error(code: u16, body: &str) -> Box<ureq::Error> {
        let response = ureq::Response::new(code, "Bad Request", body).expect("build response");
        Box::new(ureq::Error::Status(code, response))
    }

    #[test]
    fn cache_breakpoints_land_on_the_tool_block_and_the_system_message() {
        let mut body = json!({
            "messages": [
                { "role": "system", "content": "the rules" },
                { "role": "user", "content": "hello" }
            ],
            "tools": [ { "function": { "name": "read" } }, { "function": { "name": "list" } } ]
        });
        mark_cache_breakpoints(&mut body);

        // The LAST tool, not the first: a breakpoint covers everything before
        // it, so marking the first would cache one schema and re-send the rest.
        assert_eq!(body["tools"][1]["cache_control"]["type"], "ephemeral");
        assert!(body["tools"][0]["cache_control"].is_null(), "only the last tool carries it");

        // A bare string has nowhere to hang the field, so the system content
        // becomes a one-element text part.
        assert_eq!(body["messages"][0]["content"][0]["text"], "the rules");
        assert_eq!(body["messages"][0]["content"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(body["messages"][1]["content"], "hello", "the only user turn is the new one");
    }

    #[test]
    fn history_gets_its_own_breakpoint_but_the_new_prompt_does_not() {
        // The prefix breakpoints cover a fixed ~1.2k tokens. The transcript is
        // the part that grows, so without a breakpoint in it a long
        // conversation re-sends everything at full price every turn.
        let mut body = json!({ "messages": [
            { "role": "system", "content": "rules" },
            { "role": "user", "content": "first question" },
            { "role": "assistant", "content": "first answer" },
            { "role": "user", "content": "the new prompt" }
        ]});
        mark_cache_breakpoints(&mut body);

        assert_eq!(body["messages"][0]["content"][0]["cache_control"]["type"], "ephemeral", "system");
        assert_eq!(
            body["messages"][2]["content"][0]["cache_control"]["type"], "ephemeral",
            "the last settled turn ends the history this and the next turn share"
        );
        assert_eq!(
            body["messages"][3]["content"], "the new prompt",
            "the newest message has never been sent, so caching it would burn a slot for nothing"
        );
        assert!(body["messages"][1]["cache_control"].is_null(), "one history breakpoint, not many");
    }

    #[test]
    fn a_first_turn_marks_only_the_prefix() {
        // system + the first prompt: there is no settled history to cache yet.
        let mut body = json!({ "messages": [
            { "role": "system", "content": "rules" },
            { "role": "user", "content": "hello" }
        ]});
        mark_cache_breakpoints(&mut body);
        assert_eq!(body["messages"][0]["content"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(body["messages"][1]["content"], "hello");
    }

    #[test]
    fn a_tool_call_turn_without_text_is_left_alone() {
        // An assistant turn that only calls tools has no content string to
        // promote; writing a parts array over it would drop the tool calls.
        let mut body = json!({ "messages": [
            { "role": "system", "content": "rules" },
            { "role": "assistant", "tool_calls": [ { "id": "1" } ] },
            { "role": "user", "content": "next" }
        ]});
        mark_cache_breakpoints(&mut body);
        assert!(body["messages"][1]["tool_calls"].is_array(), "tool calls survive");
        assert!(body["messages"][1]["content"].is_null(), "nothing invented to carry the field");
    }

    #[test]
    fn marking_a_request_without_tools_or_system_is_a_no_op() {
        let mut body = json!({ "messages": [ { "role": "user", "content": "hi" } ] });
        let before = body.clone();
        mark_cache_breakpoints(&mut body);
        assert_eq!(body, before, "nothing to mark must not corrupt the request");
    }

    #[test]
    fn a_rejected_request_quotes_the_providers_explanation() {
        // The reason this exists: `ureq`'s Display stops at the status code, so
        // a real llama.cpp context overflow read as a bare "status code 400" —
        // indistinguishable from a bad key or an unknown model.
        let body = r#"{"error":{"message":"request (6406 tokens) exceeds the available context size (4096 tokens)","type":"exceed_context_size_error"}}"#;
        let message = describe_failure("http://localhost:8080/v1/chat/completions", status_error(400, body));

        assert!(message.contains("status 400"), "keeps the code: {message}");
        assert!(message.contains("exceeds the available context size"), "quotes the body: {message}");
        assert!(message.contains("localhost:8080"), "names the endpoint: {message}");
    }

    #[test]
    fn a_long_body_is_truncated_rather_than_dumped() {
        let body = "x".repeat(MAX_BODY_SNIPPET * 3);
        let message = describe_failure("http://host/v1", status_error(400, &body));

        assert!(message.ends_with('…'), "marks the truncation: {message}");
        assert!(
            message.len() < body.len(),
            "a provider that returns an HTML error page must not become the whole message"
        );
    }

    #[test]
    fn an_empty_body_still_reports_the_status() {
        let message = describe_failure("http://host/v1", status_error(401, ""));
        assert!(message.contains("status 401"), "{message}");
        assert!(!message.trim_end().ends_with(':'), "no dangling colon: {message}");
    }
}
