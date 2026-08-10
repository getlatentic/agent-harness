//! One chat request, in either dialect this adapter speaks.
//!
//! The OpenAI `/v1/chat/completions` shape and Ollama's native `/api/chat`
//! differ in the envelope, not in the exchange: the same conversation, the same
//! tools, the same streamed assembly. They were nonetheless two functions with
//! one skeleton, kept in step by hand and by a comment asking future readers to
//! keep their signatures symmetric — which nothing enforced, and which had
//! already drifted.
//!
//! Everything that genuinely differs is data, and [`Dialect`] holds it: the
//! path, the fields that open a stream, where a JSON Schema goes, and how the
//! response frames arrive. The native endpoint exists because `/v1` silently
//! caps `num_ctx` at 4096 and truncates the prompt — a protocol difference, so
//! the protocol is what names it, rather than a context-window number standing
//! in for one.

use std::io::{BufRead, BufReader};
use std::sync::atomic::AtomicBool;

use serde_json::{json, Value};

use super::ollama;
use super::wire::{self, ChatMessage, Fragment, Usage};
use super::PromptCache;

/// One chat exchange, independent of the dialect it will be sent in.
pub(crate) struct ChatRequest<'a> {
    /// Endpoint root, carrying no trailing slash.
    pub base: &'a str,
    pub model: &'a str,
    pub messages: &'a [ChatMessage],
    /// The OpenAI `tools` array (built in [`super::tools`]); omitted when empty.
    pub tools: &'a [Value],
    /// Bearer credential, when the endpoint wants one. A locally served model
    /// usually does not, but a proxy in front of one may — so this follows the
    /// endpoint rather than the dialect.
    pub api_key: Option<&'a str>,
    pub extras: RequestExtras<'a>,
}

/// Optional request shaping beyond the messages and tools.
#[derive(Default)]
pub(crate) struct RequestExtras<'a> {
    /// JSON Schema the answer must conform to, in the OpenAI `response_format`
    /// wrapping. [`Dialect`] unwraps it where the endpoint wants it bare.
    pub response_format: Option<&'a Value>,
    /// Image data URIs to attach to the first user message (multimodal input).
    pub image_data_uris: &'a [String],
    /// Inline reasoning tag to lift out of the stream (e.g. `Some("think")` for
    /// `<think>…</think>`); `None` disables extraction.
    pub reasoning_tag: Option<&'a str>,
    /// Whether to mark the prompt prefix as cacheable.
    pub cache: PromptCache,
}

/// Which wire protocol an endpoint speaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dialect {
    /// `/v1/chat/completions`, SSE — what every hosted provider and most local
    /// servers offer.
    OpenAi,
    /// Ollama's own `/api/chat`, NDJSON, which takes the context window per
    /// request instead of ignoring it.
    OllamaNative { num_ctx: u64 },
}

impl Dialect {
    fn path(self) -> &'static str {
        match self {
            Self::OpenAi => "/v1/chat/completions",
            Self::OllamaNative { .. } => "/api/chat",
        }
    }

    /// The base object: what identifies the exchange, and what opens the stream.
    fn envelope(self, request: &ChatRequest<'_>) -> Value {
        let ChatRequest { model, messages, extras, .. } = request;
        match self {
            Self::OpenAi => json!({
                "model": model,
                "messages": messages,
                "stream": true,
                "stream_options": { "include_usage": true },
            }),
            Self::OllamaNative { num_ctx } => json!({
                "model": model,
                "messages": ollama::to_native_messages(messages, extras.image_data_uris),
                "stream": true,
                "keep_alive": ollama::KEEP_ALIVE,
                "options": { "num_ctx": num_ctx, "temperature": ollama::TEMPERATURE },
            }),
        }
    }

    /// Where a schema goes: OpenAI takes it wrapped as `response_format`, native
    /// Ollama takes the bare JSON Schema in `format`.
    fn apply_schema(self, body: &mut Value, response_format: &Value) {
        match self {
            Self::OpenAi => body["response_format"] = response_format.clone(),
            Self::OllamaNative { .. } => {
                if let Some(bare) = ollama::native_format(response_format) {
                    body["format"] = bare;
                }
            }
        }
    }

    fn body(self, request: &ChatRequest<'_>) -> Value {
        let extras = &request.extras;
        let mut body = self.envelope(request);
        if !request.tools.is_empty() {
            body["tools"] = Value::Array(request.tools.to_vec());
        }
        if let Some(schema) = extras.response_format {
            self.apply_schema(&mut body, schema);
        }
        if self == Self::OpenAi {
            // The native envelope carries images inside the message objects
            // already; the OpenAI shape rewrites the first user message here.
            if !extras.image_data_uris.is_empty() {
                wire::attach_images(&mut body, extras.image_data_uris);
            }
            // Last, so the breakpoint lands on the final tool once the list has
            // settled.
            if extras.cache == PromptCache::Ephemeral {
                wire::mark_cache_breakpoints(&mut body);
            }
        }
        body
    }

    fn drain(
        self,
        lines: impl Iterator<Item = String>,
        reasoning_tag: Option<&str>,
        cancel: &AtomicBool,
        on_delta: impl FnMut(Fragment),
    ) -> Result<(ChatMessage, Option<Usage>), String> {
        match self {
            Self::OpenAi => Ok(wire::drain_stream(lines, reasoning_tag, cancel, on_delta)),
            Self::OllamaNative { .. } => {
                ollama::drain_native_stream(lines, reasoning_tag, cancel, on_delta)
            }
        }
    }
}

/// Send `request` in `dialect` and stream the reply. `on_delta` sees each text
/// or reasoning fragment as it arrives; the return is the assembled assistant
/// message (content plus tool calls) with whatever usage the endpoint reported.
/// Blocking — driven on the worker thread.
pub(crate) fn post_chat_stream(
    request: ChatRequest<'_>,
    dialect: Dialect,
    cancel: &AtomicBool,
    on_delta: impl FnMut(Fragment),
) -> Result<(ChatMessage, Option<Usage>), String> {
    let url = format!("{}{}", request.base, dialect.path());
    let body = dialect.body(&request);
    let api_key = request.api_key;
    let response = wire::send_with_retry(&url, || {
        let mut post = ureq::post(&url);
        if let Some(key) = api_key {
            post = post.set("Authorization", &format!("Bearer {key}"));
        }
        post.send_json(body.clone()).map_err(Box::new)
    })?;
    let lines = BufReader::new(response.into_reader()).lines().map_while(Result::ok);
    dialect.drain(lines, request.extras.reasoning_tag, cancel, on_delta)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request<'a>(messages: &'a [ChatMessage], tools: &'a [Value], extras: RequestExtras<'a>) -> ChatRequest<'a> {
        ChatRequest { base: "http://host", model: "m", messages, tools, api_key: None, extras }
    }

    #[test]
    fn both_dialects_carry_the_same_conversation_and_tools() {
        // The property the two hand-synced builders were supposed to hold and
        // nothing checked: whatever is common to the exchange must survive into
        // either envelope.
        let messages = [ChatMessage::user("hello")];
        let tools = [json!({ "type": "function", "function": { "name": "read" } })];
        let native = Dialect::OllamaNative { num_ctx: 8_192 };

        for dialect in [Dialect::OpenAi, native] {
            let body = dialect.body(&request(&messages, &tools, RequestExtras::default()));
            assert_eq!(body["model"], "m", "{dialect:?}");
            assert_eq!(body["stream"], true, "{dialect:?}");
            assert_eq!(body["messages"].as_array().map(Vec::len), Some(1), "{dialect:?}");
            assert_eq!(body["tools"].as_array().map(Vec::len), Some(1), "{dialect:?}");
        }
    }

    #[test]
    fn an_empty_tool_list_is_omitted_rather_than_sent_empty() {
        // Some endpoints reject `"tools": []` outright, and it costs nothing to
        // send in the case where the model has no tools to call anyway.
        for dialect in [Dialect::OpenAi, Dialect::OllamaNative { num_ctx: 8_192 }] {
            let body = dialect.body(&request(&[ChatMessage::user("hi")], &[], RequestExtras::default()));
            assert!(body.get("tools").is_none(), "{dialect:?}");
        }
    }

    #[test]
    fn each_dialect_puts_the_schema_where_its_endpoint_wants_it() {
        let schema = json!({
            "type": "json_schema",
            "json_schema": { "name": "answer", "schema": { "type": "object" } }
        });
        let messages = [ChatMessage::user("hi")];
        let extras = RequestExtras { response_format: Some(&schema), ..Default::default() };
        let open_ai = Dialect::OpenAi.body(&request(&messages, &[], extras));
        assert_eq!(open_ai["response_format"], schema, "OpenAI takes it wrapped");
        assert!(open_ai.get("format").is_none());

        let extras = RequestExtras { response_format: Some(&schema), ..Default::default() };
        let native = Dialect::OllamaNative { num_ctx: 8_192 }.body(&request(&messages, &[], extras));
        assert_eq!(native["format"], json!({ "type": "object" }), "Ollama takes it bare");
        assert!(native.get("response_format").is_none());
    }

    #[test]
    fn the_native_dialect_asks_for_the_window_it_was_given() {
        // The reason this dialect exists: `/v1` accepts the request and loads
        // the model at 4096 regardless, truncating the prompt in silence.
        let body = Dialect::OllamaNative { num_ctx: 32_768 }
            .body(&request(&[ChatMessage::user("hi")], &[], RequestExtras::default()));
        assert_eq!(body["options"]["num_ctx"], 32_768);
        assert_eq!(Dialect::OllamaNative { num_ctx: 1 }.path(), "/api/chat");
        assert_eq!(Dialect::OpenAi.path(), "/v1/chat/completions");
    }

    #[test]
    fn implicit_is_the_default_so_an_unmarked_request_stays_unmarked() {
        // Correct everywhere; marking is wasted on providers that cache
        // implicitly and restructures a message they never asked to change.
        assert_eq!(PromptCache::default(), PromptCache::Implicit);
        let body = Dialect::OpenAi.body(&request(&[ChatMessage::user("hi")], &[], RequestExtras::default()));
        assert!(body["messages"][0]["content"].is_string(), "left as it was written");
    }

    #[test]
    fn cache_breakpoints_and_usage_are_asked_for_only_where_they_exist() {
        let messages = [
            ChatMessage::system("rules"),
            ChatMessage::user("first"),
            ChatMessage::tool_result("c1", "done"),
            ChatMessage::user("second"),
        ];
        let extras = RequestExtras { cache: PromptCache::Ephemeral, ..Default::default() };
        let open_ai = Dialect::OpenAi.body(&request(&messages, &[], extras));
        assert_eq!(open_ai["stream_options"]["include_usage"], true);
        assert!(
            open_ai["messages"][0]["content"][0]["cache_control"].is_object(),
            "the system prefix is marked: {open_ai}"
        );

        // Ollama reports usage on its own terminal frame and has no notion of a
        // cache breakpoint, so asking for either would be noise it rejects.
        let extras = RequestExtras { cache: PromptCache::Ephemeral, ..Default::default() };
        let native = Dialect::OllamaNative { num_ctx: 8_192 }.body(&request(&messages, &[], extras));
        assert!(native.get("stream_options").is_none());
        assert!(!native.to_string().contains("cache_control"), "nothing to mark natively: {native}");
    }
}
