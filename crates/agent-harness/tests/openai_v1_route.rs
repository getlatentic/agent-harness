//! The OpenAI `/v1` route against a stand-in server, asserting **what we send**.
//!
//! `tests/openai_v1_live.rs` drives real providers, but it costs tokens, needs
//! keys, and can only check that a run completes — it cannot look at the
//! request. Everything this crate decides before the wire (the route, the auth
//! header, cache breakpoints, where a schema goes, how an image is attached) is
//! invisible to it.
//!
//! So this is the `/v1` sibling of `tests/ollama_route.rs`: a `tiny_http`
//! stand-in speaking the real wire format, asserting the request we produced
//! rather than what a mock was told to return.

#![cfg(feature = "openai-compatible")]

use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use harness::{
    ApiKey, Attachment, Harness, HarnessModel, OpenHarness, OpenHarnessConfig, PromptCache, RunEvent, RunMode,
    RunRequest, RunTuning,
};
use serde_json::{json, Value};

/// Every request the harness made: URL, `Authorization` header, decoded body.
type Seen = Arc<Mutex<Vec<Request>>>;

struct Request {
    url: String,
    authorization: Option<String>,
    body: Value,
}

/// A server answering `/v1/chat/completions` with the queued responses in turn.
/// Anything past the end gets an empty completion, so an extra turn ends the
/// loop rather than hanging it.
fn fake_openai(responses: Vec<(u16, String)>) -> (String, Seen) {
    let server = tiny_http::Server::http("127.0.0.1:0").expect("bind ephemeral port");
    let base = format!("http://{}", server.server_addr());
    let seen: Seen = Arc::default();
    let log = Arc::clone(&seen);
    let mut queued = responses.into_iter();

    thread::spawn(move || {
        while let Ok(mut request) = server.recv() {
            let url = request.url().to_owned();
            let authorization = request
                .headers()
                .iter()
                .find(|h| h.field.equiv("Authorization"))
                .map(|h| h.value.as_str().to_owned());
            let mut raw = String::new();
            let _ = std::io::Read::read_to_string(request.as_reader(), &mut raw);
            let body = serde_json::from_str(&raw).unwrap_or(Value::Null);
            log.lock().unwrap().push(Request { url, authorization, body });

            let (status, payload) = queued.next().unwrap_or_else(|| (200, sse(&[])));
            // A real 429 carries `Retry-After`, and honouring it is what the
            // backoff defers to — so saying "now" keeps these tests instant
            // instead of sleeping out the exponential default.
            let headers = match status {
                429 => vec![tiny_http::Header::from_bytes(&b"Retry-After"[..], &b"0"[..]).unwrap()],
                _ => Vec::new(),
            };
            let _ = request.respond(tiny_http::Response::new(
                tiny_http::StatusCode(status),
                headers,
                Cursor::new(payload.into_bytes()),
                None,
                None,
            ));
        }
    });
    (base, seen)
}

/// Frames in the SSE envelope every OpenAI-compatible endpoint streams.
fn sse(frames: &[Value]) -> String {
    let mut out = String::new();
    for frame in frames {
        out.push_str(&format!("data: {frame}\n\n"));
    }
    out.push_str("data: [DONE]\n\n");
    out
}

fn text(chunk: &str) -> Value {
    json!({ "choices": [{ "delta": { "content": chunk } }] })
}

fn usage(prompt: u64, completion: u64) -> Value {
    json!({
        "choices": [],
        "usage": { "prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion }
    })
}

/// A reply carrying visible text and a token count — the ordinary case.
fn answer(chunk: &str) -> (u16, String) {
    (200, sse(&[text(chunk), usage(11, 7)]))
}

fn harness_at(base: &str, api_key: ApiKey, prompt_cache: PromptCache) -> OpenHarness {
    OpenHarness::custom(OpenHarnessConfig {
        id: "stand-in".to_owned(),
        display_name: "Stand-in".to_owned(),
        base_url: base.to_owned(),
        api_key,
        prompt_cache,
        models: vec![HarnessModel { value: "test-model".to_owned(), label: "Test model".to_owned() }],
        ..Default::default()
    })
    // Also skips the llama.cpp `/props` probe: 127.0.0.1 reads as a local
    // endpoint, and the stand-in does not answer that route.
    .with_context_tokens(32_000)
}

fn collect(harness: &OpenHarness, request: RunRequest) -> Vec<RunEvent> {
    let events: Arc<Mutex<Vec<RunEvent>>> = Arc::default();
    let sink = Arc::clone(&events);
    let done = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&done);
    let handle = harness
        .start(
            request,
            Arc::new(move |event| {
                if matches!(event, RunEvent::Exited { .. } | RunEvent::Error { .. }) {
                    flag.store(true, Ordering::SeqCst);
                }
                sink.lock().unwrap().push(event);
            }),
        )
        .expect("run should start");
    for _ in 0..600 {
        if done.load(Ordering::SeqCst) {
            break;
        }
        thread::sleep(std::time::Duration::from_millis(25));
    }
    let _ = handle;
    let out = events.lock().unwrap().clone();
    out
}

fn ask(prompt: &str) -> RunRequest {
    RunRequest {
        run_id: "test-run".to_owned(),
        prompt: prompt.to_owned(),
        cwd: Some(std::env::temp_dir()),
        mode: RunMode::Ask,
        tuning: RunTuning { model: Some("test-model".to_owned()), ..Default::default() },
        ..Default::default()
    }
}

fn streamed_text(events: &[RunEvent]) -> String {
    events
        .iter()
        .filter_map(|e| match e {
            RunEvent::Text { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn a_run_takes_the_v1_route_and_asks_for_a_stream_with_usage() {
    let (base, seen) = fake_openai(vec![answer("pong")]);
    let events = collect(&harness_at(&base, ApiKey::NotNeeded, PromptCache::default()), ask("ping"));

    assert_eq!(streamed_text(&events), "pong");
    assert!(events.iter().any(|e| matches!(e, RunEvent::Exited { .. })), "{events:?}");

    let seen = seen.lock().unwrap();
    let request = seen.first().expect("one chat request");
    assert_eq!(request.url, "/v1/chat/completions");
    assert_eq!(request.body["stream"], true);
    assert_eq!(
        request.body["stream_options"]["include_usage"], true,
        "usage on a stream is opt-in; without it the run reports no tokens"
    );
    assert_eq!(request.body["model"], "test-model");
    assert!(events.iter().any(|e| matches!(e, RunEvent::Usage { .. })), "usage reaches the host");
}

#[test]
fn a_key_travels_as_a_bearer_header_and_is_absent_when_there_is_none() {
    let (base, seen) = fake_openai(vec![answer("ok")]);
    collect(&harness_at(&base, ApiKey::Value("s3cret".to_owned()), PromptCache::default()), ask("hi"));
    assert_eq!(
        seen.lock().unwrap().first().and_then(|r| r.authorization.clone()).as_deref(),
        Some("Bearer s3cret")
    );

    // A locally served model usually wants no key, and sending an empty bearer
    // is how a request gets rejected by something that would have allowed it.
    let (base, seen) = fake_openai(vec![answer("ok")]);
    collect(&harness_at(&base, ApiKey::NotNeeded, PromptCache::default()), ask("hi"));
    assert_eq!(seen.lock().unwrap().first().and_then(|r| r.authorization.clone()), None);
}

#[test]
fn only_an_ephemeral_cache_marks_the_prefix() {
    // The cost of getting this wrong is silent and recurring: an unmarked
    // request to an Anthropic model through a gateway re-charges the whole
    // system prompt and tool block at full input price every turn.
    let (base, seen) = fake_openai(vec![answer("ok")]);
    collect(&harness_at(&base, ApiKey::NotNeeded, PromptCache::Ephemeral), ask("hi"));
    let marked = seen.lock().unwrap().first().map(|r| r.body.to_string()).unwrap_or_default();
    assert!(marked.contains("cache_control"), "the prefix is marked: {marked}");
    let body: Value = serde_json::from_str(&marked).unwrap();
    let tools = body["tools"].as_array().expect("tools are offered");
    assert!(
        tools.last().expect("at least one tool")["cache_control"].is_object(),
        "the breakpoint sits on the LAST tool, so it covers the whole schema block"
    );

    let (base, seen) = fake_openai(vec![answer("ok")]);
    collect(&harness_at(&base, ApiKey::NotNeeded, PromptCache::Implicit), ask("hi"));
    let unmarked = seen.lock().unwrap().first().map(|r| r.body.to_string()).unwrap_or_default();
    assert!(
        !unmarked.contains("cache_control"),
        "providers that cache implicitly never asked for this, and it restructures their messages"
    );
}

#[test]
fn a_requested_schema_reaches_the_provider_in_the_response_format_wrapper() {
    let schema = json!({ "type": "object", "properties": { "answer": { "type": "string" } } });
    let (base, seen) = fake_openai(vec![answer("{}")]);
    let request = RunRequest {
        tuning: RunTuning {
            model: Some("test-model".to_owned()),
            output_schema: Some(schema.clone()),
            ..Default::default()
        },
        ..ask("hi")
    };
    collect(&harness_at(&base, ApiKey::NotNeeded, PromptCache::default()), request);

    let seen = seen.lock().unwrap();
    let sent = &seen.first().expect("one chat request").body["response_format"];
    assert_eq!(sent["type"], "json_schema", "the OpenAI wrapper, not the bare schema");
    assert_eq!(sent["json_schema"]["schema"], schema);
}

#[test]
fn an_attached_image_rides_on_the_first_user_message() {
    let (base, seen) = fake_openai(vec![answer("a cat")]);
    let request = RunRequest {
        attachments: vec![Attachment { mime_type: "image/png".to_owned(), data: vec![1, 2, 3] }],
        ..ask("what is this?")
    };
    collect(&harness_at(&base, ApiKey::NotNeeded, PromptCache::default()), request);

    let seen = seen.lock().unwrap();
    let messages = seen.first().expect("one chat request").body["messages"].as_array().unwrap().clone();
    let user = messages.iter().find(|m| m["role"] == "user").expect("a user message");
    let parts = user["content"].as_array().expect("content becomes a parts array to hold the image");
    assert_eq!(parts[0]["type"], "text", "the prompt text survives alongside the image");
    assert_eq!(parts[0]["text"], "what is this?");
    assert_eq!(parts[1]["type"], "image_url");
    assert!(
        parts[1]["image_url"]["url"].as_str().unwrap().starts_with("data:image/png;base64,"),
        "sent inline as a data URI"
    );
}

#[test]
fn a_tool_call_split_across_frames_is_reassembled_and_answered() {
    // OpenAI streams a call's arguments in pieces; a run that assembled them
    // wrongly would call the tool with truncated JSON.
    let path = std::env::temp_dir().join("hl-v1-route-read.txt");
    std::fs::write(&path, "file contents").unwrap();
    let arguments = json!({ "path": path.to_str().unwrap() }).to_string();
    let (head, tail) = arguments.split_at(arguments.len() / 2);

    let (base, seen) = fake_openai(vec![
        (
            200,
            sse(&[
                json!({ "choices": [{ "delta": { "tool_calls": [{ "index": 0, "id": "c1", "function": { "name": "read", "arguments": head } }] } }] }),
                json!({ "choices": [{ "delta": { "tool_calls": [{ "index": 0, "function": { "arguments": tail } }] } }] }),
            ]),
        ),
        answer("it says: file contents"),
    ]);
    let events = collect(&harness_at(&base, ApiKey::NotNeeded, PromptCache::default()), ask("read it"));

    assert!(
        events.iter().any(|e| matches!(e, RunEvent::ToolStart { title, .. } if title == "read")),
        "the call is dispatched: {events:?}"
    );
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2, "the result goes back for a second turn");
    let followup = seen[1].body["messages"].as_array().unwrap();
    let result = followup.iter().find(|m| m["role"] == "tool").expect("the tool result is fed back");
    assert!(
        result["content"].as_str().unwrap_or_default().contains("file contents"),
        "with the file it actually read: {result}"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_rejected_request_surfaces_what_the_provider_said() {
    // `ureq`'s Display stops at the status code, so a real context overflow read
    // as a bare "status code 400" — indistinguishable from a bad key.
    let (base, _seen) = fake_openai(vec![(
        400,
        json!({ "error": { "message": "this model's maximum context length is 8192 tokens" } }).to_string(),
    )]);
    let events = collect(&harness_at(&base, ApiKey::NotNeeded, PromptCache::default()), ask("hi"));

    let reported = events.iter().find_map(|e| match e {
        RunEvent::Error { message, .. } => Some(message.clone()),
        _ => None,
    });
    let reported = reported.expect("the run reports an error");
    assert!(reported.contains("maximum context length"), "got {reported:?}");
}

#[test]
fn a_transient_failure_is_retried_rather_than_ending_the_run() {
    // 429 and 5xx are the provider saying "later", not "no". Ending the run on
    // one turns a rate limit into a lost conversation.
    let (base, seen) = fake_openai(vec![(429, "slow down".to_owned()), answer("pong")]);
    let events = collect(&harness_at(&base, ApiKey::NotNeeded, PromptCache::default()), ask("ping"));

    assert_eq!(streamed_text(&events), "pong", "the retry is what the caller sees: {events:?}");
    assert!(!events.iter().any(|e| matches!(e, RunEvent::Error { .. })), "{events:?}");
    assert_eq!(seen.lock().unwrap().len(), 2, "refused once, then sent again");
}

#[test]
fn retrying_gives_up_rather_than_hammering_a_provider_that_keeps_refusing() {
    // The other end of the same policy. Retrying without a bound is how a
    // rate-limited client becomes the reason it stays rate-limited.
    let refusals = std::iter::repeat_with(|| (429, "slow down".to_owned())).take(10).collect();
    let (base, seen) = fake_openai(refusals);
    let events = collect(&harness_at(&base, ApiKey::NotNeeded, PromptCache::default()), ask("ping"));

    assert_eq!(seen.lock().unwrap().len(), 4, "the first attempt plus three retries, then it stops");
    assert!(events.iter().any(|e| matches!(e, RunEvent::Error { .. })), "and says so: {events:?}");
}
