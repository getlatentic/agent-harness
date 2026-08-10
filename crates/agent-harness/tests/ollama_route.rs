//! The Ollama route, end to end, against a stand-in Ollama.
//!
//! Everything else in this crate's `openai_compatible` tests is a pure
//! function — line parsers, session records, tool schemas. Nothing exercised
//! `run()`: the actual agent loop making HTTP requests and turning responses
//! into [`RunEvent`]s. That left the two things most likely to break silently
//! uncovered — *which* endpoint we call, and whether a tool result actually
//! makes it back to the model.
//!
//! A real Ollama is not required (and not wanted in CI: a model pull per job,
//! and non-deterministic output). A `tiny_http` server speaking Ollama's native
//! wire format pins the contract deterministically, on every platform, in
//! milliseconds. See the `ollama-live` CI job for the real-vendor counterpart.

#![cfg(feature = "openai-compatible")]

use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use harness::{Harness, OpenHarness, RunEvent, RunMode, RunRequest, RunTuning};
use serde_json::{json, Value};

/// Requests the fake server received, so a test can assert on what we SENT —
/// the path, and the message history on the follow-up turn.
type Seen = Arc<Mutex<Vec<(String, Value)>>>;

/// A stand-in Ollama. `chat_turns` is popped per `/api/chat` call, so a test can
/// script a multi-turn exchange (turn 1 asks for a tool, turn 2 answers).
/// Returns the base URL and the request log. The server thread parks on
/// `recv()` for the life of the test binary — cheap, and it can't race a test
/// that makes more requests than the script anticipated.
fn fake_ollama(tags: Value, chat_turns: Vec<String>) -> (String, Seen) {
    fake_ollama_with_context(tags, chat_turns, 8192)
}

/// As [`fake_ollama`], with the context window `/api/show` reports — the signal
/// `PromptProfile::Auto` keys on.
fn fake_ollama_with_context(tags: Value, chat_turns: Vec<String>, context_length: u64) -> (String, Seen) {
    fake_ollama_full(tags, chat_turns, context_length, 24_000_000_000)
}

/// As [`fake_ollama`], reporting both facts `PromptProfile::Auto` keys on.
fn fake_ollama_full(
    tags: Value,
    chat_turns: Vec<String>,
    context_length: u64,
    parameter_count: u64,
) -> (String, Seen) {
    let mut turns = chat_turns.into_iter();
    fake_ollama_responding(tags, context_length, parameter_count, move || {
        // Extra turns end the loop rather than hanging it.
        (200, turns.next().unwrap_or_else(|| done_line("")))
    })
}

/// The stand-in itself: `/api/tags` and `/api/show` answer from the given facts,
/// and every `/api/chat` is answered by `chat`, which chooses the status as well
/// as the body — a provider refusing an over-long prompt is a 400, not a stream,
/// so a helper that can only return 200 cannot express it.
fn fake_ollama_responding(
    tags: Value,
    context_length: u64,
    parameter_count: u64,
    mut chat: impl FnMut() -> (u16, String) + Send + 'static,
) -> (String, Seen) {
    let server = tiny_http::Server::http("127.0.0.1:0").expect("bind ephemeral port");
    let base = format!("http://{}", server.server_addr());
    let seen: Seen = Arc::default();
    let log = Arc::clone(&seen);

    thread::spawn(move || {
        while let Ok(mut request) = server.recv() {
            let url = request.url().to_owned();
            let mut raw = String::new();
            let _ = std::io::Read::read_to_string(request.as_reader(), &mut raw);
            let parsed: Value = serde_json::from_str(&raw).unwrap_or(Value::Null);
            log.lock().unwrap().push((url.clone(), parsed));

            let (status, body) = if url.starts_with("/api/tags") {
                (200, tags.to_string())
            } else if url.starts_with("/api/show") {
                // Ollama keys this by architecture (`qwen2.context_length`), and the
                // parser scans for that suffix — a bare `context_length` silently
                // falls back to the default and makes this stub decorative.
                let info = json!({
                    "model_info": {
                        "qwen2.context_length": context_length,
                        "general.parameter_count": parameter_count,
                    }
                });
                (200, info.to_string())
            } else if url.starts_with("/api/chat") {
                chat()
            } else {
                (200, String::new())
            };
            let _ = request.respond(tiny_http::Response::new(
                tiny_http::StatusCode(status),
                Vec::new(),
                Cursor::new(body.into_bytes()),
                None,
                None,
            ));
        }
    });
    (base, seen)
}

/// One streaming NDJSON chunk carrying visible text.
fn text_line(text: &str) -> String {
    json!({ "message": { "role": "assistant", "content": text }, "done": false }).to_string()
}

/// The terminal chunk, carrying token counts.
fn done_line(text: &str) -> String {
    json!({
        "message": { "role": "assistant", "content": text },
        "done": true,
        "prompt_eval_count": 11,
        "eval_count": 7
    })
    .to_string()
}

fn collect(harness: &dyn Harness, prompt: &str) -> Vec<RunEvent> {
    collect_resuming(harness, prompt, None)
}

/// As [`collect`], continuing a stored session.
fn collect_resuming(harness: &dyn Harness, prompt: &str, resume: Option<String>) -> Vec<RunEvent> {
    collect_inner(harness, prompt, resume)
}

fn collect_inner(harness: &dyn Harness, prompt: &str, resume: Option<String>) -> Vec<RunEvent> {
    let events: Arc<Mutex<Vec<RunEvent>>> = Arc::default();
    let sink = Arc::clone(&events);
    let done = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&done);
    let handle = harness
        .start(
            RunRequest {
                run_id: "test-run".to_owned(),
                prompt: prompt.to_owned(),
                cwd: Some(std::env::temp_dir()),
                mode: RunMode::Ask,
                tuning: RunTuning { model: Some("test-model".to_owned()), ..Default::default() },
                resume,
                attachments: Vec::new(),
            },
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

#[test]
fn readiness_and_model_list_come_from_api_tags() {
    let tags = json!({ "models": [ { "name": "qwen2.5:0.5b" }, { "name": "llama3.2:1b" } ] });
    let (base, seen) = fake_ollama(tags, Vec::new());
    let harness = OpenHarness::ollama_at(&base);

    let readiness = harness.readiness();
    assert!(readiness.ready, "a reachable Ollama is ready: {:?}", readiness.error);

    let models: Vec<String> = harness.list_models().unwrap().into_iter().map(|m| m.value).collect();
    assert_eq!(models, vec!["qwen2.5:0.5b", "llama3.2:1b"]);

    let paths: Vec<String> = seen.lock().unwrap().iter().map(|(u, _)| u.clone()).collect();
    assert!(paths.iter().all(|p| p.starts_with("/api/")), "native API only, got {paths:?}");
}

#[test]
fn a_run_streams_text_and_reports_usage() {
    let tags = json!({ "models": [ { "name": "test-model" } ] });
    let turns = vec![format!("{}\n{}", text_line("Hello, "), done_line("world."))];
    let (base, seen) = fake_ollama(tags, turns);

    let events = collect(&OpenHarness::ollama_at(&base), "say hello");

    let text: String = events
        .iter()
        .filter_map(|e| match e {
            RunEvent::Text { delta, .. } => Some(delta.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "Hello, world.");
    assert!(
        events.iter().any(|e| matches!(e, RunEvent::Exited { .. })),
        "a run must terminate with Exited, got {events:?}"
    );
    assert!(
        events.iter().any(|e| matches!(e, RunEvent::Usage { .. })),
        "token counts from the terminal chunk must surface as Usage"
    );

    // The regression that motivated the native path: `/v1/chat/completions`
    // silently caps num_ctx at 4096 and truncates the system prompt.
    let log = seen.lock().unwrap();
    let (path, body) = log.iter().find(|(u, _)| u.starts_with("/api/chat")).expect("chat call");
    assert_eq!(path, "/api/chat", "must not fall back to the /v1 shape");
    assert!(
        body["options"]["num_ctx"].as_u64().is_some_and(|n| n > 4096),
        "num_ctx must be sent and exceed Ollama's truncating 4096 default: {}",
        body["options"]
    );
    assert_eq!(body["stream"], json!(true));
}

#[test]
fn a_tool_call_round_trips_its_result_back_to_the_model() {
    let tags = json!({ "models": [ { "name": "test-model" } ] });
    // Turn 1: the model asks to list the working directory. Turn 2: it answers.
    let ask_for_tool = json!({
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [ { "function": { "name": "list", "arguments": { "path": "." } } } ]
        },
        "done": true,
        "prompt_eval_count": 5,
        "eval_count": 2
    })
    .to_string();
    let (base, seen) =
        fake_ollama(tags, vec![ask_for_tool, done_line("Listed the directory.")]);

    let events = collect(&OpenHarness::ollama_at(&base), "what is here?");

    assert!(
        events.iter().any(|e| matches!(e, RunEvent::ToolStart { .. })),
        "a tool call must surface as ToolStart, got {events:?}"
    );

    // The loop's whole point: turn 2 must carry the tool's OUTPUT, or the model
    // is answering blind and every tool call is decorative.
    let log = seen.lock().unwrap();
    let chats: Vec<&Value> = log.iter().filter(|(u, _)| u.starts_with("/api/chat")).map(|(_, b)| b).collect();
    assert_eq!(chats.len(), 2, "one turn per model reply");
    let follow_up = chats[1]["messages"].as_array().expect("messages array");
    assert!(
        follow_up.iter().any(|m| m["role"] == "tool"),
        "the second turn must include the tool result: {follow_up:?}"
    );
}

/// Tool names offered to the model in the first `/api/chat` request.
fn offered_tools(seen: &Seen) -> Vec<String> {
    let log = seen.lock().unwrap();
    let (_, body) = log.iter().find(|(u, _)| u.starts_with("/api/chat")).expect("chat call");
    body["tools"]
        .as_array()
        .map(|tools| {
            tools
                .iter()
                .filter_map(|t| t["function"]["name"].as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

/// The profile is chosen from the context window `/api/show` reports, and it
/// has to reach the wire — an enum that resolves correctly but never changes
/// the request would pass a unit test and fix nothing.
#[test]
fn a_small_context_window_narrows_the_tools_actually_sent() {
    let tags = json!({ "models": [ { "name": "test-model" } ] });

    let (small_url, small_seen) =
        fake_ollama_with_context(tags.clone(), vec![done_line("ok")], 8_192);
    let _ = collect(&OpenHarness::ollama_at(&small_url), "hello");
    let small = offered_tools(&small_seen);

    let (big_url, big_seen) =
        fake_ollama_with_context(tags, vec![done_line("ok")], 131_072);
    let _ = collect(&OpenHarness::ollama_at(&big_url), "hello");
    let big = offered_tools(&big_seen);

    assert!(!small.is_empty() && !big.is_empty(), "both runs must offer tools");
    assert!(
        small.len() < big.len(),
        "a small window must cost fewer tool schemas: {small:?} vs {big:?}"
    );
    // Whatever is trimmed, the model must keep the means to find and read a
    // file — a run that cannot do that is not narrower, it is broken.
    for essential in ["read", "list"] {
        assert!(small.contains(&essential.to_owned()), "{essential} missing from {small:?}");
    }
    for optional in ["webfetch", "todowrite"] {
        assert!(!small.contains(&optional.to_owned()), "{optional} should be withheld: {small:?}");
        assert!(big.contains(&optional.to_owned()), "{optional} expected at full surface: {big:?}");
    }
}

/// The window's blind spot: a tiny model that advertises a huge context. It
/// passes the context test and still cannot use a full tool surface, so the
/// parameter count has to be read too.
#[test]
fn a_tiny_model_gets_the_small_surface_despite_a_huge_window() {
    let tags = json!({ "models": [ { "name": "test-model" } ] });
    let (url, seen) =
        fake_ollama_full(tags, vec![done_line("ok")], 131_072, 1_200_000_000);

    let _ = collect(&OpenHarness::ollama_at(&url), "hello");

    let offered = offered_tools(&seen);
    assert!(offered.contains(&"read".to_owned()), "core survives: {offered:?}");
    assert!(
        !offered.contains(&"webfetch".to_owned()),
        "a 1.2B model must not be handed the full surface just because its window is large: {offered:?}"
    );
}

/// A session store holding one long conversation, so a resumed run has enough
/// history to be worth compacting. Returns the store root and the session id.
fn seeded_session(tag: &str, turns: usize) -> (std::path::PathBuf, String) {
    let root = std::env::temp_dir().join(format!("hl-compact-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("sessions")).unwrap();

    let id = "ses-long";
    let mut lines = vec![format!(
        r#"{{"type":"session","id":"{id}","title":"long","created_at":1,"updated_at":1}}"#
    )];
    for i in 0..turns {
        let filler = "some earlier discussion that is long enough to matter ".repeat(6);
        lines.push(format!(
            r#"{{"type":"message","role":"user","content":"question {i}: {filler}"}}"#
        ));
        lines.push(format!(
            r#"{{"type":"message","role":"assistant","content":"answer {i}: {filler}"}}"#
        ));
    }
    std::fs::write(root.join("sessions").join(format!("{id}.jsonl")), lines.join("\n") + "\n").unwrap();
    (root, id.to_owned())
}

/// Compaction, end to end. Every piece of it had unit tests — `tail_boundary`,
/// `window` — while nothing asserted that it ever *runs*. Replacing
/// `compact_if_needed`, `compact_to` or `persist` with a no-op passed the whole
/// suite, which means the feature preventing context overflow could have been
/// switched off silently.
#[test]
fn a_resumed_conversation_past_the_context_limit_is_compacted_and_saved() {
    let (root, session_id) = seeded_session("threshold", 40);
    let tags = json!({ "models": [ { "name": "test-model" } ] });
    // Turn 1 answers the summarization request; turn 2 is the real reply.
    let (base, seen) = fake_ollama_with_context(
        tags,
        vec![done_line("A summary of the earlier turns."), done_line("Answer.")],
        8_192,
    );

    let harness = OpenHarness::ollama_at(&base)
        .with_session_dir(&root)
        // Small enough that the seeded history is over the threshold.
        .with_context_tokens(2_000);

    let events = collect_resuming(&harness, "and finally?", Some(session_id.clone()));

    let compacted = events.iter().any(|e| {
        matches!(e, RunEvent::Activity { message, .. } if message.contains("compacted the conversation"))
    });
    assert!(compacted, "compaction must actually fire: {events:?}");

    assert!(
        events.iter().any(|e| matches!(e, RunEvent::Exited { .. })),
        "the run still completes after compacting"
    );

    // The summarization request goes to the model like any other turn, so a
    // compaction that never happened would leave only one chat call.
    let chats = seen.lock().unwrap().iter().filter(|(u, _)| u.starts_with("/api/chat")).count();
    assert!(chats >= 2, "summarize + answer, got {chats} chat call(s)");

    // And the compacted transcript is written back, not just held in memory.
    let saved = std::fs::read_to_string(root.join("sessions").join(format!("{session_id}.jsonl")))
        .expect("the session file");
    assert!(
        saved.contains(r#""role":"compaction""#),
        "the summary must reach disk, or the next resume replays everything it replaced"
    );

    // And exactly once. Appending a tail slice after an insert re-saves a turn
    // that was already on disk, which is how a resumed conversation grows a
    // duplicate of its own prompt.
    assert_eq!(
        saved.matches(r#""content":"and finally?""#).count(),
        1,
        "the prompt is saved once, not once per save after a mid-transcript insert"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// The provider gets the last word on whether a prompt fits. Our own estimate
/// is `len / 4`, so a run can be under the threshold by that count and still be
/// refused — and the answer is to compact against what the provider actually
/// said and retry, not to end the run on a guess that was already wrong.
#[test]
fn a_prompt_the_provider_refuses_as_too_long_is_compacted_and_retried() {
    let (root, session_id) = seeded_session("retry", 40);
    let tags = json!({ "models": [ { "name": "test-model" } ] });
    let mut call = 0;
    let (base, seen) = fake_ollama_responding(tags, 20_000, 24_000_000_000, move || {
        call += 1;
        match call {
            // llama.cpp's wording, which `is_context_overflow` matches on.
            1 => (400, json!({ "error": "the request exceeds the available context size" }).to_string()),
            2 => (200, done_line("A summary of the earlier turns.")),
            _ => (200, done_line("Answer after retrying.")),
        }
    });

    // Wide enough that the seeded history is comfortably under the threshold
    // compaction fires on by itself: the refusal has to be what triggers it,
    // or this tests the proactive path a second time.
    let harness = OpenHarness::ollama_at(&base).with_session_dir(&root).with_context_tokens(20_000);
    let events = collect_resuming(&harness, "and finally?", Some(session_id));

    assert!(
        events.iter().any(|e| matches!(e, RunEvent::Exited { .. })),
        "the refusal is recoverable, so the run finishes: {events:?}"
    );
    assert!(
        !events.iter().any(|e| matches!(e, RunEvent::Error { .. })),
        "and does not surface the refusal as the run's outcome: {events:?}"
    );
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            RunEvent::Text { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert!(text.contains("Answer after retrying"), "the retried turn is what the caller sees, got {text:?}");

    let chats = seen.lock().unwrap().iter().filter(|(u, _)| u.starts_with("/api/chat")).count();
    assert_eq!(chats, 3, "refused, summarized, retried");
    let _ = std::fs::remove_dir_all(&root);
}

/// The other half: only an overflow is worth retrying. Treating every failure
/// as one spends a summarization request on an error that will repeat, and
/// buries the provider's actual complaint behind it.
#[test]
fn a_failure_that_is_not_about_length_ends_the_run_without_compacting() {
    let (root, session_id) = seeded_session("nonoverflow", 40);
    let tags = json!({ "models": [ { "name": "test-model" } ] });
    // A 400, not a 5xx: a transient status is retried by `send_with_retry`
    // before the loop ever sees it, which is a different mechanism.
    let (base, seen) = fake_ollama_responding(tags, 20_000, 24_000_000_000, || {
        (400, json!({ "error": "model \"test-model\" not found, try pulling it first" }).to_string())
    });

    let harness = OpenHarness::ollama_at(&base).with_session_dir(&root).with_context_tokens(20_000);
    let events = collect_resuming(&harness, "and finally?", Some(session_id));

    let errored = events
        .iter()
        .any(|e| matches!(e, RunEvent::Error { message, .. } if message.contains("not found")));
    assert!(errored, "the provider's own complaint reaches the caller: {events:?}");
    assert!(
        !events.iter().any(|e| {
            matches!(e, RunEvent::Activity { message, .. } if message.contains("compacted the conversation"))
        }),
        "and nothing is summarized on the way out: {events:?}"
    );

    let chats = seen.lock().unwrap().iter().filter(|(u, _)| u.starts_with("/api/chat")).count();
    assert_eq!(chats, 1, "one attempt, no retry");
    let _ = std::fs::remove_dir_all(&root);
}

/// The same route against a **real** Ollama — what the fake can't catch: vendor
/// drift. Ollama silently capping `num_ctx` on `/v1` is exactly the class of bug
/// a hand-written stand-in reproduces faithfully and therefore misses.
///
/// Ignored by default (needs a server and a pulled model). The `ollama-live` CI
/// job runs it with `-- --ignored`. Point it elsewhere with `OLLAMA_HOST`.
#[test]
#[ignore = "live: needs a running Ollama with OLLAMA_TEST_MODEL pulled"]
fn live_ollama_streams_a_real_completion() {
    let base = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_owned());
    let base = if base.starts_with("http") { base } else { format!("http://{base}") };
    let model = std::env::var("OLLAMA_TEST_MODEL").unwrap_or_else(|_| "qwen2.5:0.5b".to_owned());

    let harness = OpenHarness::ollama_at(&base);
    let readiness = harness.readiness();
    assert!(readiness.ready, "no Ollama at {base}: {:?}", readiness.error);
    assert!(
        harness.list_models().unwrap().iter().any(|m| m.value == model),
        "pull {model} first (`ollama pull {model}`)"
    );

    let events: Arc<Mutex<Vec<RunEvent>>> = Arc::default();
    let sink = Arc::clone(&events);
    let done = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&done);
    let _handle = harness
        .start(
            RunRequest {
                run_id: "live".to_owned(),
                prompt: "Reply with the single word: pong".to_owned(),
                cwd: Some(std::env::temp_dir()),
                mode: RunMode::Ask,
                tuning: RunTuning { model: Some(model.clone()), ..Default::default() },
                resume: None,
                attachments: Vec::new(),
            },
            Arc::new(move |event| {
                if matches!(event, RunEvent::Exited { .. } | RunEvent::Error { .. }) {
                    flag.store(true, Ordering::SeqCst);
                }
                sink.lock().unwrap().push(event);
            }),
        )
        .expect("run should start");

    // Measured on an M-series Mac: ~85s for llama3.2:1b, because every run
    // prompt-evals a ~20k-char system prompt plus 9 tool schemas before the
    // model emits a token. A CI runner has no GPU, so allow well beyond that;
    // the loop exits the moment the run finishes.
    for _ in 0..12_000 {
        if done.load(Ordering::SeqCst) {
            break;
        }
        thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(done.load(Ordering::SeqCst), "live run did not finish within 10 minutes");

    let events = events.lock().unwrap().clone();
    // Assert the PROTOCOL, never the model's answer — a 0.5b model's wording is
    // not a contract, and pinning it would make this test a coin flip.
    if let Some(RunEvent::Error { message, .. }) = events.iter().find(|e| matches!(e, RunEvent::Error { .. })) {
        panic!("live run errored: {message}");
    }
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            RunEvent::Text { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert!(!text.trim().is_empty(), "a live run must produce text, got {events:?}");
    assert!(events.iter().any(|e| matches!(e, RunEvent::Exited { .. })), "must terminate");
}
