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

/// As [`collect`], with the run's tool access spelled out.
fn collect_with_tools(
    harness: &dyn Harness,
    prompt: &str,
    tools: harness::ToolAccess,
) -> Vec<RunEvent> {
    collect_inner(harness, prompt, None, tools)
}

/// As [`collect`], continuing a stored session.
fn collect_resuming(harness: &dyn Harness, prompt: &str, resume: Option<String>) -> Vec<RunEvent> {
    collect_inner(harness, prompt, resume, harness::ToolAccess::Default)
}

fn collect_inner(
    harness: &dyn Harness,
    prompt: &str,
    resume: Option<String>,
    tools: harness::ToolAccess,
) -> Vec<RunEvent> {
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
                tools,
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

    events.lock().unwrap().clone()
}

/// A stand-in serving only what the model *manager* uses: `/api/tags` for the
/// installed list, and `/api/pull` streaming the given NDJSON lines. Separate
/// from [`fake_ollama_responding`] so the chat helpers keep their signatures.
fn fake_ollama_manager(tags: Value, pull_lines: Vec<String>) -> (String, Seen) {
    let server = tiny_http::Server::http("127.0.0.1:0").expect("bind ephemeral port");
    let base = format!("http://{}", server.server_addr());
    let seen: Seen = Arc::default();
    let log = Arc::clone(&seen);

    thread::spawn(move || {
        while let Ok(mut request) = server.recv() {
            let url = request.url().to_owned();
            let mut raw = String::new();
            let _ = std::io::Read::read_to_string(request.as_reader(), &mut raw);
            log.lock().unwrap().push((url.clone(), serde_json::from_str(&raw).unwrap_or(Value::Null)));

            let body = if url.starts_with("/api/tags") {
                tags.to_string()
            } else if url.starts_with("/api/pull") {
                pull_lines.join("\n")
            } else {
                String::new()
            };
            let _ = request.respond(tiny_http::Response::new(
                tiny_http::StatusCode(200),
                Vec::new(),
                Cursor::new(body.into_bytes()),
                None,
                None,
            ));
        }
    });
    (base, seen)
}

#[test]
fn the_installed_list_carries_what_a_model_manager_shows() {
    // `list_models` (above) is the picker's name-only shape. This is the
    // manager's: without size, parameters and quantization it cannot tell a
    // 400 MB Q4 from a 40 GB f16, which is the decision the screen exists for.
    let tags = json!({ "models": [
        {
            "name": "qwen2.5:0.5b",
            "size": 397_821_319u64,
            "details": { "parameter_size": "494.03M", "quantization_level": "Q4_K_M" }
        },
        // A tag with no `details` still lists — just without the labels.
        { "name": "bare:latest", "size": 12u64 }
    ] });
    let (base, _) = fake_ollama_manager(tags, Vec::new());

    let installed = OpenHarness::ollama_at(&base).list_installed_models().expect("list");

    assert_eq!(installed.len(), 2);
    assert_eq!(installed[0].name, "qwen2.5:0.5b");
    assert_eq!(installed[0].size, 397_821_319);
    assert_eq!(installed[0].parameter_size.as_deref(), Some("494.03M"));
    assert_eq!(installed[0].quantization_level.as_deref(), Some("Q4_K_M"));

    assert_eq!(installed[1].name, "bare:latest");
    assert_eq!(installed[1].parameter_size, None, "absent details are absent, not empty strings");
}

#[test]
fn a_pull_reports_progress_per_layer_and_succeeds_on_the_success_line() {
    // The manager renders a bar from these, so a pull that reported nothing
    // until it finished would look hung for the length of a multi-GB download.
    let lines = vec![
        json!({ "status": "pulling manifest" }).to_string(),
        json!({ "status": "pulling", "digest": "sha256:aa", "completed": 50u64, "total": 200u64 })
            .to_string(),
        json!({ "status": "pulling", "digest": "sha256:aa", "completed": 200u64, "total": 200u64 })
            .to_string(),
        json!({ "status": "success" }).to_string(),
    ];
    let (base, seen) = fake_ollama_manager(json!({ "models": [] }), lines);

    let mut updates: Vec<harness::PullProgress> = Vec::new();
    let cancel = AtomicBool::new(false);

    OpenHarness::ollama_at(&base)
        .pull_model("qwen2.5:0.5b", &cancel, &mut |p| updates.push(p))
        .expect("a stream ending in success is a completed pull");

    assert!(updates.len() >= 2, "one update per line, got {updates:?}");
    assert!(
        updates.iter().any(|u| u.completed == Some(50) && u.total == Some(200)),
        "a partly-downloaded layer must report its counters: {updates:?}",
    );
    assert_eq!(updates.last().map(|u| u.status.as_str()), Some("success"));

    // What a host actually renders: one bar across layers.
    let mut bar = harness::PullProgressAggregator::default();
    let percent = updates.iter().filter_map(|u| bar.update(u)).last();
    assert_eq!(percent, Some(100.0), "a finished pull ends at 100%");

    let (url, body) = seen.lock().unwrap().iter().find(|(u, _)| u.starts_with("/api/pull")).cloned().expect("pull call");
    assert_eq!(url, "/api/pull");
    assert_eq!(body["model"], "qwen2.5:0.5b", "the model asked for is the model requested");
    assert_eq!(body["stream"], true, "a non-streaming pull reports nothing until it ends");
}

#[test]
fn a_pull_that_errors_mid_stream_is_a_failure_not_a_quiet_success() {
    // `pull_stream_surfaces_error_line` already pins the stream parser. What
    // this adds is the trip back out: Ollama reports a failed pull as an
    // `error` line on a *200* response, so the failure has to survive being
    // mapped through `pull_model` — a boundary that turned the Err into an Ok
    // would look like a download that finished.
    let lines = vec![
        json!({ "status": "pulling manifest" }).to_string(),
        json!({ "error": "model \"nope\" not found" }).to_string(),
    ];
    let (base, _) = fake_ollama_manager(json!({ "models": [] }), lines);
    let cancel = AtomicBool::new(false);

    let result = OpenHarness::ollama_at(&base).pull_model("nope", &cancel, &mut |_| {});

    let message = result.expect_err("an error line must fail the pull");
    assert!(format!("{message}").contains("not found"), "got {message}");
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
    seeded_session_parts(tag, &[Part::Turns(turns)])
}

/// A piece of a seeded conversation, so a test can place a summary marker part
/// way through and give it a tail.
enum Part {
    Turns(usize),
    Summary,
}

fn seeded_session_parts(tag: &str, parts: &[Part]) -> (std::path::PathBuf, String) {
    let root = std::env::temp_dir().join(format!("hl-compact-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("sessions")).unwrap();

    let id = "ses-long";
    let mut lines = vec![format!(
        r#"{{"type":"session","id":"{id}","title":"long","created_at":1,"updated_at":1}}"#
    )];
    let mut i = 0;
    for part in parts {
        match part {
            Part::Summary => lines.push(
                r#"{"type":"message","role":"compaction","content":"An earlier summary."}"#.to_owned(),
            ),
            Part::Turns(turns) => {
                for _ in 0..*turns {
                    let filler = "some earlier discussion that is long enough to matter ".repeat(6);
                    lines.push(format!(
                        r#"{{"type":"message","role":"user","content":"question {i}: {filler}"}}"#
                    ));
                    lines.push(format!(
                        r#"{{"type":"message","role":"assistant","content":"answer {i}: {filler}"}}"#
                    ));
                    i += 1;
                }
            }
        }
    }
    std::fs::write(root.join("sessions").join(format!("{id}.jsonl")), lines.join("\n") + "\n").unwrap();
    (root, id.to_owned())
}

fn compaction_fired(events: &[RunEvent]) -> bool {
    events.iter().any(|e| {
        matches!(e, RunEvent::Activity { message, .. } if message.contains("compacted the conversation"))
    })
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

/// Compaction reserves *half* the window, not a sliver, and fires while there
/// is still room. On a small local window a quarter is only ~1K of headroom, so
/// the request brushes the limit before compaction fires and the model
/// truncates mid-prompt — the reserve is what buys the slack.
#[test]
fn compaction_fires_while_half_the_window_is_still_free() {
    let (root, session_id) = seeded_session("halffree", 40);
    let tags = json!({ "models": [ { "name": "test-model" } ] });
    let (base, _seen) = fake_ollama_with_context(
        tags,
        vec![done_line("A summary of the earlier turns."), done_line("Answer.")],
        12_000,
    );

    // The seeded history is well under this window, and over half of it. A
    // reserve that shrank to nothing would wait until the window was full.
    let harness = OpenHarness::ollama_at(&base).with_session_dir(&root).with_context_tokens(12_000);
    let events = collect_resuming(&harness, "and finally?", Some(session_id));

    assert!(compaction_fired(&events), "a conversation past half the window is compacted: {events:?}");
    let _ = std::fs::remove_dir_all(&root);
}

/// How much of the conversation survives verbatim scales with the window. The
/// tail budget is a quarter of it, so a wide window keeps the recent turns
/// intact where a narrow one summarizes nearly everything.
#[test]
fn a_wider_window_keeps_more_of_the_conversation_verbatim() {
    let (root, session_id) = seeded_session("verbatim", 140);
    let tags = json!({ "models": [ { "name": "test-model" } ] });
    let (base, _seen) = fake_ollama_with_context(
        tags,
        vec![done_line("A summary of the earlier turns."), done_line("Answer.")],
        32_000,
    );

    let harness = OpenHarness::ollama_at(&base).with_session_dir(&root).with_context_tokens(32_000);
    let events = collect_resuming(&harness, "and finally?", Some(session_id.clone()));
    assert!(compaction_fired(&events), "the seeded history is over the threshold: {events:?}");

    let saved = std::fs::read_to_string(root.join("sessions").join(format!("{session_id}.jsonl")))
        .expect("the session file");
    // Everything after the marker was kept rather than summarized. A quarter of
    // this window is ~8k tokens of tail, which is tens of messages; a budget
    // that ignored the window would leave roughly a dozen.
    let kept = saved.lines().skip_while(|l| !l.contains(r#""role":"compaction""#)).count() - 1;
    assert!(kept >= 30, "a 32k window should keep tens of turns verbatim, kept {kept}");
    let _ = std::fs::remove_dir_all(&root);
}

/// A second compaction summarizes only what is new. The marker is where the
/// last one ended, and only turns *after* it are candidates — otherwise a long
/// session accumulates summaries of its own summaries.
///
/// Two runs, identical but for the length of the tail after the marker. The
/// short one must decline and the long one must compact: without the second
/// half this test passes just as well when the threshold is never reached at
/// all, which is exactly how its first version went vacuous.
#[test]
fn a_second_compaction_summarizes_only_what_is_new() {
    // Small enough that the windowed view — which starts at the marker, not at
    // the beginning — is over the threshold on its own.
    const WINDOW: u64 = 1_200;

    let (root, session_id) =
        seeded_session_parts("nothingnew", &[Part::Turns(10), Part::Summary, Part::Turns(4)]);
    let tags = json!({ "models": [ { "name": "test-model" } ] });
    let (base, seen) = fake_ollama_with_context(tags, vec![done_line("Answer.")], WINDOW);
    let harness = OpenHarness::ollama_at(&base).with_session_dir(&root).with_context_tokens(WINDOW);
    let events = collect_resuming(&harness, "and finally?", Some(session_id.clone()));

    assert!(!compaction_fired(&events), "the turns since the summary still fit: {events:?}");
    let chats = seen.lock().unwrap().iter().filter(|(u, _)| u.starts_with("/api/chat")).count();
    assert_eq!(chats, 1, "no summarization request, just the turn");
    let saved = std::fs::read_to_string(root.join("sessions").join(format!("{session_id}.jsonl")))
        .expect("the session file");
    assert_eq!(
        saved.matches(r#""role":"compaction""#).count(),
        1,
        "the existing summary is not joined by a summary of itself"
    );
    let _ = std::fs::remove_dir_all(&root);

    // Same window, same marker, more said since: now there is something to
    // summarize, so it fires. This is what proves the case above declined on
    // the boundary rather than never being considered.
    let (root, session_id) =
        seeded_session_parts("somethingnew", &[Part::Turns(10), Part::Summary, Part::Turns(14)]);
    let tags = json!({ "models": [ { "name": "test-model" } ] });
    let (base, _seen) = fake_ollama_with_context(
        tags,
        vec![done_line("A summary of the newer turns."), done_line("Answer.")],
        WINDOW,
    );
    let harness = OpenHarness::ollama_at(&base).with_session_dir(&root).with_context_tokens(WINDOW);
    let events = collect_resuming(&harness, "and finally?", Some(session_id));
    assert!(compaction_fired(&events), "a tail past the budget is summarized: {events:?}");
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

    // A dedicated empty directory, not the shared temp dir. `/tmp` on a CI
    // runner is full of interesting-looking things, and a 0.5b model given a
    // trivial prompt and a populated workspace goes exploring instead of
    // answering: it asked to read `/tmp` — absolute, which the tools refuse by
    // design, since a tool path is relative to the workspace — then retried the
    // same call until it hit the turn limit, producing no text at all. Nothing
    // to look at means nothing to be distracted by.
    let workspace = std::env::temp_dir().join("agent-harness-live-ollama");
    let _ = std::fs::create_dir_all(&workspace);

    let events: Arc<Mutex<Vec<RunEvent>>> = Arc::default();
    let sink = Arc::clone(&events);
    let done = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&done);
    let _handle = harness
        .start(
            RunRequest {
                run_id: "live".to_owned(),
                // Says not to use tools, because the smallest tool-capable
                // models will otherwise reach for one on any prompt at all.
                prompt: "Reply with the single word: pong. Do not use any tools."
                    .to_owned(),
                cwd: Some(workspace.clone()),
                mode: RunMode::Ask,
                tools: harness::ToolAccess::Default,
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
    if text.trim().is_empty() {
        // Name the likely cause rather than printing every event: a model that
        // spent the run calling tools never got to an answer, and forty
        // ToolStart/ToolEnd pairs bury that.
        let tools = events.iter().filter(|e| matches!(e, RunEvent::ToolStart { .. })).count();
        panic!(
            "a live run must produce text; got none after {tools} tool call(s) \
             across {} events — if that count is high the model looped instead \
             of answering",
            events.len()
        );
    }
    assert!(events.iter().any(|e| matches!(e, RunEvent::Exited { .. })), "must terminate");
}

/// Not a word, so a model cannot guess it, and absent from the prompt, so it
/// cannot be echoed. It reaches an answer only by way of the file.
/// `ToolAccess::None` has to hold at dispatch, not only at the offer. A model
/// trained with its own tool syntax calls one whether or not any was offered,
/// so "nothing advertised" is not the same guarantee as "nothing reachable" —
/// and it was the weaker one that shipped: gpt-oss, handed an empty tool array,
/// asked for `glob`, `list`, `read` and `bash`, and every call ran.
///
/// Scripted here rather than left to the live job, which is advisory: this is a
/// sandbox promise, so it needs a test that cannot be skipped or shrugged off.
#[test]
fn tool_access_none_refuses_a_call_the_model_makes_anyway() {
    let tags = json!({ "models": [ { "name": "test-model" } ] });
    let unbidden_call = json!({
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
        fake_ollama(tags, vec![unbidden_call, done_line("Nothing was available.")]);

    let events = collect_with_tools(
        &OpenHarness::ollama_at(&base),
        "what is here?",
        harness::ToolAccess::None,
    );

    assert!(offered_tools(&seen).is_empty(), "nothing may be offered");
    // The call is reported — a host should see what the model reached for —
    // but it must come back refused. `ok: true` here means it ran.
    let ran: Vec<&RunEvent> =
        events.iter().filter(|e| matches!(e, RunEvent::ToolEnd { ok: true, .. })).collect();
    assert!(ran.is_empty(), "a tool ran that was never offered: {ran:?}");
}


/// A run that ends with an error must not also report success.
///
/// A silent turn used to emit the error and exit 0 beside it — the two
/// terminal signals contradicting each other, so a caller reading the exit
/// code, which is the conventional one, still saw a success holding an empty
/// string. That is the thing naming the stop reason was added to prevent, and
/// naming it is only half of it.
#[test]
fn a_turn_that_says_nothing_ends_as_a_failure() {
    let tags = json!({ "models": [ { "name": "test-model" } ] });
    let silent = json!({
        "message": { "role": "assistant", "content": "" },
        "done": true,
        "done_reason": "stop",
        "prompt_eval_count": 5,
        "eval_count": 0
    })
    .to_string();
    let (base, _seen) = fake_ollama(tags, vec![silent]);

    let events = collect(&OpenHarness::ollama_at(&base), "say something");

    let message = events
        .iter()
        .find_map(|e| match e {
            RunEvent::Error { message, .. } => Some(message.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("a turn that said nothing must say why: {events:?}"));
    assert!(message.contains("no answer"), "the error must name the outcome: {message}");
    assert!(message.contains("stop"), "and why the provider stopped: {message}");

    let exit = events.iter().find_map(|e| match e {
        RunEvent::Exited { exit_code, .. } => Some(*exit_code),
        _ => None,
    });
    assert_eq!(exit, Some(Some(1)), "an errored run must not exit 0: {events:?}");
}


const PLANTED: &str = "quartzine-80417";

/// Plant the token, run one live prompt under `tools`, and report which tools
/// actually *ran* and what the model said. Attempts are not the measure: a
/// refused call is still reported as a start, so only a `ToolEnd` that
/// succeeded counts. Shared by the two tests below, which differ only in what
/// the run was allowed to reach.
fn live_tool_run(tools: harness::ToolAccess) -> (Vec<String>, String) {
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

    // Rebuilt per run: a leftover from the previous one would let a model that
    // never called a tool still find the token.
    let workspace = std::env::temp_dir().join(format!("agent-harness-live-tool-{tools:?}"));
    let _ = std::fs::remove_dir_all(&workspace);
    std::fs::create_dir_all(&workspace).expect("workspace");
    std::fs::write(workspace.join("note.txt"), format!("code: {PLANTED}\n")).expect("planted file");

    let events: Arc<Mutex<Vec<RunEvent>>> = Arc::default();
    let sink = Arc::clone(&events);
    let done = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&done);
    let _handle = harness
        .start(
            RunRequest {
                run_id: "live-tool".to_owned(),
                prompt: "Read the file note.txt in the working directory and reply with \
                         the code it contains, and nothing else."
                    .to_owned(),
                cwd: Some(workspace.clone()),
                mode: RunMode::Ask,
                tools,
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

    for _ in 0..12_000 {
        if done.load(Ordering::SeqCst) {
            break;
        }
        thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(done.load(Ordering::SeqCst), "live run did not finish within 10 minutes");

    let events = events.lock().unwrap().clone();
    if let Some(RunEvent::Error { message, .. }) = events.iter().find(|e| matches!(e, RunEvent::Error { .. })) {
        panic!("live run errored: {message}");
    }
    assert!(events.iter().any(|e| matches!(e, RunEvent::Exited { .. })), "must terminate");
    let started: std::collections::HashMap<&str, &str> = events
        .iter()
        .filter_map(|e| match e {
            RunEvent::ToolStart { tool_call_id, title, .. } => {
                Some((tool_call_id.as_str(), title.as_str()))
            }
            _ => None,
        })
        .collect();
    let ran = events
        .iter()
        .filter_map(|e| match e {
            RunEvent::ToolEnd { tool_call_id, ok: true, .. } => {
                Some(started.get(tool_call_id.as_str()).copied().unwrap_or("?").to_owned())
            }
            _ => None,
        })
        .collect();
    let text = events
        .iter()
        .filter_map(|e| match e {
            RunEvent::Text { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    let _ = std::fs::remove_dir_all(&workspace);
    (ran, text)
}

/// The tool loop against a **real** model, both arms in one test — the half the
/// scripted tests cannot reach. `a_tool_call_round_trips_its_result_back_to_the_model`
/// proves the loop runs a tool and carries its output into the next turn, and
/// `tool_access_none_refuses_a_call_the_model_makes_anyway` proves a call is
/// refused when nothing was offered. Both script the model, so neither says
/// anything about a real one choosing a tool.
///
/// The two arms belong together, in this order, because the second is only
/// evidence given the first. A model that never calls a tool passes the
/// withheld arm for the wrong reason — which is exactly what CI's 0.5b model
/// did — so proving the token is reachable *with* tools is what makes proving
/// it unreachable *without* them mean anything.
///
/// The assertion is the round trip, not the wording: a token appearing nowhere
/// in the prompt reaches the answer only by way of the file.
///
/// Run by hand, not by CI, and that is a measurement rather than a preference:
/// a tool round trip needs a model a CPU runner cannot host. qwen2.5:0.5b never
/// calls a tool; llama3.2:3b loops to the turn limit without answering; what
/// worked was gpt-oss:20b, at 13GB. So the promise this covers is guarded
/// deterministically by `tool_access_none_refuses_a_call_the_model_makes_anyway`
/// in the normal suite, and this stays the check a human runs against a real
/// model before a release:
///
/// ```text
/// OLLAMA_TEST_MODEL=gpt-oss:20b cargo test --all-features --test ollama_route \
///   -- --ignored live_ollama_reaches
/// ```
#[test]
#[ignore = "live: needs Ollama with a tool-capable OLLAMA_TEST_MODEL (gpt-oss:20b)"]
fn live_ollama_reaches_the_file_with_tools_and_cannot_without_them() {
    let (ran, text) = live_tool_run(harness::ToolAccess::Default);
    assert!(!ran.is_empty(), "offered tools and ran none; it cannot have read the file");
    assert!(
        text.contains(PLANTED),
        "the answer must carry what the tool read. ran {ran:?}, answered {text:?}"
    );

    let (ran, text) = live_tool_run(harness::ToolAccess::None);
    assert!(ran.is_empty(), "nothing was offered, yet these tools ran: {ran:?}");
    assert!(!text.contains(PLANTED), "the file was reached with no tools offered: {text:?}");
}
