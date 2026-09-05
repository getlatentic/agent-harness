//! The `claude` adapter against the **real CLI**.
//!
//! This adapter is where a host's no-tools runs actually leaked: `ToolAccess::None`
//! was passed as `--allowedTools ""`, which is the CLI's auto-approve list and
//! gates nothing, so 14 judging runs made 46 tool calls between them. It now
//! passes `--disallowedTools "*"`.
//!
//! That correction was measured by hand and pinned by a unit test on the argv.
//! An argv test proves what we send, never what the CLI does with it — and the
//! bug was precisely a flag that parsed, ran, and withheld nothing. So the
//! guarantee is checked here against the CLI itself.
//!
//! Ignored by default: needs `claude` installed and signed in, and costs tokens.
//!
//! ```text
//! cargo test --all-features --test claude_live -- --ignored
//! ```

#![cfg(feature = "claude")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use harness::{Claude, FnTool, Harness, RunEvent, RunMode, RunRequest, RunTuning, ToolAccess, ToolServer};

/// Not a word, and absent from the prompt: it reaches an answer only via the file.
const PLANTED: &str = "quartzine-80417";

/// Run one prompt under `tools`; report which tools actually *ran* and what was said.
///
/// Attempts are not the measure. A refused call still emits `ToolStart`, so only
/// a `ToolEnd` that succeeded counts — the distinction this adapter's bug hid.
fn run_under(tools: ToolAccess) -> (Vec<String>, String) {
    let workspace = std::env::temp_dir().join(format!("agent-harness-claude-{tools:?}"));
    let _ = std::fs::remove_dir_all(&workspace);
    std::fs::create_dir_all(&workspace).expect("workspace");
    std::fs::write(workspace.join("note.txt"), format!("code: {PLANTED}\n")).expect("planted file");

    let claude = Claude::new();
    let readiness = claude.readiness();
    assert!(readiness.ready, "claude not ready: {:?}", readiness.error);

    let events: Arc<Mutex<Vec<RunEvent>>> = Arc::default();
    let sink = Arc::clone(&events);
    let done = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&done);
    let _handle = claude
        .start(
            RunRequest {
                run_id: "claude-live-tool".to_owned(),
                prompt: "Read the file note.txt in the working directory and reply with \
                         the code it contains, and nothing else."
                    .to_owned(),
                cwd: Some(workspace.clone()),
                mode: RunMode::Ask,
                tools,
                tuning: RunTuning::default(),
                resume: None,
                attachments: Vec::new(),
            },
            Arc::new(move |event| {
                if matches!(event, RunEvent::Exited { .. }) {
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
    let started: std::collections::HashMap<String, String> = events
        .iter()
        .filter_map(|e| match e {
            RunEvent::ToolStart { tool_call_id, title, .. } => {
                Some((tool_call_id.clone(), title.clone()))
            }
            _ => None,
        })
        .collect();
    let ran = events
        .iter()
        .filter_map(|e| match e {
            RunEvent::ToolEnd { tool_call_id, ok: true, .. } => {
                Some(started.get(tool_call_id).cloned().unwrap_or_else(|| "?".to_owned()))
            }
            _ => None,
        })
        .collect();
    let said = events
        .iter()
        .filter_map(|e| match e {
            RunEvent::Text { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    let _ = std::fs::remove_dir_all(&workspace);
    (ran, said)
}

/// Offered first, withheld second — the order is the point. A run that never
/// calls a tool passes the withheld arm for the wrong reason, so proving the
/// token is reachable *with* tools is what makes it evidence when it is not.
#[test]
#[ignore = "live: needs the claude CLI installed and signed in; costs tokens"]
fn claude_reaches_the_file_with_tools_and_cannot_without_them() {
    let (ran, said) = run_under(ToolAccess::Default);
    assert!(!ran.is_empty(), "offered tools and ran none; it cannot have read the file");
    assert!(
        said.contains(PLANTED),
        "the answer must carry what the tool read — ran {ran:?}, said {said:?}"
    );

    let (ran, said) = run_under(ToolAccess::None);
    assert!(ran.is_empty(), "nothing was offered, yet these ran: {ran:?}");
    assert!(!said.contains(PLANTED), "the file was reached with no tools offered: {said:?}");
}

/// Run one prompt against a Claude carrying a host tool that returns the
/// planted token; report whether the tool was called in this process, which
/// tools ran, what was said, and how the run ended.
fn run_with_host_tool(tools: ToolAccess) -> (bool, Vec<String>, String, Option<i32>) {
    let called = Arc::new(AtomicBool::new(false));
    let seen = Arc::clone(&called);
    let server = ToolServer::new("host").with_tool(
        FnTool::new(
            "secret_word",
            "Returns the secret word.",
            serde_json::json!({ "type": "object", "properties": {} }),
            move |_| {
                seen.store(true, Ordering::SeqCst);
                Ok(PLANTED.to_owned())
            },
        )
        .as_read_only(),
    );
    let claude = Claude::new().with_tool_server(server);
    assert!(claude.readiness().ready, "claude not ready");

    let events: Arc<Mutex<Vec<RunEvent>>> = Arc::default();
    let sink = Arc::clone(&events);
    let done = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&done);
    let workspace = std::env::temp_dir().join(format!("agent-harness-claude-host-{tools:?}"));
    std::fs::create_dir_all(&workspace).expect("workspace");
    let _handle = claude
        .start(
            RunRequest {
                run_id: format!("claude-live-host-{tools:?}"),
                prompt: "Call the `secret_word` tool from the `host` MCP server, then reply with \
                         exactly the word it returns and nothing else."
                    .to_owned(),
                cwd: Some(workspace),
                mode: RunMode::Ask,
                tools,
                tuning: RunTuning { max_turns: Some(4), ..RunTuning::default() },
                ..Default::default()
            },
            Arc::new(move |event| {
                if matches!(event, RunEvent::Exited { .. }) {
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
    let ran = events
        .iter()
        .filter_map(|e| match e {
            RunEvent::ToolStart { title, .. } => Some(title.clone()),
            _ => None,
        })
        .collect();
    let said = events
        .iter()
        .filter_map(|e| match e {
            RunEvent::Text { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    let exit = events.iter().find_map(|e| match e {
        RunEvent::Exited { exit_code, .. } => Some(*exit_code),
        _ => None,
    });
    (called.load(Ordering::SeqCst), ran, said, exit.flatten())
}

/// The control channel end to end against the real CLI: the closure in this
/// test runs because Claude asked for it, and the answer carries what it
/// returned. Then the same harness under `ToolAccess::None`: the host tool must
/// not run and the run must still end cleanly, because that arm takes the
/// one-way spawn and proves it was left alone.
#[test]
#[ignore = "live: needs the claude CLI installed and signed in; costs tokens"]
fn claude_calls_a_host_tool_in_this_process_and_none_withholds_it() {
    let (called, ran, said, exit) = run_with_host_tool(ToolAccess::Default);
    assert!(called, "the host closure never ran; ran {ran:?}, said {said:?}");
    assert!(ran.iter().any(|t| t.contains("secret_word")), "no tool event for the call: {ran:?}");
    assert!(said.contains(PLANTED), "the answer must carry what the tool returned: {said:?}");
    assert_eq!(exit, Some(0), "the CLI must exit once stdin closes after the result");

    let (called, ran, said, exit) = run_with_host_tool(ToolAccess::None);
    assert!(!called && ran.is_empty(), "nothing was offered, yet the host tool ran: {ran:?}");
    assert!(!said.contains(PLANTED), "the token was reached with no tools offered: {said:?}");
    assert_eq!(exit, Some(0), "a None run with a server attached still ends cleanly");
}

/// Run one prompt under `tools` with a schema; report the structured answer, the
/// tools that started, and everything said.
fn run_with_schema(tools: ToolAccess) -> (Option<serde_json::Value>, Vec<String>, String) {
    let workspace = std::env::temp_dir().join(format!("agent-harness-claude-schema-{tools:?}"));
    let _ = std::fs::remove_dir_all(&workspace);
    std::fs::create_dir_all(&workspace).expect("workspace");
    std::fs::write(workspace.join("note.txt"), format!("code: {PLANTED}\n")).expect("planted file");

    let events: Arc<Mutex<Vec<RunEvent>>> = Arc::default();
    let sink = Arc::clone(&events);
    let done = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&done);
    let _handle = Claude::new()
        .start(
            RunRequest {
                run_id: format!("claude-live-schema-{tools:?}"),
                prompt: "Read the file note.txt in the working directory and report the code it contains. \
                         If you cannot read it, put the reason in the code field."
                    .to_owned(),
                cwd: Some(workspace),
                mode: RunMode::Ask,
                tools,
                tuning: RunTuning {
                    max_turns: Some(4),
                    output_schema: Some(serde_json::json!({
                        "type": "object", "properties": { "code": { "type": "string" } }, "required": ["code"]
                    })),
                    ..RunTuning::default()
                },
                ..Default::default()
            },
            Arc::new(move |event| {
                if matches!(event, RunEvent::Exited { .. }) {
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
    let structured = events.iter().find_map(|e| match e {
        RunEvent::StructuredOutput { value, .. } => Some(value.clone()),
        _ => None,
    });
    let started = events
        .iter()
        .filter_map(|e| match e {
            RunEvent::ToolStart { title, .. } => Some(title.clone()),
            _ => None,
        })
        .collect();
    let said = events
        .iter()
        .filter_map(|e| match e {
            RunEvent::Text { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    (structured, started, said)
}

/// The schema is honoured in both directions of the tool question. With tools,
/// the shape arrives holding what the file said. Without them — the arm that
/// used to fail, because the wildcard denied the very tool that carries the
/// schema — the shape still arrives, holding the model's account of having no
/// way to read the file, and the file is never reached.
#[test]
#[ignore = "live: needs the claude CLI installed and signed in; costs tokens"]
fn claude_fits_a_schema_with_tools_and_still_fits_it_with_none() {
    let (structured, started, said) = run_with_schema(ToolAccess::Default);
    let value = structured.expect("a structured answer with tools offered");
    assert!(value["code"].as_str().is_some_and(|c| c.contains(PLANTED)), "the shape holds the file's code: {value}");

    let (structured, started_none, said_none) = run_with_schema(ToolAccess::None);
    let value = structured.expect("a structured answer with tools withheld");
    assert!(value["code"].is_string(), "the shape is still filled: {value}");
    assert!(!value.to_string().contains(PLANTED) && !said_none.contains(PLANTED), "the file was reached with no tools: {value} / {said_none}");
    assert!(
        started_none.iter().all(|t| t == "StructuredOutput"),
        "only the tool that carries the answer may start under None: {started_none:?}"
    );
    let _ = (started, said);
}

/// A model call, not an agent run: the replaced prompt is the whole prompt and
/// thinking is off. The measure is the token count the CLI reports — the agent's
/// envelope alone is ~7,000 tokens a call, so a cap well under that is the
/// regression guard for "the envelope came back".
#[test]
#[ignore = "live: needs the claude CLI installed and signed in; costs tokens"]
fn claude_as_a_model_pays_hundreds_of_prompt_tokens_not_thousands() {
    let events: Arc<Mutex<Vec<RunEvent>>> = Arc::default();
    let sink = Arc::clone(&events);
    let done = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&done);
    let _handle = Claude::new()
        .start(
            RunRequest {
                run_id: "claude-live-model".to_owned(),
                prompt: "Reply with exactly the word MODELOK and nothing else.".to_owned(),
                cwd: Some(std::env::temp_dir()),
                mode: RunMode::Ask,
                tools: ToolAccess::None,
                tuning: RunTuning {
                    system_prompt: Some("You answer exactly as instructed, in plain text.".to_owned()),
                    max_thinking_tokens: Some(0),
                    max_turns: Some(1),
                    ..RunTuning::default()
                },
                ..Default::default()
            },
            Arc::new(move |event| {
                if matches!(event, RunEvent::Exited { .. }) {
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
    let said: String = events
        .iter()
        .filter_map(|e| match e {
            RunEvent::Text { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert!(said.contains("MODELOK"), "the replaced prompt still yields the answer: {said:?}");
    let prompt_tokens = events
        .iter()
        .find_map(|e| match e {
            RunEvent::Usage { input_tokens, cache_read_tokens, cache_write_tokens, .. } => {
                Some(input_tokens.unwrap_or(0) + cache_read_tokens.unwrap_or(0) + cache_write_tokens.unwrap_or(0))
            }
            _ => None,
        })
        .expect("a usage event");
    assert!(prompt_tokens > 0 && prompt_tokens < 1500, "the agent's envelope is back in the prompt: {prompt_tokens} tokens");
}
