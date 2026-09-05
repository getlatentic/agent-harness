//! The `codex` adapter against the **real CLI**.
//!
//! Codex cannot withhold its own tools, so `ToolAccess::None` is refused rather
//! than accepted and ignored. That refusal is only honest if the tools really
//! do run — a conservative-looking refusal covering an adapter that happened to
//! be harmless would be a different thing entirely. So both halves are checked
//! here against the CLI itself: the request is turned away before anything is
//! spawned, and a normal run reaches a file the prompt never quotes.
//!
//! Ignored by default: needs `codex` installed and signed in, and costs tokens.
//!
//! ```text
//! cargo test --all-features --test codex_live -- --ignored
//! ```

#![cfg(feature = "codex")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use harness::{Codex, Harness, RunEvent, RunMode, RunRequest, RunTuning, ToolAccess};

/// Not a word, and absent from the prompt: it reaches an answer only via the file.
const PLANTED: &str = "quartzine-80417";

/// A request codex cannot honour is turned away, not run.
///
/// Cheap and offline — the point is that it costs nothing because nothing is
/// spawned. Accepting it and running with tools anyway is the failure that let
/// 46 tool calls through 14 runs meant to have none.
#[test]
#[ignore = "live: needs the codex CLI installed and signed in"]
fn codex_refuses_a_no_tools_run_before_spawning_anything() {
    let started = std::time::Instant::now();
    let Err(error) = Codex::new().start(
        RunRequest { tools: ToolAccess::None, ..RunRequest::default() },
        Arc::new(|_| {}),
    ) else {
        panic!("codex cannot withhold its tools, so it must not accept the request");
    };
    let message = error.to_string();
    assert!(message.contains("ToolAccess::None"), "name what was asked: {message}");
    assert!(message.contains("codex"), "and which adapter refused: {message}");
    assert!(started.elapsed().as_secs() < 5, "the refusal must not spawn the CLI");
}

/// And the refusal is honest: offered its tools, codex reaches the file.
///
/// Without this the refusal above proves only that we return an error. What
/// makes it the right call is that the tools it cannot withhold genuinely run.
#[test]
#[ignore = "live: needs the codex CLI installed and signed in; costs tokens"]
fn codex_reaches_the_file_when_its_tools_are_offered() {
    let workspace = std::env::temp_dir().join("agent-harness-codex-live");
    let _ = std::fs::remove_dir_all(&workspace);
    std::fs::create_dir_all(&workspace).expect("workspace");
    std::fs::write(workspace.join("note.txt"), format!("code: {PLANTED}\n")).expect("planted file");

    let codex = Codex::new();
    let readiness = codex.readiness();
    assert!(readiness.ready, "codex not ready: {:?}", readiness.error);

    let events: Arc<Mutex<Vec<RunEvent>>> = Arc::default();
    let sink = Arc::clone(&events);
    let done = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&done);
    let _handle = codex
        .start(
            RunRequest {
                run_id: "codex-live-tool".to_owned(),
                prompt: "Read the file note.txt in the working directory and reply with \
                         the code it contains, and nothing else."
                    .to_owned(),
                cwd: Some(workspace.clone()),
                mode: RunMode::Ask,
                tools: ToolAccess::Default,
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
    if let Some(RunEvent::Error { message, .. }) =
        events.iter().find(|e| matches!(e, RunEvent::Error { .. }))
    {
        panic!("codex errored: {message}");
    }
    let said: String = events
        .iter()
        .filter_map(|e| match e {
            RunEvent::Text { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        said.contains(PLANTED),
        "codex was offered its tools and must have read the file, said {said:?}"
    );
    let _ = std::fs::remove_dir_all(&workspace);
}
