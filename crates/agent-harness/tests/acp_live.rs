//! The `acp` adapter against a **real ACP agent** (OpenCode).
//!
//! An ACP agent owns its own tool surface and the protocol has no way to say
//! "offer none", so `ToolAccess::None` is refused rather than accepted and
//! ignored. As with codex, the refusal is only honest if the tools really do
//! run — so both halves are checked against a real agent: the request is turned
//! away before anything is spawned, and a normal run reaches a file the prompt
//! never quotes.
//!
//! Ignored by default: needs `opencode` installed and configured with a model,
//! and a run costs tokens.
//!
//! ```text
//! cargo test --all-features --test acp_live -- --ignored
//! ```

#![cfg(feature = "acp")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use harness::{AcpHarness, Harness, RunEvent, RunMode, RunRequest, RunTuning, ToolAccess};

/// Not a word, and absent from the prompt: it reaches an answer only via the file.
const PLANTED: &str = "quartzine-80417";

/// A request an ACP agent cannot honour is turned away, not run.
///
/// Offline and instant — nothing is spawned, which is the point.
#[test]
#[ignore = "live: needs the opencode CLI installed"]
fn acp_refuses_a_no_tools_run_before_spawning_anything() {
    let started = std::time::Instant::now();
    let Err(error) = AcpHarness::opencode().start(
        RunRequest { tools: ToolAccess::None, ..RunRequest::default() },
        Arc::new(|_| {}),
    ) else {
        panic!("an ACP agent cannot withhold its own tools, so it must not accept the request");
    };
    let message = error.to_string();
    assert!(message.contains("ToolAccess::None"), "name what was asked: {message}");
    assert!(message.contains("acp"), "and which adapter refused: {message}");
    assert!(started.elapsed().as_secs() < 5, "the refusal must not spawn the agent");
}

/// And the refusal is honest: offered its tools, the agent reaches the file.
#[test]
#[ignore = "live: needs opencode installed and configured with a model; costs tokens"]
fn acp_reaches_the_file_when_its_tools_are_offered() {
    let workspace = std::env::temp_dir().join("agent-harness-acp-live");
    let _ = std::fs::remove_dir_all(&workspace);
    std::fs::create_dir_all(&workspace).expect("workspace");
    std::fs::write(workspace.join("note.txt"), format!("code: {PLANTED}\n")).expect("planted file");

    let agent = AcpHarness::opencode();
    let readiness = agent.readiness();
    assert!(readiness.ready, "opencode not ready: {:?}", readiness.error);

    let events: Arc<Mutex<Vec<RunEvent>>> = Arc::default();
    let sink = Arc::clone(&events);
    let done = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&done);
    let _handle = agent
        .start(
            RunRequest {
                run_id: "acp-live-tool".to_owned(),
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
        panic!("opencode errored: {message}");
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
        "the agent was offered its tools and must have read the file, said {said:?}"
    );
    let _ = std::fs::remove_dir_all(&workspace);
}
