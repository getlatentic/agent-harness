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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use harness::{Claude, Harness, RunEvent, RunMode, RunRequest, RunTuning, ToolAccess};

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
