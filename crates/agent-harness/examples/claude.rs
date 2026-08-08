//! **Claude Code** — run a prompt and stream the normalized events.
//!
//! ```text
//! cargo run --example claude   # needs the `claude` CLI, installed + signed in
//! ```
//!
//! Every other example uses this same loop. Only the constructor changes.

use harness::{Claude, Harness, HarnessError, RunEvent, RunMode, RunRequest, RunTuning};

fn main() -> Result<(), HarnessError> {
    // Drives the `claude` CLI. See `codex.rs` for the same loop, one line apart.
    let claude = Claude::new();

    // `run_channel()` starts the run and hands back the events on a channel,
    // so there's no callback/`Sender` plumbing to write by hand. It returns
    // immediately; events arrive on background threads. (`run()` is still
    // there for push semantics — forwarding straight onto a Tauri Channel or
    // SSE sink from inside a callback.)
    let (_handle, rx) = claude.run_channel(RunRequest {
        run_id: "demo".into(),
        prompt: "In one sentence, what is a Markdown heading?".into(),
        cwd: None,                    // working dir for the agent's tool calls
        mode: RunMode::Ask,           // Ask = answer only; Edit = may edit files
        tuning: RunTuning::default(), // optional: model / effort / max_turns
        resume: None,                 // Some(session_id) to continue a prior run
        attachments: Vec::new(),      // images for multimodal models; none here
    })?; // keep `_handle` to `.cancel()`; dropping it does NOT stop the run

    // ONE normalized event stream, regardless of the backing CLI. `rx` hangs
    // up on its own when the run ends, so this loop terminates without
    // touching the handle:
    for ev in rx {
        match ev {
            RunEvent::Text { delta, .. } => print!("{delta}"), // the answer
            RunEvent::Thinking { delta, .. } => eprint!("{delta}"), // model reasoning
            RunEvent::ToolStart { title, .. } => eprintln!("\n[tool] {title}"),
            RunEvent::Error { message, .. } => eprintln!("\n[error] {message}"),
            RunEvent::Exited { .. } => break,
            _ => {}
        }
    }
    Ok(())
}
