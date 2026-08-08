//! **Claude Code** — run a prompt and stream the normalized events.
//!
//! ```text
//! cargo run --example claude   # needs the `claude` CLI, installed + signed in
//! ```
//!
//! Every other example uses this same loop. Only the constructor changes.

use harness::{Claude, Harness, HarnessError, RunEvent, RunRequest};

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
        // The rest have defaults worth knowing: no `cwd` (the agent's tools
        // run where the process does), `mode: RunMode::Ask` (answer only —
        // `Edit` lets it write files), no `resume` (pass a session id to
        // continue a prior run), and no `attachments` (images, for multimodal
        // models). `tuning` carries model, effort and turn cap.
        ..Default::default()
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
