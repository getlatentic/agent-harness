//! **OpenAI Codex** — the same loop as `claude.rs`, one line apart.
//!
//! ```text
//! cargo run --example codex   # needs the `codex` CLI, installed + signed in
//! ```
//!
//! This file exists to make the point concrete: swapping the agent is a
//! constructor change. Diff it against `claude.rs` — nothing else moves.

use harness::{Codex, Harness, HarnessError, RunEvent, RunRequest};

fn main() -> Result<(), HarnessError> {
    let codex = Codex::new();

    let (_handle, events) = codex.run(RunRequest {
        run_id: "demo".into(),
        prompt: "In one sentence, what is a Markdown heading?".into(),
        ..Default::default()
    })?;

    for event in events {
        match event {
            RunEvent::Text { delta, .. } => print!("{delta}"),
            RunEvent::Thinking { delta, .. } => eprint!("{delta}"),
            RunEvent::ToolStart { title, .. } => eprintln!("\n[tool] {title}"),
            RunEvent::Error { message, .. } => eprintln!("\n[error] {message}"),
            RunEvent::Exited { .. } => break,
            _ => {}
        }
    }
    println!();
    Ok(())
}
