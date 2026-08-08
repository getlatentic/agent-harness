//! **OpenAI Codex** — the same loop as `claude.rs`, one line apart.
//!
//! ```text
//! cargo run --example codex   # needs the `codex` CLI, installed + signed in
//! ```
//!
//! This file exists to make the point concrete: swapping the agent is a
//! constructor change. Diff it against `claude.rs` — nothing else moves.

use harness::{Codex, Harness, HarnessError, RunEvent, RunMode, RunRequest, RunTuning};

fn main() -> Result<(), HarnessError> {
    let codex = Codex::new();

    let (_handle, rx) = codex.run_channel(RunRequest {
        run_id: "demo".into(),
        prompt: "In one sentence, what is a Markdown heading?".into(),
        cwd: None,
        mode: RunMode::Ask,
        // Codex exposes reasoning effort but no turn cap. Model ids change
        // often, so free text is accepted: RunTuning { model: Some("o4-mini".into()), .. }
        tuning: RunTuning::default(),
        resume: None,
        attachments: Vec::new(),
    })?;

    for ev in rx {
        match ev {
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
