//! Drive an external **ACP agent** (OpenCode, Gemini, Goose, …) over the Agent
//! Client Protocol and stream the same normalized [`RunEvent`]s as every other
//! harness — agent-harness spawns the agent and relays its session.
//!
//! ```text
//! cargo run --example acp --features acp   # needs `opencode` on PATH
//! ```

use harness::{AcpHarness, Harness, HarnessError, RunEvent, RunRequest};

fn main() -> Result<(), HarnessError> {
    // OpenCode over ACP — spawns `opencode acp` and relays its session stream.
    // Any other ACP agent works the same way — see `gemini.rs`.
    let agent = AcpHarness::opencode();

    let (_handle, events) = agent.run(RunRequest {
        run_id: "demo".into(),
        prompt: "In one sentence, what is the Agent Client Protocol?".into(),
        ..Default::default()
    })?;

    // One normalized stream, whichever ACP agent produced it.
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
    Ok(())
}
