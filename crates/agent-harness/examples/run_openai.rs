//! Drive an **OpenAI-compatible / local model** with the built-in runtime
//! ([`OpenHarness`], the `openai-compatible` feature) — it *owns* the agent loop
//! and tool surface, no external CLI to wrap, and emits the same [`RunEvent`]s.
//!
//! ```text
//! cargo run --example run_openai --features openai-compatible
//! # needs a reachable model, e.g. `ollama serve` + `ollama pull qwen2.5-coder`
//! ```

use harness::{Harness, HarnessError, OpenHarness, RunEvent, RunMode, RunRequest, RunTuning};

fn main() -> Result<(), HarnessError> {
    // Local Ollama on its default port (no API key). For a hosted endpoint:
    //   OpenHarness::custom("openrouter", "OpenRouter",
    //       "https://openrouter.ai/api", Some("OPENROUTER_API_KEY".into()), vec![])
    let model = OpenHarness::ollama();

    let (_handle, rx) = model.run_channel(RunRequest {
        run_id: "demo".into(),
        prompt: "In one sentence, what is an OpenAI-compatible API?".into(),
        cwd: None,
        mode: RunMode::Ask, // Ask = read-only tools; Edit = + write/edit/bash
        // OpenHarness owns the model, so pick one per run (Ollama has no default).
        tuning: RunTuning { model: Some("qwen2.5-coder".into()), ..Default::default() },
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
    Ok(())
}
