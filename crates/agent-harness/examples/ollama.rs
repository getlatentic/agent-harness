//! **Ollama** — a local model on the built-in runtime ([`OpenHarness`], the
//! `openai-compatible` feature). There is no CLI to wrap here: this crate owns
//! the agent loop and the tool surface, and emits the same [`RunEvent`]s.
//!
//! Ollama is the one provider with a native path. It uses `/api/*` rather than
//! `/v1`, so the context window is set correctly and models can be pulled.
//! Every other provider is plain configuration — see `openrouter.rs`.
//!
//! ```text
//! cargo run --example ollama --features openai-compatible
//! # needs `ollama serve` and at least one pulled model
//! # pick one explicitly with OLLAMA_MODEL=llama3.2:1b
//! ```

use harness::{Harness, HarnessError, OpenHarness, RunEvent, RunMode, RunRequest, RunTuning};

fn main() -> Result<(), HarnessError> {
    // Local Ollama on its default port (no API key). For a hosted endpoint:
    //   OpenHarness::custom(OpenHarnessConfig {
    //       id: "openrouter".into(), display_name: "OpenRouter".into(),
    //       base_url: "https://openrouter.ai/api".into(),
    //       api_key_env: Some("OPENROUTER_API_KEY".into()), ..Default::default() })
    let model = OpenHarness::ollama();

    let readiness = model.readiness();
    if !readiness.ready {
        eprintln!("Ollama is not reachable: {}", readiness.error.unwrap_or_default());
        if let Some(hint) = model.info().install_hint {
            eprintln!("Get it from {}", hint.url);
        }
        return Ok(());
    }

    // Ollama has no default model, so a run must name one. Take it from the
    // environment, else the first model actually installed — hardcoding an id
    // here would fail on any machine that pulled something else.
    let chosen = match std::env::var("OLLAMA_MODEL") {
        Ok(model) if !model.trim().is_empty() => model,
        _ => match model.list_models()?.into_iter().next() {
            Some(first) => first.value,
            None => {
                eprintln!("No models installed. Try `ollama pull llama3.2:1b`.");
                return Ok(());
            }
        },
    };
    eprintln!("[model] {chosen}");

    let (_handle, rx) = model.run(RunRequest {
        run_id: "demo".into(),
        prompt: "In one sentence, what is an OpenAI-compatible API?".into(),
        mode: RunMode::Ask, // Ask = read-only tools; Edit = + write/edit/bash
        tuning: RunTuning { model: Some(chosen), ..Default::default() },
        ..Default::default()
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
