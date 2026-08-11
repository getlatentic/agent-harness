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

use harness::{Harness, Error, OpenHarness, RunEvent, RunMode, RunRequest, RunTuning};

#[path = "common/mod.rs"]
mod common;

fn main() -> Result<(), Error> {
    // Local Ollama on its default port (no API key). For a hosted endpoint:
    //   OpenHarness::custom(OpenHarnessConfig {
    //       id: "openrouter".into(), display_name: "OpenRouter".into(),
    //       base_url: "https://openrouter.ai/api".into(),
    //       api_key_env: Some("OPENROUTER_API_KEY".into()), ..Default::default() })
    let model = OpenHarness::ollama();

    let readiness = model.readiness();
    if !readiness.ready {
        eprintln!("Ollama is not reachable: {}", readiness.error.unwrap_or_default());
        if let Some(hint) = model.manifest().install_hint {
            eprintln!("Get it from {}", hint.url);
        }
        return Ok(());
    }

    // Ollama has no default model, so a run must name one. Take it from the
    // environment, else the largest model installed — hardcoding an id here
    // would fail on any machine that pulled something else.
    //
    // Largest, not first: this loop offers the model nine tool schemas, and a
    // very small model answers by reciting them back as prose instead of
    // calling one. Ollama's `capabilities` does not separate the two — a 1B
    // model reports `tools` because its template has the syntax, not because
    // it can use it — so size is the signal available.
    let chosen = match std::env::var("OLLAMA_MODEL") {
        Ok(model) if !model.trim().is_empty() => model,
        _ => match common::largest_installed(&model)? {
            Some(name) => name,
            None => {
                eprintln!("No models installed. Try `ollama pull qwen2.5:7b-instruct`.");
                return Ok(());
            }
        },
    };
    eprintln!("[model] {chosen}");

    let (_handle, events) = model.run(RunRequest {
        run_id: "demo".into(),
        prompt: "In one sentence, what is an OpenAI-compatible API?".into(),
        mode: RunMode::Ask, // Ask = read-only tools; Edit = + write/edit/bash
        tuning: RunTuning { model: Some(chosen), ..Default::default() },
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
