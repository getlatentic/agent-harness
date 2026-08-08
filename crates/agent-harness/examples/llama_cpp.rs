//! **llama.cpp** — a local `llama-server` on the built-in runtime.
//!
//! ```text
//! llama-server -m model.gguf --port 8080 --jinja   # in another terminal
//! cargo run --example llama_cpp --features openai-compatible
//! ```
//!
//! `llama-server` serves the OpenAI chat API, so it needs no adapter — only a
//! `base_url`. It takes no API key, which is what `api_key_env: None` means.
//!
//! Two notes for local servers. Start `llama-server` with `--jinja` or it will
//! not emit tool calls, and the agent loop needs them. And llama.cpp serves one
//! model per process, so the model id is whatever that process loaded; the name
//! below is only a label.

use harness::{
    Harness, HarnessError, OpenHarness, OpenHarnessConfig, RunEvent, RunRequest, RunTuning,
};

fn main() -> Result<(), HarnessError> {
    let base_url = std::env::var("LLAMA_SERVER_URL")
        .unwrap_or_else(|_| "http://localhost:8080".into());

    let llama = OpenHarness::custom(OpenHarnessConfig {
        id: "llama-cpp".into(),
        display_name: "llama.cpp".into(),
        base_url: base_url.clone(),
        api_key_env: None, // a local server needs no key
        ..Default::default()
    });

    // Note: `readiness()` does NOT probe here. Ollama has a native endpoint to
    // ask, so `OpenHarness::ollama()` reports real reachability. A generic
    // OpenAI-compatible endpoint has nothing cheap to probe, so readiness
    // assumes a reachable cloud provider and reports ready. For a local server
    // that assumption is wrong, and the first run is what tells you it is down.
    // Watch for `RunEvent::Error` below instead.
    eprintln!("[endpoint] {base_url}");

    let (_handle, events) = llama.run(RunRequest {
        run_id: "demo".into(),
        prompt: "In one sentence, what is llama.cpp?".into(),
        // Whatever the server loaded. llama.cpp ignores this and serves its
        // one model, so it is a label rather than a selection.
        tuning: RunTuning { model: Some("local".into()), ..Default::default() },
        ..Default::default()
    })?;

    for event in events {
        match event {
            RunEvent::Text { delta, .. } => print!("{delta}"),
            RunEvent::Thinking { delta, .. } => eprint!("{delta}"),
            RunEvent::ToolStart { title, .. } => eprintln!("\n[tool] {title}"),
            RunEvent::Error { message, .. } => {
                eprintln!("\n[error] {message}");
                eprintln!("Is llama-server running at {base_url}?");
                eprintln!("Start one with: llama-server -m model.gguf --port 8080 --jinja");
            }
            RunEvent::Exited { .. } => break,
            _ => {}
        }
    }
    println!();
    Ok(())
}
