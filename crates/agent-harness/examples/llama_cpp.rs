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

use harness::{ApiKey, 
    Harness, HarnessError, OpenHarness, OpenHarnessConfig, RunEvent, RunRequest, RunTuning,
};

fn main() -> Result<(), HarnessError> {
    let base_url = std::env::var("LLAMA_SERVER_URL")
        .unwrap_or_else(|_| "http://localhost:8080".into());

    // Every tool the loop offers costs prompt tokens before the model reads a
    // word of your prompt, and a local server is usually started on a small
    // context. Name the ones you do not need for the job:
    //   LLAMA_DISABLE_TOOLS=webfetch,todowrite,bash,write,edit
    // `OpenHarness::builtin_tool_names()` lists what you can name here.
    let disabled_tools: Vec<String> = std::env::var("LLAMA_DISABLE_TOOLS")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect();

    let llama = OpenHarness::custom(OpenHarnessConfig {
        id: "llama-cpp".into(),
        display_name: "llama.cpp".into(),
        base_url: base_url.clone(),
        api_key: ApiKey::NotNeeded, // a local server needs no key
        disabled_tools,
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
        // A plain `llama-server` serves the one model it loaded and ignores
        // this, so `"local"` is a label. A front end that fronts several
        // (`llama serve --models-preset`, LM Studio) does resolve it, and
        // rejects a name it does not know — set `LLAMA_MODEL` there.
        tuning: RunTuning {
            model: Some(std::env::var("LLAMA_MODEL").unwrap_or_else(|_| "local".into())),
            ..Default::default()
        },
        ..Default::default()
    })?;

    for event in events {
        match event {
            RunEvent::Text { delta, .. } => print!("{delta}"),
            RunEvent::Thinking { delta, .. } => eprint!("{delta}"),
            RunEvent::ToolStart { title, .. } => eprintln!("\n[tool] {title}"),
            RunEvent::Error { message, .. } => {
                eprintln!("\n[error] {message}");
                // A local server fails in two ways worth telling apart, and a
                // rejected request means it is running, so do not ask whether
                // it is. The agent loop sends the system prompt plus nine tool
                // schemas — several thousand tokens before your prompt — which
                // overflows a server started on the 4096-token default.
                if message.contains("context size") {
                    eprintln!("Give llama-server a bigger context: -c 16384.");
                    eprintln!("Or send fewer tools: OpenHarnessConfig::disabled_tools.");
                } else if message.contains("status ") {
                    eprintln!("The server rejected the request; the body above says why.");
                } else {
                    eprintln!("Is llama-server running at {base_url}?");
                    eprintln!("Start one with: llama-server -m model.gguf --port 8080 --jinja");
                }
            }
            RunEvent::Exited { .. } => break,
            _ => {}
        }
    }
    println!();
    Ok(())
}
