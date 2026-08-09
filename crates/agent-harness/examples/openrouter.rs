//! **OpenRouter** — a hosted provider on the built-in runtime.
//!
//! ```text
//! OPENROUTER_API_KEY=sk-or-… cargo run --example openrouter --features openai-compatible
//! ```
//!
//! There is no OpenRouter adapter in this crate. There is no OpenRouter code at
//! all. A provider is a `base_url` and a key, so this example is the whole
//! integration. The same few lines reach DeepSeek, Together, Groq, Fireworks,
//! or anything else that speaks the OpenAI chat API — change the URL.
//!
//! The key is read here and passed by value. An environment variable is a fine
//! place for a demo to keep one; a real host reads it from an OS vault. Either
//! way the secret goes into the config, not into the process environment,
//! because every child the agent spawns inherits that environment.

use harness::{ApiKey, 
    Harness, HarnessError, OpenHarness, OpenHarnessConfig, RunEvent, RunRequest, RunTuning,
};

fn main() -> Result<(), HarnessError> {
    let Ok(key) = std::env::var("OPENROUTER_API_KEY") else {
        eprintln!("Set OPENROUTER_API_KEY first. Get one at https://openrouter.ai/keys");
        return Ok(());
    };

    // The whole integration.
    let openrouter = OpenHarness::custom(OpenHarnessConfig {
        id: "openrouter".into(),
        display_name: "OpenRouter".into(),
        base_url: "https://openrouter.ai/api".into(),
        api_key: ApiKey::Value(key),
        ..Default::default()
    })
    // Optional: fill the model picker from the models.dev catalog rather than
    // a hardcoded list. Needs the `models-dev` feature.
    .with_models_dev("openrouter");

    let model = std::env::var("OPENROUTER_MODEL").unwrap_or_else(|_| "openai/gpt-oss-120b".into());
    eprintln!("[model] {model}");

    let (_handle, events) = openrouter.run(RunRequest {
        run_id: "demo".into(),
        prompt: "In one sentence, what is an OpenAI-compatible API?".into(),
        tuning: RunTuning { model: Some(model), ..Default::default() },
        ..Default::default()
    })?;

    for event in events {
        match event {
            RunEvent::Text { delta, .. } => print!("{delta}"),
            RunEvent::Thinking { delta, .. } => eprint!("{delta}"),
            RunEvent::ToolStart { title, .. } => eprintln!("\n[tool] {title}"),
            RunEvent::Usage { input_tokens, output_tokens, .. } => {
                eprintln!("\n[usage] in={input_tokens:?} out={output_tokens:?}")
            }
            RunEvent::Error { message, .. } => eprintln!("\n[error] {message}"),
            RunEvent::Exited { .. } => break,
            _ => {}
        }
    }
    println!();
    Ok(())
}
