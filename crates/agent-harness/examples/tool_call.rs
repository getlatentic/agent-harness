//! **Tool calls** — watch the agent loop read the filesystem and report back.
//!
//! ```text
//! cargo run --example tool_call --features openai-compatible
//! # needs `ollama serve` and a tool-capable model, e.g. `ollama pull qwen3:4b`
//! # OLLAMA_MODEL=gpt-oss:20b cargo run --example tool_call --features openai-compatible
//! ```
//!
//! The other examples ask a question the model can answer from memory. This one
//! asks about the working directory, which it cannot, so it has to use a tool.
//! That round trip — model asks for a tool, we run it, the result goes back —
//! is the thing that makes this an agent rather than a chat completion.
//!
//! Pick a model that supports tool calling. A very small model will echo the
//! tool schemas back at you instead of calling them.

use harness::{Harness, HarnessError, OpenHarness, RunEvent, RunMode, RunRequest, RunTuning};

fn main() -> Result<(), HarnessError> {
    let agent = OpenHarness::ollama();

    let readiness = agent.readiness();
    if !readiness.ready {
        eprintln!("Ollama is not reachable: {}", readiness.error.unwrap_or_default());
        return Ok(());
    }

    let model = match std::env::var("OLLAMA_MODEL") {
        Ok(m) if !m.trim().is_empty() => m,
        _ => match agent.list_models()?.into_iter().next() {
            Some(first) => first.value,
            None => {
                eprintln!("No models installed. Try `ollama pull qwen3:4b`.");
                return Ok(());
            }
        },
    };
    eprintln!("[model] {model}\n");

    let (_handle, events) = agent.run(RunRequest {
        run_id: "tools".into(),
        prompt: "List the files in the current directory, then say how many there are.".into(),
        // The tools run here. Point it anywhere you do not mind being read.
        cwd: Some(std::env::current_dir().unwrap_or_default()),
        // Spelled out because it is the subject of this example: Ask offers
        // the read-only tools (read, glob, grep, list); Edit adds write, edit
        // and bash. Ask is the default, so the other examples omit it.
        mode: RunMode::Ask,
        // Ask offers read-only tools: read, glob, grep, list.
        // Edit adds the mutating ones: write, edit, bash.
        tuning: RunTuning { model: Some(model), ..Default::default() },
        ..Default::default()
    })?;

    let mut calls = 0usize;
    for event in events {
        match event {
            // A tool call starts. `locations` carries the paths it touches, so
            // a UI can show the subject rather than a bare tool name.
            RunEvent::ToolStart { title, tool_kind, locations, .. } => {
                calls += 1;
                let where_ = locations
                    .iter()
                    .map(|l| l.path.clone())
                    .collect::<Vec<_>>()
                    .join(", ");
                eprintln!("[tool {calls}] {title} ({tool_kind:?}) {where_}");
            }
            // The result. `content` is what goes back to the model, so this is
            // literally what it sees before it writes its answer.
            RunEvent::ToolEnd { ok, content, .. } => {
                let preview = content.unwrap_or_default();
                let first =
                    preview.lines().next().unwrap_or("").chars().take(72).collect::<String>();
                eprintln!("[done] ok={ok} -> {first}…");
            }
            RunEvent::Text { delta, .. } => print!("{delta}"),
            RunEvent::Thinking { delta, .. } => eprint!("{delta}"),
            RunEvent::Error { message, .. } => eprintln!("\n[error] {message}"),
            RunEvent::Exited { .. } => break,
            _ => {}
        }
    }
    eprintln!("\n\n{calls} tool call(s).");
    if calls == 0 {
        eprintln!("The model answered without using a tool. Try a larger one.");
    }
    Ok(())
}
