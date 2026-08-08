# agent-harness

**Drive coding agents from Rust, behind one trait.**

Claude Code, OpenAI Codex, any ACP agent, and any OpenAI-compatible model —
Ollama, OpenRouter, vLLM, LM Studio, llama-server. Each agent streams a
different format. This crate turns them all into one event stream, so your app
learns the shape once.

```toml
agent-harness = "0.4"
```

## Highlights

- **One trait, every agent.** Implement against `Harness`. Swap the agent
  underneath without touching your loop.
- **One event stream.** Text, reasoning, tool calls, plans, token usage, and
  lifecycle arrive as `RunEvent`, whatever runs underneath.
- **A built-in agent for open models.** The `openai-compatible` feature is not
  a wrapper. It speaks the chat API and runs the tool loop in Rust.
- **Open registry.** Add your own agent in your own crate. No fork.
- **Discovery, not installation.** `readiness()` reports what is on the
  machine. `install_hint` says where to get what is missing. This crate never
  installs anything.
- **Cancellable.** Every run returns a handle. Call `cancel()` to stop it.

## Install

```sh
cargo add agent-harness
```

The CLI adapters are on by default. Add the others as features:

```toml
agent-harness = { version = "0.4", features = ["openai-compatible", "acp"] }
```

| feature | what it adds |
|---|---|
| `claude`, `codex` | the CLI adapters (default) |
| `acp` | any [Agent Client Protocol](https://agentclientprotocol.com) agent — OpenCode, Gemini, Goose |
| `openai-compatible` | the built-in agent for open models |
| `models-dev` | live model lists from the [models.dev](https://models.dev) catalog |

## Getting started

Build an agent. Call `run_channel`. Read events until the run exits.

```rust
use harness::{Claude, Harness, HarnessError, RunEvent, RunMode, RunRequest, RunTuning};

fn main() -> Result<(), HarnessError> {
    let (_handle, events) = Claude::new().run_channel(RunRequest {
        run_id: "demo".into(),
        prompt: "In one sentence, what is a Markdown heading?".into(),
        attachments: Vec::new(),
        cwd: None,                     // where the agent's tools run
        mode: RunMode::Ask,            // Ask reads. Edit also writes.
        tuning: RunTuning::default(),  // model, effort, max turns
        resume: None,                  // Some(session_id) continues a run
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
    Ok(())
}
```

Keep `_handle` if you want to stop the run. Dropping it does not cancel.

To use Codex instead, change one line. The loop does not change.

```rust
let (_handle, events) = harness::Codex::new().run_channel(request)?;
```

## Open models

The `openai-compatible` feature is the agent, not a wrapper around one. It
calls the chat API and runs the tool loop in Rust: `read`, `glob`, `grep`,
`list`, `write`, `edit`, `bash`, `webfetch`, `websearch`, `todowrite`,
`question`, and `apply_patch`. It also handles sessions, skills, subagents,
and MCP servers.

```rust
let ollama = harness::OpenHarness::ollama();
let (_handle, events) = ollama.run_channel(RunRequest {
    prompt: "List the Rust files and say what each one does.".into(),
    mode: RunMode::Edit,
    tuning: RunTuning { model: Some("qwen2.5-coder".into()), ..Default::default() },
    ..request
})?;
```

A provider is configuration, not code. Point the same type at any
OpenAI-compatible endpoint:

```rust
use harness::{OpenHarness, OpenHarnessConfig};

let openrouter = OpenHarness::custom(OpenHarnessConfig {
    id: "openrouter".into(),
    display_name: "OpenRouter".into(),
    base_url: "https://openrouter.ai/api".into(),
    api_key_env: Some("OPENROUTER_API_KEY".into()),
    ..Default::default()
});
```

Ollama is the one exception. It gets its native `/api/*` endpoints, so the
context window is set correctly and local models can be pulled and deleted.
Use `ollama_at(url)` if your server is not on the default port.

## Setup and sign-in

`readiness()` tells you what is on the machine. It reports `installed`,
`auth_configured`, and a version.

This crate does not install agents. When one is missing, `info().install_hint`
says where to get it:

```rust
if !harness.readiness().installed {
    if let Some(hint) = harness.info().install_hint {
        println!("Get {} from {}", harness.info().display_name, hint.url);
        if let Some(command) = hint.command {
            println!("  {command}");
        }
    }
}
```

`login()` runs an agent's own sign-in flow, which opens a browser. For CI, set
the agent's API key in the environment instead — `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY`. Then `readiness()` reports ready.

## Bring your own agent

`Harness::run` only emits `RunEvent`s. An implementation can spawn a CLI or
call an HTTP API. Write it in your own crate and register it:

```rust
let registry = harness::default_registry().register(MyAgent::new());
```

Reuse the parts: `spawn_streaming` runs a subprocess and streams its output.
`normalize_process_event` wraps a stateless line parser.
`run_events_from_parsed` wraps a stateful one. See
[`examples/custom_harness.rs`](crates/agent-harness/examples/custom_harness.rs).

## Crates

| crate | what it is |
|---|---|
| [`agent-harness`](crates/agent-harness) | the framework. Imported as `harness`. |
| [`cli-stream`](crates/cli-stream) | a standalone subprocess engine. Spawns a CLI, streams stdout and stderr line by line, and cancels it. It knows nothing about agents. |

## Status

Pre-1.0. The API can still change. See [CHANGELOG.md](CHANGELOG.md) for
breaking changes and how to migrate.

Built for the Compose writing app. Designed to work on its own.

## License

[MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
