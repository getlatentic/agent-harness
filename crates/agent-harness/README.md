# agent-harness

**Use popular coding agents from your Rust code.**

Claude Code, OpenAI Codex, OpenCode, and any OpenAI-compatible model. Call them
from your program instead of opening a terminal. You do not shell out, and you
do not parse each agent's own output format. Every agent returns the same event
stream, so adding an agent means writing a parser into `RunEvent` — not
teaching your UI another format.

Imported as `harness`.

```sh
cargo add agent-harness
```

## Highlights

- **No terminal.** Run an agent from your code and read its output as typed
  events, not scraped text.
- **Every agent looks the same.** Text, reasoning, tool calls, plans, token
  usage, and lifecycle all arrive as `RunEvent`.
- **Swap agents without rewriting.** Change the constructor. Your loop stays.
- **A built-in agent for open models.** The `openai-compatible` feature is not a
  wrapper. It speaks the chat API and runs the tool loop in Rust.
- **Add your own agent.** Implement `Harness` in your own crate and register it.
  No fork.
- **Stable wire format.** `RunEvent` serializes to camelCase JSON, so HTTP, SSE,
  and IPC transports all emit the same shape.

## Features

The CLI adapters are on by default. Everything else is opt-in, so the core
crate stays small.

```toml
# Claude Code and Codex
agent-harness = "0.4"

# Add ACP agents and the built-in agent for open models
agent-harness = { version = "0.4", features = ["acp", "openai-compatible"] }

# Or take just one adapter
agent-harness = { version = "0.4", default-features = false, features = ["claude"] }
```

| feature | what it adds |
|---|---|
| `claude`, `codex` | the CLI adapters (default) |
| `acp` | any [Agent Client Protocol](https://agentclientprotocol.com) agent — OpenCode, Gemini, Goose |
| `openai-compatible` | the built-in agent for open models. Pulls an HTTP client and search libraries. |
| `models-dev` | live model lists from the [models.dev](https://models.dev) catalog |

## Getting started

Build an agent. Call `run`. Read events.

(`run` hands back a receiver to loop over. `start` takes a callback instead —
reach for it when you are forwarding events straight onto a socket and an
intermediate hop would be waste. An adapter implements only `start`; every
harness gets `run` for free.)

```rust
use harness::{Claude, Harness, RunEvent, RunRequest};

let (_handle, events) = Claude::new().run(RunRequest {
    run_id: "demo".into(),
    prompt: "In one sentence, what is a Markdown heading?".into(),
    ..Default::default()
})?;

for event in events {
    match event {
        RunEvent::Text { delta, .. } => print!("{delta}"),
        RunEvent::Exited { .. } => break,
        _ => {}
    }
}
```

Name what you mean; the rest defaults — no working directory, `RunMode::Ask`
(answer only; `Edit` lets it write files), no resumed session, no attachments.

Keep `_handle` if you want to stop the run. Dropping it does not cancel.

To use Codex instead, change one line:

```rust
let (_handle, events) = harness::Codex::new().run(request)?;
```

## Open models

The CLI adapters wrap an external agent. An open model has no agent to wrap,
so the `openai-compatible` feature ships one. It calls the chat API and runs
the tool loop in Rust: `read`, `glob`, `grep`, `list`, `write`, `edit`,
`bash`, `webfetch`, `websearch`, `todowrite`, `question`, and `apply_patch`.
It also handles sessions, skills, subagents, and MCP servers.

```rust
let ollama = harness::OpenHarness::ollama();
let (_handle, events) = ollama.run(request)?;
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

Ollama is the one exception. It uses its native `/api/*` endpoints, so the
context window is set correctly and local models can be pulled and deleted.
Use `ollama_at(url)` when your server is not on the default port.

## Setup and sign-in

`readiness()` reports what is on the machine: `installed`, `auth_configured`,
and a version.

This crate does not install agents. When one is missing, `info().install_hint`
says where to get it.

```rust
if !harness.readiness().installed {
    if let Some(hint) = harness.manifest().install_hint {
        println!("Get it from {}", hint.url);
        if let Some(command) = hint.command {
            println!("  {command}");
        }
    }
}
```

`login()` runs an agent's own sign-in flow, which opens a browser. In CI, set
the agent's API key in the environment instead. Then `readiness()` reports
ready.

## Bring your own agent

`Harness` only emits `RunEvent`s. Your implementation can spawn a CLI or call
an HTTP API. Register it next to the built-ins:

```rust
let registry = harness::Registry::new()
    .register(harness::Claude::new())
    .register(MyAgent::new());
```

## What the trait gives you

- **`Harness`** — `info`, `readiness`, `run`, `credential`, `login`,
  `list_models`. Object-safe, so `Box<dyn Harness>` works.
- **`RunEvent`** — text, thinking, tool start and end, plan, usage with cache
  tokens, suggested edits, questions, and lifecycle. It follows the [Agent
  Client Protocol](https://agentclientprotocol.com) vocabulary.
- **`Registry`** — an open set. Built-ins and your own agents sit together.

The subprocess engine is a separate crate:
[`cli-stream`](https://crates.io/crates/cli-stream).

## Examples

One per harness. Run with `cargo run --example <name>`.

| example | harness | features |
|---|---|---|
| `claude` | Claude Code | default |
| `codex` | OpenAI Codex | default |
| `acp` | OpenCode over ACP | `acp` |
| `gemini` | Gemini CLI over ACP | `acp` |
| `ollama` | a local model on Ollama | `openai-compatible` |
| `openrouter` | a hosted model on OpenRouter | `openai-compatible models-dev` |
| `llama_cpp` | a local `llama-server` | `openai-compatible` |
| `tool_call` | the agent loop calling tools and reading the results | `openai-compatible` |
| `setup` | readiness, install hints, and sign-in | default |
| `custom_harness` | your own agent | default |
| `playground` | a browser UI that streams a live run over SSE | `openai-compatible acp` |

`claude` and `codex` differ by one line. `ollama`, `openrouter`, and
`llama_cpp` differ only in configuration — no provider has its own adapter.

## License

MIT or Apache-2.0, at your option.
