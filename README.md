# agent-harness

**Drive LLM coding agents — Claude Code, OpenAI Codex, and local Ollama / any
OpenAI-compatible model — from Rust, behind one trait.**

Instead of shelling out to `claude -p …` / `codex exec …` and hand-parsing each
tool's bespoke stream, you drive them through one `Harness` trait and consume a
single normalized `RunEvent` stream — text, reasoning, tool start/end, plan,
token usage, lifecycle — whichever agent runs underneath. Your app learns the
event shape **once**.

```toml
agent-harness = "0.4"   # Claude Code, Codex, bob & ACP adapters
```

## Crates

| crate | what it is |
|---|---|
| [`agent-harness`](crates/agent-harness) | the framework — the `Harness` trait, the normalized `RunEvent` stream, an open `Registry`, and the backend adapters: **claude / codex / bob / acp** (wrap an external agent) and **`openai-compatible`** (a full local-/OpenAI-compatible runtime that *is* the agent — the Claude Code / Codex SDK, model-agnostic). All feature-gated. Imported as **`harness`**. |
| [`cli-stream`](crates/cli-stream) | a standalone streaming-subprocess engine — spawn a CLI, stream stdout/stderr line-by-line, cancel it. No agent knowledge; useful on its own. |
| [`bob-rs`](crates/bob-rs) | an **unofficial**, community Rust SDK for the `bob` agent CLI. Not affiliated with bob. |

## Quickstart

Construct an agent, then `run_channel` streams normalized events. **Swap the
agent, keep the loop.**

### Claude Code

```rust
use harness::{Claude, Harness, HarnessError, RunEvent, RunMode, RunRequest, RunTuning};

fn main() -> Result<(), HarnessError> {
    let (_handle, events) = Claude::new().run_channel(RunRequest {
        run_id: "demo".into(),
        prompt: "In one sentence, what is a Markdown heading?".into(),
        cwd: None,                     // working dir for the agent's tool calls
        mode: RunMode::Ask,            // Ask = answer only; Edit = may edit files
        tuning: RunTuning::default(),  // optional: model / effort / max_turns
        resume: None,                  // Some(session_id) to continue a prior run
    })?; // keep `_handle` to `.cancel()`; dropping it does NOT stop the run

    for ev in events {
        match ev {
            RunEvent::Text { delta, .. }     => print!("{delta}"),
            RunEvent::Thinking { delta, .. } => eprint!("{delta}"),
            RunEvent::ToolStart { name, .. } => eprintln!("\n[tool] {name}"),
            RunEvent::Error { message, .. }  => eprintln!("\n[error] {message}"),
            RunEvent::Exited { .. }          => break,
            _ => {}
        }
    }
    Ok(())
}
```

### OpenAI Codex

Identical loop — construct `Codex` instead:

```rust
let (_handle, events) = harness::Codex::new().run_channel(request)?;
```

### Open models — the `openai-compatible` feature

The **`openai-compatible`** feature (`OpenHarness`) is the **agent SDK for open
models** — the Claude Code / Codex SDK equivalent for any open-source or
OpenAI-compatible model. It speaks the OpenAI chat API directly and runs the tool
loop (read/glob/grep/write/edit/bash, plus webfetch/websearch/todowrite/question/
apply_patch) in Rust — point it at Ollama, OpenRouter, vLLM, or LM Studio. It's
the same `Harness` trait, so the loop is identical (enable it with
`features = ["openai-compatible"]`):

```rust
let ollama = harness::OpenHarness::ollama();
let (_handle, events) = ollama.run_channel(RunRequest {
    prompt: "List the Rust files and what each does.".into(),
    mode: RunMode::Edit,
    tuning: RunTuning { model: Some("qwen2.5-coder".into()), ..Default::default() },
    ..request
})?;
```

> **OpenCode**, Gemini, Goose, and other [ACP](https://agentclientprotocol.com) agents work via the `acp` feature — `AcpHarness::opencode()` spawns `opencode acp` and drives it over the Agent Client Protocol.

Install / sign-in is per-agent and scriptable: `readiness()` reports
`installed` / `auth_configured`, `install()` installs the CLI, `login()` runs
its OAuth. In headless/CI, set the agent's API key in the env
(`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`) and `readiness()` reports ready. See
[`examples/`](crates/agent-harness/examples).

## Bring your own agent

`Harness::run` only emits `RunEvent`s, so an implementor can spawn a CLI **or**
call an HTTP API. Implement it in your own crate and register it — no fork:

```rust
let registry = harness::default_registry().register(MyAgent::new());
```

Reuse the building blocks — `spawn_streaming` (the engine),
`normalize_process_event(event, your_parser)` for a stateless parser, or
`run_events_from_parsed` for a stateful one. Runnable example:
[`examples/custom_harness.rs`](crates/agent-harness/examples/custom_harness.rs).

## Status

Pre-1.0 — the API may change. Built for the Compose writing app, designed to be
usable standalone.

## License

[MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE) at your option. `bob-rs` is
an unofficial community SDK, not affiliated with or endorsed by the bob
maintainers.
