# agent-harness

**Use popular coding agents from your Rust code.**

Claude Code, OpenAI Codex, OpenCode, and any OpenAI-compatible model — Ollama,
OpenRouter, vLLM, LM Studio, llama-server. Call them from your program instead
of opening a terminal. You do not shell out, and you do not parse each agent's
own output format. Every agent returns the same event stream.

```toml
agent-harness = "0.4"
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
- **Discovery, not installation.** `readiness()` reports what is on the machine.
  `install_hint` says where to get what is missing. This crate installs nothing.

## Example

Build an agent. Call `run_channel`. Read events until the run exits.

```rust
use harness::{Claude, Harness, RunEvent, RunMode, RunRequest, RunTuning};

let (_handle, events) = Claude::new().run_channel(RunRequest {
    run_id: "demo".into(),
    prompt: "In one sentence, what is a Markdown heading?".into(),
    attachments: Vec::new(),
    cwd: None,
    mode: RunMode::Ask,
    tuning: RunTuning::default(),
    resume: None,
})?;

for event in events {
    match event {
        RunEvent::Text { delta, .. } => print!("{delta}"),
        RunEvent::Exited { .. } => break,
        _ => {}
    }
}
```

To use Codex, an ACP agent, or a local model, change the constructor. The loop
does not change.

Full documentation, features, and provider setup:
**[`crates/agent-harness`](crates/agent-harness)** ·
[docs.rs](https://docs.rs/agent-harness)

## Crates

| crate | what it is |
|---|---|
| [`agent-harness`](crates/agent-harness) | the framework. Imported as `harness`. |
| [`cli-stream`](crates/cli-stream) | a standalone subprocess engine. Spawns a CLI, streams stdout and stderr line by line, and cancels it. It knows nothing about agents. |

## Status

In active development. The API can still change. See
[CHANGELOG.md](CHANGELOG.md) for breaking changes and how to migrate.

## License

[MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
