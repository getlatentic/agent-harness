# agent-harness

Drive **LLM coding agents — Claude Code, OpenAI Codex, bob, and any ACP agent —
from Rust** behind one trait, with a single normalized streaming event
vocabulary. Any OpenAI-compatible model — local Ollama or a hosted endpoint —
runs on the built-in **`openai-compatible`** runtime (`OpenHarness`), which owns
the agent loop in Rust. Bring your own agent too.

Imported as `harness`. One `Harness` trait, one `RunEvent` stream: your app
learns the event shape **once** and never grows a per-agent parser — adding an
agent is "write a parser into `RunEvent`," not "teach the UI another format."

```toml
# Pre-release: pin the exact version — cargo won't auto-select a pre-release.
agent-harness = "0.4.0-alpha.1"   # default: the Claude Code / Codex / bob adapters

# Add the opt-in backends — ACP agents (opencode / Gemini) + the local /
# OpenAI-compatible runtime:
agent-harness = { version = "0.4.0-alpha.1", features = ["acp", "openai-compatible"] }

# Or slim it to one adapter — no keychain SDK, no HTTP client:
agent-harness = { version = "0.4.0-alpha.1", default-features = false, features = ["claude"] }
```

## Quickstart

Construct an agent, then `run_channel` streams normalized `RunEvent`s. **Swap
the agent, keep the loop** — that's the whole point.

### Claude Code

```rust
use harness::{Claude, Harness, RunEvent, RunMode, RunRequest, RunTuning};

let (_handle, events) = Claude::new().run_channel(RunRequest {
    run_id: "1".into(),
    prompt: "Summarize the README in one sentence.".into(),
    cwd: None,
    mode: RunMode::Ask,
    tuning: RunTuning::default(),
    resume: None,
    attachments: Vec::new(),
})?;

for event in events {
    if let RunEvent::Text { delta, .. } = event {
        print!("{delta}");
    }
}
```

### OpenAI Codex

Identical loop — construct `Codex` instead:

```rust
let (_handle, events) = harness::Codex::new().run_channel(request)?;
```

### Open models — the `openai-compatible` feature

The claude/codex/acp adapters *wrap* an external agent. An OpenAI-compatible
model has none to wrap, so the **`openai-compatible`** feature ships a runtime
that *is* the agent — it speaks the OpenAI chat API and owns the tool loop
(read/write/edit/bash/glob/grep + webfetch/websearch/todos/skills/subagents/MCP)
in Rust, behind this same `Harness` trait. Think the **Claude Code / Codex SDK**,
but for any open-source or OpenAI-compatible model (Ollama, OpenRouter, vLLM, …):

```toml
agent-harness = { version = "0.4", features = ["openai-compatible"] }
```

```rust
let ollama = harness::OpenHarness::ollama();   // or ::custom(OpenHarnessConfig { id, display_name, base_url, .. })
let (_handle, events) = ollama.run_channel(request)?;
```

Off by default (it pulls a blocking HTTP client + search libs), so the core crate
stays lean — same `RunEvent` loop, no CLI required.

> **OpenCode**, Gemini, Goose, and other [ACP](https://agentclientprotocol.com) agents work via the `acp` feature — `AcpHarness::opencode()` spawns `opencode acp` and drives it over the Agent Client Protocol.

## Bring your own agent

`Harness` only emits `RunEvent`s, so an implementor can spawn a CLI **or** call
an HTTP API — both fit. Register it alongside the built-ins; nothing to fork:

```rust
let registry = harness::Registry::new()
    .register(harness::Claude::new())
    .register(MyAgent::new());            // your own `impl Harness`
```

`RunEvent` derives `Serialize` with stable camelCase field names, so any
transport (HTTP/SSE, an IPC channel) emits identical JSON.

## What you get

- **`Harness` trait** — `info` / `readiness` / `install` / `run` / `credential`
  / `login` / `list_models`. Object-safe (`Box<dyn Harness>`).
- **`RunEvent`** — text, thinking, tool start/end, plan, usage (with cache
  tokens), suggested edits, questions, lifecycle. Aligned with the [Agent
  Client Protocol](https://agentclientprotocol.com) vocabulary.
- **Open `Registry`** — compose built-ins and your own providers.

The subprocess engine is the standalone
[`cli-stream`](https://crates.io/crates/cli-stream) crate.

## Examples

Runnable demos (`cargo run --example <name>`):

- **`run_prompt`** — Claude Code (default features).
- **`run_acp`** — an ACP agent (`opencode`); add `--features acp`.
- **`run_openai`** — an OpenAI-compatible / local model; add `--features openai-compatible`.
- **`playground`** — a browser UI that streams a live run over SSE for any
  backend; add `--features "openai-compatible acp"`.

## License

MIT OR Apache-2.0.
