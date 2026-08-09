# Changelog

Notable changes to this workspace — `cli-stream` and `agent-harness` —
recorded together. Format loosely follows
[Keep a Changelog](https://keepachangelog.com). All three are on crates.io.
Unreleased changes accumulate under **Unreleased** until the next release.

## [Unreleased]

Breaking. Read **Migration from 0.4** before you upgrade.

### Migration from 0.4

**1. `Harness::run` is now `Harness::start`.** If you *implement* `Harness`,
rename your method. The signature is unchanged:

```rust
fn start(&self, request: RunRequest, on_event: RunCallback) -> Result<RunHandle, HarnessError>
```

**2. `Harness::run_channel` is now `Harness::run`.** If you *call* a harness,
`run` is the name to reach for, and it hands back a receiver:

```rust
let (_handle, events) = harness.run(request)?;   // was run_channel
for event in events { /* … */ }
```

An adapter implements only `start`; every harness gets `run` for free.

**3. `api_key`, `api_key_env` and `requires_api_key` are now one `ApiKey`.**
Three fields could represent eight combinations for four meanings, and they
could contradict each other — which they did: "needs a key" was inferred from
"names an environment variable", so a host holding its key in a vault reported
that no credential was required and looked permanently ready.

```rust
api_key: ApiKey::NotNeeded,                    // local Ollama, llama-server
api_key: ApiKey::Value(key_from_your_vault),   // the secret itself
api_key: ApiKey::Env("OPENROUTER_API_KEY".into()), // CI / headless
api_key: ApiKey::Required,                     // needed, not supplied yet
```

`Required` is the state the old `requires_api_key` existed for: readiness
reports not-ready and the credential slot stays writable, so a host can prompt.
`is_needed()` and `env_var()` derive from the variant, so no pair of values can
disagree.

**Old →  new**

| before | after |
|---|---|
| `api_key: Some(k), requires_api_key: true` | `ApiKey::Value(k)` |
| `api_key_env: Some(v)` | `ApiKey::Env(v)` |
| `requires_api_key: true` alone | `ApiKey::Required` |
| all three unset / false | `ApiKey::NotNeeded` |

**3b. Historical note — the value-vs-variable change this replaces.** `api_key_env`
named an environment variable, so the key had to be exported into the
process — and every child the agent spawned inherited it. Pass the secret
directly instead:

```rust
OpenHarnessConfig {
    api_key: Some(key_from_your_vault),   // the secret itself
    requires_api_key: true,               // was implied by api_key_env
    ..Default::default()
}
```

`api_key_env` still works and still reads the variable, for CI and headless
runs where an env var is the natural source. `api_key` wins when both are set.

**4. `requires_api_key` is now its own field.** It used to be inferred from
`api_key_env.is_some()`, which meant a host passing a key by value reported
`credential_required: false` and looked permanently ready. Set it explicitly.
Leaving it `false` while `api_key_env` is set keeps the old behaviour.

**5. Instruction files and skills no longer read `$HOME` on their own.**
Runs used to load `~/.claude/CLAUDE.md` and scan `~/.claude/skills`
unconditionally. On the machine this was found on that was ~3,400 tokens of
someone else's product's config on every turn — enough that no local server
started on the common 4096-token context could run the loop at all. A library
should not decide to read a user's home directory, so the host names the
locations now:

```rust
OpenHarnessConfig {
    instruction_sources: InstructionSources::discover_global(),  // opt back in
    global_skill_roots: harness::global_skill_roots(),
    ..Default::default()
}
```

Leave both at their defaults for project-only behaviour.

**6. `OpenHarnessConfig` gained `disabled_tools`.** Add
`disabled_tools: Vec::new()` — or use `..Default::default()`, which is now
derived — to keep every built-in tool. Name a tool to withhold it:

```rust
disabled_tools: vec!["shell".into()],
```

Disabled tools are removed at construction, so they never reach the model's
prompt. `OpenHarness::builtin_tool_names()` lists what you can name.

### Changed

- **BREAKING — `Harness::run` renamed to `start`; `run_channel` renamed to
  `run`.** The common call is the channel one, so it should have the short
  name. `run_channel` described its plumbing rather than what a caller wanted,
  and left `run` — the obvious first thing to reach for — as the awkward
  callback form.
- **BREAKING — `OpenHarnessConfig.requires_api_key` split out of
  `api_key_env`.** One field was doing four jobs: naming a variable, marking
  the host as needing a key, deciding whether a credential slot was writable,
  and gating readiness. A host holding its key in a vault got the wrong answer
  to all four.

### Added

- **`OpenHarnessConfig.instruction_sources`** and
  **`OpenHarnessConfig.global_skill_roots`** — where `AGENTS.md` / `CLAUDE.md`
  and skills are read from. Both default to the working tree only, so **nothing
  under `$HOME` is read unless the host asks**. `InstructionSources::discover_global()`
  and `global_skill_roots()` return the conventional per-user locations for a
  host that wants them.
- **`OpenHarnessConfig.prompt_cache`** — mark the prompt prefix as cacheable
  with `PromptCache::Ephemeral`. Anthropic caches only what a request marks,
  and forwards `cache_control` through OpenAI-compatible gateways, so a Claude
  model reached via OpenRouter previously re-charged the system prompt and the
  whole tool block at full input price every turn. Breakpoints land on the last
  tool and the system message. Default stays `Implicit` — correct for OpenAI and
  DeepSeek, which cache prefixes on their own, and for local servers whose KV
  cache keys on the bytes rather than a field.
- **`OpenHarnessConfig.api_key`** — the secret by value, so a host with an OS
  vault never has to put it in the environment.
- **`OpenHarnessConfig.disabled_tools`** and
  **`OpenHarness::builtin_tool_names`** — choose the tool set at construction.
  Everything is enabled by default; name what you want withheld.
- **`RunRequest` and `RunMode` now derive `Default`.** Name the fields you
  mean and let the rest default. `RunMode` defaults to `Ask`, the read-only
  mode — defaulting to `Edit` would hand write access to anyone who forgot
  the field.
- **`OpenHarness::ollama_at(base_url)`** — an Ollama on a non-default host.
- **`ClaudeHarnessConfig` / `CodexHarnessConfig`**, via `Claude::custom` and
  `Codex::custom` — name the binary to spawn. Both adapters hardcoded
  `"claude"` and `"codex"` in five places each, while the ACP adapter had taken
  its `command` from config since it shipped. An upstream rename, a fork, a
  wrapper script or a test stub now costs a field rather than a release.
  `DEFAULT_CLAUDE_COMMAND` and `DEFAULT_CODEX_COMMAND` are public, and are what
  each config's `Default` uses.

### Fixed

- **The shell tool no longer inherits the whole environment.** A command the
  model ran could read any variable the host process held, including the API
  key driving that same run. The child now gets an allowlist of about twenty
  benign variables (`PATH`, `HOME`, `LANG`, the Windows equivalents). A
  denylist was tried first and leaked in both directions — it stripped
  `TOKENIZERS_PARALLELISM` while passing `AWS_ACCESS_KEY_ID` straight through.
- **No console window flashes on Windows** (#35). Every child process spawns
  with `CREATE_NO_WINDOW`, via `cli_stream::hidden_command`.
- **Skill discovery is ordered, so the prompt prefix stays byte-stable.** The
  catalog sits ahead of the volatile working-directory block, which makes it
  part of the prefix every request shares. Discovery took whatever order the
  directory walk produced — stable enough locally to look fine, not guaranteed
  across machines — and one reordered line changes the bytes, misses the cache
  and pays a full re-prefill of everything above it.
- **A skill whose description is a YAML block scalar is now readable.** The
  frontmatter reader handled only a flat `key: value`, so the common
  `description: >-` followed by indented lines was read as the literal `">-"`.
  The skill still appeared in the catalog, so nothing looked broken — the model
  just had nothing to match a task against and never called it. Seven of
  twenty-one skills on the machine this was found on were in that state.
- **Claude's `fable` alias is selectable.** The curated list held
  `sonnet`/`opus`/`haiku` only. Online it went unnoticed, because models.dev
  supplies `claude-fable-5` — but that list *is* the picker when models.dev is
  unreachable, and `allows_custom_model` is `false` for Claude, so offline or on
  a cold cache Fable could be neither picked nor typed.
- **A rejected request now quotes the provider's explanation.** `ureq`'s
  `Display` for a status error stops at the code, so a failed run reported
  `status code 400` and nothing else — the same message whether the key was
  refused, the model id was unknown, or the prompt was too long. The body is
  where a provider says which it was, and it now reaches the `RunEvent::Error`
  message (truncated to 500 characters).

## [0.4.0] - 2026-08-08

`agent-harness` 0.4.0 and `cli-stream` 0.3.7.

The first non-alpha release since 0.3.5. It removes the `bob` adapter and
takes installation out of the framework. Read **Migration from 0.3** before
you upgrade. Consumers pinned to `^0.3` are not affected until they bump.

### Migration from 0.3

Four changes need an edit in your code.

**1. `Harness::install` is gone.** The framework no longer installs agents.
Delete your `install` implementation. To tell a user where to get an agent,
read `info().install_hint`:

```rust
if !harness.readiness().installed {
    if let Some(hint) = harness.info().install_hint {
        println!("Get it from {}", hint.url);
        if let Some(command) = hint.command {
            println!("  {command}");
        }
    }
}
```

**2. `HarnessInfo.requires_install` is now `install_hint`.** Replace
`requires_install: false` with `install_hint: None`. Replace
`requires_install: true` with a hint that says where the agent comes from:

```rust
install_hint: Some(InstallHint::url("https://example.dev/docs")
    .with_command("npm install -g example-cli")),
```

Test for a missing agent with `info().install_hint.is_some()` where you used
to test `requires_install`.

**3. `AcpHarnessConfig` has a new `install_hint` field.** Add
`install_hint: None`, or use `..Default::default()`.

**4. The `bob` adapter and the `bob` feature are gone.** Remove the feature
from your `Cargo.toml`. Use `claude`, `codex`, `acp`, or
`openai-compatible` instead.

### Changed

- **BREAKING — `Harness::install` removed.** This crate discovers and runs
  agents. It does not install them. Installing is a decision about a user's
  machine, so it belongs to the host and its user. `login()` stays: that is
  the agent authenticating itself, not us installing it.
- **BREAKING — `HarnessInfo.requires_install: bool` replaced by
  `install_hint: Option<InstallHint>`.** A boolean could only say "this needs
  installing" and stop. `InstallHint` carries a `url` and an optional
  `command`, so a host can show a next step. This also fixed a dead end:
  `AcpHarness` reported `requires_install: false` while it did need its CLI,
  so a missing OpenCode read "Not installed" with nowhere to go.
- **BREAKING — `AcpHarnessConfig` gained `install_hint`.** Set it to `None`
  for an agent the user supplies themselves.
- **BREAKING — the `bob` adapter and its feature are removed.**
- **BREAKING — `RunEvent::Usage` gained `cache_read_tokens`,
  `cache_write_tokens`, and `cost_usd`** (`cache_*` are `Option<u64>`, `cost_usd`
  is `Option<f64>`; camelCase, omitted from the wire when `None`). Prompt-cache
  counters were previously thrown away; they're the difference between a useful
  cost display and a misleading one. `cost_usd` is an estimated run cost, emitted
  by `openai-compatible` when per-model rates are configured (`with_model_cost`). A
  match on `Usage` that bound all fields without `..` must add `..` (or the new
  fields).
- **BREAKING — `RunEvent` no longer derives `Eq`** (it now carries an `f64`
  cost). `==` and `assert_eq!` still work via `PartialEq`; `RunEvent` just can no
  longer be a `HashSet`/`HashMap` key.
- **BREAKING — tool events standardized on ACP's `ToolCall` shape.**
  `RunEvent::ToolStart` now carries `title` (was `name`), `tool_kind`,
  **`locations: Vec<ToolLocation>`** (the files the call touches — ACP's
  `locations`), and `raw_input` (was `input`); `RunEvent::ToolEnd` now carries
  `content` (was `output`), **`raw_output`**, and `locations`. New `ToolLocation
  { path, line }` type (neutral mirror of ACP's `ToolCallLocation` — no
  dependency on the `acp` crate in core). This lets the ACP adapter pass an
  agent's tool calls through losslessly and lets *our own* tools report the path
  they operate on, so a host can show the call's subject and distinguish e.g.
  listing a directory from reading a file. (`tool_kind` keeps its name — ACP
  calls it `kind`, but our wire reserves `kind` for the event tag.) CLI adapters
  (Claude/Codex/bob) default `locations` empty; openai-compatible populates it from
  the tool's `path` argument; ACP passes through whatever the agent sends.

### Added

- **`OpenHarness::ollama_at(base_url)`** — Ollama on another host, port, or in
  a container. `ollama()` hardcoded `localhost:11434`, so an Ollama-flavoured
  harness could not point anywhere else.
- **`cli_stream::hidden_command(program)`** — a `Command` that does not open a
  console window on Windows. Re-exported as `harness::hidden_command`.
- **End-to-end tests for the run path.** `tests/ollama_route.rs` drives a fake
  Ollama on all platforms. It checks that discovery never falls back to `/v1`,
  that `num_ctx` above 4096 is sent, and that a tool result returns to the
  model. `tests/openai_v1_live.rs` runs the `/v1` path against real providers.
- **A `ollama-live` CI job.** It installs Ollama, pulls a small model, and runs
  the live test. It is advisory, so vendor changes do not block a pull request.

### Fixed

- **Windows: no console window when an agent runs (#35).** Nine spawn sites
  used a bare `Command::new`. A GUI host flashed a console window on every
  agent run, every version probe, every MCP server, and every `bash` tool call.
  They now use `hidden_command`, which sets `CREATE_NO_WINDOW`.
- **Every rustdoc warning cleared.** A broken `Bob` intra-doc link would have
  rendered dead on docs.rs.

- **ACP agents can be given a launch-time model.** ACP carries no model, so
  `AcpHarness` selects one *out-of-band*: `AcpHarness::opencode()` lists models
  via `opencode models` (so `list_models()` is populated instead of empty) and
  applies a chosen `RunTuning::model` by writing a temp JSON config and pointing
  `$OPENCODE_CONFIG` at it for that spawn (removed when the run ends). A generic
  `AcpHarness::custom(…)` has no model mechanism — `list_models()` stays empty and
  `RunTuning::model` is ignored. The point: which model an ACP agent uses is a
  *launcher* concern (pick the binary, then pick the model), decided before the
  ACP session — which only begins once the process is spawned — ever starts.
- **`list_models()` can pull from the models.dev catalog (opt-in `models-dev`
  feature).** Adds `harness::models_dev::provider_models(provider)` — one cached
  GET of models.dev's `api.json`, filtered to a provider's `tool_call`
  (agent-usable) models. The CLI adapters use it: **Codex** now lists the
  `openai` lineup (was empty — free-text only), **Claude** appends the live
  `anthropic` lineup after its `sonnet`/`opus` aliases, and
  `OpenHarness::with_models_dev(provider)` points a cloud endpoint at
  a provider's catalog (new `Discovery::ModelsDev`). Off by default (keeps the
  core HTTP-free — the feature pulls `ureq`); with the feature off or the catalog
  unreachable, `list_models` falls back to each adapter's static list. For
  Claude/Codex the list is runnable as-is (one CLI login covers the whole
  provider). `HarnessModel` gained `PartialEq`/`Eq`.
- **New `openai-compatible` feature — agent-harness's local-model runtime.** Where
  the claude / codex / bob / acp adapters *wrap* an external agent, the
  `openai-compatible` feature *is* the agent: it speaks the OpenAI-compatible chat
  API over blocking HTTP (`ureq`)
  and runs the agent loop in Rust, owning a built-in tool surface reimplemented
  from OpenCode's design (MIT): **`read`** (1-based `offset`/`limit`,
  line-numbered, per-line + total caps), **`glob`** (gitignore-aware file
  matching), **`grep`** (regex content search), and **`list`** (one-level
  gitignore-aware directory listing) — all via ripgrep's own libraries
  (`ignore`/`globset`/`regex`) in-process, no `rg` binary needed —
  **`write`** (overwrite + mkdir-p), **`edit`** (exact string replacement with
  uniqueness enforcement + `replace_all`, plus a whitespace-tolerant line-match
  fallback), and **`bash`** (timeout + cooperative cancel) — plus **`webfetch`**
  (URL → text/markdown/html, 5 MB cap), **`todowrite`** (→ `RunEvent::Plan`),
  **`question`** (→ `RunEvent::AskQuestion`; ends the run and resumes with the
  user's answer), **`websearch`** (Exa/Parallel over MCP, keyed by
  `EXA_API_KEY`/`PARALLEL_API_KEY`), and **`apply_patch`** (the `*** Begin Patch`
  envelope, offered to gpt-5-class models in place of `edit`/`write`). Tools live
  in a registry the loop dispatches through; each declares its parameters as a
  typed `schemars` struct (the JSON Schema is derived from the type). One type
  serves Ollama, OpenRouter, vLLM, LM Studio, … via
  `OpenHarness::ollama()` / `::custom(...)`; Ollama models are
  discovered live via `/api/tags` (`list_models()`). Read-only tools are offered
  in both modes; `RunMode::Edit` adds the mutating ones (writes land on disk
  directly — review stays in the host). It implements `agent-harness`'s
  `Harness` trait and registers through the open `Registry`; it is its own crate
  (carrying the HTTP + search deps), not a feature of `agent-harness`.
  Sessions persist + resume — opt-in via `with_session_dir` (JSON files keyed by
  id: a metadata record + the transcript; `RunRequest.resume` replays a prior
  session, `sessions()` lists them newest-first). Skills: discovers `SKILL.md`
  files (including Claude Code's `~/.claude/skills`), advertises a
  name+description catalog in the system prompt, and loads a skill's body on
  demand via the `skill` tool. Subagents: a `task` tool spawns a child agent
  (its own session with `parent_id`, the same tools minus `task`/`question`) and
  returns its result. Responses **stream** token deltas (`RunEvent::Text` per
  fragment). Context management follows OpenCode: large tool outputs are
  **capped** (2000 lines / 50 KiB) and spilled to a temp file the model can
  `read` back; and near the context limit the transcript is **compacted** by
  inserting a summary marker into the **non-lossy** full transcript and sending
  the model a windowed view (full history stays on disk and survives resume). The
  limit comes from `with_context_tokens`, or — for **Ollama** — is auto-discovered
  from the model's own `/api/show` context length, so compaction works locally
  with no manual configuration. `read` accepts absolute paths (like OpenCode; writes
  stay cwd-scoped). `apply_patch` shares `edit`'s whitespace-tolerant fallback.
  Project instruction files (`AGENTS.md` / `CLAUDE.md`, the user-global ones plus
  every one from the git root down to the cwd) load into the system prompt; HTTP
  calls **retry** transient failures (429 / 5xx / transport, honoring
  `Retry-After`, never retrying a context-overflow); model **reasoning** streams
  as `RunEvent::Thinking` — from a dedicated `reasoning_content` / `reasoning`
  field, or lifted from inline `<think>…</think>` tags (configurable via
  `with_reasoning_tag`, default `think`, `without_reasoning_extraction` to
  disable; handles tags split across SSE chunks); and hosts can register
  **named subagents** via
  `with_agent` that the `task` tool targets by `subagent_type` — each with its own
  role prompt and optional model override, advertised to the model as a catalog.
  An **MCP client** connects configured servers (`with_mcp_server`) at run start —
  over **stdio** (a launched process, newline-delimited JSON-RPC) or **HTTP** (a
  remote Streamable-HTTP endpoint, JSON or SSE) behind one transport trait — does
  the `initialize` handshake + `tools/list`, and surfaces each server's tools
  (namespaced `server_tool`) through the same tool set as the built-ins, offered +
  dispatched identically and shared with subagents — and a server's **resources**,
  if any, get a read-only `server_read_resource` tool (the resource list in its
  description), and a server's **prompts** are exposed host-side via
  `mcp_prompts()` / `get_mcp_prompt(...)` (`prompts/list` / `prompts/get`) for a
  host to surface as commands — the autonomous loop itself doesn't invoke them.
  Connection is best-effort (a server that fails is skipped with a status line,
  never fatal). It also honors `RunTuning.output_schema` for
  **structured output** (sends OpenAI `response_format: json_schema` each turn, so
  the final answer conforms; tool-call turns stay unconstrained), and emits an
  estimated **cost** on `RunEvent::Usage.cost_usd` when a model's per-token rates
  are registered via `with_model_cost`, and accepts **image input**
  (`RunRequest.attachments`) which it sends to vision models as base64 data-URI
  `image_url` parts on the first user message, and gates tool calls with
  **permission rules** (`with_permission_rule` — allow / deny / **ask** by tool +
  subject pattern, e.g. deny `bash` matching `rm -rf`, checked before execution).
  An `ask` rule consults a host **permission prompt** (`with_permission_prompt` —
  a callback invoked synchronously on the run thread, so the host blocks on its
  own confirm UI; the interactive allow/ask/deny of OpenCode). No rules =
  allow-all (the bypass posture). LSP is intentionally held. Not yet published.
- **`RunTuning.output_schema`** — an optional JSON Schema for **structured
  output**. An adapter that supports it constrains the model's final answer to the
  schema; `openai-compatible` does (OpenAI `response_format`), the CLI adapters ignore
  it for now. Additive (`Option`, defaults `None`).
- **`RunRequest.attachments` + the `Attachment` type** — **multimodal image
  input**. A multimodal adapter (`openai-compatible`) sends them to the model; the
  text CLI adapters (Claude / codex / ACP) ignore them. The `Vec` defaults empty,
  but adding the field is a breaking recompile for exhaustive `RunRequest`
  literals (add `attachments: Vec::new()`).
- **`AcpHarness` — drive external Agent Client Protocol agents** (new `acp`
  feature, off by default). Spawns an ACP agent (`::opencode()` runs
  `opencode acp`; `::custom(id, name, command, args)` for Gemini/Goose/…),
  speaks ACP (nd-JSON over stdio) via Zed's `agent-client-protocol` crate — we
  are the ACP *client* — and translates its `session/update` stream into
  `RunEvent`s using the aligned schema. Built on the crate's current **0.14**
  role/builder model: `Client.builder()` registers handlers for the agent's
  permission requests + session notifications, then `connect_with` spawns the
  agent and runs the session (`initialize` → `session/new` → `session/prompt`).
  Tracking current ACP (pinned `0.14`, not the long-stale `0.7`) is what lets it
  handshake with current agents (opencode / Zed / Gemini). The async connection
  is driven on a `smol` executor inside the worker thread, keeping the same
  thread+callback `run()` shape as the other adapters; a cancel races the
  connection and tears the agent down. `RunMode::Edit` auto-allows
  tool-permission prompts, `RunMode::Ask` rejects them (read-only). Opt-in —
  pulls the ACP crate + `smol`, off by default. Follow-ups: `session/load`
  resume, per-run model selection (a session config option).
- **`RunEvent::Plan { entries: Vec<PlanEntry> }`** — the agent's task/todo list
  (neutral shape). New `PlanEntry` + `PlanEntryStatus` + `PlanEntryPriority`
  types. The `acp` adapter maps ACP `plan` onto it; `openai-compatible` emits it
  from its `todowrite` tool. `PlanEntryStatus` gained `Cancelled` (additive —
  the enum is `#[non_exhaustive]`) to match OpenCode's todo statuses.
- **`RunEvent::SessionInfoUpdate { title, updated_at }`** — a live session
  title/timestamp update, mapping field-for-field onto ACP
  `session_info_update` (`title` + `updatedAt`), for sessions lists that want a
  title before the run ends.
- **`ToolKind` gained `Delete` / `Move` / `Fetch`** so it mirrors ACP's
  tool-call `kind` set (`read`/`edit`/`delete`/`move`/`search`/`execute`/`fetch`/
  `other`) — a `ToolStart` maps onto an ACP `tool_call` without a translation
  table. (Additive — `ToolKind` is `#[non_exhaustive]`. Our `Write` has no ACP
  kind; a bridge maps it to `edit`.)
- **`Harness::list_models()`** — a defaulted trait method returning the static
  `info().capabilities.models`. Adapters with a runtime model source (a hosted
  API's `/v1/models`, Ollama's `/api/tags`) override it; a harness with no
  model concept returns an empty list and the host hides the picker.
- The **Claude** parser now surfaces `cache_read_input_tokens` /
  `cache_creation_input_tokens`, and the **codex** parser surfaces
  `cached_input_tokens`, on `RunEvent::Usage`.

## [0.3.5] - 2026-06-12

### Fixed
- **Node-CLI spawns are paired with the node they were installed under.**
  `cli-stream` now resolves a bare program name (`bob`, `claude`, …) to its
  absolute path on the augmented PATH before spawning (`resolve_program`,
  exported), so the engine's program-dir prepend actually fires and the child's
  `#!/usr/bin/env node` picks the **sibling node from the CLI's own install
  dir** — not whichever node happens to lead the inherited PATH. Previously a
  bare name split the brain: the OS found bob under an nvm v24 dir while the
  child PATH led with v20, and bob's self-re-exec died on a newer-node-only
  flag (`--disable-sigusr1`) as an opaque "exited with code 9". Verified
  end-to-end by running bob through the harness with v20 deliberately first on
  PATH (`agent-harness/examples/run_bob.rs`).
- **bob runs preflight the Node version with a typed error.** `bob-rs`'s
  `spawn_bob` now checks the node that will actually execute bob (sibling
  first, else first on PATH) against `BOB_MIN_NODE_VERSION` and returns the new
  `BobError::NodeIncompatible` — "bob requires Node.js 22.15.0+ — found
  v20.19.2 at …" — instead of letting the child die with exit code 9. Additive
  (`BobError` is `#[non_exhaustive]`). Note: bobshell's own `engines` field
  claims `>=20.0.0`, but its runtime re-exec requires ≥22.12 in practice — the
  SDK constant is the empirically correct floor.

## [0.3.4] - 2026-06-12

### Fixed
- **bob's preamble narration no longer leaks above the answer.** Separate from
  its `<thinking>` reasoning, bob streams prose narration — bold status headings
  like `**Reading file…**` / `**Task completed successfully**` — inside assistant
  `message`s, and the parser surfaced that prose as message text next to the real
  answer. But bob's *answer* is always the `attempt_completion` result (verified
  in code AND ask mode — the assistant `message` only ever carries reasoning +
  preamble), so the `BobStreamParser` now folds **all** assistant-message content
  (reasoning + narration) into the `thinking` stream and leaves only the
  `attempt_completion` text as the visible message. A `tool_use` / non-message
  line (the answer) keeps its text. Verified by replaying a real captured bob
  stream (`examples/replay_bob.rs`): the visible text is exactly the answer; the
  narration rides in the collapsed thinking trace. Completes the 0.3.3 echo fix
  (the `[using tool …]` echo and the preamble prose were two separate leaks).

## [0.3.3] - 2026-06-11

### Fixed
- **bob's `[using tool …]` narration echoes no longer leak into the message.**
  The ported `BobStreamParser` had dropped `BobChatMapper`'s echo suppression, so
  bob's inline `[using tool read_file: …]` / `[using tool write_to_file: …]`
  lines surfaced as message text next to the real answer. Restored the
  `suppressing_echo` state machine (it handles the echo spanning deltas), so only
  `attempt_completion`'s result becomes the message text; `<thinking>` was already
  routed to its own stream. Verified by replaying a real captured bob stream
  through the parser (`examples/replay_bob.rs`).

## [0.3.2] - 2026-06-11

### Added
- **`RunRequest.resume` — continue a prior CLI session.** A host can pass the
  session id captured from an earlier run's init (`RunEvent::Session` /
  `SessionInfo`) to **resume that conversation** instead of replaying a
  transcript in the prompt, so the CLI supplies the history (full fidelity,
  fewer tokens). **All three adapters honor it uniformly**, the same way they do
  `extra_args`: Claude maps it to `--resume <id>`; Codex restructures to
  `codex exec resume <id> … <prompt>` (the id is a positional before the prompt,
  `--json`/`--skip-git-repo-check` still apply); bob threads it through
  `RunBobOptions.resume` into `--resume <id>` (bob accepts the session UUID).
  `None` → a fresh session. Additive: the field defaults `None`, so every
  existing caller is unaffected (set `resume: None`).
- **`RunEvent::AskQuestion` — neutral interactive-question event.** When an agent
  asks the user a multiple-choice question (Claude's `AskUserQuestion`), the
  adapter maps it onto `AskQuestion { run_id, request_id, questions }` carrying
  neutral `Question` / `QuestionOption` types, so a host renders chips without
  name-checking a harness's tool — the way `ToolKind` already neutralizes tool
  names. The Claude adapter emits it today; the answer travels back as the user's
  **next chat message** (the host's existing send-path resumes the session), so
  this is one event, no new control channel: `RunControl` is unchanged and no
  stdin write-back is involved. The enum is `#[non_exhaustive]`, so the new
  variant doesn't break consumers with a `_` arm.

### Changed
- **bob runs direct-write (`auto_edit`) — no more previewable-edit proposals.**
  The bob adapter now reports `previews_edits: false` and maps `RunMode::Edit` to
  `BobApprovalMode::AutoEdit`, so bob writes files directly like Claude/Codex and
  the host reviews via its own edit gate (snapshot/clone) rather than an in-stream
  preview. In Edit mode the adapter also **suppresses bob's `SuggestedEdits`** — an
  applied write is not a proposal, so it surfaces as a file-op (the
  `write_to_file` ToolStart/ToolEnd) instead. A host that branched on
  `previews_edits` now treats bob uniformly with the other write-capable harnesses,
  with no id checks. (Ask mode is read-only and unchanged.)

## [0.3.1] - 2026-06-10

### Added
- **Host-controlled CLI args via `RunTuning.extra_args`.** A host can pass raw
  flags, appended after an adapter's own argv — to add a flag (`--settings`,
  `--add-dir`) or set one the adapter otherwise defaults — without editing the
  adapter. Crucially, the adapter's defaults are *defaults, not fixed*: the
  Claude adapter omits its own `--permission-mode acceptEdits` when the host
  sets `--permission-mode` through `extra_args`, so the host fully owns the flag
  (a sensible default exists, but it's cleanly overridable — no duplicate).
  **All three adapters honor it uniformly**, so a client applies a flag the same
  way regardless of harness: Claude appends at the end of its argv; Codex before
  its trailing positional prompt; bob threads them through the new
  `RunBobOptions.extra_args` (bob-rs) into its own argv. Keeps run *policy* on
  the host: a fully-headless host that needs Bash/skills to run without an
  unanswerable permission prompt passes `--permission-mode bypassPermissions`.
  Additive: the field defaults empty, so every existing caller is unaffected.

## [0.3.0] - 2026-06-09

### Fixed
- **In-band harness failures now surface as `RunEvent::Error`.** `ParsedLine`
  gained an `error` field, and `run_events_from_parsed` — the single place a
  parsed stdout line becomes a `RunEvent` — emits `RunEvent::Error` when it's
  set. So a failure a harness reports *in its stdout stream* (not just a
  spawn/IO `ProcessEvent::Error`) now reaches the consumer. The codex adapter
  maps `codex exec --json`'s `turn.failed { error: { message } }` and
  `error { message }` lines to it: previously `turn.failed` was ignored outright
  and `error` was downgraded to a transient activity line, so a codex turn that
  failed mid-run (quota, context overflow, model error) produced no answer *and*
  no error — looking like the agent silently did nothing. Additive: parsers that
  don't set `error` are unaffected.

## [0.2.0] - 2026-06-09

### Added
- **Neutral `ToolKind` on `RunEvent::ToolStart`.** A cross-harness behaviour
  class (`Read` / `Write` / `Edit` / `Search` / `Execute` / `Other`) rides
  alongside the raw tool `name`, classified once per adapter where the wire
  format is already parsed, so a consumer can route by what a tool call *does*
  (a read → a context pill, an edit → a file-op card) without re-encoding each
  harness's native tool vocabulary (bob's `read_file`, Claude's `Read`, codex's
  `file_change`). The neutral class rides as `toolKind` on the wire — distinct
  from the `kind` event discriminator. Additive: the raw `name` / `tool_call_id`
  are unchanged, so a consumer that only reads those is unaffected.

## [0.1.0] - 2026-06-03

### Added
- **Typed errors, end to end.** Every crate's public API now returns a typed
  error carrying the real underlying source, not a flattened `String`:
  - `cli-stream` → `StreamError` (`Spawn` carries the spawn `io::Error`,
    `PipeNotCaptured`, `CancelLockPoisoned`).
  - `bob-rs` → `BobError` (`Io { context, source }`, `Keychain(keyring::Error)`,
    `Serialize(serde_json::Error)`, `Invalid`, `NoDataDir`, `Stream(StreamError)`).
  - `agent-harness` → `HarnessError`'s category variants (`Spawn`/`Install`/
    `Login`/`Cancel`) now carry a `BoxError` **source** instead of a `String`,
    so a consumer can `err.source().downcast_ref::<StreamError>()` /
    `::<BobError>()`. `Display` still flattens the source into the message, so a
    consumer that stringifies at a boundary (`.to_string()`) sees the same full
    text as before. `StreamError` and `BobError` are re-exported from
    `agent-harness` for downcasting; `HarnessError::{spawn,install,login,cancel}`
    constructors box any source.
- **`RunEvent` enrichment** — `Session` (id + model), `Usage` (input/output/total
  tokens), and tool `input`/`output`, populated by the bob/claude/codex parsers.
- **Headless auth.** `readiness()` reports authenticated when the CLI's API-key
  env var is set (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `BOBSHELL_API_KEY`) —
  so a container/CI run reports ready without an interactive browser login.
- **Login-shell PATH.** `cli-stream`'s `augmented_node_path()` resolves the
  user's real `PATH` via their login shell (finds nvm / pnpm / volta / asdf /
  Homebrew), cached once, with a hardcoded fallback.
- The harness-agnostic raw tier `parse_raw_line`, the open `Registry`, and the
  `custom_harness` example (compose your own harness from the published pieces).
- **`Harness::run()`** — a provided method that starts a run and returns
  its `RunEvent`s on an `mpsc` receiver, so callers can `for ev in rx { … }`
  instead of hand-writing the `Arc::new(move |ev| tx.send(ev))` callback. The
  receiver hangs up on its own when the run ends. `run()` stays for push
  semantics (forwarding onto a Tauri Channel / SSE sink from the callback).
- A local quality gate, `scripts/check.sh` (clippy `-D warnings` + test + build +
  feature-gate builds + `cargo deny` when installed), and a `deny.toml`.
- **Testable docs + real-I/O coverage.** Runnable/`no_run` doctests on the
  headline APIs (`spawn_streaming`, `Harness::run`, `HarnessError`,
  `Registry`) so the documented code can't drift from the API; a stub-process
  integration test (`tests/stub_run.rs`) that drives a real `sh` child through
  the full spawn → stream → normalize → channel/cancel path; and an
  env-passthrough engine test.

### Changed
- **`RunEvent` and `ProcessEvent` are `#[non_exhaustive]`** — new event kinds are
  additive (downstream matches carry a `_` arm), so future additions don't break
  consumers the way `Session`/`Usage` once did.
- The three adapters are uniform `<harness>/{mod,parser}.rs` modules.

### Fixed
- **`cli-stream::cancel()` now terminates a *running* child.** It previously held
  the child lock across a blocking `wait()`, so `cancel` couldn't send SIGTERM
  until the process exited on its own — "Stop" did nothing mid-run.
- No `unwrap`/`expect`/`panic!` remain in library (non-test) code; poisoned
  mutexes are recovered rather than panicked on.
- `augmented_node_path()` keeps only absolute `PATH` entries — a relative/empty
  entry (e.g. a direnv `node_modules/.bin`) can no longer run a planted binary
  from the spawn cwd.
- **`bob-rs` keychain now persists on Linux and Windows, not just macOS.** The
  `keyring` dependency was built with `apple-native` only, so on other platforms
  it fell back to a no-op store and silently dropped saved keys. It now selects a
  native backend per OS — Keychain (macOS), Credential Manager (Windows), Secret
  Service over D-Bus with a pure-Rust encrypted session (Linux). Headless Linux
  (no Secret Service daemon) is unaffected: the key comes from `BOBSHELL_API_KEY`.
