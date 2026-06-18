//! Compose's neutral agent-harness core.
//!
//! The library you depend on to drive — or build — an agent harness,
//! independent of any specific backend. It provides:
//!   * the [`Harness`] trait + the neutral request/metadata types
//!     ([`RunRequest`] / [`RunTuning`] / [`HarnessInfo`] / …),
//!   * the normalized [`RunEvent`] vocabulary every adapter parses into
//!     ([`normalize_process_event`] + [`ParsedLine`]),
//!   * the generic streaming subprocess engine ([`spawn_streaming`] +
//!     [`ProcessEvent`] + [`ProcessHandle`]) + the install/login event
//!     shape ([`InstallEvent`]), and
//!   * the shared interactive-login helper ([`run_login_command`]).
//!
//! The built-in per-CLI adapters live here as modules ([`bob`] / [`claude`]
//! / [`codex`]), re-exported as [`Bob`] / [`Claude`] / [`Codex`]. The
//! [`Registry`] is open: a third party adds their own provider by
//! implementing [`Harness`] in their crate and registering it — no fork.
//!
//! Wire shapes derive `Serialize` so every transport emits identical
//! JSON — keep their field names stable; the TypeScript front-end
//! consumes them verbatim.

pub mod events;
pub mod harness;
/// The untyped raw passthrough tier, harness-agnostic (any JSONL CLI).
pub mod raw;
/// models.dev catalog lookup for `list_models` (opt-in `models-dev` feature).
pub mod models_dev;

pub use events::{
    normalize_process_event, run_events_from_parsed, ByteRange, ParsedLine, PlanEntry,
    PlanEntryPriority, PlanEntryStatus, Question, QuestionOption, RunEvent, SessionInfo,
    SuggestedEdit, ToolCallEnd, ToolCallStart, ToolKind, ToolLocation, UsageInfo,
};
pub use raw::parse_raw_line;
pub use harness::{
    run_login_command, BoxError, CredentialSpec, Harness, HarnessCapabilities, HarnessError,
    HarnessInfo, HarnessModel, HarnessReadiness, InstallCallback, ReasoningEffort, RunCallback,
    RunControl, RunHandle, RunMode, RunRequest, RunTuning,
};
// The generic subprocess engine + the install/process event shapes live in
// the `cli-stream` leaf; re-export them so adapters + consumers reach them
// through the framework (e.g. `use harness::spawn_streaming`). `StreamError`
// is re-exported too so a consumer can `downcast_ref` a `HarnessError`'s
// source back to the typed spawn/cancel error.
pub use cli_stream::{
    augmented_node_path, spawn_streaming, InstallEvent, ProcessEvent, ProcessHandle, StreamError,
};

#[cfg(feature = "bob")]
pub mod bob;
#[cfg(feature = "claude")]
pub mod claude;
#[cfg(feature = "codex")]
pub mod codex;
/// The ACP-client adapter (drives an external Agent Client Protocol agent:
/// OpenCode, Gemini, Goose, …).
#[cfg(feature = "acp")]
pub mod acp;
/// The OpenAI-compatible / local-model runtime ("we are the agent" — owns the
/// loop + tool surface, unlike the CLI/ACP adapters that wrap an external agent).
#[cfg(feature = "openai-compatible")]
pub mod openai_compatible;
pub mod registry;

// The built-in adapters, re-exported as short names so consumers write
// `use harness::{Bob, Claude, Codex}` — each gated behind its feature.
#[cfg(feature = "bob")]
pub use bob::{bob_tool_kind, normalize_bob_event, BobHarness as Bob, BOB_HARNESS_ID};
// bob's typed error, re-exported so a consumer can `downcast_ref` a bob
// `HarnessError`'s source (install / keychain / spawn) back to `BobError`.
#[cfg(feature = "bob")]
pub use bob_rs::BobError;
#[cfg(feature = "claude")]
pub use claude::{ClaudeHarness as Claude, CLAUDE_HARNESS_ID};
#[cfg(feature = "codex")]
pub use codex::{CodexHarness as Codex, CODEX_HARNESS_ID};
#[cfg(feature = "acp")]
pub use acp::AcpHarness;
// The OpenAI-compatible / local-model runtime + its public surface (permission
// rules, MCP config, named subagents, model pricing, sessions).
#[cfg(feature = "openai-compatible")]
pub use openai_compatible::{
    AgentDef, McpPrompt, McpPromptArg, McpServer, McpTransport, ModelCost, OpenHarness, Permission,
    PermissionPrompt, PermissionRequest, PermissionRule, PromptMessage, SessionRecord,
};
// The open registry + convenience constructors over the built-ins.
pub use registry::{
    default_registry, harness_by_id, harness_catalog, Registry, DEFAULT_HARNESS_ID,
};
