//! The neutral harness contract: the [`Harness`] trait, the run-control
//! handle, the neutral request/metadata types, and the shared
//! interactive-login helper.
//!
//! A *harness* is whatever actually answers the user's prompt — a CLI
//! agent (bob / Claude Code / Codex today), a direct LLM API tomorrow,
//! some other runner after that. A consumer only needs to: probe whether
//! a harness is ready, run a one-time install if required, stream a run,
//! and know which credential to ask for. This module is that seam.
//!
//! ## Design rules
//!
//! - **Object-safe trait.** Consumers hold `Box<dyn Harness>`; no
//!   generics leak across the seam.
//! - **Arc callbacks, not generic closures.** Streaming methods take
//!   `Arc<dyn Fn(..) + Send + Sync>` so they stay object-safe and can be
//!   cloned onto the reader threads the subprocess engine uses.
//! - **Normalize at the adapter, not the UI.** The event enums in
//!   [`crate::events`] are harness-neutral by intent; each adapter
//!   translates its CLI's wire format into them so the front-end consumes
//!   one shape regardless of which harness produced it.

use std::path::PathBuf;
use std::sync::{mpsc, Arc, Condvar, Mutex};

use serde::{Deserialize, Serialize};

use crate::events::RunEvent;
use cli_stream::{Command, Event, InstallEvent, ProcessHandle};
use crate::program_path::ResolveCli;

// --- Streaming callbacks --------------------------------------------

/// Callback a harness invokes for each run event. `Arc<dyn Fn>` is
/// `Clone + Send + Sync`, so it can be handed to the multiple reader
/// threads a process-backed harness uses without the trait method
/// needing to be generic.
pub type RunCallback = Arc<dyn Fn(RunEvent) + Send + Sync>;

/// Callback a harness invokes for each install event.
pub type InstallCallback = Arc<dyn Fn(InstallEvent) + Send + Sync>;

// --- Errors ---------------------------------------------------------

/// A boxed, type-erased error source. The [`Error`] variants carry one
/// of these instead of `#[from]`-ing a single concrete type, because each
/// *category* can be produced by more than one underlying error: a `Spawn`
/// failure is a [`cli_stream::StreamError`] for a CLI adapter and an I/O or
/// protocol error for an ACP one. The real error stays reachable through
/// [`std::error::Error::source`] (and `downcast_ref`); the category is the
/// variant.
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Why a [`Harness`] operation failed. Returned by `install` / `run` /
/// `login` / [`RunControl::cancel`] so a consumer can branch on the *kind* of
/// failure — offer install vs sign-in vs surface the message — instead of
/// string-matching.
///
/// Each category carries the real underlying error as a [`source`] (via the
/// [`BoxError`] field), so a consumer that wants more than the category can
/// walk `.source()` or `downcast_ref::<cli_stream::StreamError>()`. The
/// `Display` still flattens the source into the
/// message (`"failed to start the agent: <source>"`), so a consumer that just
/// stringifies at a boundary (e.g. a Tauri command's `.to_string()`) gets the
/// same full message as before. `#[non_exhaustive]` so adding a variant later
/// isn't a breaking change.
///
/// ```
/// use harness::{Error, StreamError};
/// use std::error::Error as _; // for `.source()`, without shadowing `harness::Error`
///
/// // Box any typed source under a category constructor:
/// let err = Error::spawn(StreamError::PipeNotCaptured { stream: "stdout" });
///
/// // Stringifying at a boundary flattens the source into the message
/// // (so a Tauri command's `.to_string()` keeps its full text)…
/// assert!(err.to_string().starts_with("failed to start the agent: "));
///
/// // …while the real typed cause stays reachable for a consumer that wants
/// // to branch on it rather than parse a string.
/// let source = err.source().expect("Command carries a source");
/// assert!(source.downcast_ref::<StreamError>().is_some());
/// ```
///
/// [`source`]: std::error::Error::source
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The harness's CLI couldn't be started — not installed, not on `PATH`,
    /// or an OS-level spawn failure.
    #[error("failed to start the agent: {0}")]
    Spawn(#[source] BoxError),
    /// A one-time install step failed.
    #[error("install failed: {0}")]
    Install(#[source] BoxError),
    /// Interactive sign-in failed.
    #[error("sign-in failed: {0}")]
    Login(#[source] BoxError),
    /// Cancelling an in-flight run failed.
    #[error("cancel failed: {0}")]
    Cancel(#[source] BoxError),
    /// Any other adapter/runtime failure (e.g. a backend SDK error that
    /// doesn't map onto the cases above). Carries a message rather than a
    /// source — it's the catch-all when there's nothing typed to preserve.
    #[error("{0}")]
    Other(String),
}

impl Error {
    /// Categorize a source error as a [`Spawn`](Error::Spawn) failure.
    /// Accepts anything boxable — a typed `StreamError`/`BobError`, or a
    /// `String`/`&str` for adapters with nothing typed to carry.
    pub fn spawn(source: impl Into<BoxError>) -> Self {
        Self::Spawn(source.into())
    }
    /// Categorize a source error as an [`Install`](Error::Install) failure.
    pub fn install(source: impl Into<BoxError>) -> Self {
        Self::Install(source.into())
    }
    /// Categorize a source error as a [`Login`](Error::Login) failure.
    pub fn login(source: impl Into<BoxError>) -> Self {
        Self::Login(source.into())
    }
    /// Categorize a source error as a [`Cancel`](Error::Cancel) failure.
    pub fn cancel(source: impl Into<BoxError>) -> Self {
        Self::Cancel(source.into())
    }
}

// --- Run control (cancellation) -------------------------------------

/// Object-safe handle to an in-flight run. A process-backed harness
/// cancels by signalling its child; a request-backed harness (a hosted
/// LLM API) cancels by aborting its HTTP stream. The consumer only needs
/// these two operations, so the concrete mechanism stays behind the trait.
pub trait RunControl: Send + Sync {
    /// Stop the run. Best-effort; idempotent.
    fn cancel(&self) -> Result<(), Error>;
    /// Whether [`cancel`](RunControl::cancel) was called.
    fn was_cancelled(&self) -> bool;
    /// The OS process id of the underlying child while it's alive, for a
    /// process-backed run. `None` for adapters with no child process (a
    /// direct-model run aborts an HTTP stream, not a process) — so an embedder
    /// can record live pids and reap a child a hard crash orphaned.
    fn pid(&self) -> Option<u32> {
        None
    }
}

/// Boxed [`RunControl`] returned by [`Harness::run`].
pub type RunHandle = Box<dyn RunControl>;

// The engine's run handle is the canonical process-backed `RunControl`.
// Both the trait and the handle live in this crate, so this impl is here
// (orphan rule) rather than in any adapter crate.
impl RunControl for ProcessHandle {
    fn cancel(&self) -> Result<(), Error> {
        ProcessHandle::cancel(self).map_err(Error::cancel)
    }
    fn was_cancelled(&self) -> bool {
        ProcessHandle::was_cancelled(self)
    }
    fn pid(&self) -> Option<u32> {
        ProcessHandle::pid(self)
    }
}

// --- Neutral request / metadata shapes ------------------------------

/// What the user wants the harness to do with the prompt. Mirrors
/// the Ask / Edit split the comment bubble already exposes; adapters
/// map it onto their own mode vocabulary.
///
/// **How strongly `Ask` is enforced depends on the adapter**, because only one
/// of them owns the tools:
///
/// * `openai-compatible` — this crate owns the tool surface, so `Ask` simply
///   does not offer the mutating tools. The model cannot call what it was
///   never given.
/// * `acp` — the agent owns its tools. `Ask` denies ACP permission requests,
///   which works only for calls the agent chooses to ask about. An agent that
///   treats reading a file or searching the web as safe will just do it.
/// * `claude` / `codex` — mapped onto each CLI's own permission flags, so the
///   CLI enforces it.
///
/// Treat `Ask` as "do not change my files", not as a sandbox. None of these
/// adapters isolate the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    /// Answer / discuss. No file edits expected.
    ///
    /// The default, deliberately: a caller who omits the mode gets the
    /// read-only one. Defaulting to `Edit` would hand write access to anyone
    /// who forgot the field.
    #[default]
    Ask,
    /// Propose edits to the workspace.
    Edit,
}

/// Whether a run may call tools at all — the question of *reach*, kept apart
/// from [`RunMode`]'s question of *write permission*.
///
/// Every other agent SDK expresses this as a per-call parameter rather than a
/// mode: `ModelSettings.tool_choice = "none"` in the OpenAI Agents SDK,
/// `tool_choice: "none"` on the OpenAI and Anthropic APIs, `toolChoice: 'none'`
/// in the Vercel AI SDK. It is not a third rung on the Ask/Edit ladder, and
/// modelling it as one made `Ask` and `Answer` two names for adjacent ideas.
///
/// This is not `tool_choice: "none"`, which still *sends* every schema and
/// only forbids the call. Nothing is sent, because the tokens and the
/// temptation were both the problem.
///
/// A *guarantee*, in the tiers on [`Harness`]: a caller acts on it being true,
/// so an adapter that cannot keep it refuses rather than running unsandboxed,
/// and says which it is through [`Features::withheld_tools`] beforehand.
///
/// Kept by `openai-compatible` (nothing offered, dispatched or connected) and
/// `claude` (`--disallowedTools "*"`; an empty *allow* list is the CLI's
/// auto-approve list, not an availability gate, and withholds nothing).
///
/// Refused by `codex` and `acp`, for narrower reasons than "no mechanism" —
/// each has one, and neither reaches far enough. Codex has per-tool switches
/// (`features.shell_tool`, `web_search`, `--disable <feature>`) but none for
/// all of them: with the shell off, `apply_patch` still writes — 4 of 4 runs,
/// with the sandbox opened so it was not the thing refusing — and the set
/// drifts by version, so an enumeration here could not be checked against
/// anything. ACP can stop a call — `session/request_permission` runs agent to
/// client with the call and the options, and the client answers — but it cannot
/// un-offer, and this crate implements no such handler. What an agent does when
/// every request is denied has not been observed here, so the schema is the
/// claim and the behaviour is not.
/// Refuse a run asking for a tool guarantee this adapter cannot make.
///
/// Taking [`ToolAccess::None`] and running with tools anyway hands a caller a
/// sandbox that is not one, and silence is what makes it dangerous: a host
/// adopted the no-tools mode for a judging task and, ten hours later, 14 runs
/// had made 46 tool calls between them. Nothing said otherwise, because the
/// adapters took the field and dropped it.
///
/// So an adapter that cannot honour it says so at the call, where the caller
/// can still choose, rather than at the end of a run that was never sandboxed.
pub(crate) fn refuse_withheld_tools(
    adapter: &str,
    tools: ToolAccess,
    because: &str,
) -> Result<(), Error> {
    if tools == ToolAccess::None {
        return Err(Error::Other(format!(
            "{adapter}: ToolAccess::None asks for a guarantee this adapter cannot make — \
             {because}. Run it on an adapter that honours it (claude, openai-compatible), or \
             ask for ToolAccess::Default and treat every tool as reachable."
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolAccess {
    /// Whatever the mode offers: read-only tools in `Ask`, those plus the
    /// mutating ones in `Edit`.
    #[default]
    Default,
    /// No tools are offered, and none can be reached. A model trained with its
    /// own tool syntax calls one whether or not any was offered, so the offer
    /// alone is not the guarantee: a call made anyway is refused, not served.
    ///
    /// For work where everything needed is already in the message — classifying
    /// a document, extracting a field, judging whether two things are the same.
    /// A run is offered read-only file tools whether or not the task has any use
    /// for them, and a prompt full of paths and URLs invites a model to go
    /// looking: one host saw 228 records fail because a tool call consumed the
    /// only turn, and later found a job spending 17 tool calls listing a
    /// repository it had been asked nothing about.
    ///
    /// Asking a model in prose not to call tools works and is the weaker fix.
    /// This removes the choice.
    ///
    /// **What the model *says* still differs by adapter, and the answer text is
    /// not evidence of a lookup.** The `openai-compatible` adapter refuses a
    /// call the model makes anyway with a message saying the run has no tools,
    /// so the model reports that it cannot look. The `claude` CLI withholds
    /// silently, so the model narrates the call it wanted into its answer —
    /// observed emitting `Bash: {"command":"ls -la"}` as prose, and elsewhere
    /// continuing past that into invented output, reporting `total 0` for a
    /// directory that held a file. Nothing ran in either case; the guarantee
    /// holds. But a caller scraping an answer for tool syntax, or reading
    /// "I checked and found nothing" as a finding, will be misled — and for
    /// judging work an invented absence can flip the judgement. Treat the text
    /// as the model's own words.
    None,
}

/// How hard the model should think, in harness-neutral terms. Codex
/// maps this onto `model_reasoning_effort`; Claude Code has no
/// equivalent `-p` flag today and ignores it. Kept neutral so a future
/// harness that exposes effort can honor the same field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    /// The CLI/config token for this level (e.g. codex's
    /// `model_reasoning_effort="high"`).
    pub fn as_cli_value(self) -> &'static str {
        match self {
            ReasoningEffort::Minimal => "minimal",
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
        }
    }
}

/// User-chosen, harness-neutral run-shaping knobs. Every field is
/// optional; each adapter maps the ones its CLI supports and ignores
/// the rest (Claude has no reasoning-effort flag; Codex has no
/// max-turns flag). Grouped into one struct so the neutral
/// [`RunRequest`] stays open for extension — a new knob is a field
/// here, not a new positional parameter threaded through every caller.
#[derive(Debug, Clone, Default)]
pub struct RunTuning {
    /// Model id or alias passed verbatim to the CLI (`--model` /
    /// `-m`). `None` → let the CLI use its configured default.
    pub model: Option<String>,
    /// Reasoning effort (Codex: `-c model_reasoning_effort`).
    ///
    /// A preference, in the tiers on [`Harness`]: ignored where an adapter has
    /// no equivalent, because the answer is the same shape either way.
    pub effort: Option<ReasoningEffort>,
    /// Cap on agentic turns (Claude: `--max-turns`).
    ///
    /// A preference. Ignored by Codex and ACP, which have no turn cap: a run
    /// then takes more turns than asked, which the caller sees in the result.
    pub max_turns: Option<u32>,
    /// Raw CLI args the host appends verbatim **after** the adapter's own,
    /// so a host can add a flag (`--settings`, `--add-dir`) or override one
    /// it already sets — for CLIs where a repeated flag is last-wins (e.g.
    /// Claude Code / commander) — without editing the adapter. The host opts
    /// into CLI-specific flag names when it uses this; keep cross-harness
    /// knobs as their own typed fields above. Default empty.
    pub extra_args: Vec<String>,
    /// A JSON Schema the final assistant answer must conform to (structured
    /// output). Adapters that support it constrain the model's final message to
    /// this schema; the rest ignore it. `None` → free-form text.
    pub output_schema: Option<serde_json::Value>,
    /// Extra system-prompt instructions from the host — the user's per-harness
    /// "custom instructions". The `openai-compatible` adapter appends it after
    /// its base system prompt; other adapters currently ignore it (a CLI mapping
    /// such as Claude's `--append-system-prompt` can opt in later). `None` → none.
    pub extra_instructions: Option<String>,
    /// Absolute path to the agent's executable, overriding PATH resolution of
    /// the bare CLI name. `None` → resolve by name on PATH. CLI adapters
    /// (claude/codex/bob) spawn this path instead of their default program; the
    /// `openai-compatible` adapter spawns no process and ignores it.
    pub binary_path: Option<std::path::PathBuf>,
}

/// A non-text input attached to a run — currently an image. Multimodal adapters
/// (`openai-compatible`) send it to the model; text-only CLI adapters ignore it.
#[derive(Debug, Clone)]
pub struct Attachment {
    /// MIME type, e.g. `image/png` or `image/jpeg`.
    pub mime_type: String,
    /// Raw bytes; the adapter base64-encodes them into a data URI for the wire.
    pub data: Vec<u8>,
}

/// A harness-neutral run request. Adapter-specific knobs (bob's
/// approval mode, coin budget, executable override) are filled in by
/// the adapter from its own defaults; the user-facing tuning the
/// picker exposes (model, effort, turn cap) rides on `tuning`.
/// Derives `Default`, so a call site names only what it cares about and leaves
/// the rest to `..Default::default()`. Spelling out `cwd: None`, `resume:
/// None` and `attachments: Vec::new()` on every call was noise, and it made
/// each new field a breaking change for every caller.
#[derive(Debug, Clone, Default)]
pub struct RunRequest {
    /// Caller-chosen id used to correlate events with the handle.
    pub run_id: String,
    pub prompt: String,
    /// Non-text inputs (images) for multimodal models; empty for a text run.
    /// Multimodal adapters send them to the model; text-only CLI adapters
    /// ignore them.
    pub attachments: Vec<Attachment>,
    /// Working directory for the run — the workspace path, so the
    /// harness's tool calls land inside the user's vault.
    ///
    /// `None` inherits the host process's working directory, which is seldom
    /// what a caller means: an agent holding tools then has reach over
    /// wherever the host happened to be started from. Name a directory chosen
    /// for the run.
    pub cwd: Option<PathBuf>,
    pub mode: RunMode,
    /// Whether this run may call tools at all. Defaults to whatever `mode`
    /// offers.
    pub tools: ToolAccess,
    /// Optional, harness-neutral run-shaping knobs (model, effort,
    /// turn cap). Adapters honor the subset their CLI supports.
    pub tuning: RunTuning,
    /// Session id to **resume** — continue a prior run's conversation instead
    /// of starting fresh, so the CLI supplies the history (no transcript replay
    /// in the prompt). `None` → a new session. Each adapter maps it to its
    /// CLI's resume form (Claude `--resume <id>`, codex `exec resume <id>`,
    /// bob `-r <id>`); the id comes from the earlier run's init `SessionInfo`.
    pub resume: Option<String>,
}

/// Where a harness's secret lives in the OS keychain, and how to
/// label it in the UI. Lets the front-end ask for the right
/// credential per harness without hard-coding any one harness's slot.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CredentialSpec {
    /// Human label, e.g. "Bob API key" / "Anthropic API key".
    pub label: String,
    pub keychain_service: String,
    pub keychain_account: String,
    /// Whether the harness can run at all without this credential.
    pub required: bool,
}

/// Harness-neutral readiness snapshot for the UI. `details` carries
/// adapter-specific probes (bob's Node/npm) as free-form JSON so the
/// trait stays generic.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Readiness {
    pub harness_id: String,
    /// Installed *and* authenticated *and* able to run.
    pub ready: bool,
    pub installed: bool,
    pub version: Option<String>,
    pub auth_configured: bool,
    pub error: Option<String>,
    /// Adapter-specific extra fields (serialized harness snapshot).
    pub details: serde_json::Value,
}

/// A model the harness can be pointed at, for the picker's model
/// selector. `value` is passed verbatim to the CLI (`--model` / `-m`)
/// via [`RunTuning::model`]; `label` is the human-facing name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelChoice {
    pub value: String,
    pub label: String,
}

/// An installed model with the metadata a model-manager UI shows — the
/// neutral shape returned by [`Harness::list_installed_models`]. Richer than
/// [`ModelChoice`] (the picker's name-only entry): on-disk `size` in bytes plus
/// the parameter count / quantization where the backend reports them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstalledModel {
    pub name: String,
    /// On-disk size in bytes.
    pub size: u64,
    /// e.g. `"3.2B"`; `None` when the backend doesn't report it.
    pub parameter_size: Option<String>,
    /// e.g. `"Q4_K_M"`; `None` when the backend doesn't report it.
    pub quantization_level: Option<String>,
}

/// A progress update from [`Harness::pull_model`], one per chunk of a streaming
/// download. `status` is always present (`"pulling manifest"`, `"success"`, …);
/// the byte counters appear once a layer is downloading, so a host can show a
/// percentage from `completed`/`total` aggregated across `digest`s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PullProgress {
    pub status: String,
    /// The layer this line reports on; `None` for phase lines (manifest,
    /// success).
    pub digest: Option<String>,
    pub total: Option<u64>,
    pub completed: Option<u64>,
}

/// A host push-callback for [`Harness::pull_model`] download progress, invoked
/// per stream chunk on the calling thread.
pub type PullProgressCallback<'a> = &'a mut (dyn FnMut(PullProgress) + Send);

/// Folds a stream of [`PullProgress`] into a single overall percent, so a host
/// shows one progress bar for a multi-layer download. A streaming pull reports
/// `completed`/`total` *per `digest`* and resends a digest's line as it grows;
/// this keeps the latest figures per digest, so the overall percent is
/// `100 * Σcompleted / Σtotal` across the digests seen so far.
#[derive(Debug, Default)]
pub struct PullProgressAggregator {
    layers: std::collections::HashMap<String, (u64, u64)>,
}

impl PullProgressAggregator {
    /// Fold one progress update in (only `digest` lines carrying a `total`
    /// count) and return the overall percent so far — `None` until any byte
    /// total is known (e.g. during the manifest phase).
    pub fn update(&mut self, progress: &PullProgress) -> Option<f64> {
        if let (Some(digest), Some(total)) = (&progress.digest, progress.total) {
            self.layers.insert(digest.clone(), (progress.completed.unwrap_or(0), total));
        }
        self.percent()
    }

    /// Overall percent across every digest seen, clamped to 0–100; `None` until a
    /// total is known. A just-finished layer can momentarily report
    /// `completed > total`, so the ratio is capped.
    pub fn percent(&self) -> Option<f64> {
        let total: u64 = self.layers.values().map(|(_, t)| *t).sum();
        if total == 0 {
            return None;
        }
        let completed: u64 = self.layers.values().map(|(c, _)| *c).sum();
        Some((completed as f64 / total as f64 * 100.0).clamp(0.0, 100.0))
    }
}

/// What a harness's local-model management exposes — returned by
/// [`Harness::model_management`] (`None` when unsupported). Today only the
/// `openai-compatible` Ollama adapter manages models; carrying its endpoint lets
/// a host link out (e.g. "browse all models") without re-deriving it, while the
/// pull/list/delete operations themselves stay behind the trait so HTTP never
/// leaves the adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelManagement {
    /// The model server's base URL (e.g. `http://localhost:11434`), for a host
    /// to show or link to; not used to issue requests host-side.
    pub base_url: String,
}

/// What a harness supports, so every consumer (the picker, the options
/// panel, the credential preflight, the chat availability gate) adapts
/// to it *declaratively* instead of branching on the harness id. A new
/// adapter that, say, needs a stored key just sets `credential_required:
/// true` here — no `id == "bob"` checks to hunt down.
///
/// [`Default`] is "supports nothing", which is the honest starting point: an
/// adapter names what it does support and leaves the rest, rather than
/// restating eight fields and risking one being wrong by omission.
///
/// ```
/// # use harness::Features;
/// let claude_like = Features { max_turns: true, ..Default::default() };
/// assert!(!claude_like.effort);
/// ```
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Features {
    /// Compose stores this harness's credential (bob). When `false`,
    /// the CLI owns its own login (claude/codex) and Compose runs no
    /// credential/install preflight — a missing login surfaces as the
    /// harness's own run error rather than a Compose prompt.
    pub credential_required: bool,
    /// Emits previewable suggested edits the user approves before they
    /// apply (bob). When `false`, edits land on disk directly and the
    /// file watcher reflects them (claude/codex).
    pub previews_edits: bool,
    /// Curated model choices for the picker's selector. Empty → no
    /// curated list (rely on `custom_model`).
    pub models: Vec<ModelChoice>,
    /// Whether a free-text model id is accepted beyond `models` (codex,
    /// whose model names change frequently). Drives a text field vs a
    /// fixed dropdown in the picker.
    pub custom_model: bool,
    /// Honors [`RunTuning::effort`] (codex reasoning effort).
    pub effort: bool,
    /// Honors [`RunTuning::max_turns`] (claude turn cap).
    pub max_turns: bool,
    /// Honors [`ToolAccess::None`]. `false` means the adapter cannot withhold
    /// its agent's tools and **refuses** such a run rather than pretending —
    /// so a host offering a no-tools mode asks here first, instead of meeting
    /// it as a failed run. True for `claude` and `openai-compatible`.
    pub withheld_tools: bool,
    /// Supports an interactive [`Harness::login`] flow (the CLI's own
    /// OAuth, e.g. `claude auth login` / `codex login`). Drives the
    /// picker's "Sign in" affordance when installed-but-not-signed-in.
    /// `false` for harnesses Compose authenticates itself (bob).
    pub login: bool,
    /// Honors [`RunTuning::extra_instructions`] — the user's per-harness custom
    /// instructions, appended to the system prompt. `true` only for the
    /// `openai-compatible` adapter so far; the picker hides the field for the
    /// rest rather than offering a control that does nothing.
    pub custom_instructions: bool,
}

/// Where a user gets a harness that isn't on the machine yet.
///
/// This crate discovers and runs agents; it never installs them. A harness
/// that depends on an external CLI says so here and the host renders it, so
/// "not installed" is a next step rather than a dead end.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InstallHint {
    /// Where to get it. Always present — every agent has a home page, while
    /// only some have a one-liner that works on every platform.
    pub url: String,
    /// A copy-pasteable command, when one exists for every supported platform.
    pub command: Option<String>,
}

impl InstallHint {
    pub fn url(url: impl Into<String>) -> Self {
        Self { url: url.into(), command: None }
    }

    pub fn with_command(mut self, command: impl Into<String>) -> Self {
        self.command = Some(command.into());
        self
    }
}

/// Who a harness is: the identity and presentation a picker renders.
///
/// What it can *do* is [`Features`], asked for separately — that question
/// is put far more often than this one, and answering it should not mean
/// building three strings.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Info {
    pub id: String,
    pub display_name: String,
    pub description: String,
    /// How the user installs this harness themselves. `None` when there is
    /// nothing to install — a hosted endpoint, or an agent they already supply.
    pub install_hint: Option<InstallHint>,
}

// --- The trait ------------------------------------------------------

/// A pluggable agent backend. Implementors are cheap to construct
/// (they hold config, not connections) so a registry can hand out
/// fresh boxes on demand.
///
/// # What an adapter owes a request it cannot honour
///
/// No adapter supports everything: `max_turns` has no Codex flag, `effort` has
/// no ACP equivalent, and neither of those two can withhold their agent's own
/// tools. There are three honest answers, and which applies is decided by one
/// question — **if this is silently not honoured, does the caller end up
/// believing something untrue?**
///
/// | | when | how |
/// |---|---|---|
/// | Advertise | always | a [`Features`] flag, so a host asks before it runs |
/// | Refuse | a guarantee this adapter cannot keep | `Err` from [`start`](Harness::start), naming the adapter |
/// | Ignore | a preference | honour nothing, and say so in the doc comment |
/// | Say it is missing | the request cannot be *expressed* here | neither — add the capability, or decline to |
///
/// The fourth row is not a way of handling a request; it is noticing that there
/// was no request. A host wanting its system prompt to *replace* the agent's is
/// not being refused a guarantee or ignored a preference — [`RunTuning`] offers
/// `extra_instructions`, additive by name, and replacement has never been on
/// offer. Reach for this row when the tiers feel stretched: usually the thing
/// being classified is a missing capability wearing a request's clothes.
///
/// A *preference* changes what a run costs or how well it goes, and not
/// honouring it shows up in the result: a turn cap ignored means more turns,
/// which the caller can see and pay for.
///
/// A *guarantee* is something the caller acts on being true. Not honouring it
/// silently leaves them holding a false belief, and the belief is the damage. A
/// host that asked for [`ToolAccess::None`] and got tools anyway will classify
/// documents with a live filesystem underneath and never know — which is not
/// hypothetical: it ran for ten hours downstream, 46 tool calls across 14 runs
/// that had asked for none, defended by a flag that parsed and did nothing and
/// by a comment that described the gap accurately.
///
/// Prefer advertising to refusing. An `Err` mid-sweep is one failure per
/// record; a [`Features`] flag read at startup is one decision. Refusal is the
/// backstop for a host that named this adapter anyway.
///
/// A comment is not one of these answers. `tools: _` with an explanation of why
/// reads as diligence and behaves exactly like silence: no flag, no error, and
/// nothing said at the only moment a caller could act on it.
///
/// What counts instead is an artifact. A guarantee ships as a [`Features`] flag
/// *and* a row in the test that holds the flag to the behaviour, so an adapter
/// that advertises one and does the other fails rather than reads well. This
/// rule cannot make the classification mechanical — an author still picks the
/// tier — but it makes a wrong pick show up as a missing artifact instead of a
/// plausible paragraph, which is the whole difference from the comment above.
pub trait Harness: Send + Sync {
    /// Who this harness is — identity and presentation, for the picker.
    fn info(&self) -> Info;

    /// What this harness supports, so a consumer adapts to it declaratively
    /// instead of branching on [`Info::id`].
    ///
    /// Defaults to supporting nothing, which is the safe direction: an adapter
    /// names what it does, and one that has not heard of a capability added
    /// later does not claim it.
    fn features(&self) -> Features {
        Features::default()
    }

    /// Probe availability / version / auth. May shell out; callers
    /// should treat it as blocking and run it off the UI thread.
    fn readiness(&self) -> Readiness;

    /// Start a run, streaming events through `on_event`. Returns a
    /// handle immediately; work continues on background threads.
    fn start(&self, request: RunRequest, on_event: RunCallback) -> Result<RunHandle, Error>;

    /// The credential this harness needs.
    fn credential(&self) -> CredentialSpec;

    /// Enumerate the models this harness can run, *live*. The default returns
    /// the static list declared in [`Info`]
    /// (`capabilities().models`), so existing adapters need no change.
    ///
    /// Override it when the model set is discovered at runtime rather than
    /// known at compile time — a hosted-API adapter querying the provider's
    /// `/v1/models`, an Ollama adapter hitting `/api/tags`. A harness with no
    /// model-selection concept (bob runs whatever it's configured with)
    /// returns an empty list, and the host hides the picker — capability by
    /// the *absence* of models, not a separate flag. May shell out / hit the
    /// network; treat it as blocking and run it off the UI thread.
    fn list_models(&self) -> Result<Vec<ModelChoice>, Error> {
        Ok(self.features().models)
    }

    /// Whether this harness can install/list/delete its own models locally, and
    /// if so the endpoint metadata a host UI can surface (see
    /// [`ModelManagement`]). `None` (the default) means model management isn't
    /// supported — a host hides the "Manage models" surface. Only the
    /// `openai-compatible` Ollama adapter returns `Some` today.
    fn model_management(&self) -> Option<ModelManagement> {
        None
    }

    /// Installed local models with their on-disk size + details, for a manager
    /// UI (distinct from [`list_models`](Harness::list_models), the picker's
    /// name-only set). Default: unsupported — override alongside
    /// [`model_management`](Harness::model_management). Blocking (hits the local
    /// server); run it off the UI thread.
    fn list_installed_models(&self) -> Result<Vec<InstalledModel>, Error> {
        Err(Error::Other(
            "This harness does not support managing models.".to_owned(),
        ))
    }

    /// Download (install) a model, streaming progress to `on_progress`. `cancel`
    /// is polled during the download; flipping it aborts the pull. Blocking
    /// until the download finishes (or fails / is cancelled); run it off the UI
    /// thread. Default: unsupported.
    fn pull_model(
        &self,
        _model: &str,
        _cancel: &std::sync::atomic::AtomicBool,
        _on_progress: PullProgressCallback<'_>,
    ) -> Result<(), Error> {
        Err(Error::Other(
            "This harness does not support managing models.".to_owned(),
        ))
    }

    /// Remove an installed local model. Removing one that's already absent
    /// succeeds (the requested end state). Default: unsupported.
    fn delete_model(&self, _model: &str) -> Result<(), Error> {
        Err(Error::Other(
            "This harness does not support managing models.".to_owned(),
        ))
    }

    /// Trigger the harness's own interactive sign-in (its CLI's OAuth),
    /// streaming progress as [`InstallEvent`]s. The flow opens the user's
    /// browser; this blocks until the login process exits, then
    /// `Done { ok }` reports success. This is the agent authenticating
    /// itself — distinct from installing it, which the host's user does.
    /// Default: unsupported, for harnesses the host authenticates by key.
    fn login(&self, _on_event: InstallCallback) -> Result<(), Error> {
        Err(Error::login(
            "This harness does not support interactive sign-in.",
        ))
    }

    /// Convenience over [`run`](Harness::run) for callers that want to
    /// *pull* events off a channel instead of supplying a push callback.
    /// Forwards each [`RunEvent`] into an `mpsc` channel and hands the
    /// receiver back alongside the run handle, so the caller can simply
    /// `for event in rx { … }` rather than re-write the
    /// `Arc::new(move |ev| tx.send(ev))` plumbing at every call site.
    ///
    /// The receiver hangs up when the run ends — and on its own, without
    /// the caller dropping the [`RunHandle`] first. The forwarding callback
    /// (and the `Sender` it owns) lives only on the engine's reader
    /// threads; once the process exits and those threads finish, every
    /// clone of the callback drops, the `Sender` drops, and the `for` loop
    /// over `rx` terminates. (Dropping the handle never cancels a run — see
    /// [`RunControl`] — so it is safe to drain `rx` to completion while
    /// still holding the handle for a possible [`cancel`](RunControl::cancel).)
    ///
    /// Prefer [`start`](Harness::start) when you need push semantics —
    /// e.g. forwarding straight onto a Tauri `Channel` or an SSE sink from
    /// inside the callback — where an intermediate channel is just an extra
    /// hop. This is a provided method (not overridable surface): an adapter
    /// implements only [`start`](Harness::start), and every harness — built-in
    /// or third-party — gets `run` for free.
    ///
    /// ```no_run
    /// use harness::{Claude, Harness, RunEvent, RunMode, RunRequest, RunTuning};
    ///
    /// # fn main() -> Result<(), harness::Error> {
    /// let (_handle, rx) = Claude::new().run(RunRequest {
    ///     run_id: "demo".into(),
    ///     prompt: "Explain Markdown headings in one sentence.".into(),
    ///     ..Default::default()
    /// })?;
    /// for event in rx {
    ///     match event {
    ///         RunEvent::Text { delta, .. } => print!("{delta}"),
    ///         RunEvent::Exited { .. } => break,
    ///         _ => {}
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    fn run(
        &self,
        request: RunRequest,
    ) -> Result<(RunHandle, mpsc::Receiver<RunEvent>), Error> {
        let (tx, rx) = mpsc::channel();
        let handle = self.start(
            request,
            Arc::new(move |event| {
                // A hung-up receiver (consumer stopped early) is not an
                // error: the run keeps streaming; we just drop the event
                // nobody is waiting for.
                let _ = tx.send(event);
            }),
        )?;
        Ok((handle, rx))
    }
}

/// Run a harness's interactive sign-in command, streaming its output as
/// [`InstallEvent`]s and blocking until it exits. Reuses
/// [`ResolveCli`] (CLI resolution + reader threads, so a packaged
/// `.app` finds the CLI), mapping its process events onto the
/// install-stream shape (Step / Stdout / Stderr / Done). The login CLI
/// opens the user's browser for OAuth; we surface its output (incl. any
/// device-code URL) so the UI can show progress. Blocks on a condvar
/// until the process exits — the caller is a Tauri `(async)` command on
/// a worker thread, so the UI never blocks.
pub fn run_login_command(
    program: &str,
    args: &[&str],
    on_event: InstallCallback,
) -> Result<(), Error> {
    (*on_event)(InstallEvent::Step {
        text: "Opening your browser to sign in…".to_owned(),
    });
    let done = Arc::new((Mutex::new(false), Condvar::new()));
    let done_cb = Arc::clone(&done);
    let events_cb = Arc::clone(&on_event);
    // Bound, not `_`, so the handle outlives the wait (dropping it could
    // signal the child); by the time we return, the process has exited.
    let spawn = Command::new(program).cwd(std::env::current_dir().unwrap_or_default()).run_id(format!("login-{program}"))
        .args(args.iter().copied());
    let _handle = spawn.resolve_cli().stream(move |event| {
            let finished = matches!(event, Event::Exited { .. });
            if let Some(install) = login_event(&event) {
                (*events_cb)(install);
            }
            if finished {
                let (lock, cvar) = &*done_cb;
                // Recover from a poisoned lock instead of panicking on a
                // reader thread: the guarded value is a plain bool, never in a
                // half-updated state worth bailing on.
                *lock.lock().unwrap_or_else(|p| p.into_inner()) = true;
                cvar.notify_all();
            }
        },
    )
    .map_err(Error::login)?;
    let (lock, cvar) = &*done;
    let mut finished = lock.lock().unwrap_or_else(|p| p.into_inner());
    while !*finished {
        finished = cvar.wait(finished).unwrap_or_else(|p| p.into_inner());
    }
    Ok(())
}

/// One process event as an install-stream event, or `None` for the ones a user
/// has no use for.
///
/// A separate function because the interesting cases are decisions, not
/// plumbing: stderr carries text worth showing (an OAuth device code often
/// arrives there, and so does the reason a login failed), and a read or wait
/// failure is surfaced as stderr rather than dropped — a run that dies mid-way
/// would otherwise end with no explanation at all. Those paths are provoked by
/// OS-level failures no test can arrange, so the mapping is checked here with
/// values instead.
fn login_event(event: &Event) -> Option<InstallEvent> {
    match event {
        Event::Stdout { line, .. } => Some(InstallEvent::Stdout { text: line.clone() }),
        Event::Stderr { line, .. } => Some(InstallEvent::Stderr { text: line.clone() }),
        Event::Error { message, .. } => Some(InstallEvent::Stderr { text: message.clone() }),
        Event::Exited { exit_code, .. } => {
            Some(InstallEvent::Done { exit_code: *exit_code, ok: *exit_code == Some(0) })
        }
        // `Started` is the spawn itself, which the caller already knows about;
        // `Event` is #[non_exhaustive], so anything new is ignored too.
        _ => None,
    }
}

/// Whether an API-key value an adapter pulled from the environment counts as
/// authenticated — i.e. present and non-blank. Adapters OR this into their
/// [`Harness::readiness`] so a key in the env (headless / CI / container)
/// reports authenticated, not only the CLI's own interactive OAuth login —
/// which can't complete where there's no browser. Pure (the env read stays at
/// the call site) so it's unit-tested directly.
///
/// Only the claude/codex adapters OR this into readiness, so it is gated to
/// those features: without them (`--no-default-features`) it would be dead
/// code, hence the `cfg`.
#[cfg(any(feature = "claude", feature = "codex"))]
pub(crate) fn api_key_value_usable(value: Option<String>) -> bool {
    matches!(value, Some(v) if !v.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(digest: &str, completed: u64, total: u64) -> PullProgress {
        PullProgress {
            status: format!("pulling {digest}"),
            digest: Some(digest.to_owned()),
            total: Some(total),
            completed: Some(completed),
        }
    }

    #[test]
    fn pull_aggregator_sums_across_digests_keeping_latest_per_digest() {
        let mut agg = PullProgressAggregator::default();
        // Info phase: no byte totals yet → no percent.
        assert_eq!(
            agg.update(&PullProgress { status: "pulling manifest".into(), digest: None, total: None, completed: None }),
            None
        );
        // One layer half done.
        assert_eq!(agg.update(&layer("sha256:a", 50, 100)), Some(50.0));
        // A second layer appears; percent now spans both totals.
        assert_eq!(agg.update(&layer("sha256:b", 0, 100)), Some(25.0));
        // The first layer's line is resent larger — the latest figure replaces
        // it (not summed onto the prior one).
        assert_eq!(agg.update(&layer("sha256:a", 100, 100)), Some(50.0));
        assert_eq!(agg.update(&layer("sha256:b", 100, 100)), Some(100.0));
    }

    #[test]
    fn pull_aggregator_clamps_overshoot_to_100() {
        let mut agg = PullProgressAggregator::default();
        // A finished layer can report completed > total momentarily.
        assert_eq!(agg.update(&layer("sha256:a", 120, 100)), Some(100.0));
    }

    // Gated like the fn it tests — `api_key_value_usable` only exists when a
    // claude/codex adapter is compiled in.
    #[cfg(any(feature = "claude", feature = "codex"))]
    #[test]
    fn api_key_value_usable_requires_a_nonblank_value() {
        assert!(api_key_value_usable(Some("sk-abc".to_owned())));
        assert!(!api_key_value_usable(Some(String::new())));
        assert!(!api_key_value_usable(Some("   ".to_owned())));
        assert!(!api_key_value_usable(None));
    }

    /// An adapter that implements only the four required methods — the whole
    /// point of the trait's provided ones. What it gets for free is a contract
    /// third-party adapters depend on, so it is asserted rather than assumed.
    struct MinimalHarness;

    impl Harness for MinimalHarness {
        fn info(&self) -> Info {
            Info {
                id: "minimal".to_owned(),
                display_name: "Minimal".to_owned(),
                description: "implements the required surface and nothing else".to_owned(),
                install_hint: None,
            }
        }

        fn features(&self) -> Features {
            Features {
                models: vec![ModelChoice { value: "m1".to_owned(), label: "Model one".to_owned() }],
                ..Default::default()
            }
        }
        fn readiness(&self) -> Readiness {
            Readiness {
                harness_id: "minimal".to_owned(),
                ready: true,
                installed: true,
                version: None,
                auth_configured: true,
                error: None,
                details: serde_json::Value::Null,
            }
        }
        fn start(&self, _request: RunRequest, _on_event: RunCallback) -> Result<RunHandle, Error> {
            Ok(Box::new(NoopControl))
        }
        fn credential(&self) -> CredentialSpec {
            CredentialSpec {
                label: "none".to_owned(),
                keychain_service: "s".to_owned(),
                keychain_account: "a".to_owned(),
                required: false,
            }
        }
    }

    #[test]
    fn an_adapter_that_implements_only_the_required_surface_still_answers_the_rest() {
        let harness = MinimalHarness;
        // The picker asks every harness for models; the default answers from
        // the capabilities it already declared rather than making each adapter
        // write the same one-liner.
        assert_eq!(harness.list_models().unwrap(), harness.features().models);
        assert!(harness.model_management().is_none(), "no model management is the default");
        assert!(NoopControl.pid().is_none(), "a harness with no process reports no pid");
    }

    #[test]
    fn unsupported_optional_features_refuse_rather_than_pretend_to_succeed() {
        // Returning Ok(vec![]) here would read as "you have no models
        // installed" from a harness that cannot install any — the same
        // absent-versus-unsupported confusion the session store had.
        let harness = MinimalHarness;
        let cancel = std::sync::atomic::AtomicBool::new(false);

        for message in [
            harness.list_installed_models().map(|_| ()).unwrap_err().to_string(),
            harness.pull_model("m", &cancel, &mut |_| {}).unwrap_err().to_string(),
            harness.delete_model("m").unwrap_err().to_string(),
        ] {
            assert!(message.contains("does not support managing models"), "got {message}");
        }
        assert!(
            harness.login(Arc::new(|_| {})).unwrap_err().to_string().contains("interactive sign-in"),
            "and sign-in says which thing is unsupported"
        );
    }

    #[test]
    fn capabilities_default_to_supporting_nothing() {
        // The safe direction: a new field defaults to off, so an adapter that
        // has not heard of it does not silently claim it.
        let none = Features::default();
        assert!(!none.credential_required && !none.previews_edits && !none.custom_model);
        assert!(!none.effort && !none.max_turns && !none.login);
        assert!(!none.custom_instructions);
        assert!(none.models.is_empty());
    }

    #[test]
    fn reasoning_effort_keeps_the_tokens_a_cli_actually_accepts() {
        // These are sent verbatim as `model_reasoning_effort=<value>`; a
        // prettified variant name would be rejected by the CLI, not by us.
        assert_eq!(ReasoningEffort::Minimal.as_cli_value(), "minimal");
        assert_eq!(ReasoningEffort::Low.as_cli_value(), "low");
        assert_eq!(ReasoningEffort::Medium.as_cli_value(), "medium");
        assert_eq!(ReasoningEffort::High.as_cli_value(), "high");
    }

    #[test]
    fn an_install_hint_always_has_a_url_and_optionally_a_command() {
        // Not every agent has a one-liner that works on every platform, so the
        // command is optional while the home page never is.
        let bare = InstallHint::url("https://example.test/install");
        assert_eq!(bare.url, "https://example.test/install");
        assert!(bare.command.is_none());
        assert_eq!(bare.with_command("brew install thing").command.as_deref(), Some("brew install thing"));
    }

    fn login_events(program: &str, args: &[&str]) -> (Result<(), Error>, Vec<InstallEvent>) {
        let seen: Arc<Mutex<Vec<InstallEvent>>> = Arc::default();
        let sink = Arc::clone(&seen);
        let result = run_login_command(program, args, Arc::new(move |event| sink.lock().unwrap().push(event)));
        let events = seen.lock().unwrap().clone();
        (result, events)
    }

    #[test]
    fn a_sign_in_streams_the_cli_output_the_user_has_to_act_on() {
        // The whole reason this streams rather than waiting: an OAuth flow
        // prints a device code or a URL, and a user who never sees it cannot
        // finish signing in.
        let (result, events) = login_events("echo", &["visit https://example.test/device"]);
        assert!(result.is_ok(), "{result:?}");

        assert!(
            matches!(events.first(), Some(InstallEvent::Step { .. })),
            "something is said before the browser opens: {events:?}"
        );
        assert!(
            events.iter().any(|e| matches!(e, InstallEvent::Stdout { text } if text.contains("example.test/device"))),
            "the URL reaches the host: {events:?}"
        );
        assert!(
            matches!(events.last(), Some(InstallEvent::Done { ok: true, exit_code: Some(0) })),
            "and it ends exactly once, saying how: {events:?}"
        );
    }

    #[test]
    fn every_kind_of_process_output_reaches_the_user_during_sign_in() {
        // The paths that matter here cannot be provoked from a test — a wait or
        // read failure is an OS-level fault — so the mapping is checked with
        // values. Losing any of these leaves a stalled sign-in with nothing on
        // screen to explain it.
        let ev = |e: Event| login_event(&e);
        let run_id = || "r".to_owned();

        assert!(ev(Event::Started { run_id: run_id() }).is_none(), "the spawn is not news");

        let out = ev(Event::Stdout { run_id: run_id(), line: "visit https://x.test".into() });
        assert!(matches!(out, Some(InstallEvent::Stdout { text }) if text.contains("x.test")));

        // A device code arrives on stderr as often as stdout, and so does the
        // reason a login failed.
        let err = ev(Event::Stderr { run_id: run_id(), line: "code ABCD".into() });
        assert!(matches!(err, Some(InstallEvent::Stderr { text }) if text == "code ABCD"));

        // A stream that dies is reported, not swallowed.
        let broken = ev(Event::Error { run_id: run_id(), message: "stream read failed".into() });
        assert!(matches!(broken, Some(InstallEvent::Stderr { text }) if text.contains("read failed")));

        let ok = ev(Event::Exited { run_id: run_id(), exit_code: Some(0), cancelled: false });
        assert!(matches!(ok, Some(InstallEvent::Done { ok: true, exit_code: Some(0) })));
        let failed = ev(Event::Exited { run_id: run_id(), exit_code: Some(1), cancelled: false });
        assert!(matches!(failed, Some(InstallEvent::Done { ok: false, .. })), "only zero is success");
    }

    #[test]
    fn a_failed_sign_in_says_so_rather_than_completing_quietly() {
        // `false` exits non-zero without printing. Reporting ok here would
        // leave a host believing the user is signed in.
        let (result, events) = login_events("false", &[]);
        assert!(result.is_ok(), "the command ran; it is its exit code that failed");
        assert!(
            matches!(events.last(), Some(InstallEvent::Done { ok: false, .. })),
            "got {events:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_process_backed_run_reports_its_pid_and_whether_it_was_stopped() {
        // Both answers are load-bearing for an embedder. The pid is recorded so
        // a child orphaned by a hard crash can be reaped on the next launch;
        // `was_cancelled` is how a run the user stopped is told apart from one
        // that finished on its own. Forwarding either wrongly is invisible
        // until a stale agent is left running.
        let child = Command::new("sleep")
            .cwd(std::env::temp_dir())
            .run_id("pid-test")
            .args(["30"])
            .resolve_cli()
            .stream(|_| {})
        .expect("sleep should spawn");
        let run: RunHandle = Box::new(child);

        let pid = run.pid().expect("a live child has a pid");
        assert!(pid > 1, "a real OS pid, not a placeholder: {pid}");
        assert!(!run.was_cancelled(), "nothing has stopped it yet");

        run.cancel().expect("cancel");
        assert!(run.was_cancelled(), "a stopped run says so");
    }

    /// A no-op [`RunControl`] so the mock harness below can hand back a
    /// [`RunHandle`] without a real process behind it.
    struct NoopControl;
    impl RunControl for NoopControl {
        fn cancel(&self) -> Result<(), Error> {
            Ok(())
        }
        fn was_cancelled(&self) -> bool {
            false
        }
    }

    /// A minimal in-memory harness whose `run()` pushes a fixed event
    /// sequence straight to the callback, synchronously, then returns —
    /// dropping its only `RunCallback` clone. That's exactly the ownership
    /// shape `run` relies on, with no subprocess to spawn, so it
    /// pins down the contract: events are forwarded, and the receiver hangs
    /// up on its own once the run's callback ownership ends.
    struct MockHarness {
        events: Vec<RunEvent>,
    }
    impl Harness for MockHarness {
        fn info(&self) -> Info {
            unreachable!("not exercised by run")
        }
        fn readiness(&self) -> Readiness {
            unreachable!("not exercised by run")
        }
        fn start(
            &self,
            _request: RunRequest,
            on_event: RunCallback,
        ) -> Result<RunHandle, Error> {
            for event in &self.events {
                on_event(event.clone());
            }
            // `on_event` (the lone RunCallback clone, owning the channel's
            // Sender) drops as this returns → the receiver closes.
            Ok(Box::new(NoopControl))
        }
        fn credential(&self) -> CredentialSpec {
            unreachable!("not exercised by run")
        }
    }

    fn demo_request() -> RunRequest {
        RunRequest {
            run_id: "t".to_owned(),
            prompt: "hi".to_owned(),
            cwd: None,
            mode: RunMode::Ask,
            tools: ToolAccess::Default,
            tuning: RunTuning::default(),
            resume: None,
            attachments: Vec::new(),
        }
    }

    #[test]
    fn run_forwards_every_event_then_closes() {
        let harness = MockHarness {
            events: vec![
                RunEvent::Text {
                    run_id: "t".to_owned(),
                    delta: "hello".to_owned(),
                },
                RunEvent::Exited {
                    run_id: "t".to_owned(),
                    exit_code: Some(0),
                    cancelled: false,
                },
            ],
        };
        let (_handle, rx) = harness.run(demo_request()).expect("run ok");
        // Draining to completion *terminates* — proof the channel closed
        // without us dropping the handle.
        let collected: Vec<RunEvent> = rx.into_iter().collect();
        assert_eq!(
            collected,
            vec![
                RunEvent::Text {
                    run_id: "t".to_owned(),
                    delta: "hello".to_owned(),
                },
                RunEvent::Exited {
                    run_id: "t".to_owned(),
                    exit_code: Some(0),
                    cancelled: false,
                },
            ]
        );
    }

    #[test]
    fn run_receiver_closes_even_with_no_events() {
        let harness = MockHarness { events: Vec::new() };
        let (_handle, rx) = harness.run(demo_request()).expect("run ok");
        assert_eq!(rx.into_iter().count(), 0); // closes immediately, doesn't hang
    }

    /// One row per guarantee the crate makes: the flag that advertises it, and
    /// a request that an adapter without it must refuse. The contract on
    /// [`Harness`] says a guarantee ships as a flag *and* a row here, so adding
    /// the flag alone leaves this table short and the omission is visible.
    ///
    /// `Features` is exhaustively destructured on purpose. A new flag makes
    /// this fail to compile, which is the question being asked at the only
    /// moment anyone will think about it: is this a guarantee, and if so where
    /// is its row?
    fn guarantees() -> Vec<&'static str> {
        let Features {
            credential_required: _,
            previews_edits: _,
            models: _,
            custom_model: _,
            effort: _,
            max_turns: _,
            withheld_tools: _,
            login: _,
            custom_instructions: _,
        } = Features::default();
        vec!["withheld_tools"]
    }

    /// A guarantee is refused, never dropped. Held for every adapter the
    /// registry knows, including ones not yet written: if the flag says no, the
    /// run must fail rather than proceed unguarded.
    #[test]
    fn an_adapter_that_cannot_keep_a_guarantee_refuses_rather_than_dropping_it() {
        assert!(!guarantees().is_empty(), "the crate makes at least one guarantee");
        for name in guarantees() {
            assert_eq!(name, "withheld_tools", "a new guarantee needs its own arm below");
            for adapter in ["codex", "acp"] {
                for mode in [RunMode::Ask, RunMode::Edit] {
                    let refused = refuse_withheld_tools(adapter, ToolAccess::None, "cannot");
                    assert!(refused.is_err(), "{adapter} in {mode:?} must refuse, not drop");
                    assert!(
                        format!("{}", refused.unwrap_err()).contains(adapter),
                        "the error names the adapter, so a host knows which one"
                    );
                }
            }
        }
        // …and a run that asked for nothing special is never refused.
        assert!(refuse_withheld_tools("codex", ToolAccess::Default, "cannot").is_ok());
    }

    #[test]
    fn harness_error_preserves_typed_source_and_flattened_message() {
        use std::error::Error as _;

        // Categorize a real typed engine error as a Command failure.
        let err = Error::spawn(cli_stream::StreamError::PipeNotCaptured { stream: "stdout" });

        // Display still flattens the source into the message, so a consumer
        // that just `.to_string()`s at a boundary (a Tauri command) gets the
        // category prefix *and* the full underlying detail — unchanged from
        // when the variant held a String.
        let message = err.to_string();
        assert!(message.starts_with("failed to start the agent: "), "got {message:?}");
        assert!(message.contains("stdout pipe was not captured"), "got {message:?}");

        // And the real typed error is reachable via the source chain — the
        // whole point of carrying a source instead of a flattened string.
        let source = err.source().expect("Error::Spawn has a source");
        assert!(
            source.downcast_ref::<cli_stream::StreamError>().is_some(),
            "source should downcast back to the typed StreamError"
        );
    }
}
