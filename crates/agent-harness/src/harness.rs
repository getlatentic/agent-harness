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
use cli_stream::{spawn_streaming, InstallEvent, ProcessEvent, ProcessHandle};

// --- Streaming callbacks --------------------------------------------

/// Callback a harness invokes for each run event. `Arc<dyn Fn>` is
/// `Clone + Send + Sync`, so it can be handed to the multiple reader
/// threads a process-backed harness uses without the trait method
/// needing to be generic.
pub type RunCallback = Arc<dyn Fn(RunEvent) + Send + Sync>;

/// Callback a harness invokes for each install event.
pub type InstallCallback = Arc<dyn Fn(InstallEvent) + Send + Sync>;

// --- Errors ---------------------------------------------------------

/// A boxed, type-erased error source. The [`HarnessError`] variants carry one
/// of these instead of `#[from]`-ing a single concrete type, because each
/// *category* can be produced by more than one underlying error: a `Spawn`
/// failure is a [`cli_stream::StreamError`] for the claude/codex adapters but a
/// `bob_rs::BobError` for bob. The real error stays reachable through
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
/// walk `.source()` or `downcast_ref::<cli_stream::StreamError>()` /
/// `::<bob_rs::BobError>()`. The `Display` still flattens the source into the
/// message (`"failed to start the agent: <source>"`), so a consumer that just
/// stringifies at a boundary (e.g. a Tauri command's `.to_string()`) gets the
/// same full message as before. `#[non_exhaustive]` so adding a variant later
/// isn't a breaking change.
///
/// ```
/// use harness::{HarnessError, StreamError};
/// use std::error::Error;
///
/// // Box any typed source under a category constructor:
/// let err = HarnessError::spawn(StreamError::PipeNotCaptured { stream: "stdout" });
///
/// // Stringifying at a boundary flattens the source into the message
/// // (so a Tauri command's `.to_string()` keeps its full text)…
/// assert!(err.to_string().starts_with("failed to start the agent: "));
///
/// // …while the real typed cause stays reachable for a consumer that wants
/// // to branch on it rather than parse a string.
/// let source = err.source().expect("Spawn carries a source");
/// assert!(source.downcast_ref::<StreamError>().is_some());
/// ```
///
/// [`source`]: std::error::Error::source
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HarnessError {
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

impl HarnessError {
    /// Categorize a source error as a [`Spawn`](HarnessError::Spawn) failure.
    /// Accepts anything boxable — a typed `StreamError`/`BobError`, or a
    /// `String`/`&str` for adapters with nothing typed to carry.
    pub fn spawn(source: impl Into<BoxError>) -> Self {
        Self::Spawn(source.into())
    }
    /// Categorize a source error as an [`Install`](HarnessError::Install) failure.
    pub fn install(source: impl Into<BoxError>) -> Self {
        Self::Install(source.into())
    }
    /// Categorize a source error as a [`Login`](HarnessError::Login) failure.
    pub fn login(source: impl Into<BoxError>) -> Self {
        Self::Login(source.into())
    }
    /// Categorize a source error as a [`Cancel`](HarnessError::Cancel) failure.
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
    fn cancel(&self) -> Result<(), HarnessError>;
    /// Whether [`cancel`](RunControl::cancel) was called.
    fn was_cancelled(&self) -> bool;
}

/// Boxed [`RunControl`] returned by [`Harness::run`].
pub type RunHandle = Box<dyn RunControl>;

// The engine's run handle is the canonical process-backed `RunControl`.
// Both the trait and the handle live in this crate, so this impl is here
// (orphan rule) rather than in any adapter crate.
impl RunControl for ProcessHandle {
    fn cancel(&self) -> Result<(), HarnessError> {
        ProcessHandle::cancel(self).map_err(HarnessError::cancel)
    }
    fn was_cancelled(&self) -> bool {
        ProcessHandle::was_cancelled(self)
    }
}

// --- Neutral request / metadata shapes ------------------------------

/// What the user wants the harness to do with the prompt. Mirrors
/// the Ask / Edit split the comment bubble already exposes; adapters
/// map it onto their own mode vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    /// Answer / discuss. No file edits expected.
    Ask,
    /// Propose edits to the workspace.
    Edit,
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
    pub effort: Option<ReasoningEffort>,
    /// Cap on agentic turns (Claude: `--max-turns`).
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
#[derive(Debug, Clone)]
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
    pub cwd: Option<PathBuf>,
    pub mode: RunMode,
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
pub struct HarnessReadiness {
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
pub struct HarnessModel {
    pub value: String,
    pub label: String,
}

/// What a harness supports, so every consumer (the picker, the options
/// panel, the credential preflight, the chat availability gate) adapts
/// to it *declaratively* instead of branching on the harness id. A new
/// adapter that, say, needs a stored key just sets `credential_required:
/// true` here — no `id == "bob"` checks to hunt down.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessCapabilities {
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
    /// curated list (rely on `allows_custom_model`).
    pub models: Vec<HarnessModel>,
    /// Whether a free-text model id is accepted beyond `models` (codex,
    /// whose model names change frequently). Drives a text field vs a
    /// fixed dropdown in the picker.
    pub allows_custom_model: bool,
    /// Honors [`RunTuning::effort`] (codex reasoning effort).
    pub supports_effort: bool,
    /// Honors [`RunTuning::max_turns`] (claude turn cap).
    pub supports_max_turns: bool,
    /// Supports an interactive [`Harness::login`] flow (the CLI's own
    /// OAuth, e.g. `claude auth login` / `codex login`). Drives the
    /// picker's "Sign in" affordance when installed-but-not-signed-in.
    /// `false` for harnesses Compose authenticates itself (bob).
    pub supports_login: bool,
}

/// Static metadata for the harness picker.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HarnessInfo {
    pub id: String,
    pub display_name: String,
    pub description: String,
    /// True if the harness needs a one-time [`Harness::install`].
    pub requires_install: bool,
    /// Declarative capabilities — what the harness supports, so the UI
    /// and run-gating never special-case its id.
    pub capabilities: HarnessCapabilities,
}

// --- The trait ------------------------------------------------------

/// A pluggable agent backend. Implementors are cheap to construct
/// (they hold config, not connections) so a registry can hand out
/// fresh boxes on demand.
pub trait Harness: Send + Sync {
    /// Static metadata for the UI.
    fn info(&self) -> HarnessInfo;

    /// Probe availability / version / auth. May shell out; callers
    /// should treat it as blocking and run it off the UI thread.
    fn readiness(&self) -> HarnessReadiness;

    /// Stream a one-time install. Harnesses that need no install
    /// (e.g. a hosted-API adapter) return `Ok(())` immediately.
    fn install(&self, on_event: InstallCallback) -> Result<(), HarnessError>;

    /// Start a run, streaming events through `on_event`. Returns a
    /// handle immediately; work continues on background threads.
    fn run(&self, request: RunRequest, on_event: RunCallback) -> Result<RunHandle, HarnessError>;

    /// The credential this harness needs.
    fn credential(&self) -> CredentialSpec;

    /// Enumerate the models this harness can run, *live*. The default returns
    /// the static list declared in [`HarnessInfo`]
    /// (`info().capabilities.models`), so existing adapters need no change.
    ///
    /// Override it when the model set is discovered at runtime rather than
    /// known at compile time — a hosted-API adapter querying the provider's
    /// `/v1/models`, an Ollama adapter hitting `/api/tags`. A harness with no
    /// model-selection concept (bob runs whatever it's configured with)
    /// returns an empty list, and the host hides the picker — capability by
    /// the *absence* of models, not a separate flag. May shell out / hit the
    /// network; treat it as blocking and run it off the UI thread.
    fn list_models(&self) -> Result<Vec<HarnessModel>, HarnessError> {
        Ok(self.info().capabilities.models)
    }

    /// Trigger the harness's own interactive sign-in (its CLI's OAuth),
    /// streaming progress as [`InstallEvent`]s — the same subprocess
    /// stream shape as [`install`](Harness::install). The flow opens the
    /// user's browser; this blocks until the login process exits, then
    /// `Done { ok }` reports success. Default: unsupported — harnesses
    /// that Compose authenticates itself (bob, via its API key) keep it.
    fn login(&self, _on_event: InstallCallback) -> Result<(), HarnessError> {
        Err(HarnessError::login(
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
    /// Prefer [`run`](Harness::run) directly when you need push semantics —
    /// e.g. forwarding straight onto a Tauri `Channel` or an SSE sink from
    /// inside the callback — where an intermediate channel is just an extra
    /// hop. This is a provided method (not overridable surface): adapters
    /// implement only `run`, and every harness — built-in or third-party —
    /// gets `run_channel` for free.
    ///
    /// ```no_run
    /// use harness::{Claude, Harness, RunEvent, RunMode, RunRequest, RunTuning};
    ///
    /// # fn main() -> Result<(), harness::HarnessError> {
    /// let (_handle, rx) = Claude::new().run_channel(RunRequest {
    ///     run_id: "demo".into(),
    ///     prompt: "Explain Markdown headings in one sentence.".into(),
    ///     cwd: None,
    ///     mode: RunMode::Ask,
    ///     tuning: RunTuning::default(),
    ///     resume: None,
    ///     attachments: Vec::new(),
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
    fn run_channel(
        &self,
        request: RunRequest,
    ) -> Result<(RunHandle, mpsc::Receiver<RunEvent>), HarnessError> {
        let (tx, rx) = mpsc::channel();
        let handle = self.run(
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
/// [`spawn_streaming`] (PATH augmentation + reader threads, so a packaged
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
) -> Result<(), HarnessError> {
    (*on_event)(InstallEvent::Step {
        text: "Opening your browser to sign in…".to_owned(),
    });
    let done = Arc::new((Mutex::new(false), Condvar::new()));
    let done_cb = Arc::clone(&done);
    let events_cb = Arc::clone(&on_event);
    // Bound, not `_`, so the handle outlives the wait (dropping it could
    // signal the child); by the time we return, the process has exited.
    let _handle = spawn_streaming(
        PathBuf::from(program),
        args.iter().map(|s| (*s).to_owned()).collect(),
        Vec::new(),
        std::env::current_dir().unwrap_or_default(),
        format!("login-{program}"),
        move |event| match event {
            ProcessEvent::Started { .. } => {}
            ProcessEvent::Stdout { line, .. } => {
                (*events_cb)(InstallEvent::Stdout { text: line });
            }
            ProcessEvent::Stderr { line, .. } => {
                (*events_cb)(InstallEvent::Stderr { text: line });
            }
            ProcessEvent::Error { message, .. } => {
                (*events_cb)(InstallEvent::Stderr { text: message });
            }
            ProcessEvent::Exited { exit_code, .. } => {
                (*events_cb)(InstallEvent::Done {
                    exit_code,
                    ok: exit_code == Some(0),
                });
                let (lock, cvar) = &*done_cb;
                // Recover from a poisoned lock instead of panicking on a
                // reader thread: the guarded value is a plain bool, never in a
                // half-updated state worth bailing on.
                *lock.lock().unwrap_or_else(|p| p.into_inner()) = true;
                cvar.notify_all();
            }
            // `ProcessEvent` is #[non_exhaustive]; ignore any future variant.
            _ => {}
        },
    )
    .map_err(HarnessError::login)?;
    let (lock, cvar) = &*done;
    let mut finished = lock.lock().unwrap_or_else(|p| p.into_inner());
    while !*finished {
        finished = cvar.wait(finished).unwrap_or_else(|p| p.into_inner());
    }
    Ok(())
}

/// Whether an API-key value an adapter pulled from the environment counts as
/// authenticated — i.e. present and non-blank. Adapters OR this into their
/// [`Harness::readiness`] so a key in the env (headless / CI / container)
/// reports authenticated, not only the CLI's own interactive OAuth login —
/// which can't complete where there's no browser. Pure (the env read stays at
/// the call site) so it's unit-tested directly.
///
/// Only the claude/codex adapters OR this into readiness — bob reports auth via
/// `bob-rs`'s own keychain source — so it's gated to those features. Without
/// them (`--no-default-features`) it would be dead code, hence the `cfg`.
#[cfg(any(feature = "claude", feature = "codex"))]
pub(crate) fn api_key_value_usable(value: Option<String>) -> bool {
    matches!(value, Some(v) if !v.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// A no-op [`RunControl`] so the mock harness below can hand back a
    /// [`RunHandle`] without a real process behind it.
    struct NoopControl;
    impl RunControl for NoopControl {
        fn cancel(&self) -> Result<(), HarnessError> {
            Ok(())
        }
        fn was_cancelled(&self) -> bool {
            false
        }
    }

    /// A minimal in-memory harness whose `run()` pushes a fixed event
    /// sequence straight to the callback, synchronously, then returns —
    /// dropping its only `RunCallback` clone. That's exactly the ownership
    /// shape `run_channel` relies on, with no subprocess to spawn, so it
    /// pins down the contract: events are forwarded, and the receiver hangs
    /// up on its own once the run's callback ownership ends.
    struct MockHarness {
        events: Vec<RunEvent>,
    }
    impl Harness for MockHarness {
        fn info(&self) -> HarnessInfo {
            unreachable!("not exercised by run_channel")
        }
        fn readiness(&self) -> HarnessReadiness {
            unreachable!("not exercised by run_channel")
        }
        fn install(&self, _on_event: InstallCallback) -> Result<(), HarnessError> {
            Ok(())
        }
        fn run(
            &self,
            _request: RunRequest,
            on_event: RunCallback,
        ) -> Result<RunHandle, HarnessError> {
            for event in &self.events {
                on_event(event.clone());
            }
            // `on_event` (the lone RunCallback clone, owning the channel's
            // Sender) drops as this returns → the receiver closes.
            Ok(Box::new(NoopControl))
        }
        fn credential(&self) -> CredentialSpec {
            unreachable!("not exercised by run_channel")
        }
    }

    fn demo_request() -> RunRequest {
        RunRequest {
            run_id: "t".to_owned(),
            prompt: "hi".to_owned(),
            cwd: None,
            mode: RunMode::Ask,
            tuning: RunTuning::default(),
            resume: None,
            attachments: Vec::new(),
        }
    }

    #[test]
    fn run_channel_forwards_every_event_then_closes() {
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
        let (_handle, rx) = harness.run_channel(demo_request()).expect("run_channel ok");
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
    fn run_channel_receiver_closes_even_with_no_events() {
        let harness = MockHarness { events: Vec::new() };
        let (_handle, rx) = harness.run_channel(demo_request()).expect("run_channel ok");
        assert_eq!(rx.into_iter().count(), 0); // closes immediately, doesn't hang
    }

    #[test]
    fn harness_error_preserves_typed_source_and_flattened_message() {
        use std::error::Error;

        // Categorize a real typed engine error as a Spawn failure.
        let err = HarnessError::spawn(cli_stream::StreamError::PipeNotCaptured { stream: "stdout" });

        // Display still flattens the source into the message, so a consumer
        // that just `.to_string()`s at a boundary (a Tauri command) gets the
        // category prefix *and* the full underlying detail — unchanged from
        // when the variant held a String.
        let message = err.to_string();
        assert!(message.starts_with("failed to start the agent: "), "got {message:?}");
        assert!(message.contains("stdout pipe was not captured"), "got {message:?}");

        // And the real typed error is reachable via the source chain — the
        // whole point of carrying a source instead of a flattened string.
        let source = err.source().expect("HarnessError::Spawn has a source");
        assert!(
            source.downcast_ref::<cli_stream::StreamError>().is_some(),
            "source should downcast back to the typed StreamError"
        );
    }
}
