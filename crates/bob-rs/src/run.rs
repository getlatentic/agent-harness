//! Build the bob CLI argv and spawn it via the shared streaming engine.
//!
//! Both `bob-api` (browser preview HTTP) and `src-tauri` (desktop IPC)
//! consume this. The generic subprocess engine (`spawn_streaming`, the
//! process-event type, the run handle) lives in `agent-harness`; this
//! module is the bob-specific layer on top — the chat-mode / approval
//! flags, `RunBobOptions`, and injecting bob's `BOBSHELL_API_KEY`.

use crate::check::semver_at_least;
use crate::error::BobError;
use crate::keychain::resolve_api_key;
use crate::BOB_MIN_NODE_VERSION;
use cli_stream::{resolve_program, spawn_streaming, ProcessEvent, ProcessHandle};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

// --- Wire shapes (bob-specific) -------------------------------------

/// Bob chat mode CLI flag. `--chat-mode <value>` accepts the snake_case
/// forms below.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BobChatMode {
    Plan,
    Code,
    Advanced,
    Ask,
}

impl BobChatMode {
    pub fn as_cli_value(self) -> &'static str {
        match self {
            Self::Plan => "plan",
            Self::Code => "code",
            Self::Advanced => "advanced",
            Self::Ask => "ask",
        }
    }
}

/// Bob's approval flow. `default` prompts the user via bob's UI; `yolo`
/// skips prompts. We only use `default` and `yolo` today (the legacy
/// `auto_edit` mode kept for back-compat with the existing Tauri command
/// surface).
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BobApprovalMode {
    Default,
    AutoEdit,
    Yolo,
}

impl BobApprovalMode {
    pub fn as_cli_value(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::AutoEdit => "auto_edit",
            Self::Yolo => "yolo",
        }
    }
}

/// Options for a single bob run. Built by both the axum endpoint (from
/// JSON body) and the Tauri command (from invoke args).
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RunBobOptions {
    pub prompt: String,
    #[serde(default = "default_chat_mode")]
    pub chat_mode: BobChatMode,
    #[serde(default = "default_approval_mode")]
    pub approval_mode: BobApprovalMode,
    #[serde(default = "default_max_coins")]
    pub max_coins: u32,
    /// Working directory the bob process runs in. Defaults to the
    /// caller's cwd. For workspace-scoped runs, pass the workspace path
    /// so bob's tool calls land inside that workspace.
    pub cwd: Option<PathBuf>,
    /// Override the bob executable path. Mainly for tests + when the
    /// caller has already resolved bob (e.g. Tauri's locator). Defaults
    /// to `bob` on PATH.
    #[serde(default)]
    pub bob_executable: Option<PathBuf>,
    /// Extra CLI args the caller appends verbatim after bob's own argv —
    /// the same host-controlled passthrough the other adapters expose, so
    /// a client can apply a flag uniformly across harnesses. Default empty.
    #[serde(default)]
    pub extra_args: Vec<String>,
    /// Session id to **resume** (`-r <id>`) instead of starting fresh — continue
    /// a prior conversation so bob supplies the history rather than the caller
    /// replaying a transcript. `None` → a new session. Default `None`.
    #[serde(default)]
    pub resume: Option<String>,
}

fn default_chat_mode() -> BobChatMode { BobChatMode::Ask }
fn default_approval_mode() -> BobApprovalMode { BobApprovalMode::Default }
fn default_max_coins() -> u32 { 30 }

// --- Spawn ----------------------------------------------------------

/// Spawn bob and stream output through `callback` until the child exits.
/// Returns a [`ProcessHandle`] immediately — the reader + wait threads
/// continue in the background.
///
/// `run_id` is opaque to bob-rs; the caller chooses the identifier and
/// uses it to correlate events with the handle.
pub fn spawn_bob<F>(
    opts: RunBobOptions,
    run_id: String,
    callback: F,
) -> Result<ProcessHandle, BobError>
where
    F: FnMut(ProcessEvent) + Send + Sync + Clone + 'static,
{
    let args = build_args(&opts);
    let api_key = resolve_api_key().map(|(value, _)| value).unwrap_or_default();
    // Resolve a bare `bob` to its absolute install path up front. The engine
    // then prepends that directory to the child PATH, so bob's
    // `#!/usr/bin/env node` re-exec picks the sibling node it was installed
    // under — not whichever (possibly ancient) node leads the inherited PATH.
    let program: PathBuf = resolve_program(
        opts.bob_executable.clone().unwrap_or_else(|| PathBuf::from("bob")),
    );
    // Fail with the real cause ("Node 24+ required, found v20") instead of
    // letting bob's re-exec die on a new-node-only flag as an opaque exit 9.
    ensure_node_compatible(&program)?;
    let cwd = opts.cwd.unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
    spawn_bob_raw(program, args, api_key, cwd, run_id, callback)
}

/// Which `node` will actually execute `program` (a Node-CLI script): the one
/// sitting next to it if there is one — the engine prepends the program's
/// own dir to the child PATH, so a sibling always wins — else the first
/// `node` on the augmented PATH (what the `#!/usr/bin/env node` shebang
/// would find). `None` when no node is reachable at all.
fn node_for_program(program: &Path) -> Option<PathBuf> {
    if let Some(dir) = program.parent().filter(|p| !p.as_os_str().is_empty()) {
        let sibling = dir.join("node");
        if sibling.is_file() {
            return Some(sibling);
        }
    }
    let resolved = resolve_program(PathBuf::from("node"));
    resolved.is_absolute().then_some(resolved)
}

/// Preflight: verify the node that will run bob meets
/// [`BOB_MIN_NODE_VERSION`]. bob re-launches itself with flags only newer
/// nodes accept (`--disable-sigusr1`), so an old runtime dies with an opaque
/// "exited with code 9" — this turns that into a typed, actionable error
/// *before* the spawn. Deliberately permissive at the edges: an unresolved
/// bob (bare name — the spawn's own "not found" error is clearer) or a node
/// that won't answer `--version` (broken probe shouldn't block a run that
/// might work) both pass through.
fn ensure_node_compatible(program: &Path) -> Result<(), BobError> {
    if program.parent().map_or(true, |p| p.as_os_str().is_empty()) {
        return Ok(());
    }
    let Some(node) = node_for_program(program) else {
        return Err(BobError::NodeIncompatible {
            minimum: BOB_MIN_NODE_VERSION.to_owned(),
            detail: "no `node` found on PATH".to_owned(),
        });
    };
    let Some(version) = probe_node_version(&node) else {
        return Ok(());
    };
    if semver_at_least(&version, BOB_MIN_NODE_VERSION) {
        Ok(())
    } else {
        Err(BobError::NodeIncompatible {
            minimum: BOB_MIN_NODE_VERSION.to_owned(),
            detail: format!("found {version} at {}", node.display()),
        })
    }
}

/// `<node> --version` → `v24.13.0`-style string, `None` on any failure.
fn probe_node_version(node: &Path) -> Option<String> {
    let output = Command::new(node)
        .arg("--version")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let version = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    (!version.is_empty()).then_some(version)
}

/// Lower-level spawn for callers that have already built the argv,
/// resolved the bob executable path, and loaded the API key themselves
/// (the Tauri runner, which carries its own locator + workspace-aware
/// argv builder). Thin bob-specific wrapper over
/// [`cli_stream::spawn_streaming`]: sets bob's `BOBSHELL_API_KEY` env
/// var, otherwise identical.
pub fn spawn_bob_raw<F>(
    program: PathBuf,
    args: Vec<String>,
    api_key: String,
    cwd: PathBuf,
    run_id: String,
    callback: F,
) -> Result<ProcessHandle, BobError>
where
    F: FnMut(ProcessEvent) + Send + Sync + Clone + 'static,
{
    let handle = spawn_streaming(
        program,
        args,
        vec![("BOBSHELL_API_KEY".to_owned(), api_key)],
        cwd,
        run_id,
        callback,
    )?; // cli_stream::StreamError → BobError::Stream
    Ok(handle)
}

/// Build the bob CLI argv. Mirrors the structure used by both the Vite
/// `bobRunPlugin` and the Tauri `build_bob_command`.
fn build_args(opts: &RunBobOptions) -> Vec<String> {
    let mut args = vec![
        opts.prompt.clone(),
        "--chat-mode".to_owned(),
        opts.chat_mode.as_cli_value().to_owned(),
        "--output-format".to_owned(),
        "stream-json".to_owned(),
        "--approval-mode".to_owned(),
        opts.approval_mode.as_cli_value().to_owned(),
        "--accept-license".to_owned(),
        "--max-coins".to_owned(),
        opts.max_coins.to_string(),
    ];
    // Continue a prior session instead of starting fresh (bob accepts the
    // session UUID, per `--resume {number|uuid|latest}`).
    if let Some(session_id) = &opts.resume {
        args.push("--resume".to_owned());
        args.push(session_id.clone());
    }
    // Host passthrough, appended verbatim after bob's own argv.
    args.extend(opts.extra_args.iter().cloned());
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(extra_args: Vec<String>) -> RunBobOptions {
        RunBobOptions {
            prompt: "hi".to_owned(),
            chat_mode: BobChatMode::Ask,
            approval_mode: BobApprovalMode::Default,
            max_coins: 30,
            cwd: None,
            bob_executable: None,
            extra_args,
            resume: None,
        }
    }

    #[test]
    fn build_args_appends_extra_args_after_bobs_own() {
        let args = build_args(&opts(vec!["--foo".to_owned(), "bar".to_owned()]));
        // bob's own argv stays intact (prompt positional first, format flag present)…
        assert_eq!(args.first().map(String::as_str), Some("hi"));
        assert!(args.contains(&"stream-json".to_owned()));
        // …and the host's flags are appended verbatim at the end.
        assert!(args.ends_with(&["--foo".to_owned(), "bar".to_owned()]));
    }

    #[test]
    fn build_args_with_no_extra_is_unchanged() {
        let args = build_args(&opts(Vec::new()));
        assert_eq!(args.last().map(String::as_str), Some("30"));
    }

    #[test]
    fn build_args_resume_adds_session_flag() {
        let mut o = opts(Vec::new());
        o.resume = Some("sess-7".to_owned());
        let args = build_args(&o);
        let i = args.iter().position(|a| a == "--resume").expect("--resume");
        assert_eq!(args.get(i + 1).map(String::as_str), Some("sess-7"));
        // The prompt positional is still first.
        assert_eq!(args.first().map(String::as_str), Some("hi"));
    }

    /// Lay out a fake `<dir>/{bob,node}` toolchain where `node --version`
    /// prints `version` — the sibling pairing `ensure_node_compatible`
    /// actually probes. Returns the fake bob's path.
    #[cfg(unix)]
    fn fake_toolchain(dir: &Path, version: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let exec = std::fs::Permissions::from_mode(0o755);
        let node = dir.join("node");
        std::fs::write(&node, format!("#!/bin/sh\necho {version}\n")).unwrap();
        std::fs::set_permissions(&node, exec.clone()).unwrap();
        let bob = dir.join("bob");
        std::fs::write(&bob, "#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&bob, exec).unwrap();
        bob
    }

    // The regression behind these: bob found under an nvm v24 dir was
    // re-exec'd on the v20 node leading the inherited PATH and died with
    // "bad option: --disable-sigusr1" → an opaque "exited with code 9".

    #[cfg(unix)]
    #[test]
    fn node_preflight_rejects_a_too_old_sibling_node() {
        let dir = tempfile::tempdir().unwrap();
        let bob = fake_toolchain(dir.path(), "v20.19.2");
        let err = ensure_node_compatible(&bob).expect_err("v20 must be rejected");
        match err {
            BobError::NodeIncompatible { minimum, detail } => {
                assert_eq!(minimum, BOB_MIN_NODE_VERSION);
                assert!(detail.contains("v20.19.2"), "detail names the bad version: {detail}");
            }
            other => panic!("expected NodeIncompatible, got {other:?}"),
        }
    }

    #[cfg(unix)]
    #[test]
    fn node_preflight_accepts_a_satisfying_sibling_node() {
        let dir = tempfile::tempdir().unwrap();
        // Exactly the minimum is inclusive…
        let bob = fake_toolchain(dir.path(), &format!("v{BOB_MIN_NODE_VERSION}"));
        ensure_node_compatible(&bob).expect("minimum version passes");
        // …and anything newer passes too.
        let bob = fake_toolchain(dir.path(), "v24.13.0");
        ensure_node_compatible(&bob).expect("newer version passes");
    }

    #[test]
    fn node_preflight_skips_an_unresolved_bare_program() {
        // A bare name means bob itself wasn't found — the spawn's own
        // "not found" error is the clearer signal, so the preflight defers.
        ensure_node_compatible(Path::new("bob")).expect("bare name passes through");
    }
}
