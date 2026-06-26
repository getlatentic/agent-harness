//! Claude Code (`claude`) as a [`Harness`].
//!
//! Same process-spawn shape as the bob adapter — a different binary,
//! flags, and stdout parser. We invoke `claude -p` in headless
//! streaming mode and parse its NDJSON into the shared normalized
//! [`crate::RunEvent`] stream, so the front-end treats Claude exactly
//! like any other harness.
//!
//! Auth: Claude Code manages its own credentials (its OAuth login or
//! its own `ANTHROPIC_API_KEY` in the environment), so Compose does
//! not store or inject a key — `credential().required` is `false`.
//!
//! The stdout wire format and its decode into [`crate::RunEvent`]s live in
//! [`parser`] ([`parse_claude_line`]).

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

use crate::{
    normalize_process_event, spawn_streaming, CredentialSpec, Harness, HarnessCapabilities,
    HarnessError, HarnessInfo, HarnessModel, HarnessReadiness, InstallCallback, InstallEvent,
    RunCallback, RunHandle, RunMode, RunRequest, RunTuning,
};

mod parser;
// Shared with the Codex adapter so it can report its install-kind too. (The
// binary resolve + classify logic is harness-agnostic; it lives here for now.)
pub(crate) mod resolve;
pub use parser::parse_claude_line;

/// Registry id for the Claude Code harness.
pub const CLAUDE_HARNESS_ID: &str = "claude";

/// Claude Code CLI as a [`Harness`].
#[derive(Debug, Default, Clone)]
pub struct ClaudeHarness;

impl ClaudeHarness {
    pub fn new() -> Self {
        Self
    }
}

impl Harness for ClaudeHarness {
    fn info(&self) -> HarnessInfo {
        HarnessInfo {
            id: CLAUDE_HARNESS_ID.to_owned(),
            display_name: "Claude Code".to_owned(),
            description: "Anthropic's Claude Code agent CLI. Uses your existing Claude Code login."
                .to_owned(),
            requires_install: true,
            capabilities: HarnessCapabilities {
                // Claude Code owns its own login; it edits files
                // directly (no previews). Curated model aliases (no
                // free-text) + a turn cap; no reasoning-effort flag.
                credential_required: false,
                previews_edits: false,
                models: vec![
                    HarnessModel { value: "sonnet".to_owned(), label: "Sonnet (latest)".to_owned() },
                    HarnessModel { value: "opus".to_owned(), label: "Opus (latest)".to_owned() },
                    HarnessModel { value: "haiku".to_owned(), label: "Haiku (latest)".to_owned() },
                ],
                allows_custom_model: false,
                supports_effort: false,
                supports_max_turns: true,
                supports_login: true,
                supports_custom_instructions: false,
            },
        }
    }

    fn list_models(&self) -> Result<Vec<HarnessModel>, HarnessError> {
        // Keep the curated aliases first (`sonnet`/`opus` track "latest" and don't
        // churn), then append models.dev's current `anthropic` lineup (exact ids)
        // when the `models-dev` feature is on. Offline / feature-off → just aliases.
        let mut models = self.info().capabilities.models;
        models.extend(crate::models_dev::provider_models("anthropic"));
        Ok(models)
    }

    fn readiness(&self) -> HarnessReadiness {
        let Some(version) = probe_version("claude") else {
            return HarnessReadiness {
                harness_id: CLAUDE_HARNESS_ID.to_owned(),
                ready: false,
                installed: false,
                version: None,
                auth_configured: false,
                error: Some("Claude Code (`claude`) is not installed or not on PATH.".to_owned()),
                details: Value::Null,
            };
        };
        // Installed — now distinguish signed-in from not, so the picker
        // can offer "Sign in" instead of failing the first run. Either the
        // CLI's own OAuth login OR an `ANTHROPIC_API_KEY` in the environment
        // counts: the env key is how you run headless (a container / CI),
        // where `claude auth login` can't open a browser. `claude auth status`
        // only sees the OAuth state, so we OR in the env key ourselves.
        let signed_in = probe_claude_signed_in()
            || crate::harness::api_key_value_usable(std::env::var("ANTHROPIC_API_KEY").ok());
        HarnessReadiness {
            harness_id: CLAUDE_HARNESS_ID.to_owned(),
            ready: signed_in,
            installed: true,
            version: Some(version),
            auth_configured: signed_in,
            error: if signed_in {
                None
            } else {
                Some(
                    "Claude Code is installed but not signed in. Click Sign in to connect your Anthropic account, or set ANTHROPIC_API_KEY."
                        .to_owned(),
                )
            },
            details: resolved_details(),
        }
    }

    fn install(&self, on_event: InstallCallback) -> Result<(), HarnessError> {
        // Claude Code ships as a native, Developer-ID-signed, self-updating
        // binary (`~/.local/bin/claude`); the npm `@anthropic-ai/claude-code`
        // package is frozen at 1.0.x and must never be (re)installed. If a
        // `claude` is already on PATH (e.g. an old npm one), migrate it to the
        // native build via `claude install`; otherwise bootstrap the native
        // installer. Blocking (matches the `install` contract); output is
        // forwarded as install events.
        let path = crate::augmented_node_path();
        let output = if resolve::resolve_on_path("claude", &path).is_some() {
            (*on_event)(InstallEvent::Step {
                text: "Migrating Claude Code to the native installer…".to_owned(),
            });
            Command::new("claude")
                .arg("install")
                .env("PATH", &path)
                .output()
                .map_err(|e| HarnessError::install(format!("failed to run claude install: {e}")))?
        } else {
            (*on_event)(InstallEvent::Step {
                text: "Downloading the Claude Code installer…".to_owned(),
            });
            let script = download_install_script(&path)?;
            (*on_event)(InstallEvent::Step {
                text: "Running the Claude Code installer…".to_owned(),
            });
            // Run the downloaded script with `bash <file>` — NOT a `curl | bash`
            // pipe (the single most EDR-flagged pattern); download-then-execute
            // gives endpoint security a file to inspect first.
            Command::new("bash")
                .arg(script.path())
                .env("PATH", &path)
                .output()
                .map_err(|e| HarnessError::install(format!("failed to run installer: {e}")))?
        };
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            (*on_event)(InstallEvent::Stdout {
                text: line.to_owned(),
            });
        }
        for line in String::from_utf8_lossy(&output.stderr).lines() {
            (*on_event)(InstallEvent::Stderr {
                text: line.to_owned(),
            });
        }
        (*on_event)(InstallEvent::Done {
            exit_code: output.status.code(),
            ok: output.status.success(),
        });
        Ok(())
    }

    fn run(&self, request: RunRequest, on_event: RunCallback) -> Result<RunHandle, HarnessError> {
        // `attachments` ignored: Claude Code is a text CLI (no image input here).
        let RunRequest { run_id, prompt, cwd, mode, tuning, resume, attachments: _ } = request;
        let args = build_claude_args(prompt, mode, &tuning, resume.as_deref());
        let cwd = cwd.unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

        // No env injected — Claude Code uses its own auth. PATH
        // augmentation inside `spawn_streaming` ensures `node` is
        // found for a Finder-launched .app.
        let program = tuning.binary_path.clone().unwrap_or_else(|| PathBuf::from("claude"));
        let handle = spawn_streaming(
            program,
            args,
            Vec::new(),
            cwd,
            run_id,
            move |event| {
                for normalized in normalize_process_event(event, parse_claude_line) {
                    (*on_event)(normalized);
                }
            },
        )
        .map_err(HarnessError::spawn)?;
        Ok(Box::new(handle))
    }

    fn credential(&self) -> CredentialSpec {
        CredentialSpec {
            label: "Claude Code login (managed by the claude CLI)".to_owned(),
            keychain_service: "anthropic".to_owned(),
            keychain_account: "ANTHROPIC_API_KEY".to_owned(),
            // Claude Code authenticates itself; Compose need not store
            // a key for it.
            required: false,
        }
    }

    fn login(&self, on_event: InstallCallback) -> Result<(), HarnessError> {
        // `claude auth login` runs the CLI's OAuth flow (opens the
        // browser); streamed + blocked-until-exit by the shared helper.
        crate::run_login_command("claude", &["auth", "login"], on_event)
    }
}

/// Probe Claude Code's auth: `claude auth status` prints JSON with a
/// `loggedIn` boolean (exit 0 when signed in). Returns true only when
/// signed in; defensively falls back to the exit code if the JSON is
/// unexpected. Lets [`ClaudeHarness::readiness`] distinguish installed
/// from signed-in.
fn probe_claude_signed_in() -> bool {
    let Ok(output) = Command::new("claude")
        .args(["auth", "status"])
        .env("PATH", crate::augmented_node_path())
        .output()
    else {
        return false;
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(stdout.trim()) {
        if let Some(logged_in) = map.get("loggedIn").and_then(Value::as_bool) {
            return logged_in;
        }
    }
    // Fallback: exit 0 with non-empty output ≈ signed in.
    output.status.success() && !stdout.trim().is_empty()
}

/// Build readiness `details` carrying where `claude` resolves on the augmented
/// PATH and how it was installed (native / npm-global / homebrew / bundled /
/// unknown). Attached as a `serde_json::Value` object so it rides the existing
/// `HarnessReadiness.details` without a struct change. `details.resolved_path`
/// is absent when the binary can't be located despite a successful
/// `--version` (e.g. a PATH entry the resolver can't read) — the host renders
/// version + status regardless.
fn resolved_details() -> Value {
    let path = crate::augmented_node_path();
    let Some(resolved) = resolve::resolve_on_path("claude", &path) else {
        return Value::Null;
    };
    let mut details = serde_json::Map::new();
    details.insert(
        "resolved_path".to_owned(),
        Value::String(resolved.to_string_lossy().into_owned()),
    );
    if let Ok(home) = std::env::var("HOME") {
        let kind = resolve::classify(&resolved, std::path::Path::new(&home), None);
        details.insert("install_kind".to_owned(), Value::String(kind.as_str().to_owned()));
    }
    Value::Object(details)
}

/// Build the argv for a `claude -p` headless run. Kept pure (no
/// spawn) so the flag mapping is unit-tested. `tuning.model` →
/// `--model`, `tuning.max_turns` → `--max-turns`; Claude Code has no
/// reasoning-effort `-p` flag, so `tuning.effort` is intentionally
/// ignored here.
fn build_claude_args(
    prompt: String,
    mode: RunMode,
    tuning: &RunTuning,
    resume: Option<&str>,
) -> Vec<String> {
    let mut args = vec![
        "-p".to_owned(),
        prompt,
        "--output-format".to_owned(),
        "stream-json".to_owned(),
        "--verbose".to_owned(),
        "--include-partial-messages".to_owned(),
    ];
    // Continue a prior session instead of replaying history in the prompt.
    if let Some(session_id) = resume {
        args.push("--resume".to_owned());
        args.push(session_id.to_owned());
    }
    if let Some(model) = tuning.model.as_deref().map(str::trim).filter(|m| !m.is_empty()) {
        args.push("--model".to_owned());
        args.push(model.to_owned());
    }
    if let Some(max_turns) = tuning.max_turns {
        args.push("--max-turns".to_owned());
        args.push(max_turns.to_string());
    }
    // Conservative *default* permission mode (auto-approve edits; Bash etc.
    // stay gated), emitted only when the caller hasn't set `--permission-mode`
    // through `extra_args`. So there is a sensible default, but a host fully
    // controls the mode — `bypassPermissions` for headless, `auto`, … — by
    // passing its own, with no adapter edit and no duplicate flag. In Ask mode
    // the CLI stays read-only by default.
    if matches!(mode, RunMode::Edit) && !extra_args_sets(&tuning.extra_args, "--permission-mode") {
        args.push("--permission-mode".to_owned());
        args.push("acceptEdits".to_owned());
    }
    // Host passthrough/overrides, appended verbatim after the adapter's own.
    args.extend(tuning.extra_args.iter().cloned());
    args
}

/// Whether the host's `extra_args` already sets `flag` (so the adapter should
/// not also emit its own default for it). Matches `--flag` and `--flag=value`.
fn extra_args_sets(extra_args: &[String], flag: &str) -> bool {
    let with_eq = format!("{flag}=");
    extra_args.iter().any(|a| a == flag || a.starts_with(&with_eq))
}

/// The official Claude Code bootstrap script. A `GET` 302-redirects to
/// `downloads.claude.ai/.../bootstrap.sh`, which installs the native binary.
const CLAUDE_INSTALL_URL: &str = "https://claude.ai/install.sh";

/// Download the native-installer bootstrap to a temp file via `curl -fsSL`
/// (TLS-verified, follows the 302), returning a guard that deletes the file on
/// drop. Downloading to a file — rather than piping `curl … | bash` — lets EDR
/// inspect the script before it runs. Errors if curl fails or the script is
/// empty (a redirect/transport failure that left no body), so we never hand
/// `bash` an empty file and report a spurious success.
fn download_install_script(path: &str) -> Result<TempScript, HarnessError> {
    let script = TempScript::new()
        .map_err(|e| HarnessError::install(format!("could not create a temp file: {e}")))?;
    let output = Command::new("curl")
        .args(["-fsSL", CLAUDE_INSTALL_URL, "-o"])
        .arg(script.path())
        .env("PATH", path)
        .output()
        .map_err(|e| HarnessError::install(format!("failed to run curl: {e}")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(HarnessError::install(format!(
            "downloading {CLAUDE_INSTALL_URL} failed: {}",
            stderr.trim()
        )));
    }
    let empty = std::fs::metadata(script.path()).map(|m| m.len() == 0).unwrap_or(true);
    if empty {
        return Err(HarnessError::install(format!(
            "{CLAUDE_INSTALL_URL} returned an empty installer script"
        )));
    }
    Ok(script)
}

/// A uniquely-named temp file deleted when dropped — holds the downloaded
/// installer for the duration of the `bash <file>` run. A tiny RAII guard so
/// the `claude` feature needn't pull a tempfile dependency.
struct TempScript {
    path: PathBuf,
}

impl TempScript {
    fn new() -> std::io::Result<Self> {
        // Process id + a monotonic counter keep concurrent installs from
        // colliding without needing a randomness dependency.
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "claude-install-{}-{}.sh",
            std::process::id(),
            n
        ));
        // Create it now so the file exists with restrictive defaults before curl
        // writes to it.
        std::fs::File::create(&path)?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempScript {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Run `<program> --version`, returning the trimmed stdout on
/// success. Used by readiness to detect the CLI on PATH.
fn probe_version(program: &str) -> Option<String> {
    // Augment PATH so a packaged `.app` (minimal launchd PATH) can find a
    // CLI installed via nvm / Homebrew / official installer — otherwise an
    // installed CLI is mis-reported as "not installed".
    let output = Command::new(program)
        .arg("--version")
        .env("PATH", crate::augmented_node_path())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ReasoningEffort;

    #[test]
    fn claude_info_and_credential() {
        let h = ClaudeHarness::new();
        assert_eq!(h.info().id, CLAUDE_HARNESS_ID);
        assert!(h.info().requires_install);
        // Claude manages its own auth — Compose doesn't require a key.
        assert!(!h.credential().required);
    }

    /// Value of the arg immediately following `flag`, if present.
    fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .map(String::as_str)
    }

    #[test]
    fn claude_args_default_omit_model_and_turn_cap() {
        let args = build_claude_args("hi".to_owned(), RunMode::Ask, &RunTuning::default(), None);
        // Prompt is the positional right after `-p`.
        assert_eq!(args[0], "-p");
        assert_eq!(args[1], "hi");
        assert!(!args.iter().any(|a| a == "--model"));
        assert!(!args.iter().any(|a| a == "--max-turns"));
        assert!(!args.iter().any(|a| a == "--permission-mode"));
    }

    #[test]
    fn claude_resume_adds_session_flag() {
        let args =
            build_claude_args("hi".to_owned(), RunMode::Ask, &RunTuning::default(), Some("sess-123"));
        assert_eq!(flag_value(&args, "--resume"), Some("sess-123"));
        // The prompt + headless stream flags are untouched.
        assert_eq!(args[0], "-p");
        assert_eq!(args[1], "hi");
    }

    #[test]
    fn claude_args_carry_model_and_max_turns_and_ignore_effort() {
        let tuning = RunTuning {
            model: Some("opus".to_owned()),
            effort: Some(ReasoningEffort::High),
            max_turns: Some(5),
            ..RunTuning::default()
        };
        let args = build_claude_args("hi".to_owned(), RunMode::Ask, &tuning, None);
        assert_eq!(flag_value(&args, "--model"), Some("opus"));
        assert_eq!(flag_value(&args, "--max-turns"), Some("5"));
        // Claude Code has no reasoning-effort `-p` flag — it must not leak.
        assert!(!args.iter().any(|a| a.contains("reasoning_effort")));
    }

    #[test]
    fn claude_blank_model_is_treated_as_unset() {
        let tuning = RunTuning { model: Some("   ".to_owned()), ..RunTuning::default() };
        let args = build_claude_args("hi".to_owned(), RunMode::Ask, &tuning, None);
        assert!(!args.iter().any(|a| a == "--model"));
    }

    #[test]
    fn claude_edit_mode_defaults_to_accept_edits() {
        // Conservative built-in default; a host overrides via extra_args.
        let args = build_claude_args("hi".to_owned(), RunMode::Edit, &RunTuning::default(), None);
        assert_eq!(flag_value(&args, "--permission-mode"), Some("acceptEdits"));
    }

    #[test]
    fn host_extra_args_are_appended_verbatim() {
        // A host adds flags the adapter doesn't manage — appended as given.
        let tuning = RunTuning {
            extra_args: vec!["--add-dir".to_owned(), "/extra".to_owned()],
            ..RunTuning::default()
        };
        let args = build_claude_args("hi".to_owned(), RunMode::Ask, &tuning, None);
        assert!(args.ends_with(&["--add-dir".to_owned(), "/extra".to_owned()]));
    }

    #[test]
    fn host_permission_mode_replaces_the_default_cleanly() {
        // When the host sets --permission-mode, the adapter does NOT also emit
        // its acceptEdits default — the host fully owns the flag, no duplicate.
        let tuning = RunTuning {
            extra_args: vec!["--permission-mode".to_owned(), "bypassPermissions".to_owned()],
            ..RunTuning::default()
        };
        let args = build_claude_args("hi".to_owned(), RunMode::Edit, &tuning, None);
        let modes: Vec<usize> = args
            .iter()
            .enumerate()
            .filter(|(_, a)| a.as_str() == "--permission-mode")
            .map(|(i, _)| i)
            .collect();
        assert_eq!(modes.len(), 1, "exactly one --permission-mode (the host's)");
        assert_eq!(args[modes[0] + 1], "bypassPermissions");
        assert!(!args.iter().any(|a| a == "acceptEdits"));
    }

    #[test]
    fn extra_args_sets_matches_flag_and_flag_eq_value() {
        assert!(extra_args_sets(&["--permission-mode".to_owned()], "--permission-mode"));
        assert!(extra_args_sets(&["--permission-mode=auto".to_owned()], "--permission-mode"));
        assert!(!extra_args_sets(&["--add-dir".to_owned()], "--permission-mode"));
    }
}
