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
//! `parser` (`parse_claude_line`).

use std::path::PathBuf;

use serde_json::Value;

use crate::{
    normalize_process_event, probe_version, spawn_streaming, CredentialSpec, Harness,
    Capabilities, Error, Manifest, ModelChoice, Readiness,
    InstallCallback, InstallHint, RunCallback, RunHandle, RunMode, RunRequest, RunTuning,
};

mod parser;
// Shared with the Codex adapter so it can report its install-kind too. (The
// binary resolve + classify logic is harness-agnostic; it lives here for now.)
pub(crate) mod resolve;
pub use parser::parse_claude_line;

/// Registry id for the Claude Code harness.
pub const CLAUDE_HARNESS_ID: &str = "claude";

/// The program spawned when the host doesn't name one.
pub const DEFAULT_CLAUDE_COMMAND: &str = "claude";

/// Claude Code CLI as a [`Harness`].
#[derive(Debug, Clone)]
pub struct ClaudeHarness {
    command: String,
}

impl Default for ClaudeHarness {
    // Not derived: a derived `Default` would leave `command` empty and every
    // spawn would fail on a name nobody chose.
    fn default() -> Self {
        Self::new()
    }
}

/// What this adapter is, as opposed to what is layered onto it — the same
/// split as [`AcpHarnessConfig`](crate::AcpHarnessConfig) and
/// [`OpenHarnessConfig`](crate::OpenHarnessConfig).
#[derive(Clone, Debug)]
pub struct ClaudeHarnessConfig {
    /// Program to spawn. A bare name is resolved on PATH; a path is used as
    /// given. Everything else about the adapter — arguments, output parsing,
    /// auth probing — is unchanged, so a rename upstream, a fork, a wrapper
    /// script or a test stub costs a field here rather than a release.
    pub command: String,
}

impl Default for ClaudeHarnessConfig {
    fn default() -> Self {
        Self { command: DEFAULT_CLAUDE_COMMAND.to_owned() }
    }
}

impl ClaudeHarness {
    /// Drives `claude` from PATH.
    pub fn new() -> Self {
        Self::custom(ClaudeHarnessConfig::default())
    }

    /// Drives a binary the host names:
    ///
    /// ```no_run
    /// use harness::{ClaudeHarness, ClaudeHarnessConfig};
    /// let claude = ClaudeHarness::custom(ClaudeHarnessConfig {
    ///     command: "/opt/forks/claude-next".into(),
    /// });
    /// ```
    pub fn custom(config: ClaudeHarnessConfig) -> Self {
        Self { command: config.command }
    }
}

impl Harness for ClaudeHarness {
    fn manifest(&self) -> Manifest {
        Manifest {
            id: CLAUDE_HARNESS_ID.to_owned(),
            display_name: "Claude Code".to_owned(),
            description: "Anthropic's Claude Code agent CLI. Uses your existing Claude Code login."
                .to_owned(),
            install_hint: Some(
                InstallHint::url("https://code.claude.com/docs")
                    .with_command("curl -fsSL https://claude.ai/install.sh | bash"),
            ),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            // Claude Code owns its own login; it edits files directly, so
            // no previews and no stored credential. Everything it does not
            // support is left to `Default`.
            //
            // The aliases `claude --help` documents. This list is the whole
            // picker when models.dev is unreachable, and `allows_custom_model`
            // stays off, so anything missing here is unreachable — not merely
            // unlisted.
            models: vec![
                ModelChoice { value: "sonnet".to_owned(), label: "Sonnet (latest)".to_owned() },
                ModelChoice { value: "opus".to_owned(), label: "Opus (latest)".to_owned() },
                ModelChoice { value: "fable".to_owned(), label: "Fable (latest)".to_owned() },
                ModelChoice { value: "haiku".to_owned(), label: "Haiku (latest)".to_owned() },
            ],
            supports_max_turns: true,
            supports_login: true,
            ..Default::default()
        }
    }

    fn list_models(&self) -> Result<Vec<ModelChoice>, Error> {
        // Keep the curated aliases first (`sonnet`/`opus` track "latest" and don't
        // churn), then append models.dev's current `anthropic` lineup (exact ids)
        // when the `models-dev` feature is on. Offline / feature-off → just aliases.
        let mut models = self.capabilities().models;
        models.extend(crate::models_dev::provider_models("anthropic"));
        Ok(models)
    }

    fn readiness(&self) -> Readiness {
        let Some(version) = probe_version(&self.command) else {
            return Readiness {
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
        let signed_in = probe_claude_signed_in(&self.command)
            || crate::harness::api_key_value_usable(std::env::var("ANTHROPIC_API_KEY").ok());
        Readiness {
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
            details: resolved_details(&self.command),
        }
    }

    fn start(&self, request: RunRequest, on_event: RunCallback) -> Result<RunHandle, Error> {
        // `attachments` ignored: Claude Code is a text CLI (no image input here).
        let RunRequest { run_id, prompt, cwd, mode, tuning, resume, attachments: _ } = request;
        let args = build_claude_args(prompt, mode, &tuning, resume.as_deref());
        let cwd = cwd.unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

        // No env injected — Claude Code uses its own auth. PATH
        // augmentation inside `spawn_streaming` ensures `node` is
        // found for a Finder-launched .app.
        let program = tuning.binary_path.clone().unwrap_or_else(|| PathBuf::from(&self.command));
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
        .map_err(Error::spawn)?;
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

    fn login(&self, on_event: InstallCallback) -> Result<(), Error> {
        // `claude auth login` runs the CLI's OAuth flow (opens the
        // browser); streamed + blocked-until-exit by the shared helper.
        crate::run_login_command(&self.command, &["auth", "login"], on_event)
    }
}

/// Probe Claude Code's auth: `claude auth status` prints JSON with a
/// `loggedIn` boolean (exit 0 when signed in). Returns true only when
/// signed in; defensively falls back to the exit code if the JSON is
/// unexpected. Lets [`ClaudeHarness::readiness`] distinguish installed
/// from signed-in.
fn probe_claude_signed_in(command: &str) -> bool {
    let Ok(output) = crate::hidden_command(command)
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
/// `Readiness.details` without a struct change. `details.resolved_path`
/// is absent when the binary can't be located despite a successful
/// `--version` (e.g. a PATH entry the resolver can't read) — the host renders
/// version + status regardless.
fn resolved_details(command: &str) -> Value {
    let path = crate::augmented_node_path();
    let Some(resolved) = resolve::resolve_on_path(command, &path) else {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ReasoningEffort;

    #[test]
    fn claude_info_and_credential() {
        let h = ClaudeHarness::new();
        assert_eq!(h.manifest().id, CLAUDE_HARNESS_ID);
        let hint = h.manifest().install_hint.expect("Claude Code is a CLI the user installs");
        assert!(hint.command.is_some_and(|c| c.contains("claude.ai/install.sh")));
        // Claude manages its own auth — Compose doesn't require a key.
        assert!(!h.credential().required);
    }

    #[test]
    fn a_renamed_binary_is_what_gets_probed() {
        // The point of the field: if the CLI is renamed upstream, or a user
        // keeps a fork or wrapper under another name, that costs a call here
        // rather than a release. A name nothing can resolve must read as "not
        // installed" — proving readiness consults the configured command and
        // not a baked-in "claude".
        let renamed = ClaudeHarness::custom(ClaudeHarnessConfig {
            command: "definitely-not-a-real-binary-xyz".into(),
        });
        let readiness = renamed.readiness();
        assert!(!readiness.installed, "an unresolvable command cannot report installed");

        // And the default still targets the real one.
        assert_eq!(ClaudeHarness::new().command, DEFAULT_CLAUDE_COMMAND);
        assert_eq!(ClaudeHarness::default().command, DEFAULT_CLAUDE_COMMAND);
    }

    /// A throwaway CLI that behaves however the test needs. The probes take the
    /// command as an argument, so no PATH juggling is involved — the same trick
    /// the MCP client uses to test a protocol against a real process.
    #[cfg(unix)]
    fn fake_cli(tag: &str, script: &str) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("hl-claude-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cli");
        std::fs::write(&path, format!("#!/bin/sh\n{script}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[cfg(unix)]
    #[test]
    fn a_signed_out_cli_is_believed_over_the_exit_code() {
        // `auth status` answering `{"loggedIn": false}` while exiting 0 is the
        // case that matters: the fallback below would read that as signed in,
        // and the user would be told to try a run that cannot work.
        let out = fake_cli("signedout", r#"echo '{"loggedIn": false}'"#);
        assert!(!probe_claude_signed_in(out.to_str().unwrap()));

        let inn = fake_cli("signedin", r#"echo '{"loggedIn": true}'"#);
        assert!(probe_claude_signed_in(inn.to_str().unwrap()));
    }

    #[cfg(unix)]
    #[test]
    fn a_cli_that_does_not_answer_in_json_falls_back_to_how_it_exited() {
        // Older builds print prose. Exit 0 with something to say is the best
        // available evidence; silence or a failure is not.
        let prose = fake_cli("prose", "echo 'Logged in as someone@example.test'");
        assert!(probe_claude_signed_in(prose.to_str().unwrap()));

        let silent = fake_cli("silent", "exit 0");
        assert!(!probe_claude_signed_in(silent.to_str().unwrap()), "exit 0 with nothing said proves nothing");

        let failed = fake_cli("authfail", "echo 'not logged in'; exit 1");
        assert!(!probe_claude_signed_in(failed.to_str().unwrap()));

        assert!(!probe_claude_signed_in("definitely-not-a-real-binary-xyz"), "an absent CLI is not signed in");
    }


    #[test]
    fn every_documented_alias_is_offered() {
        // These are the aliases `claude --help` names. The list matters more
        // than it looks: models.dev supplies the exact ids, but when it is
        // unreachable — offline, or a first launch with a cold cache — this
        // vec IS the picker. Combined with `allows_custom_model: false`, an
        // alias missing here cannot be selected or typed. `fable` was absent
        // and therefore unreachable in exactly that state.
        let caps = ClaudeHarness::new().capabilities();
        let offered: Vec<&str> = caps.models.iter().map(|m| m.value.as_str()).collect();
        for alias in ["sonnet", "opus", "fable", "haiku"] {
            assert!(offered.contains(&alias), "`--model {alias}` is documented, got {offered:?}");
        }
        assert!(
            !caps.allows_custom_model,
            "if free-text entry is ever allowed, an omission above stops being unreachable \
             and this test can relax"
        );
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
