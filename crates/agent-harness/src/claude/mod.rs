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
    normalize_process_event, probe_version, Command, ResolveCli, CredentialSpec, Harness,
    Features, Error, Info, ModelChoice, Readiness,
    InstallCallback, InstallHint, RunCallback, RunHandle, RunMode, RunRequest, RunTuning,
    ToolAccess, ToolServer,
};

mod control;
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
    /// Host tools served to the CLI over its control protocol — see
    /// [`Self::with_tool_server`].
    tool_servers: Vec<ToolServer>,
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
        Self { command: config.command, tool_servers: Vec::new() }
    }

    /// Offer the agent a [`ToolServer`] of functions this program implements.
    ///
    /// Claude Code sees it as an MCP server named after the server, with each
    /// tool as `mcp__<server>__<tool>`, and calls back into this process over
    /// its control protocol — no server process, no port. Attaching one opens
    /// that channel for every run that may call tools; a run asking for
    /// [`ToolAccess::None`] is unchanged, since it may call nothing.
    pub fn with_tool_server(mut self, server: ToolServer) -> Self {
        self.tool_servers.push(server);
        self
    }
}

impl Harness for ClaudeHarness {
    fn info(&self) -> Info {
        Info {
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

    fn features(&self) -> Features {
        Features {
            // Claude Code owns its own login; it edits files directly, so
            // no previews and no stored credential. Everything it does not
            // support is left to `Default`.
            //
            // The aliases `claude --help` documents. This list is the whole
            // picker when models.dev is unreachable, and `custom_model`
            // stays off, so anything missing here is unreachable — not merely
            // unlisted.
            models: vec![
                ModelChoice { value: "sonnet".to_owned(), label: "Sonnet (latest)".to_owned() },
                ModelChoice { value: "opus".to_owned(), label: "Opus (latest)".to_owned() },
                ModelChoice { value: "fable".to_owned(), label: "Fable (latest)".to_owned() },
                ModelChoice { value: "haiku".to_owned(), label: "Haiku (latest)".to_owned() },
            ],
            max_turns: true,
            withheld_tools: true,
            login: true,
            custom_instructions: true,
            host_tools: true,
            ..Default::default()
        }
    }

    fn list_models(&self) -> Result<Vec<ModelChoice>, Error> {
        // Keep the curated aliases first (`sonnet`/`opus` track "latest" and don't
        // churn), then append models.dev's current `anthropic` lineup (exact ids)
        // when the `models-dev` feature is on. Offline / feature-off → just aliases.
        let mut models = self.features().models;
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
        let RunRequest { run_id, prompt, cwd, mode, tools, tuning, resume, attachments: _ } = request;
        // Host tools ride the control channel, and so does the prompt, as a
        // message — under stream-json input the CLI ignores the positional. A
        // run that may call nothing has no use for the channel and keeps the
        // one-way spawn.
        let over_control_channel = !self.tool_servers.is_empty() && tools != ToolAccess::None;
        let positional = if over_control_channel { None } else { Some(prompt.clone()) };
        let args = build_claude_args(positional, mode, tools, &tuning, resume.as_deref());
        let cwd = cwd.unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

        // No env injected — Claude Code uses its own auth. PATH
        // augmentation inside `spawn_streaming` ensures `node` is
        // found for a Finder-launched .app.
        let program = tuning.binary_path.clone().unwrap_or_else(|| PathBuf::from(&self.command));
        let command = Command::new(program).cwd(cwd).run_id(run_id.clone()).args(args).resolve_cli();
        if over_control_channel {
            let handle = control::start(command, &run_id, prompt, self.tool_servers.clone(), on_event)?;
            return Ok(Box::new(handle));
        }
        let handle = command
            .stream(move |event| {
                for normalized in normalize_process_event(event, parse_claude_line) {
                    (*on_event)(normalized);
                }
            })
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
        .env("PATH", crate::augmented_path())
        .output()
    else {
        return false;
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(stdout.trim())
        && let Some(logged_in) = map.get("loggedIn").and_then(Value::as_bool)
    {
        return logged_in;
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
    let path = crate::augmented_path();
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
    prompt: Option<String>,
    mode: RunMode,
    tools: ToolAccess,
    tuning: &RunTuning,
    resume: Option<&str>,
) -> Vec<String> {
    // `None` means the prompt arrives on stdin as a stream-json message, which
    // is how the control channel is opened — see [`control`]. The CLI ignores
    // a positional prompt in that mode, so none is sent.
    let mut args = vec!["-p".to_owned()];
    match prompt {
        Some(prompt) => args.push(prompt),
        None => args.extend(["--input-format".to_owned(), "stream-json".to_owned()]),
    }
    args.extend(
        ["--output-format", "stream-json", "--verbose", "--include-partial-messages"].map(str::to_owned),
    );
    // Continue a prior session instead of replaying history in the prompt.
    if let Some(session_id) = resume {
        args.push("--resume".to_owned());
        args.push(session_id.to_owned());
    }
    if let Some(model) = crate::harness::nonblank(tuning.model.as_deref()) {
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
    // `ToolAccess::None` means the prompt is the whole job.
    //
    // Measured against the CLI, planting a unique string in a file and asking
    // for it back — each arm in a fresh directory, because a stale session
    // repeating an earlier answer looks exactly like a successful read:
    //
    //   no flag                          read it   2/2
    //   --allowedTools ""                read it   2/2   (auto-approve list,
    //                                                     not an availability
    //                                                     gate; `-p` permits
    //                                                     reads anyway)
    //   --disallowedTools <10 names>     read it   1/1   (10 turns — it looked
    //                                                     for another way in)
    //   --disallowedTools <19 names>     read it   2/3
    //   --disallowedTools "*"            blocked   4/4   (1 turn, no attempt)
    //
    // A name list is a speed bump, not a gate: given turns, the model routes
    // around whatever is not on it, and every arm above was a guess at a list
    // that a new built-in would invalidate anyway. The wildcard needs no list
    // and still answers an ordinary question (`pong`, one turn, no error).
    //
    // What it guarantees is that no tool *runs*, not that no tool-shaped text
    // comes back. Under the wildcard the model wrote the call out as prose —
    // `<invoke name="Bash">…` — and then invented its output, reporting a
    // directory as empty that held the planted file. Nothing was read, which
    // is the guarantee; but a caller that scrapes an answer for tool syntax,
    // or trusts a transcript quoted inside one, will find both there.
    if tools == ToolAccess::None && !extra_args_sets(&tuning.extra_args, "--disallowedTools") {
        args.push("--disallowedTools".to_owned());
        args.push("*".to_owned());
    }
    // A run that may call nothing has no use for connected MCP servers, and
    // pays for them: their definitions are assembled for every invocation, so
    // the saving scales with what is mounted and is nothing on a machine with
    // none. Measured twice, two prompts, against the same two root servers —
    // total input tokens per run:
    //
    //   8,638 -> 7,369   a judging prompt   (-1,269)
    //   8,249 -> 6,984   a trivial prompt   (-1,265)
    //
    // The saving replicates; its size is a property of that server set rather
    // than of the flag, which is why the conditions are written beside it.
    //
    // Skipped when the host manages MCP itself, since it may be mounting a
    // server for a *later* turn or another run in the same configuration.
    if tools == ToolAccess::None
        && !extra_args_sets(&tuning.extra_args, "--strict-mcp-config")
        && !extra_args_sets(&tuning.extra_args, "--mcp-config")
    {
        args.push("--strict-mcp-config".to_owned());
    }
    // The host's custom instructions, appended to the agent's own system prompt.
    // `--append-system-prompt` is additive, which is what `extra_instructions`
    // has always been: replacing Claude's prompt is not on offer here, and
    // `--system-prompt` is the flag that would do it.
    //
    // This matters where a caller cannot steer the model any other way. A DSPy
    // ReActV2 loop over this adapter renders its tools as text, and its
    // `forced_tool` steering is dropped on that path — the provider is never
    // told to submit — so the only remaining lever on the final turn is prose in
    // the system prompt.
    if let Some(extra) = crate::harness::nonblank(tuning.extra_instructions.as_deref())
        && !extra_args_sets(&tuning.extra_args, "--append-system-prompt")
    {
        args.push("--append-system-prompt".to_owned());
        args.push(extra.to_owned());
    }
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
    #[cfg(unix)]
    use crate::events::RunEvent;
    use crate::ReasoningEffort;

    /// A stand-in `claude` answering the three ways the adapter invokes it:
    /// `--version`, `auth status` (JSON with `loggedIn`), and a `-p` run that
    /// records the argv it was handed.
    #[cfg(unix)]
    fn fake_claude(tag: &str, signed_in: bool, emits: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let dir = crate::test_support::fixture_dir(tag);
        let argv = dir.join("argv");
        let cli = dir.join("claude");
        crate::test_support::install_script(
            &cli,
            &format!(
                "case \"$1\" in\n\
                 --version) echo '1.2.3 (Claude Code)'; exit 0 ;;\n\
                 auth) echo '{{\"loggedIn\":{signed_in}}}'; exit 0 ;;\n\
                 -p) : > '{argv}'; for a in \"$@\"; do printf '%s\\n' \"$a\" >> '{argv}'; done\n\
                 {emits}\n\
                 exit 0 ;;\n\
                 esac\n\
                 exit 1\n",
                argv = argv.display(),
            ),
        );
        (cli, argv)
    }

    #[cfg(unix)]
    fn drive(cli: &std::path::Path, request: RunRequest) -> Vec<RunEvent> {
        use std::sync::{Arc, Mutex};
        let seen: Arc<Mutex<Vec<RunEvent>>> = Arc::default();
        let sink = Arc::clone(&seen);
        let harness = ClaudeHarness::custom(ClaudeHarnessConfig {
            command: cli.display().to_string(),
        });
        let handle = harness
            .start(request, Arc::new(move |event| sink.lock().unwrap().push(event)))
            .expect("the stand-in should spawn");

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let done = seen
                .lock()
                .unwrap()
                .iter()
                .any(|event| matches!(event, RunEvent::Exited { .. }));
            if done {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "the run never exited");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let _ = handle.cancel();

        seen.lock().unwrap().clone()
    }

    #[cfg(unix)]
    #[test]
    fn a_run_reaches_the_cli_as_stream_json_and_comes_back_as_events() {
        // The argv tests elsewhere check `build_claude_args` in isolation; this
        // is the only one proving the process receives that argv, and that its
        // NDJSON comes back through the parser as normalized events.
        let delta = r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"hi there"}}}"#;
        let (cli, argv) = fake_claude("run", true, &format!("printf '%s\\n' '{delta}'"));

        let events = drive(
            &cli,
            RunRequest {
                run_id: "r1".to_owned(),
                prompt: "greet me".to_owned(),
                cwd: Some(std::env::temp_dir()),
                tuning: RunTuning { model: Some("opus".to_owned()), ..RunTuning::default() },
                ..RunRequest::default()
            },
        );

        let passed: Vec<String> =
            std::fs::read_to_string(&argv).unwrap().lines().map(str::to_owned).collect();
        assert_eq!(passed.first().map(String::as_str), Some("-p"));
        assert_eq!(
            passed.get(1).map(String::as_str),
            Some("greet me"),
            "the prompt follows -p: {passed:?}",
        );
        assert!(passed.windows(2).any(|w| w[0] == "--output-format" && w[1] == "stream-json"));
        assert!(passed.iter().any(|a| a == "--include-partial-messages"), "{passed:?}");
        assert!(passed.windows(2).any(|w| w[0] == "--model" && w[1] == "opus"));

        assert!(
            events.iter().any(|e| matches!(e, RunEvent::Text { delta, .. } if delta == "hi there")),
            "the CLI's delta should arrive as text: {events:?}",
        );
        assert!(events.iter().any(|e| matches!(
            e,
            RunEvent::Exited { exit_code: Some(0), cancelled: false, .. }
        )));
    }

    #[cfg(unix)]
    #[test]
    fn readiness_reads_logged_in_from_the_auth_probe() {
        // `auth status` answers JSON, so this covers the parse as well as the
        // spawn. Signed-out still reads as installed: the UI offers "Sign in"
        // only once it knows the binary is there.
        let (yes, _) = fake_claude("in", true, ":");
        let ready =
            ClaudeHarness::custom(ClaudeHarnessConfig { command: yes.display().to_string() })
                .readiness();
        assert!(ready.installed && ready.ready && ready.auth_configured);
        assert_eq!(ready.version.as_deref(), Some("1.2.3 (Claude Code)"));

        let (no, _) = fake_claude("out", false, ":");
        let ready =
            ClaudeHarness::custom(ClaudeHarnessConfig { command: no.display().to_string() })
                .readiness();
        assert!(ready.installed, "the binary is present either way");
        assert!(!ready.ready && !ready.auth_configured);
        assert!(ready.error.is_some(), "a signed-out CLI must say what to do");
    }

    #[test]
    fn claude_info_and_credential() {
        let h = ClaudeHarness::new();
        assert_eq!(h.info().id, CLAUDE_HARNESS_ID);
        let hint = h.info().install_hint.expect("Claude Code is a CLI the user installs");
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

    #[cfg(unix)]
    use crate::test_support::fake_cli;

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
        // vec IS the picker. Combined with `custom_model: false`, an
        // alias missing here cannot be selected or typed. `fable` was absent
        // and therefore unreachable in exactly that state.
        let caps = ClaudeHarness::new().features();
        let offered: Vec<&str> = caps.models.iter().map(|m| m.value.as_str()).collect();
        for alias in ["sonnet", "opus", "fable", "haiku"] {
            assert!(offered.contains(&alias), "`--model {alias}` is documented, got {offered:?}");
        }
        assert!(
            !caps.custom_model,
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

    /// `--allowedTools ""` was the obvious spelling and does nothing — measured
    /// against the CLI, a run with it set still read a file and reported the
    /// contents. It is the auto-approve list, not an availability gate.
    #[test]
    fn withheld_tools_are_denied_by_name_not_by_an_empty_allowlist() {
        let args =
            build_claude_args(Some("hi".into()), RunMode::Ask, ToolAccess::None, &RunTuning::default(), None);
        assert!(!args.iter().any(|a| a == "--allowedTools"), "that flag does not gate anything");
        let i = args.iter().position(|a| a == "--disallowedTools").expect("the flag");
        assert_eq!(args[i + 1], "*", "a wildcard: a name list leaked 2 of 3 live runs");
        assert!(!args.iter().any(|a| a == "--permission-mode"), "nothing to permit");
    }

    /// The two questions are separate: `Edit` says the run may change things,
    /// `ToolAccess::None` says it has nothing to change them with. Modelling
    /// them as one enum made that combination unsayable.
    #[test]
    fn write_permission_and_tool_access_are_independent() {
        let args =
            build_claude_args(Some("hi".into()), RunMode::Edit, ToolAccess::None, &RunTuning::default(), None);
        let i = args.iter().position(|a| a == "--disallowedTools").expect("no tools offered");
        assert_eq!(args[i + 1], "*");
        let j = args.iter().position(|a| a == "--permission-mode").expect("still an edit run");
        assert_eq!(args[j + 1], "acceptEdits");
    }

    /// Nothing may be called, so nothing needs connecting. The cost is real —
    /// MCP definitions are assembled per invocation — and it cannot change what
    /// a run can do, because the tools were already refused.
    #[test]
    fn withheld_tools_do_not_connect_mcp_servers_either() {
        let args =
            build_claude_args(Some("hi".into()), RunMode::Ask, ToolAccess::None, &RunTuning::default(), None);
        assert!(args.iter().any(|a| a == "--strict-mcp-config"), "{args:?}");
        // …and an ordinary run is untouched: a host mounting servers keeps them.
        let open =
            build_claude_args(Some("hi".into()), RunMode::Ask, ToolAccess::Default, &RunTuning::default(), None);
        assert!(!open.iter().any(|a| a == "--strict-mcp-config"), "{open:?}");
    }

    /// A host that manages MCP itself is left alone: it may be mounting a
    /// server for another run sharing this configuration.
    #[test]
    fn a_host_that_configures_mcp_itself_is_not_overridden() {
        let tuning = RunTuning {
            extra_args: vec!["--mcp-config".into(), "servers.json".into()],
            ..Default::default()
        };
        let args = build_claude_args(Some("hi".into()), RunMode::Ask, ToolAccess::None, &tuning, None);
        assert!(!args.iter().any(|a| a == "--strict-mcp-config"), "{args:?}");
    }

    #[test]
    fn a_host_that_names_its_own_tools_keeps_them() {
        let tuning = RunTuning {
            extra_args: vec!["--disallowedTools".into(), "Bash".into()],
            ..Default::default()
        };
        let args = build_claude_args(Some("hi".into()), RunMode::Ask, ToolAccess::None, &tuning, None);
        let flags = args.iter().filter(|a| *a == "--disallowedTools").count();
        assert_eq!(flags, 1, "exactly one, and it is the host's");
        assert!(args.contains(&"Bash".to_owned()));
    }

    #[test]
    fn claude_args_default_omit_model_and_turn_cap() {
        let args = build_claude_args(Some("hi".to_owned()), RunMode::Ask, ToolAccess::Default, &RunTuning::default(), None);
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
            build_claude_args(Some("hi".to_owned()), RunMode::Ask, ToolAccess::Default, &RunTuning::default(), Some("sess-123"));
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
        let args = build_claude_args(Some("hi".to_owned()), RunMode::Ask, ToolAccess::Default, &tuning, None);
        assert_eq!(flag_value(&args, "--model"), Some("opus"));
        assert_eq!(flag_value(&args, "--max-turns"), Some("5"));
        // Claude Code has no reasoning-effort `-p` flag — it must not leak.
        assert!(!args.iter().any(|a| a.contains("reasoning_effort")));
    }

    #[test]
    fn claude_blank_model_is_treated_as_unset() {
        let tuning = RunTuning { model: Some("   ".to_owned()), ..RunTuning::default() };
        let args = build_claude_args(Some("hi".to_owned()), RunMode::Ask, ToolAccess::Default, &tuning, None);
        assert!(!args.iter().any(|a| a == "--model"));
    }

    #[test]
    fn claude_edit_mode_defaults_to_accept_edits() {
        // Conservative built-in default; a host overrides via extra_args.
        let args = build_claude_args(Some("hi".to_owned()), RunMode::Edit, ToolAccess::Default, &RunTuning::default(), None);
        assert_eq!(flag_value(&args, "--permission-mode"), Some("acceptEdits"));
    }

    #[test]
    fn a_prompt_on_stdin_opens_the_control_channel_and_sends_no_positional() {
        let args = build_claude_args(None, RunMode::Ask, ToolAccess::Default, &RunTuning::default(), None);
        assert_eq!(&args[..3], &["-p", "--input-format", "stream-json"], "no positional, input on stdin");
        // The ordinary spawn is untouched: prompt positional, no input format.
        let args = build_claude_args(Some("hi".into()), RunMode::Ask, ToolAccess::Default, &RunTuning::default(), None);
        assert_eq!(&args[..2], &["-p", "hi"]);
        assert!(!args.iter().any(|a| a == "--input-format"));
    }

    /// A stand-in for `claude` speaking just enough of the control protocol to
    /// call one host tool: it checks each message the adapter must send, in the
    /// order the CLI needs them, and exits with a distinct code at the first
    /// one that is wrong — so a failure names the step. It then waits for EOF,
    /// which is how the adapter is supposed to end the conversation.
    ///
    /// Keys are matched one at a time because `serde_json` writes them sorted.
    #[cfg(unix)]
    fn fake_control_claude() -> std::path::PathBuf {
        crate::test_support::fake_cli(
            "claude-control",
            r#"read -r init
case "$init" in *'"subtype":"initialize"'*) ;; *) exit 3;; esac
case "$init" in *'"sdkMcpServers":["shop"]'*) ;; *) exit 3;; esac
printf '%s\n' '{"type":"control_response","response":{"subtype":"success","request_id":"run-1-initialize","response":{}}}'
read -r user
case "$user" in *'"type":"user"'*) ;; *) exit 4;; esac
case "$user" in *'"content":"count the stock"'*) ;; *) exit 4;; esac
printf '%s\n' '{"type":"system","subtype":"init","session_id":"s1","tools":[],"mcp_servers":[{"name":"shop","status":"connected"}]}'
printf '%s\n' '{"type":"control_request","request_id":"c1","request":{"subtype":"mcp_message","server_name":"shop","message":{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"stock","arguments":{"sku":"A"}}}}}'
read -r reply
case "$reply" in *'"request_id":"c1"'*) ;; *) exit 5;; esac
case "$reply" in *'"mcp_response"'*) ;; *) exit 5;; esac
text=$(printf '%s' "$reply" | sed -n 's/.*"text":"\([^"]*\)".*/\1/p')
printf '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"%s"}}}\n' "$text"
printf '%s\n' '{"type":"result","subtype":"success","is_error":false}'
cat >/dev/null
"#,
        )
    }

    #[cfg(unix)]
    #[test]
    fn a_host_tool_is_served_over_the_control_channel_and_stdin_is_closed_after_the_result() {
        use std::sync::mpsc;
        use std::time::{Duration, Instant};

        let cli = fake_control_claude();
        let claude = ClaudeHarness::custom(ClaudeHarnessConfig { command: cli.to_string_lossy().into_owned() })
            .with_tool_server(ToolServer::new("shop").with_tool(crate::FnTool::new(
                "stock",
                "units on hand",
                serde_json::json!({"type": "object"}),
                |args| Ok(format!("{} units", args["sku"].as_str().unwrap_or("").len())),
            )));
        let (tx, rx) = mpsc::channel();
        let _handle = claude
            .start(
                RunRequest {
                    run_id: "run-1".to_owned(),
                    prompt: "count the stock".to_owned(),
                    cwd: Some(std::env::temp_dir()),
                    ..Default::default()
                },
                std::sync::Arc::new(move |event| {
                    let _ = tx.send(event);
                }),
            )
            .expect("spawns");

        let deadline = Instant::now() + Duration::from_secs(10);
        let mut events = Vec::new();
        loop {
            let left = deadline
                .checked_duration_since(Instant::now())
                .unwrap_or_else(|| panic!("no Exited: stdin was never closed after the result — {events:?}"));
            match rx.recv_timeout(left) {
                Ok(event) => {
                    let done = matches!(event, crate::RunEvent::Exited { .. });
                    events.push(event);
                    if done {
                        break;
                    }
                }
                Err(err) => panic!("stream ended without Exited ({err}): {events:?}"),
            }
        }

        let said: String = events
            .iter()
            .filter_map(|e| match e {
                crate::RunEvent::Text { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(said, "1 units", "the tool's answer reached the CLI and came back: {events:?}");
        assert!(
            events.iter().any(|e| matches!(e, crate::RunEvent::Exited { exit_code: Some(0), cancelled: false, .. })),
            "exit 3/4/5 names the protocol step the adapter got wrong: {events:?}"
        );
    }

    #[test]
    fn extra_instructions_are_appended_to_the_system_prompt() {
        let tuning = RunTuning {
            extra_instructions: Some("  always emit a submit tool call  ".to_owned()),
            ..RunTuning::default()
        };
        let args = build_claude_args(Some("hi".to_owned()), RunMode::Ask, ToolAccess::Default, &tuning, None);
        assert_eq!(flag_value(&args, "--append-system-prompt"), Some("always emit a submit tool call"));
    }

    #[test]
    fn blank_extra_instructions_emit_no_flag_but_real_ones_do() {
        // An empty string is the host saying "no custom instructions", not a
        // request to append nothing — an empty flag value would cost a prompt
        // section and change the run for no reason.
        //
        // The last arm is the positive one, and it is why this test can fail: an
        // adapter that emitted the flag *never* would satisfy the blank arms
        // alone, which is a negative claim answered by a mechanism that does
        // nothing.
        for (instructions, expected) in [("", None), ("   ", None), ("real", Some("real"))] {
            let tuning =
                RunTuning { extra_instructions: Some(instructions.to_owned()), ..RunTuning::default() };
            let args = build_claude_args(Some("hi".to_owned()), RunMode::Ask, ToolAccess::Default, &tuning, None);
            assert_eq!(
                flag_value(&args, "--append-system-prompt"),
                expected,
                "instructions {instructions:?}"
            );
        }
    }

    #[test]
    fn a_host_that_sets_append_system_prompt_itself_gets_no_duplicate() {
        // Same rule as --permission-mode: the host fully owns a flag it sets.
        let tuning = RunTuning {
            extra_instructions: Some("adapter copy".to_owned()),
            extra_args: vec!["--append-system-prompt".to_owned(), "host copy".to_owned()],
            ..RunTuning::default()
        };
        let args = build_claude_args(Some("hi".to_owned()), RunMode::Ask, ToolAccess::Default, &tuning, None);
        assert_eq!(args.iter().filter(|a| *a == "--append-system-prompt").count(), 1);
        assert_eq!(flag_value(&args, "--append-system-prompt"), Some("host copy"));

        // The positive arm: the same tuning WITHOUT the host's flag emits the
        // adapter's own copy. Without this, an adapter that never emitted
        // anything would pass the deduplication assertion above.
        let adapter_only = RunTuning { extra_args: Vec::new(), ..tuning };
        let args = build_claude_args(Some("hi".to_owned()), RunMode::Ask, ToolAccess::Default, &adapter_only, None);
        assert_eq!(flag_value(&args, "--append-system-prompt"), Some("adapter copy"));
    }

    #[test]
    fn host_extra_args_are_appended_verbatim() {
        // A host adds flags the adapter doesn't manage — appended as given.
        let tuning = RunTuning {
            extra_args: vec!["--add-dir".to_owned(), "/extra".to_owned()],
            ..RunTuning::default()
        };
        let args = build_claude_args(Some("hi".to_owned()), RunMode::Ask, ToolAccess::Default, &tuning, None);
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
        let args = build_claude_args(Some("hi".to_owned()), RunMode::Edit, ToolAccess::Default, &tuning, None);
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
