//! OpenAI Codex (`codex`) as a [`Harness`].
//!
//! Same process-spawn shape as the bob and Claude adapters — a
//! different binary, flags, and stdout parser. We invoke
//! `codex exec --json` and parse its JSONL into the shared
//! normalized [`crate::RunEvent`] stream.
//!
//! Auth: like Claude Code, Codex manages its own credentials (its
//! `codex login` / ChatGPT auth or its own `OPENAI_API_KEY` in the
//! environment), so Compose does not store or inject a key —
//! `credential().required` is `false`.
//!
//! The stdout wire format and its decode — including the stateful
//! [`CodexStreamParser`] that resolves codex's preamble-vs-answer
//! ambiguity — live in `parser`.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::{
    probe_version, Command, ResolveCli, CredentialSpec, Harness, Features, Error,
    Info, ModelChoice, Readiness, InstallCallback, InstallHint, RunCallback,
    RunHandle, RunMode, RunRequest, RunTuning,
};

mod parser;
pub use parser::{parse_codex_line, CodexStreamParser};

/// Registry id for the Codex harness.
pub const CODEX_HARNESS_ID: &str = "codex";

/// The program spawned when the host doesn't name one.
pub const DEFAULT_CODEX_COMMAND: &str = "codex";

/// OpenAI Codex CLI as a [`Harness`].
#[derive(Debug, Clone)]
pub struct CodexHarness {
    command: String,
}

impl Default for CodexHarness {
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
pub struct CodexHarnessConfig {
    /// Program to spawn. A bare name is resolved on PATH; a path is used as
    /// given. A rename upstream, a fork, a wrapper script or a test stub costs
    /// a field here rather than a release.
    pub command: String,
}

impl Default for CodexHarnessConfig {
    fn default() -> Self {
        Self { command: DEFAULT_CODEX_COMMAND.to_owned() }
    }
}

impl CodexHarness {
    /// Drives `codex` from PATH.
    pub fn new() -> Self {
        Self::custom(CodexHarnessConfig::default())
    }

    /// Drives a binary the host names.
    pub fn custom(config: CodexHarnessConfig) -> Self {
        Self { command: config.command }
    }
}

impl Harness for CodexHarness {
    fn info(&self) -> Info {
        Info {
            id: CODEX_HARNESS_ID.to_owned(),
            display_name: "Codex".to_owned(),
            description: "OpenAI's Codex agent CLI. Uses your existing Codex login.".to_owned(),
            install_hint: Some(
                InstallHint::url("https://developers.openai.com/codex")
                    .with_command("npm install -g @openai/codex"),
            ),
        }
    }

    fn features(&self) -> Features {
        Features {
            // Codex owns its own login and edits files directly. Model
            // names change often, so it takes free-text entry rather than a
            // curated list, and it exposes reasoning effort. What it does
            // not support — a turn cap, previews, a stored credential — is
            // left to `Default`.
            custom_model: true,
            effort: true,
            login: true,
            ..Default::default()
        }
    }

    fn list_models(&self) -> Result<Vec<ModelChoice>, Error> {
        // Codex declares no static models (ids churn → free-text entry); fill the
        // picker from models.dev's `openai` lineup when the `models-dev` feature is
        // on (empty otherwise → the user types an id).
        Ok(crate::models_dev::provider_models("openai"))
    }

    fn readiness(&self) -> Readiness {
        let Some(version) = probe_version(&self.command) else {
            return Readiness {
                harness_id: CODEX_HARNESS_ID.to_owned(),
                ready: false,
                installed: false,
                version: None,
                auth_configured: false,
                error: Some("Codex (`codex`) is not installed or not on PATH.".to_owned()),
                details: Value::Null,
            };
        };
        // Installed — distinguish signed-in from not so the picker can
        // offer "Sign in" instead of failing the first run. Either the CLI's
        // own login OR an `OPENAI_API_KEY` in the environment counts: the env
        // key is how you run headless (a container / CI), where `codex login`
        // can't open a browser. `codex login status` only sees the OAuth
        // state, so we OR in the env key ourselves.
        let signed_in = probe_codex_signed_in(&self.command)
            || crate::harness::api_key_value_usable(std::env::var("OPENAI_API_KEY").ok());
        Readiness {
            harness_id: CODEX_HARNESS_ID.to_owned(),
            ready: signed_in,
            installed: true,
            version: Some(version),
            auth_configured: signed_in,
            error: if signed_in {
                None
            } else {
                Some(
                    "Codex is installed but not signed in. Click Sign in to connect your ChatGPT/OpenAI account, or set OPENAI_API_KEY."
                        .to_owned(),
                )
            },
            details: codex_resolved_details(&self.command),
        }
    }

    fn start(&self, request: RunRequest, on_event: RunCallback) -> Result<RunHandle, Error> {
        // `attachments` ignored: codex exec is a text CLI (no image input here).
        let RunRequest { run_id, prompt, cwd, mode, tuning, resume, attachments: _ } = request;
        let args = build_codex_args(prompt, mode, &tuning, resume.as_deref());
        let cwd = cwd.unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

        // No env injected — Codex uses its own auth. PATH augmentation
        // in spawn_streaming ensures `node` is found for a
        // Finder-launched .app.
        //
        // Codex needs a *stateful* parser (one per run): it emits several
        // complete `agent_message` items per turn — short preambles before
        // tool calls and a final answer — that must not be concatenated into
        // the answer, and its stderr is tracing noise to drop (see
        // [`CodexStreamParser`]). The callback runs on cli-stream's reader
        // threads, so the parser is held behind an `Arc<Mutex>` — the same
        // shape as bob's.
        let parser = Arc::new(Mutex::new(CodexStreamParser::new()));
        let program = tuning.binary_path.clone().unwrap_or_else(|| PathBuf::from(&self.command));
        let handle = Command::new(program)
            .cwd(cwd)
            .run_id(run_id)
            .args(args)
            .resolve_cli()
            .stream(move |event| {
                // Recover a poisoned lock rather than panic on a reader
                // thread — parsing is total, so the parser is never
                // mid-corruption.
                let mut parser = parser.lock().unwrap_or_else(|p| p.into_inner());
                for normalized in parser.on_process_event(event) {
                    (*on_event)(normalized);
                }
            },
        )
        .map_err(Error::spawn)?;
        Ok(Box::new(handle))
    }

    fn credential(&self) -> CredentialSpec {
        CredentialSpec {
            label: "Codex login (managed by the codex CLI)".to_owned(),
            keychain_service: "openai".to_owned(),
            keychain_account: "OPENAI_API_KEY".to_owned(),
            required: false,
        }
    }

    fn login(&self, on_event: InstallCallback) -> Result<(), Error> {
        // `codex login` runs the CLI's OAuth flow (opens the browser).
        crate::run_login_command(&self.command, &["login"], on_event)
    }
}

/// Resolve the `codex` binary and classify its install kind for the readiness
/// `details`, mirroring the Claude adapter — so the Runtimes UI can surface
/// "npm — can go stale / Update to native" instead of a bare, ambiguous
/// "Update". Reuses the shared resolve/classify in `crate::claude::resolve`.
fn codex_resolved_details(command: &str) -> Value {
    let path = crate::augmented_path();
    let Some(resolved) = crate::claude::resolve::resolve_on_path(command, &path) else {
        return Value::Null;
    };
    let mut details = serde_json::Map::new();
    details.insert(
        "resolved_path".to_owned(),
        Value::String(resolved.to_string_lossy().into_owned()),
    );
    if let Ok(home) = std::env::var("HOME") {
        let kind = crate::claude::resolve::classify(&resolved, std::path::Path::new(&home), None);
        details.insert(
            "install_kind".to_owned(),
            Value::String(kind.as_str().to_owned()),
        );
    }
    Value::Object(details)
}

/// Probe Codex's auth: `codex login status` exits 0 when signed in.
/// Lets [`CodexHarness::readiness`] distinguish installed from signed-in
/// (so the picker can offer "Sign in").
fn probe_codex_signed_in(command: &str) -> bool {
    crate::hidden_command(command)
        .args(["login", "status"])
        .env("PATH", crate::augmented_path())
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Build the argv for a `codex exec --json` headless run. Kept pure
/// (no spawn) so the flag mapping is unit-tested. `tuning.model` →
/// `--model`; `tuning.effort` → `-c model_reasoning_effort="..."`
/// (codex's config override, value parsed as TOML), defaulting to `low`
/// when unset so codex's built-in tools don't reject its `minimal`
/// default; Codex has no turn-cap flag, so `tuning.max_turns` is
/// intentionally ignored. Options precede the positional prompt, as
/// `codex exec` expects.
fn build_codex_args(
    prompt: String,
    mode: RunMode,
    tuning: &RunTuning,
    resume: Option<&str>,
) -> Vec<String> {
    // `exec` always; `exec resume <id>` to continue a prior session instead of
    // replaying history in the prompt. The session id is a positional *after*
    // the options and *before* the prompt (`codex exec resume [OPTIONS]
    // [SESSION_ID] [PROMPT]`), so it's appended at the tail below.
    let mut args = vec!["exec".to_owned()];
    if resume.is_some() {
        args.push("resume".to_owned());
    }
    // `--skip-git-repo-check`: `codex exec` otherwise refuses to run unless
    // the cwd is a git repo ("Not inside a trusted directory and
    // --skip-git-repo-check was not specified.", exit 1). A harness runs in
    // whatever working directory the consumer hands it — often not a git repo
    // (notes, drafts, a fresh folder) — so that interactive guardrail is
    // wrong here. This skips only the is-this-a-repo gate; the execution
    // sandbox (mode → `--full-auto`) is unaffected. Both flags are valid on
    // `exec resume` too.
    args.push("--json".to_owned());
    args.push("--skip-git-repo-check".to_owned());
    if let Some(model) = tuning.model.as_deref().map(str::trim).filter(|m| !m.is_empty()) {
        args.push("--model".to_owned());
        args.push(model.to_owned());
    }
    // Codex's own default reasoning effort is `minimal`, which its built-in
    // `image_gen`/`web_search` tools reject ("cannot be used with
    // reasoning.effort 'minimal'", a 400 that breaks a default run). So when
    // the user picks no effort, send `low` rather than leaving codex on
    // `minimal`. Only `minimal` when explicitly chosen.
    let effort = tuning.effort.unwrap_or(crate::ReasoningEffort::Low);
    args.push("-c".to_owned());
    args.push(format!("model_reasoning_effort=\"{}\"", effort.as_cli_value()));
    if matches!(mode, RunMode::Edit) {
        // Low-friction sandboxed auto-execution so Codex can apply
        // edits without interactive approval. (Exact sandbox flags
        // vary by codex version; --full-auto is the stable one.)
        args.push("--full-auto".to_owned());
    }
    // Host passthrough/overrides — before the trailing positionals.
    args.extend(tuning.extra_args.iter().cloned());
    // Positionals last: the session id (resume only) precedes the prompt.
    if let Some(session_id) = resume {
        args.push(session_id.to_owned());
    }
    args.push(prompt);
    args
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use crate::events::RunEvent;

    /// A stand-in `codex` that answers the three ways the adapter invokes it:
    /// `--version`, `login status`, and a real `exec` run. The `exec` case
    /// records the argv it was handed, so a test can assert what the CLI
    /// actually received rather than what the pure builder returned.
    #[cfg(unix)]
    fn fake_codex(tag: &str, signed_in: bool, emits: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("codex-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let argv = dir.join("argv");
        let cli = dir.join("codex");
        let login_exit = i32::from(!signed_in);
        std::fs::write(
            &cli,
            format!(
                "#!/bin/sh\n\
                 case \"$1\" in\n\
                 --version) echo 'codex-cli 9.9.9'; exit 0 ;;\n\
                 login) exit {login_exit} ;;\n\
                 exec) : > '{argv}'; for a in \"$@\"; do printf '%s\\n' \"$a\" >> '{argv}'; done\n\
                 {emits}\n\
                 exit 0 ;;\n\
                 esac\n\
                 exit 1\n",
                argv = argv.display(),
            ),
        )
        .unwrap();
        std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o755)).unwrap();
        (cli, argv)
    }

    #[cfg(unix)]
    fn drive(cli: &std::path::Path, request: RunRequest) -> Vec<RunEvent> {
        use std::sync::{Arc, Mutex};
        let seen: Arc<Mutex<Vec<RunEvent>>> = Arc::default();
        let sink = Arc::clone(&seen);
        let harness = CodexHarness::custom(CodexHarnessConfig {
            command: cli.display().to_string(),
        });
        let handle = harness
            .start(request, Arc::new(move |event| sink.lock().unwrap().push(event)))
            .expect("the stand-in should spawn");

        // Bounded: a fixture that never exits must fail the test, not hang it.
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
        let events = seen.lock().unwrap().clone();
        events
    }

    #[cfg(unix)]
    #[test]
    fn a_run_reaches_the_cli_as_codex_exec_and_comes_back_as_events() {
        // The argv assertions elsewhere test `build_codex_args` in isolation.
        // This is the only test that proves the argv the *process* receives is
        // that one — spawn, stream, parse and normalize included.
        let message = r#"{"type":"item.completed","item":{"id":"i1","type":"agent_message","text":"done"}}"#;
        let (cli, argv) = fake_codex("run", true, &format!("printf '%s\\n' '{message}'"));

        let events = drive(
            &cli,
            RunRequest {
                run_id: "r1".to_owned(),
                prompt: "say something".to_owned(),
                cwd: Some(std::env::temp_dir()),
                tuning: RunTuning { model: Some("o4-mini".to_owned()), ..RunTuning::default() },
                ..RunRequest::default()
            },
        );

        let passed: Vec<String> =
            std::fs::read_to_string(&argv).unwrap().lines().map(str::to_owned).collect();
        assert_eq!(passed.first().map(String::as_str), Some("exec"));
        assert!(passed.iter().any(|a| a == "--json"), "argv was {passed:?}");
        assert!(passed.iter().any(|a| a == "--skip-git-repo-check"), "argv was {passed:?}");
        assert!(passed.windows(2).any(|w| w[0] == "--model" && w[1] == "o4-mini"));
        assert_eq!(
            passed.last().map(String::as_str),
            Some("say something"),
            "the prompt is the trailing positional",
        );

        assert!(
            events.iter().any(|e| matches!(e, RunEvent::Text { delta, .. } if delta == "done")),
            "the CLI's message should arrive as text: {events:?}",
        );
        assert!(events.iter().any(|e| matches!(
            e,
            RunEvent::Exited { exit_code: Some(0), cancelled: false, .. }
        )));
    }

    #[cfg(unix)]
    #[test]
    fn a_nonzero_exit_is_reported_rather_than_read_as_a_finished_run() {
        // `codex exec` failing (bad flag, refused sandbox) must not look like a
        // run that simply produced nothing.
        let (cli, _) = fake_codex("fail", true, "exit 3; :");
        let events = drive(&cli, RunRequest { prompt: "hi".to_owned(), ..RunRequest::default() });
        assert!(
            events.iter().any(|e| matches!(e, RunEvent::Exited { exit_code: Some(3), .. })),
            "{events:?}",
        );
    }

    #[cfg(unix)]
    #[test]
    fn readiness_reports_the_version_and_whether_login_status_succeeded() {
        // Both halves come from spawning the CLI, so neither is covered by the
        // pure argv tests. Signed-out must still read as installed — the UI
        // offers "Sign in" only when it knows the binary is there.
        let (yes, _) = fake_codex("in", true, ":");
        let ready = CodexHarness::custom(CodexHarnessConfig { command: yes.display().to_string() })
            .readiness();
        assert!(ready.installed && ready.ready && ready.auth_configured);
        assert_eq!(ready.version.as_deref(), Some("codex-cli 9.9.9"));
        assert!(ready.error.is_none());

        let (no, _) = fake_codex("out", false, ":");
        let ready = CodexHarness::custom(CodexHarnessConfig { command: no.display().to_string() })
            .readiness();
        assert!(ready.installed, "the binary is present either way");
        assert!(!ready.ready && !ready.auth_configured);
        assert!(ready.error.is_some(), "a signed-out CLI must say what to do");
    }

    #[test]
    fn a_renamed_binary_is_what_gets_probed() {
        let renamed = CodexHarness::custom(CodexHarnessConfig {
            command: "definitely-not-a-real-binary-xyz".into(),
        });
        assert!(!renamed.readiness().installed, "an unresolvable command cannot report installed");
        assert_eq!(CodexHarness::new().command, DEFAULT_CODEX_COMMAND);
        assert_eq!(CodexHarness::default().command, DEFAULT_CODEX_COMMAND);
    }
    use crate::ReasoningEffort;

    /// A throwaway CLI standing in for `codex login status`.
    #[cfg(unix)]
    fn fake_cli(tag: &str, script: &str) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("hl-codex-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cli");
        std::fs::write(&path, format!("#!/bin/sh\n{script}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[cfg(unix)]
    #[test]
    fn sign_in_is_read_from_the_exit_code_of_login_status() {
        // Codex answers in prose rather than JSON, so the exit code is the
        // whole signal. Getting it backwards sends a signed-in user to a Sign
        // in button, or lets a signed-out one start a run that cannot work.
        let signed_in = fake_cli("in", "echo 'Logged in'; exit 0");
        assert!(probe_codex_signed_in(signed_in.to_str().unwrap()));

        let signed_out = fake_cli("out", "echo 'Not logged in'; exit 1");
        assert!(!probe_codex_signed_in(signed_out.to_str().unwrap()));

        assert!(!probe_codex_signed_in("definitely-not-a-real-binary-xyz"), "an absent CLI is not signed in");
    }

    #[test]
    fn codex_info_and_credential() {
        let h = CodexHarness::new();
        assert_eq!(h.info().id, CODEX_HARNESS_ID);
        let hint = h.info().install_hint.expect("Codex is a CLI the user installs");
        assert_eq!(hint.command.as_deref(), Some("npm install -g @openai/codex"));
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
    fn codex_args_default_omit_model_but_force_low_effort() {
        let args = build_codex_args("hi".to_owned(), RunMode::Ask, &RunTuning::default(), None);
        assert_eq!(args[0], "exec");
        assert!(!args.contains(&"resume".to_owned()));
        assert!(args.contains(&"--json".to_owned()));
        // Always present: a harness's cwd is often not a git repo, and
        // without this `codex exec` exits 1 ("Not inside a trusted
        // directory …"). Independent of run mode.
        assert!(args.contains(&"--skip-git-repo-check".to_owned()));
        assert!(!args.iter().any(|a| a == "--model"));
        // No explicit effort → `low`, never codex's `minimal` default (which
        // its built-in image_gen/web_search tools reject with a 400).
        assert_eq!(flag_value(&args, "-c"), Some("model_reasoning_effort=\"low\""));
        assert!(!args.iter().any(|a| a.contains("minimal")));
        assert!(!args.iter().any(|a| a == "--full-auto"));
        // Prompt is the trailing positional arg.
        assert_eq!(args.last().map(String::as_str), Some("hi"));
    }

    #[test]
    fn codex_args_explicit_minimal_effort_is_honored() {
        let tuning =
            RunTuning { effort: Some(ReasoningEffort::Minimal), ..RunTuning::default() };
        let args = build_codex_args("hi".to_owned(), RunMode::Ask, &tuning, None);
        assert_eq!(flag_value(&args, "-c"), Some("model_reasoning_effort=\"minimal\""));
    }

    #[test]
    fn codex_args_carry_model_and_effort_and_ignore_max_turns() {
        let tuning = RunTuning {
            model: Some("gpt-5-codex".to_owned()),
            effort: Some(ReasoningEffort::High),
            max_turns: Some(5),
            ..RunTuning::default()
        };
        let args = build_codex_args("hi".to_owned(), RunMode::Edit, &tuning, None);
        assert_eq!(flag_value(&args, "--model"), Some("gpt-5-codex"));
        assert_eq!(flag_value(&args, "-c"), Some("model_reasoning_effort=\"high\""));
        assert!(args.contains(&"--full-auto".to_owned()));
        // Codex has no turn-cap flag — max_turns must not leak.
        assert!(!args.iter().any(|a| a == "--max-turns"));
        // Options precede the prompt; the prompt stays last.
        assert_eq!(args.last().map(String::as_str), Some("hi"));
    }

    #[test]
    fn codex_resume_uses_the_resume_subcommand_with_id_before_prompt() {
        let args =
            build_codex_args("hi".to_owned(), RunMode::Ask, &RunTuning::default(), Some("sess-9"));
        // `exec resume` subcommand, JSON stream + git-skip still present.
        assert_eq!(args[0], "exec");
        assert_eq!(args[1], "resume");
        assert!(args.contains(&"--json".to_owned()));
        assert!(args.contains(&"--skip-git-repo-check".to_owned()));
        // Positionals: the session id immediately precedes the prompt (tail).
        let last_two = &args[args.len() - 2..];
        assert_eq!(last_two, &["sess-9".to_owned(), "hi".to_owned()]);
    }
}
