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
//! ambiguity — live in [`parser`].

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::{
    spawn_streaming, CredentialSpec, Harness, HarnessCapabilities, HarnessError, HarnessInfo,
    HarnessModel, HarnessReadiness, InstallCallback, InstallEvent, RunCallback, RunHandle, RunMode,
    RunRequest, RunTuning,
};

mod parser;
pub use parser::{parse_codex_line, CodexStreamParser};

/// Registry id for the Codex harness.
pub const CODEX_HARNESS_ID: &str = "codex";

/// OpenAI Codex CLI as a [`Harness`].
#[derive(Debug, Default, Clone)]
pub struct CodexHarness;

impl CodexHarness {
    pub fn new() -> Self {
        Self
    }
}

impl Harness for CodexHarness {
    fn info(&self) -> HarnessInfo {
        HarnessInfo {
            id: CODEX_HARNESS_ID.to_owned(),
            display_name: "Codex".to_owned(),
            description: "OpenAI's Codex agent CLI. Uses your existing Codex login.".to_owned(),
            requires_install: true,
            capabilities: HarnessCapabilities {
                // Codex owns its own login and edits files directly.
                // Model names change often, so allow free-text entry
                // rather than a curated list; it exposes reasoning
                // effort but no turn cap.
                credential_required: false,
                previews_edits: false,
                models: Vec::new(),
                allows_custom_model: true,
                supports_effort: true,
                supports_max_turns: false,
                supports_login: true,
                supports_custom_instructions: false,
            },
        }
    }

    fn list_models(&self) -> Result<Vec<HarnessModel>, HarnessError> {
        // Codex declares no static models (ids churn → free-text entry); fill the
        // picker from models.dev's `openai` lineup when the `models-dev` feature is
        // on (empty otherwise → the user types an id).
        Ok(crate::models_dev::provider_models("openai"))
    }

    fn readiness(&self) -> HarnessReadiness {
        let Some(version) = probe_version("codex") else {
            return HarnessReadiness {
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
        let signed_in = probe_codex_signed_in()
            || crate::harness::api_key_value_usable(std::env::var("OPENAI_API_KEY").ok());
        HarnessReadiness {
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
            details: codex_resolved_details(),
        }
    }

    fn install(&self, on_event: InstallCallback) -> Result<(), HarnessError> {
        (*on_event)(InstallEvent::Step {
            text: "Installing Codex via npm…".to_owned(),
        });
        let output = Command::new("npm")
            .args(["install", "-g", "@openai/codex"])
            .env("PATH", crate::augmented_node_path())
            .output()
            .map_err(|e| HarnessError::install(format!("failed to run npm: {e}")))?;
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
        let program = tuning.binary_path.clone().unwrap_or_else(|| PathBuf::from("codex"));
        let handle = spawn_streaming(
            program,
            args,
            Vec::new(),
            cwd,
            run_id,
            move |event| {
                // Recover a poisoned lock rather than panic on a reader
                // thread — parsing is total, so the parser is never
                // mid-corruption.
                let mut parser = parser.lock().unwrap_or_else(|p| p.into_inner());
                for normalized in parser.on_process_event(event) {
                    (*on_event)(normalized);
                }
            },
        )
        .map_err(HarnessError::spawn)?;
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

    fn login(&self, on_event: InstallCallback) -> Result<(), HarnessError> {
        // `codex login` runs the CLI's OAuth flow (opens the browser).
        crate::run_login_command("codex", &["login"], on_event)
    }
}

/// Resolve the `codex` binary and classify its install kind for the readiness
/// `details`, mirroring the Claude adapter — so the Runtimes UI can surface
/// "npm — can go stale / Update to native" instead of a bare, ambiguous
/// "Update". Reuses the shared resolve/classify in `crate::claude::resolve`.
fn codex_resolved_details() -> Value {
    let path = crate::augmented_node_path();
    let Some(resolved) = crate::claude::resolve::resolve_on_path("codex", &path) else {
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

/// Probe Codex's auth: `codex login status` exits 0 when signed in.
/// Lets [`CodexHarness::readiness`] distinguish installed from signed-in
/// (so the picker can offer "Sign in").
fn probe_codex_signed_in() -> bool {
    Command::new("codex")
        .args(["login", "status"])
        .env("PATH", crate::augmented_node_path())
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
    use crate::ReasoningEffort;

    #[test]
    fn codex_info_and_credential() {
        let h = CodexHarness::new();
        assert_eq!(h.info().id, CODEX_HARNESS_ID);
        assert!(h.info().requires_install);
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
