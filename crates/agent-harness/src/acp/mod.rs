//! ACP-client adapter: drive an external **Agent Client Protocol** agent
//! (OpenCode, Gemini CLI, Goose, …) over JSON-RPC/stdio and normalize its
//! `session/update` stream into [`crate::RunEvent`]s — the third adapter
//! archetype (the CLI-wrapping adapters parse a stdout stream; the
//! OpenAI-compatible adapter owns the loop; this one relays an ACP agent).
//!
//! Built on Zed's `agent-client-protocol` crate (we are the *client*; the
//! external process is the *agent*). It uses the 0.14 role/builder model: a
//! `Client.builder()` registers handlers for the agent's incoming
//! requests/notifications, then `connect_with` spawns the agent (an [`AcpAgent`]
//! stdio transport) and runs the session (`initialize` → `new_session` →
//! `prompt`). The connection is async + runtime-agnostic, so `run()` drives it
//! on a `smol` executor inside the worker thread it spawns — keeping the same
//! thread+callback shape as the other adapters.
//!
//! Opt-in behind the `acp` feature (it pulls the ACP crate + a small async
//! runtime and spawns external agents).
//!
//! [`AcpAgent`]: agent_client_protocol::AcpAgent

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agent_client_protocol as acp;
use serde_json::Value;
use smol::Timer;

use crate::{
    CredentialSpec, Harness, Features, Error, Info, ModelChoice,
    Readiness, InstallHint, RunCallback, RunControl, RunEvent, RunHandle, RunMode,
    RunRequest,
};

mod translate;

/// An ACP agent driven as a [`Harness`]. The vendor is configuration (the
/// command to spawn), not a type — `::opencode()` and `::custom(...)` are
/// constructors over one adapter.
pub struct AcpHarness {
    id: String,
    display_name: String,
    description: String,
    /// Program to spawn (e.g. `opencode`) …
    command: String,
    /// … and the args that put it in ACP mode (e.g. `["acp"]`).
    args: Vec<String>,
    /// Where the user gets this agent when `command` isn't on PATH.
    install_hint: Option<InstallHint>,
    /// How this vendor exposes launch-time model selection — ACP itself carries
    /// no model, so listing + selecting a model happen out-of-band. `None` → a
    /// generic ACP agent: no model list, and the per-run model
    /// ([`RunTuning`](crate::RunTuning)) is ignored.
    model_control: Option<ModelControl>,
}

/// How a config-file ACP agent (opencode) exposes a launch-time model. ACP has
/// no model field, so we **list** models via a vendor CLI subcommand and
/// **select** one by writing a JSON config file and pointing an env var at it
/// just before spawn. Set by [`AcpHarness::opencode`]; absent on a generic
/// [`AcpHarness::custom`], which then has no model picker.
struct ModelControl {
    /// CLI subcommand that prints one `provider/model` per line, used by
    /// [`AcpHarness::list_models`]. opencode: `["models"]` → `opencode models`.
    list_subcommand: Vec<String>,
    /// Env var the agent reads its JSON config path from. opencode: `OPENCODE_CONFIG`.
    config_env: String,
    /// JSON field in that config that selects the model. opencode: `model`.
    config_field: String,
}

/// Configuration for [`AcpHarness::custom`] — named fields rather than a
/// positional list, so a call site reads unambiguously. Derives `Default`.
#[derive(Clone, Debug, Default)]
pub struct AcpHarnessConfig {
    /// Stable id used in the registry / picker (e.g. `"gemini"`).
    pub id: String,
    /// Human-readable name shown in the UI (e.g. `"Gemini"`).
    pub display_name: String,
    /// Program to spawn (e.g. `"gemini"`).
    pub command: String,
    /// Args that launch it in ACP mode (e.g. `["--experimental-acp"]`).
    pub args: Vec<String>,
    /// Where the user gets this agent. `None` leaves the picker saying only
    /// that the command is missing.
    pub install_hint: Option<InstallHint>,
}

impl AcpHarness {
    /// OpenCode over ACP — spawns `opencode acp`. opencode reads a JSON config
    /// file named by `$OPENCODE_CONFIG`; its `model` field (a `provider/model`
    /// id) selects the model, and `opencode models` lists the available ids — so
    /// this constructor wires launch-time model selection (`list_models` + the
    /// per-run [`RunTuning`](crate::RunTuning) model).
    pub fn opencode() -> Self {
        let mut harness = Self::custom(AcpHarnessConfig {
            id: "opencode".to_owned(),
            display_name: "OpenCode".to_owned(),
            command: "opencode".to_owned(),
            args: vec!["acp".to_owned()],
            install_hint: Some(InstallHint::url("https://github.com/sst/opencode")),
        });
        harness.model_control = Some(ModelControl {
            list_subcommand: vec!["models".to_owned()],
            config_env: "OPENCODE_CONFIG".to_owned(),
            config_field: "model".to_owned(),
        });
        harness
    }

    /// Any ACP agent, configured by an [`AcpHarnessConfig`]: `command` + `args`
    /// that launch it as an ACP server over stdio, with named fields so the call
    /// site reads clearly.
    pub fn custom(config: AcpHarnessConfig) -> Self {
        let AcpHarnessConfig { id, display_name, command, args, install_hint } = config;
        Self {
            id,
            description: format!("{display_name} via the Agent Client Protocol."),
            display_name,
            command,
            args,
            install_hint,
            model_control: None,
        }
    }
}

impl Harness for AcpHarness {
    fn info(&self) -> Info {
        Info {
            id: self.id.clone(),
            display_name: self.display_name.clone(),
            description: self.description.clone(),
            install_hint: self.install_hint.clone(),
        }
    }

    fn features(&self) -> Features {
        Features {
            // Models are discovered live via `list_models()` (opencode lists
            // its own; a generic ACP agent has none), so the static list is
            // left empty by `Default`; a free-text model id is accepted.
            custom_model: true,
            ..Default::default()
        }
    }

    fn readiness(&self) -> Readiness {
        let installed = probe_command(&self.command);
        Readiness {
            harness_id: self.id.clone(),
            ready: installed,
            installed,
            version: None,
            auth_configured: installed,
            error: if installed {
                None
            } else {
                Some(format!(
                    "`{}` is not installed or not on PATH (needed to run {} over ACP).",
                    self.command, self.display_name
                ))
            },
            details: Value::Null,
        }
    }

    fn start(&self, request: RunRequest, on_event: RunCallback) -> Result<RunHandle, Error> {
        // resume (session/load) is a follow-up; this first cut runs a fresh
        // session. `attachments` ignored: a text prompt only.
        let RunRequest { run_id, prompt, cwd, mode, tools, tuning, resume: _, attachments: _ } =
            request;
        crate::harness::refuse_withheld_tools(
            "acp",
            tools,
            "an ACP agent owns its own tool surface and the protocol has no way to say \"offer none\"",
        )?;
        // ACP carries no model. For a config-file vendor (opencode), select the
        // chosen model out-of-band: write a temp JSON config `{ <field>: <model> }`
        // and point the agent's config env var at it for this spawn. Other tuning
        // knobs (effort, max_turns, …) have no ACP equivalent and are ignored.
        let (env, model_config_file) = match (&self.model_control, tuning.model) {
            (Some(mc), Some(model)) => {
                let path = write_model_config(&run_id, &mc.config_field, &model)
                    .map_err(Error::spawn)?;
                (vec![(mc.config_env.clone(), path.to_string_lossy().into_owned())], Some(path))
            }
            _ => (Vec::new(), None),
        };
        let cfg = AcpRunCfg {
            command: self.command.clone(),
            args: self.args.clone(),
            run_id,
            prompt,
            cwd: cwd.unwrap_or_else(|| std::env::current_dir().unwrap_or_default()),
            mode,
            env,
            model_config_file,
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let thread_cancel = Arc::clone(&cancel);
        // Same thread+callback shape as the other adapters: the async ACP
        // connection is driven on a smol executor on a worker thread, so run()
        // returns immediately.
        std::thread::spawn(move || run_acp(cfg, thread_cancel, on_event));
        Ok(Box::new(AcpRun { cancel }))
    }

    fn credential(&self) -> CredentialSpec {
        CredentialSpec {
            label: format!("{} (manages its own auth)", self.display_name),
            keychain_service: self.id.clone(),
            keychain_account: String::new(),
            required: false,
        }
    }

    fn list_models(&self) -> Result<Vec<ModelChoice>, Error> {
        // ACP exposes no model list; a config-file vendor (opencode) lists via
        // its own CLI subcommand. A generic ACP agent has none → empty (the host
        // hides the picker). PATH is augmented so a packaged `.app` finds the CLI.
        let Some(mc) = &self.model_control else {
            return Ok(Vec::new());
        };
        let output = crate::hidden_command(&self.command)
            .args(&mc.list_subcommand)
            .env("PATH", crate::augmented_path())
            .output()
            .map_err(|e| {
                Error::spawn(format!(
                    "`{} {}` failed: {e}",
                    self.command,
                    mc.list_subcommand.join(" ")
                ))
            })?;
        Ok(models_from_listing(output.status.success(), &String::from_utf8_lossy(&output.stdout)))
    }
}

/// The models a listing subcommand reported, one id per line.
///
/// A non-zero exit is deliberately not an error: an agent that is offline or
/// signed out should leave the picker empty, not fail the harness that owns it.
/// Separate from the spawn because both of those are decisions, and neither is
/// reachable through a process.
fn models_from_listing(succeeded: bool, stdout: &str) -> Vec<ModelChoice> {
    if !succeeded {
        return Vec::new();
    }
    stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        // No prettier name is on offer, so the id is also the label.
        .map(|line| ModelChoice { value: line.to_owned(), label: line.to_owned() })
        .collect()
}

/// Whether `command` is runnable on the (augmented) PATH — `<command> --version`
/// exits without a spawn error. Augmented PATH so a packaged `.app` finds a
/// CLI installed via nvm / Homebrew / etc.
fn probe_command(command: &str) -> bool {
    crate::hidden_command(command)
        .arg("--version")
        .env("PATH", crate::augmented_path())
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Write a one-off JSON config selecting `model` (under `field`) to a temp file
/// keyed by `run_id`, returning its path. An ACP agent that reads a config file
/// (opencode, via `$OPENCODE_CONFIG`) picks the model from it — the out-of-band
/// way to choose a model the protocol itself can't carry. Removed when the run
/// ends (see [`run_acp`]).
fn write_model_config(run_id: &str, field: &str, model: &str) -> Result<PathBuf, String> {
    let path = std::env::temp_dir().join(format!("harness-acp-model-{run_id}.json"));
    let body = serde_json::json!({ field: model }).to_string();
    std::fs::write(&path, body)
        .map_err(|e| format!("writing ACP model config to {}: {e}", path.display()))?;
    Ok(path)
}

/// Everything `run_acp` needs, assembled by `run()` from the `RunRequest`.
struct AcpRunCfg {
    command: String,
    args: Vec<String>,
    run_id: String,
    prompt: String,
    cwd: PathBuf,
    mode: RunMode,
    /// Extra env for the spawned agent — e.g. opencode's `OPENCODE_CONFIG`
    /// pointing at the temp model-config file. Empty when no model was chosen.
    env: Vec<(String, String)>,
    /// Temp model-config file to delete when the run ends, if one was written.
    model_config_file: Option<PathBuf>,
}

/// Cancel handle for an in-flight ACP run. Cooperative: a watcher future races
/// the connection and, on cancel, drops it (tearing down the agent process).
struct AcpRun {
    cancel: Arc<AtomicBool>,
}

impl RunControl for AcpRun {
    fn cancel(&self) -> Result<(), Error> {
        self.cancel.store(true, Ordering::SeqCst);
        Ok(())
    }
    fn was_cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

/// Drive the ACP connection to completion on a `smol` executor, emitting the
/// normalized event stream. Always ends with exactly one `RunEvent::Exited`.
fn run_acp(cfg: AcpRunCfg, cancel: Arc<AtomicBool>, on_event: RunCallback) {
    (*on_event)(RunEvent::Started { run_id: cfg.run_id.clone() });

    let perm_mode = cfg.mode;
    let notif_on_event = on_event.clone();
    let notif_rid = cfg.run_id.clone();
    let prompt = cfg.prompt.clone();
    let cwd = cfg.cwd.clone();

    // Transport: spawn `command args…` as a stdio ACP agent. The session's
    // working directory is carried in `new_session`, not the spawn dir.
    let env_vars: Vec<acp::schema::EnvVariable> = cfg
        .env
        .iter()
        .map(|(name, value)| acp::schema::EnvVariable::new(name.clone(), value.clone()))
        .collect();
    let server = acp::schema::McpServer::Stdio(
        acp::schema::McpServerStdio::new(cfg.command.clone(), cfg.command.clone())
            .args(cfg.args.clone())
            .env(env_vars),
    );
    let agent = acp::AcpAgent::new(server);

    // The connection: register handlers for the agent's incoming
    // requests/notifications, then run the session inside `connect_with`.
    let connect = async move {
        acp::Client
            .builder()
            .name("openai-compatible")
            .on_receive_request(
                move |req: acp::schema::RequestPermissionRequest,
                      responder: acp::Responder<acp::schema::RequestPermissionResponse>,
                      _cx: acp::ConnectionTo<acp::Agent>| {
                    let mode = perm_mode;
                    async move {
                        // Edit mode → allow; Ask mode (read-only) → reject. Pick
                        // an option of the matching kind, else the first offered.
                        let allow = matches!(mode, RunMode::Edit);
                        let pick =
                            req.options.iter().find(|o| is_allow(&o.kind) == allow).or_else(|| req.options.first());
                        let outcome = match pick {
                            Some(o) => acp::schema::RequestPermissionOutcome::Selected(
                                acp::schema::SelectedPermissionOutcome::new(o.option_id.clone()),
                            ),
                            None => acp::schema::RequestPermissionOutcome::Cancelled,
                        };
                        responder.respond(acp::schema::RequestPermissionResponse::new(outcome))
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_notification(
                move |notif: acp::schema::SessionNotification, _cx: acp::ConnectionTo<acp::Agent>| {
                    let on_event = notif_on_event.clone();
                    let rid = notif_rid.clone();
                    async move {
                        for event in translate::session_update_to_events(&rid, notif.update) {
                            (*on_event)(event);
                        }
                        Ok(())
                    }
                },
                acp::on_receive_notification!(),
            )
            .connect_with(agent, move |cx: acp::ConnectionTo<acp::Agent>| async move {
                cx.send_request(acp::schema::InitializeRequest::new(acp::schema::ProtocolVersion::LATEST))
                    .block_task()
                    .await?;
                let session =
                    cx.send_request(acp::schema::NewSessionRequest::new(cwd.clone())).block_task().await?;
                let resp = cx
                    .send_request(acp::schema::PromptRequest::new(session.session_id, vec![prompt.clone().into()]))
                    .block_task()
                    .await?;
                Ok(resp.stop_reason)
            })
            .await
            .map_err(|e| format!("ACP run failed: {e}"))
    };

    // Cooperative cancel: race the connection against the cancel flag. If cancel
    // wins, the connection future is dropped, tearing down the agent process.
    let cancel_fut = {
        let cancel = Arc::clone(&cancel);
        async move {
            loop {
                if cancel.load(Ordering::SeqCst) {
                    return Err("cancelled".to_owned());
                }
                Timer::after(Duration::from_millis(50)).await;
            }
        }
    };

    let outcome: Result<acp::schema::StopReason, String> =
        smol::block_on(futures_lite::future::or(connect, cancel_fut));

    let run_id = cfg.run_id;
    // Remove the temp model-config file (if one was written for this run).
    if let Some(path) = cfg.model_config_file {
        let _ = std::fs::remove_file(path);
    }
    match outcome {
        Ok(stop) => {
            let cancelled =
                cancel.load(Ordering::SeqCst) || matches!(stop, acp::schema::StopReason::Cancelled);
            (*on_event)(RunEvent::Exited { run_id, exit_code: Some(0), cancelled });
        }
        Err(_) if cancel.load(Ordering::SeqCst) => {
            // The error is just the agent being torn down by the cancel race.
            (*on_event)(RunEvent::Exited { run_id, exit_code: None, cancelled: true });
        }
        Err(message) => {
            (*on_event)(RunEvent::Error { run_id: run_id.clone(), message });
            (*on_event)(RunEvent::Exited { run_id, exit_code: Some(1), cancelled: false });
        }
    }
}

/// Whether a permission option allows the action (vs. rejects it).
fn is_allow(kind: &acp::schema::PermissionOptionKind) -> bool {
    matches!(
        kind,
        acp::schema::PermissionOptionKind::AllowOnce | acp::schema::PermissionOptionKind::AllowAlways
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Harness;

    /// As with codex: an agent that owns its own tool surface cannot promise to
    /// offer none, so it says so at the call rather than running unsandboxed.
    #[test]
    fn a_request_for_no_tools_is_refused_rather_than_run_with_them() {
        let harness = AcpHarness::custom(AcpHarnessConfig {
            id: "fake".to_owned(),
            display_name: "Fake".to_owned(),
            command: "agent".to_owned(),
            args: Vec::new(),
            install_hint: None,
        });
        let Err(error) = harness.start(
            RunRequest { tools: crate::ToolAccess::None, ..RunRequest::default() },
            std::sync::Arc::new(|_| {}),
        ) else {
            panic!("an ACP agent cannot withhold its own tools");
        };
        assert!(error.to_string().contains("ToolAccess::None"), "{error}");
    }

    #[cfg(unix)]
    use crate::test_support::fake_cli;

    #[cfg(unix)]
    #[test]
    fn an_agent_is_present_only_if_its_command_runs() {
        // This is what the picker shows as installed-or-not, and the only
        // evidence is whether the binary answers at all.
        let present = fake_cli("present", "exit 0");
        assert!(probe_command(present.to_str().unwrap()));

        let broken = fake_cli("broken", "exit 1");
        assert!(!probe_command(broken.to_str().unwrap()), "a command that fails is not usable");
        assert!(!probe_command("definitely-not-a-real-command"), "and one that is absent is not there");
    }

    #[test]
    fn an_agent_that_cannot_list_leaves_the_picker_empty_rather_than_failing() {
        // Offline or signed out is not a broken harness. Failing here would take
        // the whole picker down over a model list nobody asked for yet.
        assert!(models_from_listing(false, "anthropic/claude\nopenai/gpt").is_empty());
    }

    #[test]
    fn a_model_listing_is_one_id_per_line_with_the_blanks_dropped() {
        let models = models_from_listing(true, "  anthropic/claude  \n\n openai/gpt \n   \n");
        assert_eq!(models.len(), 2, "blank lines are not models: {models:?}");
        assert_eq!(models[0].value, "anthropic/claude", "trimmed");
        assert_eq!(models[0].label, models[0].value, "no prettier name is on offer, so the id is the label");
        assert_eq!(models[1].value, "openai/gpt");
        assert!(models_from_listing(true, "").is_empty());
    }

    #[test]
    fn only_an_allow_option_counts_as_permission() {
        // The agent offers a list and we pick one. Choosing a reject as though
        // it were an allow would silently deny every tool call; the reverse
        // would approve them without asking.
        use acp::schema::PermissionOptionKind as Kind;
        assert!(is_allow(&Kind::AllowOnce));
        assert!(is_allow(&Kind::AllowAlways));
        assert!(!is_allow(&Kind::RejectOnce));
        assert!(!is_allow(&Kind::RejectAlways));
    }

    #[test]
    fn cancelling_a_run_is_visible_to_whoever_asks_afterwards() {
        let run = AcpRun { cancel: Arc::new(AtomicBool::new(false)) };
        assert!(!run.was_cancelled());
        run.cancel().expect("cancel");
        assert!(run.was_cancelled(), "a stopped run says so");
    }

    /// An ACP agent that speaks just enough of the protocol to complete one
    /// run: the three requests a prompt needs, plus a streamed reply.
    ///
    /// It echoes back each request's id rather than assuming a sequence — the
    /// client sends UUIDs, and a reply carrying the wrong id is simply never
    /// matched, so the run hangs with no error to explain it.
    #[cfg(unix)]
    fn fake_acp_agent(reply: &str) -> std::path::PathBuf {
        let path = crate::test_support::fixture_dir("acpagent").join("agent");
        let script = format!(
            r#"while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":"\([^"]*\)".*/\1/p')
  case "$line" in
    *'"initialize"'*)
      printf '{{"jsonrpc":"2.0","id":"%s","result":{{"protocolVersion":1,"agentCapabilities":{{}},"authMethods":[]}}}}\n' "$id" ;;
    *'"session/new"'*)
      printf '{{"jsonrpc":"2.0","id":"%s","result":{{"sessionId":"ses-1"}}}}\n' "$id" ;;
    *'"session/prompt"'*)
      printf '%s\n' '{{"jsonrpc":"2.0","method":"session/update","params":{{"sessionId":"ses-1","update":{{"sessionUpdate":"agent_message_chunk","content":{{"type":"text","text":"{reply}"}}}}}}}}'
      printf '{{"jsonrpc":"2.0","id":"%s","result":{{"stopReason":"end_turn"}}}}\n' "$id" ;;
  esac
done
"#
        );
        crate::test_support::install_script(&path, &script);
        path
    }

    #[cfg(unix)]
    fn collect_run(agent: &std::path::Path) -> Vec<RunEvent> {
        use std::sync::atomic::AtomicBool as Flag;
        use std::sync::Mutex;

        let harness = AcpHarness::custom(AcpHarnessConfig {
            id: "fake".to_owned(),
            display_name: "Fake".to_owned(),
            command: agent.to_string_lossy().into_owned(),
            args: Vec::new(),
            install_hint: None,
        });
        let events: Arc<Mutex<Vec<RunEvent>>> = Arc::default();
        let sink = Arc::clone(&events);
        let done = Arc::new(Flag::new(false));
        let flag = Arc::clone(&done);
        let handle = harness
            .start(
                RunRequest {
                    run_id: "acp-run".to_owned(),
                    prompt: "hello".to_owned(),
                    cwd: Some(std::env::temp_dir()),
                    mode: RunMode::Ask,
                    ..Default::default()
                },
                Arc::new(move |event| {
                    if matches!(event, RunEvent::Exited { .. }) {
                        flag.store(true, Ordering::SeqCst);
                    }
                    sink.lock().unwrap().push(event);
                }),
            )
            .expect("the run should start");
        for _ in 0..400 {
            if done.load(Ordering::SeqCst) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        let _ = handle;
        let out = events.lock().unwrap().clone();
        out
    }

    #[cfg(unix)]
    #[test]
    fn a_run_against_a_real_acp_agent_streams_its_reply_and_finishes() {
        // Everything under `run_acp` — the handshake, the session, the prompt,
        // and the notification translation — only happens together. Its pieces
        // being tested says nothing about whether the sequence works.
        let agent = fake_acp_agent("the answer");
        let events = collect_run(&agent);

        assert!(
            events.iter().any(|e| matches!(e, RunEvent::Started { .. })),
            "the run announces itself: {events:?}"
        );
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                RunEvent::Text { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, "the answer", "the agent's reply reaches the caller: {events:?}");
        assert!(
            matches!(events.last(), Some(RunEvent::Exited { .. })),
            "and exactly one Exited ends it: {events:?}"
        );
        let _ = std::fs::remove_dir_all(agent.parent().unwrap());
    }

    #[test]
    fn generic_acp_agent_lists_no_models_without_shelling_out() {
        // A `custom` agent has no model_control, so list_models() short-circuits
        // to empty — it never spawns the command (here a bogus one) — and the
        // host hides the picker on the *absence* of models.
        let harness = AcpHarness::custom(AcpHarnessConfig {
            id: "x".to_owned(),
            display_name: "X".to_owned(),
            command: "definitely-not-a-real-command".to_owned(),
            args: vec!["acp".to_owned()],
            install_hint: None,
        });
        assert!(harness.list_models().expect("ok").is_empty());
        let caps = harness.features();
        assert!(caps.models.is_empty());
        assert!(caps.custom_model, "ACP agents accept a free-text model");
    }

    #[test]
    fn write_model_config_emits_field_and_model_keyed_by_run_id() {
        let path = write_model_config("run-abc", "model", "opencode/big-pickle")
            .expect("writes the config file");
        assert!(
            path.to_string_lossy().contains("run-abc"),
            "temp file is keyed by run_id: {path:?}"
        );
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read back"))
                .expect("valid JSON");
        assert_eq!(json["model"], "opencode/big-pickle");
        let _ = std::fs::remove_file(&path);
    }
}
