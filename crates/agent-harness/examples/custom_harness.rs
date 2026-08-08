//! Bring your own harness — implement [`Harness`] by *composing* the published
//! building blocks, no fork required.
//!
//! `cargo run --example custom_harness`
//!
//! This toy harness "answers" by echoing the prompt: it spawns `printf` to
//! emit a couple of JSON "wire" lines, then decodes them with its own parser.
//! The point is the composition — it reuses the framework's engine
//! ([`spawn_streaming`]), the neutral types ([`ParsedLine`] / [`RunEvent`]),
//! and the canonical [`run_events_from_parsed`] expansion, and registers
//! itself in a [`Registry`] alongside the built-ins. A real provider swaps
//! `printf` for its CLI (or an HTTP API) and `parse_line` for that wire format.
//!
//! Two patterns this demonstrates:
//!   * **new provider** — `impl Harness` from scratch, reusing the pieces;
//!   * **stateful parser** — hold per-run state (here: announce the session
//!     once) and call `run_events_from_parsed` so the `ParsedLine → RunEvent`
//!     ordering stays canonical. (For a *stateless* parser, the one-liner
//!     `normalize_process_event(event, my_fn)` does the same without the Mutex.)

use std::path::PathBuf;
use std::sync::{mpsc::sync_channel, Arc, Mutex};

use harness::{
    run_events_from_parsed, spawn_streaming, CredentialSpec, Harness, HarnessCapabilities,
    HarnessError, HarnessInfo, HarnessReadiness, ParsedLine, ProcessEvent,
    RunCallback, RunEvent, RunHandle, RunMode, RunRequest, RunTuning, Registry, SessionInfo,
};
use serde_json::Value;

const ECHO_ID: &str = "echo";

/// A minimal third-party harness. Cheap to construct (holds config, not
/// connections), so the registry can hand out fresh instances.
struct EchoHarness;

impl Harness for EchoHarness {
    fn info(&self) -> HarnessInfo {
        HarnessInfo {
            id: ECHO_ID.to_owned(),
            display_name: "Echo".to_owned(),
            description: "A toy harness that echoes the prompt — a template for your own."
                .to_owned(),
            install_hint: None,
            capabilities: HarnessCapabilities {
                credential_required: false,
                previews_edits: false,
                models: Vec::new(),
                allows_custom_model: false,
                supports_effort: false,
                supports_max_turns: false,
                supports_login: false,
                supports_custom_instructions: false,
            },
        }
    }

    fn readiness(&self) -> HarnessReadiness {
        HarnessReadiness {
            harness_id: ECHO_ID.to_owned(),
            ready: true,
            installed: true,
            version: Some("0.0.0".to_owned()),
            auth_configured: true,
            error: None,
            details: Value::Null,
        }
    }

    fn credential(&self) -> CredentialSpec {
        CredentialSpec {
            label: "none".to_owned(),
            keychain_service: String::new(),
            keychain_account: String::new(),
            required: false,
        }
    }

    fn run(&self, request: RunRequest, on_event: RunCallback) -> Result<RunHandle, HarnessError> {
        // A real harness spawns its CLI here. We spawn `printf` to emit two
        // JSON lines: an init (→ Session) and the answer (→ Text).
        let answer = format!(r#"{{"text":"echo: {}"}}"#, request.prompt.replace('"', "'"));
        let args = vec![
            "%s\n".to_owned(),
            r#"{"type":"init","model":"echo-1"}"#.to_owned(),
            answer,
        ];
        let cwd = request
            .cwd
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

        // One parser per run. The engine's callback is `Fn + Send + Sync`
        // (invoked from reader threads), so per-run state lives behind an
        // `Arc<Mutex>` — the same shape the built-in codex adapter uses.
        let parser = Arc::new(Mutex::new(EchoParser::default()));
        let handle = spawn_streaming(
            PathBuf::from("printf"),
            args,
            Vec::new(),
            cwd,
            request.run_id,
            move |event| {
                let mut parser = parser.lock().expect("echo parser mutex");
                for ev in parser.on_process_event(event) {
                    (*on_event)(ev);
                }
            },
        )
        .map_err(HarnessError::spawn)?;
        Ok(Box::new(handle))
    }
}

/// A tiny stateful parser: announce the session once (from the init line),
/// then map each `{"text":…}` line to assistant text. Lifecycle events are
/// neutral; stdout is decoded here and expanded via [`run_events_from_parsed`]
/// so the event ordering matches every other harness.
#[derive(Default)]
struct EchoParser {
    announced: bool,
}

impl EchoParser {
    fn on_process_event(&mut self, event: ProcessEvent) -> Vec<RunEvent> {
        match event {
            ProcessEvent::Started { run_id } => vec![RunEvent::Started { run_id }],
            ProcessEvent::Exited {
                run_id,
                exit_code,
                cancelled,
            } => vec![RunEvent::Exited {
                run_id,
                exit_code,
                cancelled,
            }],
            ProcessEvent::Error { run_id, message } => vec![RunEvent::Error { run_id, message }],
            ProcessEvent::Stderr { .. } => Vec::new(),
            ProcessEvent::Stdout { run_id, line } => {
                run_events_from_parsed(&run_id, self.parse_line(&line))
            }
            // `ProcessEvent` is #[non_exhaustive]; ignore any future variant.
            _ => Vec::new(),
        }
    }

    fn parse_line(&mut self, line: &str) -> ParsedLine {
        let value = serde_json::from_str::<Value>(line.trim()).unwrap_or(Value::Null);
        // The first line announces the session (stateful: only once).
        if !self.announced {
            if let Some(model) = value.get("model").and_then(Value::as_str) {
                self.announced = true;
                return ParsedLine {
                    session: Some(SessionInfo {
                        session_id: None,
                        model: Some(model.to_owned()),
                    }),
                    ..ParsedLine::default()
                };
            }
        }
        if let Some(text) = value.get("text").and_then(Value::as_str) {
            return ParsedLine {
                text: Some(text.to_owned()),
                ..ParsedLine::default()
            };
        }
        ParsedLine::default()
    }
}

fn main() -> Result<(), String> {
    // Register your harness alongside (or instead of) the built-ins — no fork.
    let reg = Registry::new().register(EchoHarness);
    let h = reg.by_id(ECHO_ID).expect("registered");
    println!("harness: {} — {}", h.info().display_name, h.info().description);

    let (tx, rx) = sync_channel::<RunEvent>(64);
    let on_event: RunCallback = Arc::new(move |ev| {
        let _ = tx.send(ev);
    });
    let _handle = h.run(
        RunRequest {
            run_id: "demo".into(),
            prompt: "hello".into(),
            cwd: None,
            mode: RunMode::Ask,
            tuning: RunTuning::default(),
            resume: None,
            attachments: Vec::new(),
        },
        on_event,
    )
    .map_err(|e| e.to_string())?;

    // One normalized stream — same as for any built-in harness.
    for ev in rx {
        match ev {
            RunEvent::Session { model, .. } => println!("[session] model={model:?}"),
            RunEvent::Text { delta, .. } => println!("[answer] {delta}"),
            RunEvent::Exited { exit_code, .. } => {
                println!("[exited] {exit_code:?}");
                break;
            }
            other => println!("{other:?}"),
        }
    }
    Ok(())
}
