//! End-to-end I/O test of the framework's run path against a **real stub
//! process** — no mocking of the OS. A minimal [`Harness`] spawns `sh -c …`
//! through the engine, and we assert the normalized [`RunEvent`] stream that
//! comes back over [`Harness::run`]. This exercises the whole public
//! chain — spawn → stream stdout → parse → normalize → channel close — plus
//! the cancel path, with actual process I/O rather than an in-memory mock.
//!
//! Unix-only: it drives `sh` / `printf` / `sleep`, and cancel is signal-based.
#![cfg(unix)]

use std::time::{Duration, Instant};

use harness::{
    normalize_process_event, Command, CredentialSpec, Harness,
    Error, Info, Readiness, ParsedLine, RunCallback,
    RunEvent, RunHandle, RunMode, RunRequest, RunTuning,
};

/// The smallest real process-backed harness: runs an arbitrary `sh -c
/// <script>` and turns each stdout line into a `RunEvent::Text` via the same
/// `normalize_process_event` skeleton the built-in adapters use.
struct StubHarness {
    script: String,
}

impl Harness for StubHarness {
    fn info(&self) -> Info {
        Info {
            id: "stub".to_owned(),
            display_name: "Stub".to_owned(),
            description: "integration-test stub harness".to_owned(),
            install_hint: None,
        }
    }

    fn readiness(&self) -> Readiness {
        Readiness {
            harness_id: "stub".to_owned(),
            ready: true,
            installed: true,
            version: None,
            auth_configured: true,
            error: None,
            details: serde_json::Value::Null,
        }
    }

    fn start(&self, request: RunRequest, on_event: RunCallback) -> Result<RunHandle, Error> {
        let handle = Command::new("sh")
            .cwd(std::env::current_dir().unwrap_or_default())
            .run_id(request.run_id)
            .args(["-c", &self.script])
            .stream(move |event| {
                for ev in normalize_process_event(event, |line| ParsedLine {
                    text: Some(line.to_owned()),
                    ..Default::default()
                }) {
                    on_event(ev);
                }
            },
        )
        .map_err(Error::spawn)?;
        Ok(Box::new(handle))
    }

    fn credential(&self) -> CredentialSpec {
        CredentialSpec {
            label: "none".to_owned(),
            keychain_service: String::new(),
            keychain_account: String::new(),
            required: false,
        }
    }
}

fn request() -> RunRequest {
    RunRequest {
        run_id: "stub-run".to_owned(),
        prompt: String::new(),
        cwd: None,
        mode: RunMode::Ask,
        tuning: RunTuning::default(),
        resume: None,
        attachments: Vec::new(),
    }
}

#[test]
fn streams_normalized_events_then_closes() {
    let harness = StubHarness {
        script: "printf '%s\\n' alpha beta".to_owned(),
    };
    let (_handle, events) = harness.run(request()).expect("run");

    // Draining to completion proves the channel closed on its own when the
    // real process exited.
    let events: Vec<RunEvent> = events.into_iter().collect();

    assert!(
        matches!(events.first(), Some(RunEvent::Started { .. })),
        "stream should lead with Started, got {:?}",
        events.first()
    );
    assert!(
        matches!(
            events.last(),
            Some(RunEvent::Exited { exit_code: Some(0), cancelled: false, .. })
        ),
        "stream should close with Exited(0, not cancelled), got {:?}",
        events.last()
    );
    // The two printed lines arrive as Text, in order — the parse + normalize
    // path ran over real stdout.
    let text: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            RunEvent::Text { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, vec!["alpha", "beta"]);
}

#[test]
fn run_id_is_threaded_onto_every_event() {
    let harness = StubHarness {
        script: "printf '%s\\n' one".to_owned(),
    };
    let (_handle, events) = harness.run(request()).expect("run");
    for event in events {
        let id = match &event {
            RunEvent::Started { run_id }
            | RunEvent::Text { run_id, .. }
            | RunEvent::Exited { run_id, .. } => run_id.as_str(),
            other => panic!("unexpected event from stub: {other:?}"),
        };
        assert_eq!(id, "stub-run");
    }
}

#[test]
fn cancel_flows_through_run_and_flags_exited() {
    // A long sleeper we cancel almost immediately. Cancelling via the
    // RunHandle must terminate the real child and surface as
    // Exited(cancelled=true) on the run receiver. (Promptness of the
    // SIGTERM→SIGKILL is covered by cli-stream's own lifecycle test; here we
    // assert the cancel *path through the framework* works and the flag rides
    // the normalized stream.)
    let harness = StubHarness {
        script: "exec sleep 5".to_owned(),
    };
    let (handle, events) = harness.run(request()).expect("run");

    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(150));
        let _ = handle.cancel();
    });

    let started = Instant::now();
    let events: Vec<RunEvent> = events.into_iter().collect();
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "cancel should end the run well before the 5s sleep finishes"
    );
    assert!(
        matches!(events.last(), Some(RunEvent::Exited { cancelled: true, .. })),
        "cancelled run should close with Exited(cancelled=true), got {:?}",
        events.last()
    );
}
