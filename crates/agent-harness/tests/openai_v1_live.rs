//! The generic OpenAI-compatible `/v1` route against **real hosted providers**.
//!
//! `tests/ollama_route.rs` covers the native `/api/*` path Ollama gets. This
//! covers the other branch — the `OpenAi` dialect — which every non-Ollama
//! provider takes: DeepSeek, OpenRouter, Bedrock, vLLM, LM Studio,
//! llama-server.
//!
//! Two kinds of test live here. `exercise` asserts the protocol of a plain
//! completion. `exercise_tools` asserts the tool loop, which the dialect had no
//! coverage of at all — and a guarantee proven on Ollama's native route is not
//! thereby proven here, as the `tool_calls` workaround in `wire.rs` (found
//! against a Bedrock gateway, not against Ollama) already showed.
//!
//! These providers are *configuration*, not code: each is an `OpenHarness` with
//! a different `base_url` + `api_key`. So one parameterised test covers all
//! of them, and adding a provider means adding a row — not an adapter.
//!
//! Ignored by default: each run costs real tokens and needs a key in the
//! environment. Run one with, e.g.:
//!
//! ```text
//! OPENROUTER_API_KEY=… cargo test --all-features --test openai_v1_live -- --ignored openrouter
//! ```

#![cfg(feature = "openai-compatible")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use harness::{ApiKey, 
    Harness, OpenHarness, OpenHarnessConfig, RunEvent, RunMode, RunRequest, RunTuning,
};

/// Start one run against a hosted provider and drain it to completion.
///
/// `patience` is in 50ms ticks: a plain completion answers in seconds, while a
/// tool round trip spends a turn on the call and another on the answer.
#[allow(clippy::too_many_arguments)]
fn live_run(
    provider: &str,
    base_url: &str,
    key_env: &str,
    model: &str,
    cwd: std::path::PathBuf,
    prompt: &str,
    tools: harness::ToolAccess,
    patience: usize,
) -> Vec<RunEvent> {
    let harness = OpenHarness::custom(OpenHarnessConfig {
        id: provider.to_owned(),
        display_name: provider.to_owned(),
        base_url: base_url.to_owned(),
        api_key: ApiKey::Env(key_env.to_owned()),
        ..Default::default()
    });

    let readiness = harness.readiness();
    assert!(readiness.ready, "{provider} not ready: {:?}", readiness.error);

    let events: Arc<Mutex<Vec<RunEvent>>> = Arc::default();
    let sink = Arc::clone(&events);
    let done = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&done);
    let _handle = harness
        .start(
            RunRequest {
                run_id: format!("live-{provider}"),
                prompt: prompt.to_owned(),
                cwd: Some(cwd),
                mode: RunMode::Ask,
                tools,
                tuning: RunTuning { model: Some(model.to_owned()), ..Default::default() },
                resume: None,
                attachments: Vec::new(),
            },
            Arc::new(move |event| {
                if matches!(event, RunEvent::Exited { .. } | RunEvent::Error { .. }) {
                    flag.store(true, Ordering::SeqCst);
                }
                sink.lock().unwrap().push(event);
            }),
        )
        .unwrap_or_else(|e| panic!("{provider}: run failed to start: {e}"));

    for _ in 0..patience {
        if done.load(Ordering::SeqCst) {
            break;
        }
        thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(done.load(Ordering::SeqCst), "{provider}: run did not finish in time");
    let events = events.lock().unwrap().clone();
    events
}

/// Drive one hosted provider through a full run and assert the *protocol*.
///
/// Never asserts what the model said: these are real models whose wording is
/// not a contract. What must hold is that text streams back, tokens are
/// reported, and the run terminates.
fn exercise(provider: &str, base_url: &str, key_env: &str, model: &str) {
    let Ok(key) = std::env::var(key_env) else {
        panic!("{provider}: set {key_env} to run this test");
    };
    assert!(!key.trim().is_empty(), "{provider}: {key_env} is empty");

    let events = live_run(
        provider,
        base_url,
        key_env,
        model,
        std::env::temp_dir(),
        "Reply with the single word: pong",
        harness::ToolAccess::Default,
        3600,
    );
    if let Some(RunEvent::Error { message, .. }) =
        events.iter().find(|e| matches!(e, RunEvent::Error { .. }))
    {
        panic!("{provider} errored: {message}");
    }
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            RunEvent::Text { delta, .. } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert!(!text.trim().is_empty(), "{provider}: no text streamed back");
    assert!(
        events.iter().any(|e| matches!(e, RunEvent::Usage { .. })),
        "{provider}: a hosted provider must report token usage"
    );
    assert!(
        events.iter().any(|e| matches!(e, RunEvent::Exited { .. })),
        "{provider}: run must terminate"
    );
    println!("{provider} replied: {}", text.trim());
}

#[test]
#[ignore = "live: costs tokens, needs DEEPSEEK_API_KEY"]
fn deepseek_v1_route() {
    exercise("deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-v4-flash");
}

#[test]
#[ignore = "live: costs tokens, needs OPENROUTER_API_KEY"]
fn openrouter_v1_route() {
    exercise(
        "openrouter",
        "https://openrouter.ai/api",
        "OPENROUTER_API_KEY",
        "openai/gpt-oss-120b",
    );
}

/// A token no model can guess and that appears nowhere in the prompt, so it
/// reaches an answer only by way of the file.
const PLANTED: &str = "quartzine-80417";

/// The tool loop on the hosted `/v1` dialect, both arms.
///
/// `ollama_route.rs` covers this contrast on the native `/api/chat` branch.
/// This is the other branch, and the one every hosted provider takes — the
/// branch a Bedrock gateway was already found to validate more strictly than
/// Ollama does (see `wire.rs`, `tool_calls` missing `type`). A guarantee that
/// holds on one dialect is not thereby proven on the other.
///
/// Offered arm first: it is what makes the withheld arm evidence. A model that
/// never calls a tool passes "nothing ran" for the wrong reason.
fn exercise_tools(provider: &str, base_url: &str, key_env: &str, model: &str) {
    let workspace = std::env::temp_dir().join(format!("agent-harness-live-{provider}"));
    let _ = std::fs::remove_dir_all(&workspace);
    std::fs::create_dir_all(&workspace).expect("workspace");
    std::fs::write(workspace.join("note.txt"), format!("code: {PLANTED}\n")).expect("planted file");
    let prompt = "Read the file note.txt in the working directory and reply with \
                  the code it contains, and nothing else.";

    let ran = |events: &[RunEvent]| -> Vec<String> {
        let started: std::collections::HashMap<&str, &str> = events
            .iter()
            .filter_map(|e| match e {
                RunEvent::ToolStart { tool_call_id, title, .. } => {
                    Some((tool_call_id.as_str(), title.as_str()))
                }
                _ => None,
            })
            .collect();
        events
            .iter()
            .filter_map(|e| match e {
                RunEvent::ToolEnd { tool_call_id, ok: true, .. } => {
                    Some(started.get(tool_call_id.as_str()).copied().unwrap_or("?").to_owned())
                }
                _ => None,
            })
            .collect()
    };
    let said = |events: &[RunEvent]| -> String {
        events
            .iter()
            .filter_map(|e| match e {
                RunEvent::Text { delta, .. } => Some(delta.as_str()),
                _ => None,
            })
            .collect()
    };
    // Named per arm: the two differ only in tool access, so a failure that does
    // not say which one is a failure you have to re-run to understand.
    let no_error = |arm: &str, events: &[RunEvent]| {
        if let Some(RunEvent::Error { message, .. }) =
            events.iter().find(|e| matches!(e, RunEvent::Error { .. }))
        {
            panic!("{provider} [{arm}] errored: {message}");
        }
    };

    let offered = live_run(
        provider,
        base_url,
        key_env,
        model,
        workspace.clone(),
        prompt,
        harness::ToolAccess::Default,
        12_000,
    );
    no_error("tools offered", &offered);
    assert!(
        !ran(&offered).is_empty(),
        "{provider}: offered tools and ran none; it cannot have read the file"
    );
    assert!(
        said(&offered).contains(PLANTED),
        "{provider}: the answer must carry what the tool read — ran {:?}, said {:?}",
        ran(&offered),
        said(&offered)
    );

    let withheld = live_run(
        provider,
        base_url,
        key_env,
        model,
        workspace.clone(),
        prompt,
        harness::ToolAccess::None,
        12_000,
    );
    // Not `no_error`: with nothing offered the model cannot do what the prompt
    // asks, so ending with nothing to say is the honest outcome — and 0.6
    // reports that rather than handing back an empty string as a success. Any
    // *other* error is still a failure.
    if let Some(RunEvent::Error { message, .. }) =
        withheld.iter().find(|e| matches!(e, RunEvent::Error { .. }))
    {
        assert!(
            message.contains("no answer"),
            "{provider} [tools withheld] errored: {message}"
        );
    }
    assert!(
        ran(&withheld).is_empty(),
        "{provider}: nothing was offered, yet these ran: {:?}",
        ran(&withheld)
    );
    assert!(
        !said(&withheld).contains(PLANTED),
        "{provider}: the file was reached with no tools offered: {:?}",
        said(&withheld)
    );
    let _ = std::fs::remove_dir_all(&workspace);
}

#[test]
#[ignore = "live: costs tokens, needs BEDROCK_MANTLE_API_KEY"]
fn bedrock_mantle_v1_route() {
    exercise(
        "bedrock-mantle",
        "https://bedrock-mantle.us-east-1.api.aws",
        "BEDROCK_MANTLE_API_KEY",
        "openai.gpt-oss-120b",
    );
}

/// Bedrock's OpenAI-compatible gateway. The strict `tool_calls` validation that
/// `wire.rs` works around was observed here, which is why the tool contrast is
/// worth running against this endpoint specifically rather than only Ollama.
#[test]
#[ignore = "live: costs tokens, needs BEDROCK_MANTLE_API_KEY"]
fn bedrock_mantle_tool_contrast() {
    exercise_tools(
        "bedrock-mantle",
        "https://bedrock-mantle.us-east-1.api.aws",
        "BEDROCK_MANTLE_API_KEY",
        "openai.gpt-oss-120b",
    );
}
