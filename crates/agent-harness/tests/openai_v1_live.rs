//! The generic OpenAI-compatible `/v1` route against **real hosted providers**.
//!
//! `tests/ollama_route.rs` covers the native `/api/*` path Ollama gets. This
//! covers the other branch — the `OpenAi` dialect — which every non-Ollama
//! provider takes: DeepSeek, OpenRouter, vLLM, LM Studio, llama-server.
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
                prompt: "Reply with the single word: pong".to_owned(),
                cwd: Some(std::env::temp_dir()),
                // Read-only: a live test must never be able to touch the disk.
                mode: RunMode::Ask,
                tools: harness::ToolAccess::Default,
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

    for _ in 0..3600 {
        if done.load(Ordering::SeqCst) {
            break;
        }
        thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(done.load(Ordering::SeqCst), "{provider}: run did not finish within 3 minutes");

    let events = events.lock().unwrap().clone();
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
