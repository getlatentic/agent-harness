//! Command bob with a prompt and stream its raw stdout.
//!
//! `cargo run --example run_bob`
//! (requires `bob` on PATH + a key: `BOBSHELL_API_KEY` in the env, or one
//! stored in the OS keychain via `bob_rs::write_api_key`).
//!
//! bob-rs hands you bob's stdout RAW (one JSON object per line from its
//! `--output-format stream-json`). For a *normalized* event stream, use the
//! `agent-harness` crate, whose `bob` adapter parses these for you.

use std::sync::mpsc::sync_channel;

use bob_rs::{spawn_bob, BobApprovalMode, BobChatMode, BobError, ProcessEvent, RunBobOptions};

fn main() -> Result<(), BobError> {
    let (tx, rx) = sync_channel::<ProcessEvent>(256);

    let _handle = spawn_bob(
        RunBobOptions {
            prompt: "List the files in this directory.".into(),
            chat_mode: BobChatMode::Ask,
            approval_mode: BobApprovalMode::Default,
            max_coins: 30,
            cwd: None,            // defaults to the current directory
            bob_executable: None, // defaults to `bob` on PATH
            extra_args: Vec::new(),
            resume: None, // Some(session_id) to continue a prior run
        },
        "demo".into(),
        move |ev| {
            let _ = tx.send(ev);
        }, // FnMut + Send + Sync + Clone
    )?;

    for ev in rx {
        match ev {
            ProcessEvent::Stdout { line, .. } => println!("{line}"),
            ProcessEvent::Stderr { line, .. } => eprintln!("{line}"),
            ProcessEvent::Error { message, .. } => eprintln!("error: {message}"),
            ProcessEvent::Exited { exit_code, .. } => {
                eprintln!("(exit {exit_code:?})");
                break;
            }
            ProcessEvent::Started { .. } => {}
            // `ProcessEvent` is #[non_exhaustive]; ignore any future variant.
            _ => {}
        }
    }
    Ok(())
}
