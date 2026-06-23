//! Run a prompt against the bob adapter and stream the normalized events.
//!
//! `BOBSHELL_API_KEY=… cargo run --example run_bob_harness --features bob`
//!
//! Doubles as the live regression check for the node-pairing fix: run it
//! with an incompatible node deliberately leading the PATH —
//!
//! ```sh
//! PATH="$HOME/.nvm/versions/node/v20.19.2/bin:$PATH" \
//!   BOBSHELL_API_KEY=… cargo run --example run_bob_harness --features bob
//! ```
//!
//! Pre-fix, bob (installed under a v24 nvm dir) was re-exec'd on the v20
//! node leading the inherited PATH and died with
//! `bad option: --disable-sigusr1` — surfacing only as "exited with code 9".
//! With the fix, `spawn_bob` resolves bob's absolute install dir and the
//! engine prepends it to the child PATH, so bob runs on its sibling node and
//! the answer streams normally regardless of the launch environment's PATH.

use harness::{Bob, Harness, HarnessError, RunEvent, RunMode, RunRequest, RunTuning};

fn main() -> Result<(), HarnessError> {
    let bob = Bob::new();

    let (_handle, rx) = bob.run_channel(RunRequest {
        run_id: "run-bob-example".into(),
        prompt: "Reply with exactly: pairing works".into(),
        cwd: None,
        mode: RunMode::Ask,
        tuning: RunTuning::default(),
        resume: None,
        attachments: Vec::new(),
    })?;

    for ev in rx {
        match ev {
            RunEvent::Text { delta, .. } => print!("{delta}"),
            RunEvent::Thinking { delta, .. } => eprint!("{delta}"),
            RunEvent::ToolStart { title, .. } => eprintln!("\n[tool] {title}"),
            RunEvent::Error { message, .. } => eprintln!("\n[error] {message}"),
            RunEvent::Exited { exit_code, .. } => {
                println!("\n[exited: {exit_code:?}]");
                break;
            }
            _ => {}
        }
    }
    Ok(())
}
