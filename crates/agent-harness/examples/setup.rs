//! Make sure a harness is installed + signed in before you run it.
//!
//! `cargo run --example setup`
//!
//! `readiness()` reports whether the CLI is installed and authenticated;
//! `install()` installs it (npm for claude/codex)
//! and `login()` runs the CLI's own OAuth (`claude auth login` / `codex
//! login`, which opens the browser).

use std::sync::Arc;

use harness::{Claude, Harness, HarnessError, InstallEvent};

fn main() -> Result<(), HarnessError> {
    let claude = Claude::new();

    // A logger for the install/login progress stream.
    let log: harness::InstallCallback = Arc::new(|ev| match ev {
        InstallEvent::Step { text } => eprintln!("• {text}"),
        InstallEvent::Stdout { text } | InstallEvent::Stderr { text } => eprintln!("  {text}"),
        InstallEvent::Done { ok, .. } => eprintln!("done (ok={ok})"),
    });

    let r = claude.readiness();
    // Fallible calls return the typed `HarnessError`; `?` propagates it.
    if !r.installed {
        claude.install(Arc::clone(&log))?; // npm i -g @anthropic-ai/claude-code
    }
    if !r.auth_configured {
        claude.login(log)?; // `claude auth login` — opens the browser
    }

    println!("ready: {}", claude.readiness().ready);
    Ok(())
}
