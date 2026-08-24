//! Make sure a harness is installed + signed in before you run it.
//!
//! `cargo run --example setup`
//!
//! `readiness()` reports whether the CLI is installed and authenticated. This
//! crate never installs an agent — when one is missing, `manifest().install_hint`
//! says where to get it, and the host shows that to the user. `login()` still
//! runs the CLI's own OAuth (`claude auth login`, which opens the browser),
//! because that is the agent authenticating itself, not us installing it.

use std::sync::Arc;

use harness::{Claude, Harness, Error, InstallEvent};

fn main() -> Result<(), Error> {
    let claude = Claude::new();

    let r = claude.readiness();
    if !r.installed {
        // Not our job to install it — tell the user where it comes from.
        if let Some(hint) = claude.info().install_hint {
            eprintln!("Claude Code isn't installed. Get it from {}", hint.url);
            if let Some(command) = hint.command {
                eprintln!("  {command}");
            }
        }
        return Ok(());
    }

    if !r.auth_configured {
        // A logger for the login progress stream.
        let log: harness::InstallCallback = Arc::new(|ev| match ev {
            InstallEvent::Step { text } => eprintln!("• {text}"),
            InstallEvent::Stdout { text } | InstallEvent::Stderr { text } => eprintln!("  {text}"),
            InstallEvent::Done { ok, .. } => eprintln!("done (ok={ok})"),
        });
        // Fallible calls return the typed `Error`; `?` propagates it.
        claude.login(log)?; // `claude auth login` — opens the browser
    }

    println!("ready: {}", claude.readiness().ready);
    Ok(())
}
