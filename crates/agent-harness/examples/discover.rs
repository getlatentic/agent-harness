//! What is on this machine, and can it run?
//!
//! `cargo run --example discover --all-features`
//!
//! Every harness answers [`Harness::readiness`] the same way — installed,
//! version, signed in — so a host renders one table rather than special-casing
//! each agent. Nothing here spends tokens: each probe runs the CLI's own
//! `--version` and auth-status subcommand.
//!
//! It doubles as the check that resolution still works. These CLIs install to
//! genuinely different places — an official installer under `~/.local/bin`,
//! npm-global under an nvm node, Homebrew under `/opt/homebrew/bin` — and
//! finding each of them is what `ResolveCli` exists for.

use harness::{AcpHarness, AcpHarnessConfig, Claude, Codex, Harness, InstallHint};

fn main() {
    let agents: Vec<Box<dyn Harness>> = vec![
        Box::new(Claude::new()),
        Box::new(Codex::new()),
        Box::new(AcpHarness::opencode()),
        Box::new(AcpHarness::custom(AcpHarnessConfig {
            id: "gemini".into(),
            display_name: "Gemini".into(),
            command: "gemini".into(),
            args: vec!["--experimental-acp".into()],
            install_hint: Some(InstallHint::url("https://github.com/google-gemini/gemini-cli")),
        })),
    ];

    println!("{:<10} {:<10} {:<8} {:<24} MODELS", "AGENT", "INSTALLED", "SIGNED", "VERSION");
    for agent in &agents {
        let info = agent.info();
        let ready = agent.readiness();
        let models = agent.list_models().map(|m| m.len()).unwrap_or(0);
        println!(
            "{:<10} {:<10} {:<8} {:<24} {}",
            info.id,
            yes_no(ready.installed),
            yes_no(ready.auth_configured),
            ready.version.as_deref().unwrap_or("—"),
            models,
        );
        // A harness that cannot run says why, and where to get it — a missing
        // agent should be a next step rather than a dead end.
        if !ready.ready {
            if let Some(error) = &ready.error {
                println!("           ↳ {error}");
            }
            if !ready.installed {
                if let Some(hint) = info.install_hint {
                    println!("           ↳ get it from {}", hint.url);
                }
            }
        }
    }
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}
