//! **Gemini CLI** over ACP.
//!
//! ```text
//! cargo run --example gemini --features acp   # needs `gemini` on PATH
//! ```
//!
//! There is no Gemini adapter. `AcpHarness::custom` takes the command that
//! launches any agent in ACP mode, so a new ACP agent is a config literal
//! rather than a code change. Compare with `acp.rs`, which uses the
//! `opencode()` shorthand for exactly the same thing.

use harness::{
    AcpHarness, AcpHarnessConfig, Harness, HarnessError, InstallHint, RunEvent,
    RunRequest,
};

fn main() -> Result<(), HarnessError> {
    let gemini = AcpHarness::custom(AcpHarnessConfig {
        id: "gemini".into(),
        display_name: "Gemini".into(),
        command: "gemini".into(),
        args: vec!["--experimental-acp".into()],
        // Where a user gets it when the command is missing. This crate never
        // installs anything; it only says where to look.
        install_hint: Some(InstallHint::url("https://github.com/google-gemini/gemini-cli")),
    });

    let readiness = gemini.readiness();
    if !readiness.ready {
        eprintln!("{}", readiness.error.unwrap_or_default());
        if let Some(hint) = gemini.info().install_hint {
            eprintln!("Get it from {}", hint.url);
        }
        return Ok(());
    }

    let (_handle, rx) = gemini.run(RunRequest::new("demo", "In one sentence, what is the Agent Client Protocol?"))?;

    for ev in rx {
        match ev {
            RunEvent::Text { delta, .. } => print!("{delta}"),
            RunEvent::Thinking { delta, .. } => eprint!("{delta}"),
            RunEvent::ToolStart { title, .. } => eprintln!("\n[tool] {title}"),
            RunEvent::Error { message, .. } => eprintln!("\n[error] {message}"),
            RunEvent::Exited { .. } => break,
            _ => {}
        }
    }
    println!();
    Ok(())
}
