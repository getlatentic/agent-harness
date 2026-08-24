//! Live check of the RunEvent enrichment against the REAL agent CLIs.
//!
//! Runs a harness with a tool-triggering prompt and prints every `RunEvent`,
//! so you can watch Session / Usage / ToolStart.input / ToolEnd.output arrive
//! from the actual CLI wire format — not the synthetic JSON the unit tests
//! use. (Claude's tool *input* is expected to be `None`: it streams args
//! incrementally; codex delivers `command` inline, so it shows input.)
//!
//! `cargo run --example verify_enrichment -- [claude|codex] ["prompt"]`
//! (requires the CLI installed + signed in.)

use std::sync::{mpsc::sync_channel, Arc};

use harness::{default_registry, ReasoningEffort, RunCallback, RunEvent, RunRequest, RunTuning};

fn main() -> Result<(), String> {
    let id = std::env::args().nth(1).unwrap_or_else(|| "claude".to_owned());
    let prompt = std::env::args().nth(2).unwrap_or_else(|| default_prompt(&id));

    let reg = default_registry();
    let h = reg
        .by_id(&id)
        .ok_or_else(|| format!("unknown/disabled harness: {id}"))?;

    // Cheapest viable settings (per the "don't burn tokens" rule).
    let tuning = match id.as_str() {
        "claude" => RunTuning { model: Some("haiku".to_owned()), ..RunTuning::default() },
        "codex" => RunTuning { effort: Some(ReasoningEffort::Low), ..RunTuning::default() },
        _ => RunTuning::default(),
    };

    let (tx, rx) = sync_channel::<RunEvent>(256);
    let on_event: RunCallback = Arc::new(move |ev| {
        let _ = tx.send(ev);
    });

    eprintln!("── running {id} ──");
    let _handle = h.start(
        RunRequest {
            run_id: "verify".into(),
            prompt,
            cwd: Some(std::env::current_dir().map_err(|e| e.to_string())?),
            tuning,
            ..Default::default()
        },
        on_event,
    )
    .map_err(|e| e.to_string())?;

    let (mut session, mut tool_input, mut tool_output, mut usage) = (false, false, false, false);
    for ev in rx {
        print_event(&ev);
        match &ev {
            RunEvent::Session { session_id, model, .. } => {
                session = session_id.is_some() || model.is_some();
            }
            RunEvent::ToolStart { raw_input, .. } => tool_input |= raw_input.is_some(),
            RunEvent::ToolEnd { content, .. } => tool_output |= content.is_some(),
            RunEvent::Usage { input_tokens, output_tokens, total_tokens, .. } => {
                usage = input_tokens.is_some() || output_tokens.is_some() || total_tokens.is_some();
            }
            RunEvent::Exited { .. } => break,
            _ => {}
        }
    }
    eprintln!(
        "\n── enrichment seen from real {id}: session={session} tool_input={tool_input} tool_output={tool_output} usage={usage}"
    );
    Ok(())
}

fn default_prompt(id: &str) -> String {
    match id {
        "codex" => "Run the command `ls` to list the current directory, then tell me how many entries there are in one short sentence.".to_owned(),
        _ => "Use the Read tool to read the file Cargo.toml, then tell me the workspace resolver version in one short sentence.".to_owned(),
    }
}

fn print_event(ev: &RunEvent) {
    let line = match ev {
        RunEvent::Text { delta, .. } => format!("Text({:?})", trunc(delta, 40)),
        RunEvent::Thinking { delta, .. } => format!("Thinking({:?})", trunc(delta, 40)),
        RunEvent::Session { session_id, model, .. } => {
            format!("Session(session_id={session_id:?}, model={model:?})")
        }
        RunEvent::ToolStart { title, raw_input, .. } => {
            format!("ToolStart(title={title:?}, input={:?})", raw_input.as_deref().map(|s| trunc(s, 60)))
        }
        RunEvent::ToolEnd { ok, content, .. } => {
            format!("ToolEnd(ok={ok}, output={:?})", content.as_deref().map(|s| trunc(s, 60)))
        }
        RunEvent::Usage { input_tokens, output_tokens, total_tokens, .. } => {
            format!("Usage(in={input_tokens:?}, out={output_tokens:?}, total={total_tokens:?})")
        }
        other => format!("{other:?}"),
    };
    eprintln!("  {line}");
}

fn trunc(s: &str, n: usize) -> String {
    let s = s.replace('\n', "\\n");
    if s.chars().count() > n {
        format!("{}…", s.chars().take(n).collect::<String>())
    } else {
        s
    }
}
