//! **Host tools** — hand the agent a function from your own program.
//!
//! ```text
//! cargo run --example host_tools --features claude
//! # needs `claude` installed and signed in
//! ```
//!
//! The other examples give the agent a prompt and read its answer. This one
//! gives it a *tool* as well: `stock_level`, a function in this process backed
//! by a map the agent cannot see. Claude Code is told the tool exists, decides to
//! call it, and the call arrives back here — over the CLI's control protocol, with
//! no server process and no port — where the closure runs and its answer goes
//! back. The `ToolStart` / `ToolEnd` events show the round trip; the answer shows
//! it was used.

use std::collections::HashMap;

use harness::{Claude, Error, FnTool, Harness, RunEvent, RunRequest, ToolServer};
use serde_json::json;

fn main() -> Result<(), Error> {
    let stock: HashMap<&str, u32> = HashMap::from([("A-7", 42), ("B-2", 0), ("C-9", 1500)]);
    let inventory = ToolServer::new("shop").with_tool(
        FnTool::new(
            "stock_level",
            "How many units of a SKU are on hand right now. SKUs look like `A-7`.",
            json!({
                "type": "object",
                "properties": { "sku": { "type": "string", "description": "the SKU to look up" } },
                "required": ["sku"],
            }),
            move |args| {
                let sku = args["sku"].as_str().ok_or("sku must be a string")?;
                stock
                    .get(sku)
                    .map(|units| format!("{units} units of {sku} on hand"))
                    .ok_or_else(|| format!("no SKU {sku}"))
            },
        )
        .as_read_only(),
    );

    let claude = Claude::new().with_tool_server(inventory);
    let readiness = claude.readiness();
    if !readiness.ready {
        eprintln!("claude is not ready: {}", readiness.error.unwrap_or_default());
        return Ok(());
    }

    let (_handle, events) = claude.run(RunRequest {
        prompt: "How many units of SKU A-7 and SKU B-2 do we have? Use the stock tool; answer in one line.".into(),
        cwd: Some(std::env::temp_dir()),
        ..Default::default()
    })?;

    for event in events {
        match event {
            RunEvent::ToolStart { title, raw_input, .. } => {
                eprintln!("→ {title} {}", raw_input.unwrap_or_default());
            }
            RunEvent::ToolEnd { ok, content, .. } => {
                eprintln!("← {} {}", if ok { "ok" } else { "failed" }, content.unwrap_or_default());
            }
            RunEvent::Text { delta, .. } => print!("{delta}"),
            RunEvent::Error { message, .. } => eprintln!("error: {message}"),
            RunEvent::Exited { exit_code, .. } => {
                println!();
                eprintln!("exited: {exit_code:?}");
            }
            _ => {}
        }
    }
    Ok(())
}
