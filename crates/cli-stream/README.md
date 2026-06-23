# cli-stream

A small, generic **streaming subprocess engine** for Rust: spawn a child
process, stream its stdout/stderr line-by-line through a callback, and cancel
it cleanly (SIGTERM → SIGKILL). Plus an optional `PATH` helper so a packaged
GUI app (a macOS `.app` launched from Finder, with a minimal environment) can
still find CLIs installed via nvm / Homebrew / asdf / volta.

No domain knowledge — just process streaming. Useful to anything that drives a
child CLI: a task runner, a build/test wrapper, a TUI, a desktop app shelling
out to tools.

```toml
cli-stream = "0.4"
```

```rust
use cli_stream::{spawn_streaming, ProcessEvent};

let handle = spawn_streaming(
    "git".into(),
    vec!["log".into(), "--oneline".into(), "-5".into()],
    vec![],                       // extra env
    std::env::current_dir()?,     // cwd
    "git-log".to_string(),        // a run id echoed on every event
    |event| match event {
        ProcessEvent::Stdout { line, .. } => println!("{line}"),
        ProcessEvent::Stderr { line, .. } => eprintln!("{line}"),
        ProcessEvent::Exited { exit_code, .. } => println!("done: {exit_code:?}"),
        _ => {}
    },
)?;
// handle.cancel()?;  // SIGTERM, then SIGKILL after a grace period
```

## What you get

- **`spawn_streaming(program, args, env, cwd, run_id, callback)`** → a
  `ProcessHandle`, streaming `ProcessEvent`s (Started / Stdout / Stderr / Error
  / Exited) from reader threads. Errors are a typed `StreamError` — `Spawn`
  carries the underlying `io::Error`, so "not on PATH" is distinguishable from
  "permission denied".
- **`ProcessHandle::cancel()`** — SIGTERM, then SIGKILL after a grace period.
  Polls `try_wait` (no blocking `wait` under a lock), so it terminates a
  *running* child promptly, not just on its next line of output.
- **`augmented_node_path()`** — resolves the user's real `PATH` from their login
  shell (cached once, with a hardcoded fallback), so a Finder-launched `.app`
  finds `node` and other CLIs instead of mis-reporting them as "not installed".
- **`InstallEvent`** — a sibling event shape for streamed install/setup output.

Cross-platform: cancel uses SIGTERM/SIGKILL on Unix and `TerminateProcess` on
Windows.

## License

MIT OR Apache-2.0.
