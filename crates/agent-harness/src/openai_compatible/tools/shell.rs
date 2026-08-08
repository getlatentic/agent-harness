//! Shell tool — `bash`. Runs a command through the platform's shell (`sh -c`
//! on unix, `cmd /C` on Windows) in the working directory,
//! draining both pipes on threads (so a chatty command can't deadlock on a full
//! pipe buffer) and polling for completion so a timeout or cooperative cancel
//! can kill it. Ported from OpenCode's `bash` design (MIT).

use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::ToolKind;

use super::{parse_args, safe_join, schema_for, Keep, Tool, ToolCtx, ToolOutcome};

/// Default `bash` timeout (OpenCode's 2 minutes); overridable per call.
const DEFAULT_BASH_TIMEOUT_MS: u64 = 120_000;

#[derive(Deserialize, JsonSchema)]
struct BashArgs {
    /// The shell command to run.
    command: String,
    /// A short (5-10 word) description of what the command does.
    #[allow(dead_code)] // advertised so the model states intent; not acted on here
    description: Option<String>,
    /// Run directory, relative to the working directory (default: the working directory).
    workdir: Option<String>,
    /// Timeout in milliseconds (default 120000); the process is killed if it exceeds it.
    timeout: Option<u64>,
}

pub(super) struct Bash;
impl Tool for Bash {
    fn id(&self) -> &str {
        "bash"
    }
    fn description(&self) -> &str {
        "Run a shell command in the working directory and return its combined \
         output. Prefer read/write/edit for file work; use bash for builds, \
         tests, git, and search."
    }
    fn parameters(&self) -> Value {
        schema_for::<BashArgs>()
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Execute
    }
    fn mutating(&self) -> bool {
        true
    }
    fn permission_subject(&self, args: &Value) -> Option<String> {
        args.get("command").and_then(Value::as_str).map(str::to_owned)
    }
    fn keep_output(&self) -> Keep {
        // A command's exit/error lines land at the end; keep both ends so the
        // model sees the trailing diagnostic, not just the leading output.
        Keep::HeadAndTail
    }
    fn execute(&self, args: &Value, ctx: &ToolCtx) -> ToolOutcome {
        let a: BashArgs = match parse_args(args) {
            Ok(a) => a,
            Err(o) => return o,
        };
        let timeout = a.timeout.filter(|&t| t > 0).unwrap_or(DEFAULT_BASH_TIMEOUT_MS);
        run_bash(ctx, &a.command, a.workdir.as_deref(), timeout)
    }
}

/// How a `bash` run ended — drives the result framing.
enum BashEnd {
    Exited(std::process::ExitStatus),
    TimedOut,
    Cancelled,
    WaitErr(String),
}

/// The command that hands `command` to the platform's shell. Unix gets
/// `sh -c`; Windows gets `cmd /C`, which is the closest equivalent that is
/// always present (PowerShell's startup cost and profile loading make it a
/// poor fit for the per-call shell of an agent loop).
fn shell_command(command: &str) -> Command {
    #[cfg(unix)]
    {
        let mut c = Command::new("sh");
        c.arg("-c").arg(command);
        c
    }
    #[cfg(windows)]
    {
        let mut c = Command::new("cmd");
        c.arg("/C").arg(command);
        c
    }
}

fn run_bash(ctx: &ToolCtx, command: &str, workdir: Option<&str>, timeout_ms: u64) -> ToolOutcome {
    let dir = match workdir {
        Some(w) => match safe_join(ctx.cwd, w) {
            Some(d) => d,
            None => return ToolOutcome::err(format!("workdir `{w}` escapes the working directory")),
        },
        None => ctx.cwd.to_path_buf(),
    };
    let mut child = match shell_command(command)
        .current_dir(&dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return ToolOutcome::err(format!("failed to run command: {e}")),
    };

    // Drain both pipes on threads so a chatty command can't deadlock on a full
    // OS pipe buffer while the main thread polls for completion.
    let (out_buf, out_h) = drain(child.stdout.take());
    let (err_buf, err_h) = drain(child.stderr.take());

    let start = Instant::now();
    let limit = Duration::from_millis(timeout_ms);
    let end = loop {
        match child.try_wait() {
            Ok(Some(status)) => break BashEnd::Exited(status),
            Ok(None) => {}
            Err(e) => break BashEnd::WaitErr(e.to_string()),
        }
        if ctx.cancel.load(Ordering::SeqCst) {
            let _ = child.kill();
            let _ = child.wait(); // reap
            break BashEnd::Cancelled;
        }
        if start.elapsed() >= limit {
            let _ = child.kill();
            let _ = child.wait();
            break BashEnd::TimedOut;
        }
        thread::sleep(Duration::from_millis(40));
    };

    // Wait for the readers only when the child exited on its own — then EOF is
    // guaranteed and joining yields the complete output. After a kill it is
    // not, so take what has arrived instead of blocking past the timeout.
    if matches!(end, BashEnd::Exited(_)) {
        let _ = out_h.join();
        let _ = err_h.join();
    }
    let stdout = taken(&out_buf);
    let stderr = taken(&err_buf);
    let mut body = stdout;
    if !stderr.trim().is_empty() {
        if !body.is_empty() && !body.ends_with('\n') {
            body.push('\n');
        }
        body.push_str(&format!("[stderr]\n{stderr}"));
    }

    match end {
        BashEnd::Exited(s) if s.success() => {
            ToolOutcome::ok(if body.is_empty() { "(no output)".to_owned() } else { body })
        }
        BashEnd::Exited(s) => {
            let code = s.code().map_or_else(|| "signal".to_owned(), |c| c.to_string());
            ToolOutcome::err(format!("(exit {code})\n{body}"))
        }
        BashEnd::TimedOut => {
            ToolOutcome::err(format!("(timed out after {timeout_ms}ms; process killed)\n{body}"))
        }
        BashEnd::Cancelled => ToolOutcome::err(format!("(cancelled; process killed)\n{body}")),
        BashEnd::WaitErr(e) => ToolOutcome::err(format!("(error waiting on command: {e})\n{body}")),
    }
}

/// Read a child pipe on its own thread, appending into a shared buffer as data
/// arrives. The buffer — rather than the thread's return value — is what makes
/// a timeout bounded: `read_to_string` only returns at EOF, and killing the
/// child does not necessarily close the pipe (a surviving grandchild can still
/// hold the write end), so waiting for the reader would wait out the very
/// process the timeout just killed.
fn drain<R: Read + Send + 'static>(pipe: Option<R>) -> (Arc<Mutex<Vec<u8>>>, thread::JoinHandle<()>) {
    let buf = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&buf);
    let handle = thread::spawn(move || {
        if let Some(mut r) = pipe {
            let mut chunk = [0u8; 8192];
            loop {
                match r.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => lock(&sink).extend_from_slice(&chunk[..n]),
                }
            }
        }
    });
    (buf, handle)
}

/// A poisoned buffer still holds the bytes read before the panic, and partial
/// output beats none in a diagnostic.
fn lock(buf: &Arc<Mutex<Vec<u8>>>) -> std::sync::MutexGuard<'_, Vec<u8>> {
    buf.lock().unwrap_or_else(|e| e.into_inner())
}

/// Take everything read so far, leaving the buffer empty.
fn taken(buf: &Arc<Mutex<Vec<u8>>>) -> String {
    let bytes = std::mem::take(&mut *lock(buf));
    String::from_utf8_lossy(&bytes).into_owned()
}
