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
    #[expect(dead_code, reason = "advertised so the model states intent; not acted on here")]
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
    let mut c = {
        let mut c = crate::hidden_command("sh");
        c.arg("-c").arg(command);
        c
    };
    #[cfg(windows)]
    let mut c = {
        let mut c = crate::hidden_command("cmd");
        c.arg("/C").arg(command);
        c
    };
    restrict_env(&mut c);
    c
}

/// What a shell the *model* drives inherits from this process.
///
/// Deny by default, and name the exceptions. The obvious alternative — strip
/// anything whose name looks like a secret — is wrong in both directions. It
/// removes `TOKENIZERS_PARALLELISM` and `PASSWORD_STORE_DIR`, which are not
/// secrets, while passing `DATABASE_URL` (which carries `user:pass@host`),
/// `AWS_ACCESS_KEY_ID`, `GH_PAT`, and `SSH_AUTH_SOCK` — a live handle to the
/// user's SSH agent — straight through.
///
/// The two failure modes are not symmetric. Withholding something needed
/// breaks a command in the open, and it can be added back. Passing something
/// secret leaks it silently. So this keeps only what a shell needs in order to
/// function at all, and anything else is the host's explicit decision.
const ALLOWED_ENV: &[&str] = &[
    // Without these a shell cannot find a binary or read its own config.
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    // Text handling: the wrong locale mangles any non-ASCII output.
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    // Somewhere to write.
    "TMPDIR",
    "TMP",
    "TEMP",
    "TZ",
    // Windows: a process that cannot see SystemRoot generally fails to start.
    "SystemRoot",
    "SystemDrive",
    "windir",
    "COMSPEC",
    "PATHEXT",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "APPDATA",
    "LOCALAPPDATA",
    "ProgramFiles",
    "ProgramData",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
];

/// Case-insensitive: Windows environment names are, and `Path` vs `PATH`
/// differs between shells.
fn is_allowed(name: &str) -> bool {
    ALLOWED_ENV.iter().any(|allowed| allowed.eq_ignore_ascii_case(name))
}

/// Replace the child's environment with the allowed subset.
fn restrict_env(command: &mut Command) {
    let keep: Vec<(std::ffi::OsString, std::ffi::OsString)> = std::env::vars_os()
        .filter(|(name, _)| is_allowed(&name.to_string_lossy()))
        .collect();
    command.env_clear();
    for (name, value) in keep {
        command.env(name, value);
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

#[cfg(test)]
mod env_tests {
    use super::*;

    fn shell_sees(var: &str) -> String {
        let script = if cfg!(windows) { format!("echo %{var}%") } else { format!("echo ${var}") };
        let out = shell_command(&script).output().unwrap();
        String::from_utf8_lossy(&out.stdout).trim().to_owned()
    }

    #[test]
    fn a_shell_call_cannot_read_the_host_key() {
        // The exposure this closes: the host holds a provider key, and the
        // model asks the shell to echo it back.
        let _env = crate::test_env::set("OPENROUTER_API_KEY", "sk-or-v1-SENTINEL");
        assert!(!shell_sees("OPENROUTER_API_KEY").contains("SENTINEL"));
    }

    #[test]
    fn the_cases_a_secret_name_denylist_missed() {
        // Each of these passed a name-shaped denylist. A URL carries its own
        // credentials; an agent socket is a live handle to the user's keys.
        for (name, value) in [
            ("DATABASE_URL", "postgres://user:SENTINEL@host/db"),
            ("AWS_ACCESS_KEY_ID", "AKIASENTINEL"),
            ("GH_PAT", "ghp_SENTINEL"),
            ("SSH_AUTH_SOCK", "/tmp/ssh-SENTINEL/agent.1"),
            ("NPM_AUTH", "SENTINEL"),
        ] {
            let _env = crate::test_env::set(name, value);
            assert!(!shell_sees(name).contains("SENTINEL"), "{name} leaked");
        }
    }

    #[test]
    fn the_cases_a_secret_name_denylist_wrongly_stripped() {
        // These are not secrets, and a tool that needs them must still get
        // them. A denylist removed all three for containing TOKEN/PASSWORD.
        for (name, value) in [
            ("TOKENIZERS_PARALLELISM", "false"),
            ("PASSWORD_STORE_DIR", "/home/x/.password-store"),
            ("TOKEN_BUDGET", "4096"),
        ] {
            let _env = crate::test_env::set(name, value);
            // Still withheld — but by a rule that is honest about it: anything
            // not named is withheld, secret or not. The host adds what it needs.
            assert!(!is_allowed(name));
        }
    }

    #[test]
    fn a_shell_call_still_works() {
        // Deny-by-default is only viable if a shell can still run. Without
        // PATH there is no `echo` to find.
        assert!(!shell_sees("PATH").is_empty());
        assert!(is_allowed("HOME") && is_allowed("TMPDIR"));
    }

    #[test]
    fn allowed_names_ignore_case() {
        // Windows environment names are case-insensitive, and shells differ on
        // `Path` vs `PATH`.
        assert!(is_allowed("path") && is_allowed("Path") && is_allowed("systemroot"));
    }
}
