//! Streaming subprocess control: spawn a child, pipe its stdout/stderr
//! line-by-line through a callback as [`ProcessEvent`]s, and hand back a
//! [`ProcessHandle`] for cancellation (SIGTERM → SIGKILL).
//!
//! The environment is the caller's: this spawns what it is told to spawn, with
//! the `PATH` it is given. Finding a CLI a user installed — resolving a bare
//! name, locating the `node` it was installed under — is a different question,
//! and one only a caller driving such a CLI needs answered.
//!
//! Cancellation is the wrinkle: a run needs to be stoppable mid-stream
//! when the user closes the tab or hits "stop". `ProcessHandle::cancel()`
//! sends SIGTERM (with a SIGKILL fallback) and flips an atomic
//! `cancelled` flag the reader threads use to short-circuit.

use crate::error::StreamError;
use serde::Serialize;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(unix)]
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Raw events emitted to the caller's callback during a streaming run.
/// JSON-tagged so axum SSE and Tauri Channel render identical payloads
/// on the wire. Harness-neutral: a process-backed adapter parses the
/// `Stdout` lines into a normalized event vocabulary (e.g. `agent-harness`'s `RunEvent`).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
// New lifecycle events can be added without breaking downstream matches —
// consumers must carry a `_` arm. (Construction of existing variants is
// unaffected, so it's still ergonomic to build them.)
#[non_exhaustive]
pub enum ProcessEvent {
    /// First event. Sent before the child has produced any output so the
    /// UI can show a "thinking…" state.
    Started { run_id: String },
    /// Raw stdout line. Process-backed CLIs emit one JSON object per line
    /// in their streaming mode. The caller parses.
    Stdout { run_id: String, line: String },
    /// Raw stderr line. Warnings + the occasional error.
    Stderr { run_id: String, line: String },
    /// Spawn / IO failure. Terminal — followed by `Exited`.
    Error { run_id: String, message: String },
    /// Process exited. Always sent exactly once at the end.
    Exited {
        run_id: String,
        exit_code: Option<i32>,
        /// True iff `cancel()` was called before exit.
        cancelled: bool,
    },
}

/// Handle to an in-flight streaming run. Caller stores it (e.g. in a
/// runId-keyed map) so a later `cancel()` can find it.
///
/// Dropping the handle does NOT cancel the run — the reader threads +
/// wait thread continue independently. Use `cancel()` explicitly when
/// the user closes the connection.
#[derive(Clone, Debug)]
pub struct ProcessHandle {
    inner: Arc<HandleInner>,
}

/// What the child's stdin is connected to.
///
/// Most CLIs get everything as arguments and want [`Closed`](Stdin::Closed): a
/// child that inherits a terminal's stdin can block forever waiting for input
/// nobody is typing. A child that *answers* — a JSON-RPC server over stdio —
/// needs [`Piped`](Stdin::Piped) and [`ProcessHandle::write_stdin_line`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Stdin {
    #[default]
    Closed,
    Piped,
}

/// What to spawn. Named fields rather than six positional arguments, so a call
/// site says which string is the program and which is the run id, and a new
/// knob is a field with a default instead of a break.
#[derive(Debug, Clone)]
pub struct Spawn {
    pub program: PathBuf,
    pub args: Vec<String>,
    /// Extra environment for the child, applied over the inherited one.
    pub env: Vec<(String, String)>,
    pub cwd: PathBuf,
    /// The caller's correlation id, echoed on every [`ProcessEvent`].
    pub run_id: String,
    pub stdin: Stdin,
}

impl Spawn {
    /// A run of `program`, in the current directory, with stdin closed.
    ///
    /// The program is the only thing a spawn cannot default, so it is the only
    /// argument. Everything else is a named method — three bare strings in a
    /// row read as "which one was the cwd again?".
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            env: Vec::new(),
            cwd: std::env::current_dir().unwrap_or_default(),
            run_id: String::new(),
            stdin: Stdin::Closed,
        }
    }

    /// Where the child runs. Defaults to the current directory.
    #[must_use]
    pub fn cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = cwd.into();
        self
    }

    /// A correlation id echoed on every [`ProcessEvent`], for a caller
    /// multiplexing several runs through one callback. Defaults to empty —
    /// with one run the handle already identifies it.
    #[must_use]
    pub fn run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = run_id.into();
        self
    }

    /// `.args(["--stdio"])` — anything string-like, borrowed or owned.
    #[must_use]
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    /// `.env([("RUST_LOG", "info")])` — applied over the inherited environment.
    #[must_use]
    pub fn env<I, K, V>(mut self, env: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.env = env.into_iter().map(|(k, v)| (k.into(), v.into())).collect();
        self
    }

    /// What the child's stdin is connected to. `stdout` and `stderr` are always
    /// piped — streaming them is what this crate is for — and arrive as
    /// [`ProcessEvent::Stdout`] / [`ProcessEvent::Stderr`].
    #[must_use]
    pub fn stdin(mut self, stdin: Stdin) -> Self {
        self.stdin = stdin;
        self
    }
}

#[derive(Debug)]
struct HandleInner {
    child: Mutex<Option<Child>>,
    /// The child's stdin, when it was piped. Taken from the `Child` at spawn so
    /// writing never has to lock the same mutex `cancel` uses.
    stdin: Mutex<Option<std::process::ChildStdin>>,
    cancelled: AtomicBool,
}

impl ProcessHandle {
    /// SIGTERM the process, then SIGKILL after 1.5s if it's still alive.
    /// The CLI is supposed to flush a final result on SIGTERM but we
    /// don't trust it to do so forever.
    pub fn cancel(&self) -> Result<(), StreamError> {
        self.inner.cancelled.store(true, Ordering::SeqCst);
        let mut guard = self
            .inner
            .child
            .lock()
            .map_err(|_| StreamError::CancelLockPoisoned)?;
        let Some(child) = guard.as_mut() else {
            // Already exited.
            return Ok(());
        };
        // Best-effort SIGTERM. On Unix, kill() sends SIGKILL by default;
        // we use libc::kill for SIGTERM, falling back to child.kill() if
        // the libc call fails. On Windows there's only TerminateProcess
        // via .kill().
        #[cfg(unix)]
        {
            let pid = child.id() as i32;
            // SAFETY: pid is the child's PID owned by this Child; sending
            // SIGTERM is well-defined.
            unsafe { libc::kill(pid, libc::SIGTERM) };
            // Spawn the SIGKILL fallback inline to avoid holding the mutex
            // while sleeping.
            let inner = Arc::clone(&self.inner);
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(1500));
                if let Ok(mut guard) = inner.child.lock() {
                    if let Some(child) = guard.as_mut() {
                        let _ = child.kill();
                    }
                }
            });
        }
        #[cfg(not(unix))]
        {
            let _ = child.kill();
        }
        Ok(())
    }

    /// Send one line to the child's **stdin**, newline-terminated and flushed.
    ///
    /// `Err` when the child was spawned with [`Stdin::Closed`] (the default), or
    /// when it has exited and the pipe is gone — both of which a caller
    /// expecting an answer needs to hear about rather than block on.
    pub fn write_stdin_line(&self, line: &str) -> Result<(), StreamError> {
        let mut guard = self.inner.stdin.lock().map_err(|_| StreamError::CancelLockPoisoned)?;
        let stdin = guard.as_mut().ok_or(StreamError::PipeNotCaptured { stream: "stdin" })?;
        use std::io::Write;
        stdin
            .write_all(line.as_bytes())
            .and_then(|()| stdin.write_all(b"\n"))
            .and_then(|()| stdin.flush())
            .map_err(|source| StreamError::Spawn { program: "stdin".to_owned(), source })
    }

    /// Whether `cancel()` was called. Tagged on the final `Exited` event.
    pub fn was_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::SeqCst)
    }

    /// The child's OS process id while it's alive, or `None` once it has been
    /// reaped (the `Child` is taken on exit). Lets an embedder record the pid
    /// so a child orphaned by a hard crash can be killed on the next launch.
    pub fn pid(&self) -> Option<u32> {
        self.inner
            .child
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(Child::id))
    }
}

/// Spawn an arbitrary streaming child process — the generic engine behind
/// every process-backed harness (bob, Claude Code, Codex).
///
/// Pipes stdout/stderr line-by-line through `callback` using the raw
/// [`ProcessEvent`] vocabulary (Started / Stdout / Stderr / Error /
/// Exited). `env` supplies per-harness secrets (each harness's API-key
/// var, or none for self-authenticating CLIs). PATH is augmented so
/// Node-based CLIs find `node`. Returns a [`ProcessHandle`] for
/// cancellation.
///
/// `callback` is invoked from three threads (stdout reader, stderr
/// reader, exit watcher); the `Clone` bound lets us hand a copy to each.
/// `run_id` is opaque — the caller chooses it and uses it to correlate
/// events with the handle.
///
/// ```no_run
/// use cli_stream::{spawn_streaming, ProcessEvent, Spawn};
///
/// # fn main() -> Result<(), cli_stream::StreamError> {
/// let handle = spawn_streaming(
///     Spawn::new("echo").cwd(std::env::current_dir().unwrap()).run_id("run-1")
///         .args(vec!["hello".to_owned()]),
///     |event| match event {
///         ProcessEvent::Stdout { line, .. } => println!("{line}"),
///         ProcessEvent::Exited { exit_code, .. } => eprintln!("exit {exit_code:?}"),
///         _ => {}
///     },
/// )?;
/// // `handle.cancel()` stops it early; dropping the handle does not.
/// let _ = handle;
/// # Ok(())
/// # }
/// ```
///
/// Spawn and read the events off a channel, rather than supplying a callback.
///
/// The same run either way; this is the shape to reach for when the reading is
/// a loop rather than a reaction:
///
/// ```no_run
/// use cli_stream::{spawn, ProcessEvent, Spawn};
///
/// # fn main() -> Result<(), cli_stream::StreamError> {
/// let (handle, events) = spawn(Spawn::new("echo").args(["hi"]))?;
/// for event in events {
///     if let ProcessEvent::Stdout { line, .. } = event {
///         println!("{line}");
///     }
/// }
/// # let _ = handle;
/// # Ok(())
/// # }
/// ```
///
/// The channel closes on its own when the run ends: the forwarding callback is
/// the only owner of the `Sender`).run_id(and it drops with the reader threads.
pub fn spawn(spawn: Spawn) -> Result<(ProcessHandle, std::sync::mpsc::Receiver<ProcessEvent>), StreamError> {
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = spawn_streaming(spawn, move |event| {
        // A hung-up receiver is not an error: the run continues, and the event
        // nobody is waiting for is dropped.
        let _ = tx.send(event);
    })?;
    Ok((handle, rx))
}

pub fn spawn_streaming<F>(spawn: Spawn, callback: F) -> Result<ProcessHandle, StreamError>
where
    F: FnMut(ProcessEvent) + Send + Sync + Clone + 'static,
{
    let Spawn { program, args, env, cwd, run_id, stdin } = spawn;
    let mut command = hidden_command(&program);
    command
        .args(&args)
        .current_dir(&cwd)
        .stdin(match stdin {
            Stdin::Closed => Stdio::null(),
            Stdin::Piped => Stdio::piped(),
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in &env {
        command.env(key, value);
    }
    let mut child = command.spawn().map_err(|source| StreamError::Spawn {
        program: program.display().to_string(),
        source,
    })?;

    let stdout = child
        .stdout
        .take()
        .ok_or(StreamError::PipeNotCaptured { stream: "stdout" })?;
    let stderr = child
        .stderr
        .take()
        .ok_or(StreamError::PipeNotCaptured { stream: "stderr" })?;

    // Taken now so `write_line` never contends with `cancel` for the child.
    let child_stdin = child.stdin.take();
    let inner = Arc::new(HandleInner {
        child: Mutex::new(Some(child)),
        stdin: Mutex::new(child_stdin),
        cancelled: AtomicBool::new(false),
    });
    let handle = ProcessHandle {
        inner: Arc::clone(&inner),
    };

    // Emit Started immediately so the caller doesn't wait on the first
    // output line for a UI signal.
    let mut started_cb = callback.clone();
    started_cb(ProcessEvent::Started {
        run_id: run_id.clone(),
    });

    // Reader threads. Each owns its own callback clone — the Clone bound
    // is the whole point.
    let stdout_cb = callback.clone();
    let stdout_run_id = run_id.clone();
    let stdout_handle = thread::spawn(move || {
        pump_lines(stdout, stdout_run_id, true, stdout_cb);
    });

    let stderr_cb = callback.clone();
    let stderr_run_id = run_id.clone();
    let stderr_handle = thread::spawn(move || {
        pump_lines(stderr, stderr_run_id, false, stderr_cb);
    });

    // Exit watcher — emits the terminal Exited event with the cancellation
    // flag. It must NOT hold the child lock across a blocking `wait()`:
    // `cancel()` needs that same lock to signal the child, so a held lock
    // would block cancel until the process exited on its own (defeating it).
    // Instead poll `try_wait()`, locking only for each non-blocking check and
    // releasing between polls so `cancel()` can acquire the lock mid-run.
    let exit_inner = Arc::clone(&inner);
    let mut exit_cb = callback;
    let exit_run_id = run_id;
    thread::spawn(move || {
        let wait_result = loop {
            {
                let mut guard = match exit_inner.child.lock() {
                    Ok(guard) => guard,
                    Err(_) => return, // poisoned — nothing safe to do
                };
                match guard.as_mut() {
                    Some(child) => match child.try_wait() {
                        Ok(Some(status)) => break Ok(status),
                        Ok(None) => {} // still running; poll again
                        Err(err) => break Err(err),
                    },
                    None => return, // already reaped
                }
            } // lock released before sleeping, so cancel() can acquire it
            thread::sleep(Duration::from_millis(50));
        };
        let _ = stdout_handle.join();
        let _ = stderr_handle.join();
        let cancelled = exit_inner.cancelled.load(Ordering::SeqCst);

        match wait_result {
            Ok(status) => exit_cb(ProcessEvent::Exited {
                run_id: exit_run_id.clone(),
                exit_code: status.code(),
                cancelled,
            }),
            Err(err) => exit_cb(ProcessEvent::Error {
                run_id: exit_run_id.clone(),
                message: format!("wait failed: {err}"),
            }),
        }

        // Drop the child handle so subsequent cancel() calls
        // short-circuit cleanly.
        if let Ok(mut guard) = exit_inner.child.lock() {
            *guard = None;
        }
    });

    Ok(handle)
}

fn pump_lines<R, F>(reader: R, run_id: String, is_stdout: bool, mut callback: F)
where
    R: Read,
    F: FnMut(ProcessEvent),
{
    let buffered = BufReader::new(reader);
    for line in buffered.lines() {
        match line {
            Ok(text) => {
                let event = if is_stdout {
                    ProcessEvent::Stdout {
                        run_id: run_id.clone(),
                        line: text,
                    }
                } else {
                    ProcessEvent::Stderr {
                        run_id: run_id.clone(),
                        line: text,
                    }
                };
                callback(event);
            }
            Err(err) => {
                callback(ProcessEvent::Error {
                    run_id: run_id.clone(),
                    message: format!("stream read failed: {err}"),
                });
                return;
            }
        }
    }
}

/// Compose a PATH for the spawned process that always includes the
/// directory containing the program — where `node`, `npm`, and friends
/// usually live in an nvm install. The user's existing PATH stays as a
/// fallback after our prepended directory.
/// A [`Command`] that never opens a console window on Windows.
///
/// A GUI host (a Tauri app, an IDE) spawning a console-subsystem CLI gets a
/// black console flashed on screen for every agent run and every `--version`
/// probe. `CREATE_NO_WINDOW` suppresses it. Use this in place of
/// `Command::new` for anything a desktop app spawns; it is a plain
/// `Command::new` on every other platform, so call sites stay `cfg`-free.
pub fn hidden_command(program: impl AsRef<std::ffi::OsStr>) -> Command {
    #[allow(unused_mut)]
    let mut command = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // https://learn.microsoft.com/windows/win32/procthread/process-creation-flags
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    command
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Issue #35: a GUI host spawning a console-subsystem CLI flashed a console
    /// window on Windows for every run and every `--version` probe. The flag is
    /// Windows-only, so what is portable to assert is that the constructor is a
    /// drop-in for `Command::new` — it still runs, and still captures output.
    #[test]
    fn hidden_command_runs_like_a_plain_command() {
        let program = if cfg!(windows) { "cmd" } else { "echo" };
        let args: &[&str] = if cfg!(windows) {
            &["/C", "echo", "ok"]
        } else {
            &["ok"]
        };
        let out = hidden_command(program).args(args).output().unwrap();
        assert!(out.status.success());
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "ok");
    }

    use std::sync::Condvar;
    use std::time::Instant;

    type Done = Arc<(Mutex<bool>, Condvar)>;

    /// A thread-safe event collector that signals `done` on the terminal
    /// event. Returns the (cloneable) callback + the shared collections.
    fn collector() -> (
        impl FnMut(ProcessEvent) + Send + Sync + Clone + 'static,
        Arc<Mutex<Vec<ProcessEvent>>>,
        Done,
    ) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let done: Done = Arc::new((Mutex::new(false), Condvar::new()));
        let cb = {
            let events = Arc::clone(&events);
            let done = Arc::clone(&done);
            move |ev: ProcessEvent| {
                let terminal =
                    matches!(ev, ProcessEvent::Exited { .. } | ProcessEvent::Error { .. });
                events.lock().unwrap().push(ev);
                if terminal {
                    let (lock, cvar) = &*done;
                    *lock.lock().unwrap() = true;
                    cvar.notify_all();
                }
            }
        };
        (cb, events, done)
    }

    /// Block until the terminal event fires, or panic after `secs`.
    fn wait_done(done: &Done, secs: u64) {
        let (lock, cvar) = &**done;
        let mut finished = lock.lock().unwrap();
        let deadline = Instant::now() + Duration::from_secs(secs);
        while !*finished {
            let now = Instant::now();
            assert!(now < deadline, "process did not finish within {secs}s");
            let (guard, _) = cvar.wait_timeout(finished, deadline - now).unwrap();
            finished = guard;
        }
    }

    /// Spawn `program args`, block until it exits, return every event.
    fn run(program: &str, args: &[&str]) -> Vec<ProcessEvent> {
        let (cb, events, done) = collector();
        let _handle = spawn_streaming(
            Spawn::new(program).run_id("t").args(args.iter().copied()),
            cb,
        )
        .expect("spawn");
        wait_done(&done, 10);
        let events = events.lock().unwrap();
        events.clone()
    }

    #[test]
    fn streams_stdout_lines_then_exits_zero() {
        let events = run("printf", &["%s\n", "alpha", "beta"]);
        // Started leads, Exited(0, not cancelled) closes.
        assert!(matches!(events.first(), Some(ProcessEvent::Started { .. })));
        assert!(matches!(
            events.last(),
            Some(ProcessEvent::Exited {
                exit_code: Some(0),
                cancelled: false,
                ..
            })
        ));
        // Lines arrive in order, one event each.
        let lines: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                ProcessEvent::Stdout { line, .. } => Some(line.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(lines, vec!["alpha", "beta"]);
    }

    #[test]
    fn nonzero_exit_code_is_reported() {
        let events = run("sh", &["-c", "exit 3"]);
        assert!(matches!(
            events.last(),
            Some(ProcessEvent::Exited {
                exit_code: Some(3),
                cancelled: false,
                ..
            })
        ));
    }

    #[test]
    fn env_vars_are_passed_to_the_child() {
        // The `env` argument must reach the child's environment — exercise it
        // directly (the other lifecycle tests pass an empty env).
        let (cb, events, done) = collector();
        let _handle = spawn_streaming(
            Spawn::new("sh").run_id("t").args(vec![
                "-c".to_owned(),
                "printf '%s\\n' \"$CLI_STREAM_STUB\"".to_owned(),
            ]).env(vec![("CLI_STREAM_STUB".to_owned(), "from-env".to_owned())]),
            cb,
        )
        .expect("spawn");
        wait_done(&done, 10);
        let events = events.lock().unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ProcessEvent::Stdout { line, .. } if line == "from-env")),
            "child should observe the injected env var, got {events:?}"
        );
    }

    #[test]
    fn stderr_is_streamed_and_not_misrouted_to_stdout() {
        let events = run("sh", &["-c", "echo to-stderr 1>&2"]);
        assert!(events
            .iter()
            .any(|e| matches!(e, ProcessEvent::Stderr { line, .. } if line == "to-stderr")));
        assert!(!events
            .iter()
            .any(|e| matches!(e, ProcessEvent::Stdout { .. })));
        assert!(events.iter().any(|e| matches!(
            e,
            ProcessEvent::Exited {
                exit_code: Some(0),
                ..
            }
        )));
    }

    #[test]
    fn cancel_promptly_terminates_the_run_and_flags_it() {
        // A 10s sleeper we cancel ~immediately; a working engine must kill it
        // far sooner than 10s. `exec` so the process *is* sleep (no orphan).
        let (cb, events, done) = collector();
        let handle = spawn_streaming(
            Spawn::new("sh").run_id("t").args(["-c", "exec sleep 10"]),
            cb,
        )
        .expect("spawn");

        // cancel() may block until the child is reaped, so fire it off-thread.
        let canceller = handle.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(100));
            let _ = canceller.cancel();
        });

        // Correct cancellation terminates the 10s sleep within a few seconds.
        wait_done(&done, 4);
        assert!(handle.was_cancelled());
        let events = events.lock().unwrap();
        assert!(
            matches!(
                events.last(),
                Some(ProcessEvent::Exited {
                    cancelled: true,
                    ..
                })
            ),
            "expected Exited(cancelled=true), got {:?}",
            events.last()
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_live_child_reports_a_pid_and_flips_when_cancelled() {
        // An embedder records the pid so a child a hard crash orphaned can be
        // reaped on the next launch, and reads `was_cancelled` to tell a run
        // the user stopped from one that finished. Both are answered by
        // forwarding, which is exactly the kind of code that silently returns
        // the wrong constant.
        let handle = spawn_streaming(
            Spawn::new("/bin/sleep").cwd(std::env::temp_dir()).run_id("pid").args(["30"]),
            |_| {},
        )
        .expect("sleep should spawn");

        let pid = handle.pid().expect("a live child has a pid");
        assert!(pid > 1, "a real OS pid, not a placeholder: {pid}");
        assert!(!handle.was_cancelled(), "nothing has stopped it yet");

        handle.cancel().expect("cancel");
        assert!(handle.was_cancelled(), "a stopped run says so");
    }

    #[test]
    fn spawning_a_missing_binary_is_err() {
        let result = spawn_streaming(
            Spawn::new("cli-stream-no-such-binary-zzz").run_id("t"),
            |_ev: ProcessEvent| {},
        );
        // Typed: a `Spawn` error carrying the OS `NotFound` io::Error as its
        // source — the whole point of `StreamError` over a `String`. A caller
        // can branch on `ErrorKind` to tell "not installed" (NotFound) from
        // "permission denied", which a flattened string can't support.
        match result {
            Err(StreamError::Spawn { program, source }) => {
                assert!(program.contains("cli-stream-no-such-binary-zzz"));
                assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
            }
            other => panic!("expected StreamError::Spawn, got {other:?}"),
        }
    }
}
