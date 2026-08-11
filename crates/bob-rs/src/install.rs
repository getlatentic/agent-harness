//! Streaming installer for the bob CLI + nvm + Node.
//!
//! The embedded `scripts/install-bob.sh` ships as a compile-time
//! string constant (`include_str!`). At runtime we spool it to a
//! tempfile and exec it through `bash -l` so the user's login
//! init runs and `nvm`/`bob` end up on PATH.
//!
//! Streaming model: caller passes a `FnMut(InstallEvent)` closure.
//! We call it for every step / stdout / stderr line as they arrive
//! and once with `Done` when the child exits. The closure runs on
//! the spawned reader thread — consumers that need to bridge to
//! their own runtime (axum's `tokio::sync::mpsc`, Tauri's
//! `Channel`) wrap with a sender inside the closure.

use crate::error::BobError;
use cli_stream::InstallEvent;
#[cfg(unix)]
use std::io::{BufRead, BufReader, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(unix)]
use std::process::{Command, Stdio};
#[cfg(unix)]
use std::thread;
#[cfg(unix)]
use tempfile::NamedTempFile;

/// Source of truth — same bytes also read by the dev API at
/// `scripts/install-bob.sh`. The bob-api binary doesn't need a
/// separate copy because it depends on this crate.
#[cfg(unix)]
const INSTALL_SCRIPT: &str = include_str!("../scripts/install-bob.sh");

/// Run the install script, invoking `callback` for each event.
///
/// Blocks until the child process exits. Callbacks run on
/// background threads — consumers that bridge to a single-threaded
/// runtime (axum's stream, Tauri's Channel) need their own sync
/// primitive (typically `tokio::sync::mpsc` or `crossbeam_channel`).
///
/// Returns `Err` only if the install can't be *started* (tempfile
/// failure, spawn failure). Once `bash` is running, all failures
/// surface through the `Done { ok: false }` event instead — that's
/// how the script's own errors get back to the user.
#[cfg(unix)]
pub fn install_bob<F>(mut callback: F) -> Result<(), BobError>
where
    F: FnMut(InstallEvent) + Send + Sync + 'static + Clone,
{
    // 1. Spool the embedded script to a tempfile. Using a file
    //    rather than piping into bash's stdin means error
    //    messages reference a stable path on disk.
    let mut tmp = NamedTempFile::new().map_err(|source| BobError::Io {
        context: "create install tempfile",
        source,
    })?;
    tmp.write_all(INSTALL_SCRIPT.as_bytes())
        .map_err(|source| BobError::Io {
            context: "write install tempfile",
            source,
        })?;

    // 2. Make the script executable. bash -l <path> would work
    //    even without +x because we pass the path as an argument,
    //    but tooling that introspects /proc tends to expect the
    //    executable bit.
    let metadata = tmp.as_file().metadata().map_err(|source| BobError::Io {
        context: "stat install tempfile",
        source,
    })?;
    let mut perms = metadata.permissions();
    perms.set_mode(0o755);
    tmp.as_file()
        .set_permissions(perms)
        .map_err(|source| BobError::Io {
            context: "chmod install tempfile",
            source,
        })?;

    // 3. Command under `bash -l` so nvm / brew / asdf init in the
    //    user profile is loaded. The script also `source`s
    //    `nvm.sh` defensively.
    let mut child = Command::new("bash")
        .arg("-l")
        .arg(tmp.path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|source| BobError::Io {
            context: "spawn install script",
            source,
        })?;

    let stdout = child
        .stdout
        .take()
        .ok_or(BobError::PipeNotCaptured { stream: "stdout" })?;
    let stderr = child
        .stderr
        .take()
        .ok_or(BobError::PipeNotCaptured { stream: "stderr" })?;

    // 4. Two reader threads. Each owns its own clone of the
    //    callback because closures aren't shareable across threads
    //    without explicit synchronization. The `Clone` bound on
    //    the type parameter is what lets us hand one to each.
    let stdout_cb = callback.clone();
    let stdout_handle = thread::spawn(move || {
        let mut cb = stdout_cb;
        let reader = BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            let event = if let Some(text) = line.strip_prefix("[BOB-INSTALL] ") {
                InstallEvent::Step { text: text.to_owned() }
            } else {
                InstallEvent::Stdout { text: line }
            };
            cb(event);
        }
    });

    let stderr_cb = callback.clone();
    let stderr_handle = thread::spawn(move || {
        let mut cb = stderr_cb;
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            cb(InstallEvent::Stderr { text: line });
        }
    });

    // 5. Wait for the child + drain reader threads + emit done.
    let status = child.wait().map_err(|source| BobError::Io {
        context: "wait on install script",
        source,
    })?;
    let _ = stdout_handle.join();
    let _ = stderr_handle.join();
    callback(InstallEvent::Done {
        exit_code: status.code(),
        ok: status.success(),
    });
    drop(tmp);
    Ok(())
}

/// bob ships its installer as a POSIX shell script (`scripts/install-bob.sh`)
/// and runs it under `bash`, so installation is unix-only. Refuse up front on
/// other platforms: the alternative is a confusing failure deep in the spawn,
/// several steps after the real reason.
#[cfg(not(unix))]
pub fn install_bob<F>(_callback: F) -> Result<(), BobError>
where
    F: FnMut(InstallEvent) + Send + Sync + 'static + Clone,
{
    Err(BobError::Io {
        context: "install bob",
        source: std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "bob's installer is a POSIX shell script run under bash; \
             install bob manually on this platform",
        ),
    })
}
