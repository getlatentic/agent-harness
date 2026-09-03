//! Executable fixtures for the adapter tests.
//!
//! Several adapters are probes: they learn whether a CLI is installed, or
//! signed in, by running it. Testing them means writing a throwaway program
//! and pointing the probe at it. Doing that safely is subtle enough to be
//! worth one home rather than a copy per module.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Passed to a fixture by the runnable check below. A fixture exits on it
/// before reaching its own body, so the check can exercise `exec` without
/// running a script that writes files or waits on a stream.
const PROBE_ARG: &str = "--agent-harness-fixture-probe";

/// How long a fixture may take to become runnable. Generous because it only
/// ever elapses when something is genuinely wrong; the real wait is a few
/// milliseconds at most.
const RUNNABLE_TIMEOUT: Duration = Duration::from_secs(5);

/// A private directory for one fixture.
///
/// Unique per call, not per name: tags repeat across modules (`broken` is both
/// an ACP agent and a version probe), and two tests sharing one path would be
/// writing the same file at the same time.
pub(crate) fn fixture_dir(tag: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let serial = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "agent-harness-{tag}-{}-{serial}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Write `body` as an executable `/bin/sh` program at `path`, and return only
/// once it has actually run.
///
/// The wait is what makes the suite reliable on Linux. The kernel refuses to
/// `exec` a file any process still holds open for writing (`ETXTBSY`), and
/// these tests race for it: cargo runs them on many threads, so while one
/// thread has a fixture open another thread's `fork` inherits that descriptor,
/// and the `exec` behind it fails. The descriptor is close-on-exec, so the
/// window shuts as soon as that child reaches `execve` — meaning a fixture
/// that runs once is past it, and no descriptor is left for a later `fork` to
/// inherit. Waiting here therefore buys every caller a program the code under
/// test can run, instead of a spawn failure that each probe would report as an
/// absent or signed-out CLI.
pub(crate) fn install_script(path: &Path, body: &str) {
    std::fs::write(
        path,
        format!("#!/bin/sh\nif [ \"$1\" = \"{PROBE_ARG}\" ]; then exit 0; fi\n{body}\n"),
    )
    .unwrap();
    make_executable(path);
    wait_until_runnable(path);
}

/// A fixture that answers however the test needs, at `<fixture dir>/cli`.
pub(crate) fn fake_cli(tag: &str, body: &str) -> PathBuf {
    let path = fixture_dir(tag).join("cli");
    install_script(&path, body);
    path
}

fn make_executable(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

fn wait_until_runnable(path: &Path) {
    let deadline = Instant::now() + RUNNABLE_TIMEOUT;
    loop {
        let attempt = Command::new(path)
            .arg(PROBE_ARG)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        match attempt {
            Ok(_) => return,
            Err(error) if Instant::now() < deadline => {
                let _ = error;
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(error) => panic!("fixture {} never became runnable: {error}", path.display()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The guard has to be invisible to the fixture: a probe that ran the body
    /// would trip every fixture that writes a file or waits on stdin.
    #[test]
    fn the_runnable_probe_does_not_run_the_body() {
        let dir = fixture_dir("probe-guard");
        let witness = dir.join("ran");
        let cli = dir.join("cli");
        install_script(&cli, &format!("touch '{}'", witness.display()));
        assert!(!witness.exists(), "installing a fixture ran its body");

        assert!(Command::new(&cli).status().unwrap().success());
        assert!(witness.exists(), "the fixture itself no longer runs its body");
    }

    /// Two fixtures asking for the same name must not land on one file.
    #[test]
    fn a_repeated_tag_gets_its_own_directory() {
        assert_ne!(fixture_dir("same"), fixture_dir("same"));
    }
}
