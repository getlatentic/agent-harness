//! Finding an agent's CLI, and the PATH it needs to run.
//!
//! None of this is about spawning or streaming — [`cli_stream`] does that, and
//! takes the environment it is given. This is the part that only matters
//! because the CLIs we drive are Node programs a user installed themselves.
//!
//! A desktop app launched from Finder inherits the minimal launchd PATH
//! (`/usr/bin:/bin:/usr/sbin:/sbin`), so an nvm-installed `node` is invisible
//! and the CLI exits 127. Worse, a CLI installed under one node version and run
//! against whichever node leads the inherited PATH fails in subtler ways. So a
//! bare name is resolved to its absolute path first, and that program's own
//! directory — where its sibling `node` lives in an nvm install — goes to the
//! front of the child's PATH.
//!
//! Verifying any of this needs a real double-click: `open <app>` leaks the
//! launching shell's PATH and passes when the packaged app would fail.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{mpsc, OnceLock};
use std::thread;
use std::time::Duration;

use cli_stream::hidden_command;

/// Ask a CLI for its version, on the augmented PATH. `None` when it cannot be
/// run, exits non-zero, or says nothing — each of which means the same thing to
/// a caller: this is not an installed, working CLI.
///
/// Every adapter wrapping a CLI needs exactly this, and it is the probe that
/// decides whether a harness reads as installed at all — so it lives once,
/// beside [`hidden_command`] and [`augmented_node_path`], rather than being
/// copied per adapter and drifting.
pub fn probe_version(program: &str) -> Option<String> {
    let output = hidden_command(program).arg("--version").env("PATH", augmented_node_path()).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    (!text.is_empty()).then_some(text)
}

fn augment_path_for_node(program: &Path) -> String {
    prepend_program_dir(program, &augmented_node_path())
}

/// Resolve a bare program name (`bob`, `claude`) to its absolute path on the
/// augmented PATH, so the spawn and the node pairing agree on *one* location.
///
/// Without this, a bare name splits the brain: the OS resolves the *program*
/// against the parent process's PATH, while the child's `#!/usr/bin/env node`
/// shebang resolves *node* against the PATH we set — and
/// `prepend_program_dir` can't pair the program with its sibling node
/// because a bare name has no parent dir. Concretely: an nvm-installed `bob`
/// found under `v24/bin` could re-exec on a `v20` node that happened to lead
/// the inherited PATH, and die on a v24-only flag ("exited with code 9").
/// Resolving to the absolute path first means the program's own directory —
/// holding the exact `node` it was installed with — is prepended and wins.
///
/// A program given with an explicit path is returned untouched; a bare name
/// that can't be found is also returned untouched, so the spawn still fails
/// with the clear "No such file" error rather than a synthetic one here.
pub fn resolve_program(program: PathBuf) -> PathBuf {
    if program.parent().is_some_and(|p| !p.as_os_str().is_empty()) {
        return program; // explicit path — caller's choice wins
    }
    resolve_on_path(&program, &augmented_node_path()).unwrap_or(program)
}

/// Walk `path_env`'s entries for the first executable file named `name`.
/// Pure with respect to env/spawn (filesystem only) so it's unit-testable.
fn resolve_on_path(name: &Path, path_env: &str) -> Option<PathBuf> {
    path_env
        .split(':')
        .filter(|dir| !dir.is_empty())
        .map(|dir| Path::new(dir).join(name))
        .find(|candidate| is_executable_file(candidate))
}

#[cfg(unix)]
fn is_executable_file(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable_file(path: &Path) -> bool {
    path.is_file()
}

/// Prepend the directory containing `program` (where `node` also lives in an
/// nvm install) to `base_path`, so the resolved binary's own dir is searched
/// first. Pure (no env / no spawn) so it's unit-tested directly.
fn prepend_program_dir(program: &Path, base_path: &str) -> String {
    match program
        .parent()
        .map(|p| p.display().to_string())
        .filter(|s| !s.is_empty())
    {
        Some(dir) => format!("{dir}:{base_path}"),
        None => base_path.to_owned(),
    }
}

/// A PATH that resolves Node-based CLIs (bob, claude, codex) even from a
/// process launched by Finder/Launchpad, which inherits only the minimal
/// launchd PATH (`/usr/bin:/bin:/usr/sbin:/sbin`) rather than the user's
/// shell PATH.
///
/// Strategy: keep the process's own PATH first (an explicit PATH still wins),
/// then append the user's **real** PATH as resolved by their login shell —
/// which sources their rc, so it knows where nvm / pnpm / volta / asdf / fnm /
/// Homebrew put `node`, with no guessing. If the shell query is unavailable
/// (no `$SHELL`, a timeout, a sandboxed app that can't spawn, …) we fall back
/// to a hardcoded best-effort list, so we're never worse than before.
///
/// Used by the run path (which prepends the resolved binary's own dir on top
/// of this) and by readiness probes that locate `claude`/`codex` via a bare
/// `Command::new(name)`. Computed once and cached for the process — the
/// (bounded) shell spawn happens at most once per launch, lazily on the first
/// readiness/run/login, never at construction.
pub fn augmented_node_path() -> String {
    static CACHED: OnceLock<String> = OnceLock::new();
    CACHED.get_or_init(compute_augmented_node_path).clone()
}

fn compute_augmented_node_path() -> String {
    let mut parts: Vec<String> = Vec::new();
    // The process's own PATH first — anything explicitly set still wins.
    if let Ok(existing) = std::env::var("PATH") {
        if !existing.is_empty() {
            parts.push(existing);
        }
    }
    // The user's real PATH (nvm/pnpm/volta/asdf/Homebrew) via their login
    // shell; a hardcoded best-effort list if that's unavailable.
    parts.push(login_shell_path().unwrap_or_else(hardcoded_node_dirs));
    keep_absolute_entries(&parts.join(":"))
}

/// Keep only **absolute** PATH entries, dropping relative or empty ones (`.`,
/// `""`, a direnv-style `node_modules/.bin`). Security: we spawn with
/// `current_dir` set to the user's workspace — where the agent itself writes
/// files and synced/downloaded content lands — so a relative/empty PATH entry
/// (which resolves against that cwd) could run a planted `node`/`claude`. An
/// empty entry is the classic implicit-cwd vector. Absolute dirs only.
fn keep_absolute_entries(path: &str) -> String {
    path.split(':')
        .filter(|entry| entry.starts_with('/'))
        .collect::<Vec<_>>()
        .join(":")
}

/// Resolve PATH by asking the user's login + interactive shell — it sources
/// their rc, so it knows wherever any node manager (nvm / pnpm / volta / asdf /
/// fnm / Homebrew) put `node`, without us guessing. Bounded by a timeout so a
/// slow or interactive rc can't hang us; returns `None` (→ hardcoded fallback)
/// on any failure: no `$SHELL`, spawn refused (e.g. a sandboxed app), timeout,
/// or no PATH in the output. Reads PATH from `env` (OS colon format,
/// shell-agnostic — works for fish too) rather than expanding `$PATH`.
///
/// This *executes the user's shell rc*, exactly as opening a terminal does —
/// their own shell, on their own machine. It is not a privilege/auth step: no
/// "login session" is created; `-l`/`-i` only select which startup files are
/// sourced (login profiles + the interactive rc where nvm usually lives).
/// Printed on its own line right before `env`, so the parser can skip any
/// shell-init chatter / terminal escape sequences (e.g. iTerm2 shell
/// integration's `]1337;…` OSC codes) the interactive shell emits before our
/// command runs — which would otherwise prepend to the `PATH=` line.
#[cfg(unix)]
const PATH_SENTINEL: &str = "__CLI_STREAM_PATH__";

#[cfg(unix)]
fn login_shell_path() -> Option<String> {
    let shell = std::env::var("SHELL").ok().filter(|s| !s.is_empty())?;
    // Print a sentinel line, then dump the environment. Reading PATH from `env`
    // (not by expanding `$PATH`) keeps it OS colon format and shell-agnostic
    // (fish stores PATH as a list); the sentinel lets the parser ignore
    // anything the interactive shell prints at startup before `env` runs.
    let script = format!("printf '\\n{PATH_SENTINEL}\\n'; env");
    let mut child = Command::new(&shell)
        .arg("-lic") // -l: login profiles, -i: interactive rc (nvm), -c: command
        .arg(&script)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    // Read on a worker thread so the whole query can be bounded by a timeout —
    // a misbehaving rc must not hang the app. Read bytes + lossy-decode (rather
    // than `read_to_string`) so non-UTF-8 in the env dump degrades to
    // replacement chars instead of discarding the whole output.
    let mut stdout = child.stdout.take()?;
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout.read_to_end(&mut buf);
        let _ = tx.send(String::from_utf8_lossy(&buf).into_owned());
    });
    // 4s: generous enough for a heavy rc (oh-my-zsh + plugins + nvm lazy-load)
    // to finish, since this is paid at most once (cached); on timeout we kill
    // the shell and fall back to the hardcoded list.
    let output = match rx.recv_timeout(Duration::from_secs(4)) {
        Ok(buf) => buf,
        Err(_) => {
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
    };
    let _ = child.wait();
    parse_path_from_shell_output(&output)
}

#[cfg(not(unix))]
fn login_shell_path() -> Option<String> {
    None
}

/// Extract the `PATH=…` value from the shell's `printf <sentinel>; env` output.
/// Everything up to (and including) the last sentinel is discarded — that's
/// where shell-init chatter and terminal escape sequences live — then the
/// `PATH=` line is read from the clean `env` dump that follows. `None` if the
/// sentinel is missing (query misbehaved) or PATH is absent/empty.
#[cfg(unix)]
fn parse_path_from_shell_output(output: &str) -> Option<String> {
    output
        .rsplit_once(PATH_SENTINEL)?
        .1
        .lines()
        .find_map(|line| line.strip_prefix("PATH="))
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(str::to_owned)
}

/// Hardcoded best-effort node locations — the fallback when the login-shell
/// query is unavailable. Leans on the *universal* dirs every distro + macOS
/// share: `/usr/bin` + `/usr/local/bin` are where apt/dnf/yum/pacman and the
/// official Node tarball install, so the common Linux container case is covered
/// without distro-specific guessing. Plus macOS Homebrew, the official-installer
/// dir, and any nvm-managed node. Anything manager-specific (pnpm/volta/asdf,
/// Linuxbrew, snap, …) is what the login-shell query is for — and a missing
/// dir is just skipped, so this is never worse than the bare launchd PATH.
fn hardcoded_node_dirs() -> String {
    let mut parts: Vec<String> =
        vec!["/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin".to_owned()];
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            let home_path = Path::new(&home);
            // Official-installer location for several agent CLIs.
            parts.push(home_path.join(".local/bin").display().to_string());
            // nvm: ~/.nvm/versions/node/<version>/bin — where npm-global
            // CLIs (bob, claude, codex) live under an nvm-managed node.
            if let Ok(entries) = std::fs::read_dir(home_path.join(".nvm/versions/node")) {
                for entry in entries.flatten() {
                    let bin = entry.path().join("bin");
                    if bin.is_dir() {
                        parts.push(bin.display().to_string());
                    }
                }
            }
        }
    }
    parts.join(":")
}


/// Spawn an agent CLI and stream it: resolve a bare name to its absolute path,
/// put that program's own directory at the front of `PATH`, then hand off to
/// [`cli_stream::spawn_streaming`].
///
/// Every adapter driving a CLI goes through here rather than calling
/// `spawn_streaming` directly. The engine takes the environment it is given —
/// spawning is its job, knowing where a user's nvm lives is not — so this is
/// the one place that knowledge is applied, and the one place to look when a
/// packaged app cannot find a CLI a terminal finds fine.
///
/// A `PATH` the caller supplies still wins: it is applied after this one.
pub fn spawn_cli<F>(
    program: PathBuf,
    args: Vec<String>,
    env: Vec<(String, String)>,
    cwd: PathBuf,
    run_id: String,
    callback: F,
) -> Result<cli_stream::ProcessHandle, cli_stream::StreamError>
where
    F: FnMut(cli_stream::ProcessEvent) + Send + Sync + Clone + 'static,
{
    let program = resolve_program(program);
    let mut with_path = vec![("PATH".to_owned(), augment_path_for_node(&program))];
    with_path.extend(env);
    cli_stream::spawn_streaming(program, args, with_path, cwd, run_id, callback)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A throwaway CLI that answers however the test needs.
    #[cfg(unix)]
    fn fake_cli(tag: &str, script: &str) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("cs-probe-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cli");
        std::fs::write(&path, format!("#!/bin/sh\n{script}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[cfg(unix)]
    #[test]
    fn a_version_is_only_reported_when_the_cli_actually_gave_one() {
        // This is what "installed" means to every adapter that wraps a CLI, so
        // an empty or failed `--version` must not read as a successful probe.
        let ok = fake_cli("version", "echo '1.2.3 (Some CLI)'");
        assert_eq!(probe_version(ok.to_str().unwrap()).as_deref(), Some("1.2.3 (Some CLI)"));

        let blank = fake_cli("blank", "exit 0");
        assert_eq!(probe_version(blank.to_str().unwrap()), None, "no version is not a version");

        let broken = fake_cli("broken", "echo 9.9.9; exit 3");
        assert_eq!(probe_version(broken.to_str().unwrap()), None, "a failed probe is not installed");

        assert_eq!(probe_version("definitely-not-a-real-binary-xyz"), None, "and neither is an absent one");
    }

    #[test]
    fn hardcoded_fallback_includes_macos_defaults() {
        // The fallback (used when the login-shell query is unavailable) must
        // still carry Homebrew + the system bins, so a launchd-spawned `.app`
        // resolves CLIs even without a usable shell — the original
        // "not installed" fix.
        let path = hardcoded_node_dirs();
        assert!(
            path.contains("/opt/homebrew/bin"),
            "missing Apple-Silicon Homebrew bin"
        );
        assert!(
            path.contains("/usr/local/bin"),
            "missing Intel Homebrew / system bin"
        );
        assert!(path.contains("/usr/bin"), "missing system bin");
    }

    #[cfg(unix)]
    #[test]
    fn parse_path_from_shell_output_skips_chatter_before_the_sentinel() {
        // Real-world shape: iTerm2 OSC escapes + a banner emitted at shell
        // startup, BEFORE our sentinel + `env` dump. Only the post-sentinel
        // PATH= line counts — note the pre-sentinel "PATH=/decoy" is ignored.
        let output = "\u{1b}]1337;RemoteHost=x\u{7}welcome banner\nPATH=/decoy\n__CLI_STREAM_PATH__\nHOME=/Users/x\nPATH=/opt/homebrew/bin:/usr/bin\nLANG=en_US";
        assert_eq!(
            parse_path_from_shell_output(output).as_deref(),
            Some("/opt/homebrew/bin:/usr/bin")
        );
        // No sentinel (query misbehaved) → None, so the caller falls back —
        // even if a bare PATH= is present.
        assert_eq!(parse_path_from_shell_output("PATH=/usr/bin"), None);
        // Sentinel present but PATH absent/empty → None.
        assert_eq!(
            parse_path_from_shell_output("__CLI_STREAM_PATH__\nFOO=bar"),
            None
        );
        assert_eq!(
            parse_path_from_shell_output("__CLI_STREAM_PATH__\nPATH=\nFOO=bar"),
            None
        );
    }

    #[test]
    fn keep_absolute_entries_drops_relative_and_empty() {
        // Relative (`node_modules/.bin`, `.`) and empty entries — which resolve
        // against the spawn cwd (the user's workspace) — are dropped; absolute
        // dirs survive in order.
        assert_eq!(
            keep_absolute_entries("/opt/homebrew/bin:node_modules/.bin:/usr/bin:.::/bin"),
            "/opt/homebrew/bin:/usr/bin:/bin"
        );
        assert_eq!(keep_absolute_entries("/usr/bin"), "/usr/bin");
        // All-relative → empty (caller still has the process PATH ahead of it).
        assert_eq!(keep_absolute_entries(".:rel:"), "");
    }

    #[test]
    fn prepend_program_dir_puts_the_binary_dir_first() {
        let combined = prepend_program_dir(
            Path::new("/Users/x/.nvm/versions/node/v22/bin/bob"),
            "/opt/homebrew/bin:/usr/bin",
        );
        assert!(combined.starts_with("/Users/x/.nvm/versions/node/v22/bin:"));
        assert!(combined.contains("/opt/homebrew/bin"));
        // A bare program name has no parent dir → base path unchanged.
        assert_eq!(
            prepend_program_dir(Path::new("bob"), "/usr/bin"),
            "/usr/bin"
        );
    }

    #[test]
    fn augmented_node_path_is_nonempty_and_resolves_system_bin() {
        // Exercises the cached public path once. `/usr/bin` is present whether
        // the shell query succeeds (real PATH) or falls back (hardcoded), and
        // is on the bare launchd PATH too — so this holds in any environment.
        let path = augmented_node_path();
        assert!(!path.is_empty());
        assert!(path.contains("/usr/bin"), "system bin must always resolve");
    }

    #[test]
    fn resolve_program_returns_explicit_paths_untouched() {
        // A caller-supplied path is the caller's choice — no PATH lookup.
        let explicit = PathBuf::from("/opt/somewhere/bob");
        assert_eq!(resolve_program(explicit.clone()), explicit);
        let relative = PathBuf::from("./bin/bob");
        assert_eq!(resolve_program(relative.clone()), relative);
    }

    #[cfg(unix)]
    #[test]
    fn resolve_on_path_finds_the_first_executable_match() {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().expect("tempdir");
        // dir_a holds a NON-executable `bob` (must be skipped); dir_b an
        // executable one (must win even though dir_a comes first on PATH).
        let dir_a = root.path().join("a");
        let dir_b = root.path().join("b");
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();
        std::fs::write(dir_a.join("bob"), "#!/bin/sh\n").unwrap();
        let exec = dir_b.join("bob");
        std::fs::write(&exec, "#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&exec, std::fs::Permissions::from_mode(0o755)).unwrap();

        let path_env = format!("{}:{}", dir_a.display(), dir_b.display());
        assert_eq!(resolve_on_path(Path::new("bob"), &path_env), Some(exec));
        // An unknown name resolves to nothing.
        assert_eq!(
            resolve_on_path(Path::new("definitely-missing"), &path_env),
            None
        );
    }
}

/// End-to-end lifecycle tests that spawn real processes. Unix-only: they use
/// `printf` / `sh` / `sleep`, and the cancel path is signal-based here.
#[cfg(all(test, unix))]
mod lifecycle {
}

