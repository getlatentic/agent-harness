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
//!
//! # Platforms
//!
//! Resolution is portable — `PATH` is split with [`std::env::split_paths`] and
//! Windows names are tried with each `PATHEXT` suffix. The **fallback list is
//! not**: [`hardcoded_node_dirs`] names Homebrew, `~/.local/bin` and
//! `~/.nvm/versions/node/*/bin`, which are macOS and Linux locations, and
//! [`login_shell_path`] asks a POSIX login shell. On Windows both come back
//! empty or unhelpful, so a CLI outside the inherited `PATH` will not be found
//! — nvm-windows keeps its versions under `%APPDATA%\nvm`, which nothing here
//! looks for yet.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

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

/// The first runnable file called `name` on `path_env`. Pure with respect to
/// env and spawn (filesystem only), so it is unit-testable.
///
/// Entries are split with [`std::env::split_paths`] rather than on `:`, because
/// Windows separates with `;` — and on Windows a bare name is not the file
/// name: `claude` is `claude.exe` or `claude.cmd`, so each `PATHEXT` suffix is
/// tried in turn.
fn resolve_on_path(name: &Path, path_env: &str) -> Option<PathBuf> {
    // Once, not once per directory: on Windows this reads an environment
    // variable, and a PATH routinely has dozens of entries.
    let extensions = split_extensions(&pathext());
    std::env::split_paths(path_env)
        .filter(|dir| !dir.as_os_str().is_empty())
        .flat_map(|dir| {
            let base = dir.join(name);
            let mut candidates = vec![base.clone()];
            for extension in &extensions {
                let mut with_extension = base.clone().into_os_string();
                with_extension.push(extension);
                candidates.push(PathBuf::from(with_extension));
            }
            candidates
        })
        .find(|candidate| is_executable_file(candidate))
}

/// Unix has no extension convention for programs — a program's name is its
/// file name.
#[cfg(unix)]
fn pathext() -> String {
    String::new()
}

/// Windows names its programs `claude.exe` / `claude.cmd`, and `PATHEXT` lists
/// the suffixes to try; the literal is what the OS falls back to when it is
/// unset.
#[cfg(not(unix))]
fn pathext() -> String {
    std::env::var("PATHEXT").unwrap_or_else(|_| ".EXE;.CMD;.BAT;.COM".to_owned())
}

/// Split a `PATHEXT` value into suffixes.
///
/// Separated from [`pathext`] so the parsing compiles and is tested on every
/// platform, not only the one it ships on: behind a `cfg` it was unreachable
/// from any test here, which reads as untested rather than as passing.
fn split_extensions(pathext: &str) -> Vec<String> {
    pathext
        .split(';')
        .filter(|extension| !extension.is_empty())
        .map(str::to_owned)
        .collect()
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

/// `base_path` with the program's own directory in front, so the `node` it was
/// installed beside — the sibling in an nvm install — is the one its shebang
/// finds. Pure (no env, no spawn), so it is unit-tested directly.
///
/// Only an **absolute** directory is prepended. This runs after
/// [`keep_absolute_entries`] and lands at the front, so a relative one would
/// outrank every filtered entry and reopen exactly the hole that filter
/// closes: we spawn with `current_dir` set to the user's workspace, which the
/// agent itself can write to, so `node_modules/.bin/claude` would put a
/// workspace-relative directory first on PATH.
///
/// Joined with [`std::env::join_paths`] rather than `:` — Windows separates
/// with `;`, where a hardcoded colon builds a PATH the OS reads as one
/// nonexistent directory. A directory that cannot be expressed in a PATH at
/// all (it contains the separator) yields `base_path` unchanged: without the
/// prepend we lose the node pairing, but a corrupt PATH loses everything.
fn prepend_program_dir(program: &Path, base_path: &str) -> String {
    let Some(dir) = program.parent().filter(|dir| dir.is_absolute()) else {
        return base_path.to_owned();
    };
    let entries = std::iter::once(dir.to_path_buf()).chain(std::env::split_paths(base_path));
    std::env::join_paths(entries)
        .map_or_else(|_| base_path.to_owned(), |joined| joined.to_string_lossy().into_owned())
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
    // The user's real PATH (nvm/pnpm/volta/asdf/Homebrew) via their login
    // shell; a hardcoded best-effort list if that's unavailable.
    let discovered = login_shell_path().unwrap_or_else(hardcoded_node_dirs);
    compose_augmented_path(std::env::var("PATH").ok(), discovered)
}

/// The process's own PATH first — anything explicitly set still wins — then
/// whatever discovery turned up.
///
/// Takes both as arguments rather than reading the environment, so the
/// "already set" case can be tested with a PATH this process does not have.
/// Reading it directly, the only assertion available was that some entry of
/// the real PATH survived — and the discovered PATH contains those same
/// entries, so the test passed whether the guard worked or not.
fn compose_augmented_path(process_path: Option<String>, discovered: String) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(existing) = process_path.filter(|path| !path.is_empty()) {
        parts.push(existing);
    }
    parts.push(discovered);
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
/// Unix only, and a module rather than scattered `#[cfg]` attributes: the
/// imports this needs are unused on Windows, and gating them one by one is how
/// `Arc`/`Mutex` ended up gated in `cli-stream` while the code using them was
/// not — a break nobody sees without cross-compiling. Kept together, a mismatch
/// cannot compile on either platform.
#[cfg(unix)]
mod login_shell {
    use std::io::Read;
    use std::process::{Command, Stdio};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    const PATH_SENTINEL: &str = "__CLI_STREAM_PATH__";

    pub(super) fn query() -> Option<String> {
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

    /// Extract the `PATH=…` value from the shell's `printf <sentinel>; env` output.
    /// Everything up to (and including) the last sentinel is discarded — that's
    /// where shell-init chatter and terminal escape sequences live — then the
    /// `PATH=` line is read from the clean `env` dump that follows. `None` if the
    /// sentinel is missing (query misbehaved) or PATH is absent/empty.
    pub(super) fn parse_path_from_shell_output(output: &str) -> Option<String> {
        output
            .rsplit_once(PATH_SENTINEL)?
            .1
            .lines()
            .find_map(|line| line.strip_prefix("PATH="))
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(str::to_owned)
    }

} // mod login_shell

/// The user's real PATH, or `None` where we cannot ask: the query is a POSIX
/// shell invocation, so Windows falls straight through to the hardcoded list.
fn login_shell_path() -> Option<String> {
    #[cfg(unix)]
    {
        login_shell::query()
    }
    #[cfg(not(unix))]
    {
        None
    }
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


/// Resolving an agent's CLI before running it.
///
/// An extension on [`cli_stream::Command`], so resolution reads as a step in
/// the same builder rather than a function wrapping it:
///
/// ```no_run
/// use cli_stream::Command;
/// use harness::ResolveCli;
///
/// # fn main() -> Result<(), cli_stream::StreamError> {
/// let handle = Command::new("claude").args(["-p", "hi"]).resolve_cli().stream(|_| {})?;
/// # let _ = handle;
/// # Ok(())
/// # }
/// ```
pub trait ResolveCli {
    /// Resolve a bare program name to its absolute path, and put that program's
    /// own directory at the front of `PATH`.
    ///
    /// Every adapter driving a CLI goes through here. The engine takes the
    /// environment it is given — spawning is its job, knowing where a user's
    /// nvm lives is not — so this is the one place that knowledge is applied,
    /// and the first place to look when a packaged app cannot find a CLI that a
    /// terminal finds fine.
    ///
    /// A `PATH` the caller set still wins: it is applied after this one.
    #[must_use]
    fn resolve_cli(self) -> Self;
}

impl ResolveCli for cli_stream::Command {
    fn resolve_cli(self) -> Self {
        let program = resolve_program(self.program);
        let mut env = vec![("PATH".to_owned(), augment_path_for_node(&program))];
        env.extend(self.env);
        cli_stream::Command { program, env, ..self }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// PATH-ish entries: absolute dirs, plus every shape the filter exists to
    /// reject — relative, bare, dot, and empty.
    fn path_entry() -> impl Strategy<Value = String> {
        prop_oneof![
            4 => "/(usr|opt|home)(/[a-z]{1,6}){0,3}",
            1 => "[a-z]{1,6}(/[a-z]{1,6}){0,2}",
            1 => Just(".".to_owned()),
            1 => Just(String::new()),
        ]
    }

    fn path_string() -> impl Strategy<Value = String> {
        prop::collection::vec(path_entry(), 0..8).prop_map(|entries| entries.join(":"))
    }

    fn entries(path: &str) -> Vec<&str> {
        path.split(':').collect()
    }

    proptest! {
        /// The security invariant, stated once for every input rather than for
        /// three examples: nothing that could resolve against the spawn cwd —
        /// the user's workspace, where the agent itself writes files — survives.
        #[test]
        fn no_entry_that_resolves_against_the_cwd_survives(path in path_string()) {
            let kept = keep_absolute_entries(&path);
            if kept.is_empty() {
                return Ok(());
            }
            for entry in entries(&kept) {
                prop_assert!(entry.starts_with('/'), "{entry:?} is not absolute");
            }
        }

        /// The other half, and the one a safety property cannot state: every
        /// real directory survives. "Drop everything" satisfies "nothing
        /// relative survives" perfectly, and would present every installed CLI
        /// as missing — so the filter has to be pinned from both sides.
        #[test]
        fn every_absolute_directory_survives(path in path_string()) {
            let kept = keep_absolute_entries(&path);
            let survivors = entries(&kept);
            for entry in entries(&path).into_iter().filter(|entry| entry.starts_with('/')) {
                prop_assert!(survivors.contains(&entry), "dropped {entry:?}");
            }
        }

        /// And it only ever removes: no entry is invented or rewritten.
        #[test]
        fn filtering_never_invents_an_entry(path in path_string()) {
            let kept = keep_absolute_entries(&path);
            if kept.is_empty() {
                return Ok(());
            }
            let original = entries(&path);
            for entry in entries(&kept) {
                prop_assert!(original.contains(&entry), "{entry:?} was not in the input");
            }
        }

        /// Prepending the program's own directory must not undo the filter.
        /// It runs *after* `keep_absolute_entries` and lands at the front, so a
        /// relative directory here outranks every real one.
        #[test]
        fn prepending_cannot_reintroduce_a_cwd_relative_entry(
            program in "([a-z]{1,6}/){0,3}[a-z]{1,6}",
            base in path_string(),
        ) {
            let base = keep_absolute_entries(&base);
            let combined = prepend_program_dir(Path::new(&program), &base);
            if combined.is_empty() {
                return Ok(());
            }
            for entry in entries(&combined) {
                prop_assert!(entry.starts_with('/'), "{entry:?} is not absolute");
            }
        }

        /// Prepending is additive: every directory already on the path is still
        /// on it. Losing one silently makes a CLI "not installed".
        #[test]
        fn prepending_keeps_every_directory_it_was_given(base in path_string()) {
            let base = keep_absolute_entries(&base);
            let combined = prepend_program_dir(Path::new("/opt/tool/bin/claude"), &base);
            for entry in entries(&base).into_iter().filter(|entry| !entry.is_empty()) {
                prop_assert!(entries(&combined).contains(&entry), "lost {entry:?}");
            }
        }
    }

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
        use super::login_shell::parse_path_from_shell_output;

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
    fn pathext_becomes_suffixes_with_the_empty_ones_dropped() {
        // A trailing or doubled `;` is ordinary in a real PATHEXT, and an empty
        // suffix would probe the bare name a second time rather than a variant.
        assert_eq!(
            split_extensions(".EXE;.CMD;;.BAT;"),
            [".EXE", ".CMD", ".BAT"],
        );
        // What unix supplies: no suffixes, so only the bare name is tried.
        assert!(split_extensions("").is_empty());
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

    #[cfg(unix)]
    #[test]
    fn only_a_runnable_file_counts_as_the_program() {
        // This is the filter that decides whether a name found on PATH is a CLI
        // we can run. Saying yes to a directory or an unexecutable file picks it
        // over the real binary further down PATH.
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("hl-exec-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let runnable = dir.join("runnable");
        std::fs::write(&runnable, "#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&runnable, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(is_executable_file(&runnable));

        let plain = dir.join("plain.txt");
        std::fs::write(&plain, "not a program").unwrap();
        assert!(!is_executable_file(&plain), "a readable file is not a runnable one");
        assert!(!is_executable_file(&dir), "a directory is not a program");
        assert!(!is_executable_file(&dir.join("absent")), "and neither is nothing");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_bare_name_is_resolved_to_the_binary_it_will_actually_run() {
        // The point of resolving before spawning: the absolute path is what
        // pairs a CLI with the `node` beside it. Left as a bare name, the child
        // resolves it against whatever PATH it ends up with instead.
        let resolved = resolve_program(PathBuf::from("sh"));
        assert!(resolved.is_absolute(), "a name on PATH resolves to its real location: {resolved:?}");
        assert!(resolved.ends_with("sh"), "and to the right binary: {resolved:?}");

        let unknown = PathBuf::from("definitely-not-a-real-binary-xyz");
        assert_eq!(
            resolve_program(unknown.clone()),
            unknown,
            "an unresolvable name is left alone so the spawn reports the real error"
        );
    }

    #[test]
    fn the_augmented_path_extends_the_one_we_already_have() {
        // Augmenting must add, never replace: a PATH the host deliberately set
        // has to keep working, or a run that was fine becomes "not installed".
        let existing = std::env::var("PATH").expect("a test process has a PATH");
        let augmented = compute_augmented_node_path();
        let first = existing.split(':').find(|e| e.starts_with('/')).expect("an absolute entry");
        assert!(augmented.contains(first), "{first} must survive into {augmented}");
    }

    #[test]
    fn the_process_path_leads_and_an_absent_one_contributes_nothing() {
        // Order is the whole point: a PATH the host deliberately set has to be
        // searched before anything we discovered, or we override a deliberate
        // choice. Asserted against directories this process does not have, so
        // it cannot pass because the real PATH happened to contain them.
        assert_eq!(
            compose_augmented_path(Some("/host/bin".to_owned()), "/found/bin".to_owned()),
            "/host/bin:/found/bin",
        );
        // Unset and empty both mean "nothing to keep" — and must not leave an
        // empty entry behind, which is the implicit-cwd vector.
        assert_eq!(compose_augmented_path(None, "/found/bin".to_owned()), "/found/bin");
        assert_eq!(
            compose_augmented_path(Some(String::new()), "/found/bin".to_owned()),
            "/found/bin",
        );
    }

    #[test]
    fn the_fallback_looks_where_agent_clis_are_actually_installed() {
        // Used when the login shell cannot be asked. Missing the home-relative
        // directories is what leaves an nvm-installed CLI invisible.
        let dirs = hardcoded_node_dirs();
        assert!(dirs.contains("/usr/local/bin") && dirs.contains("/opt/homebrew/bin"));
        if let Ok(home) = std::env::var("HOME") {
            if !home.is_empty() {
                assert!(
                    dirs.contains(&format!("{home}/.local/bin")),
                    "the official-installer location is where several agent CLIs land: {dirs}"
                );
            }
        }
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

    /// Write a file the resolver should accept as runnable, named the way the
    /// platform names programs, and answer with the bare name to look it up by.
    /// On Windows those differ — that gap *is* what `PATHEXT` probing closes.
    fn install_runnable(dir: &Path, stem: &str) -> PathBuf {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let path = dir.join(stem);
            std::fs::write(&path, "#!/bin/sh\n").unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
            path
        }
        #[cfg(not(unix))]
        {
            let path = dir.join(format!("{stem}.EXE"));
            std::fs::write(&path, "").unwrap();
            path
        }
    }

    #[test]
    fn a_bare_name_resolves_however_the_platform_spells_the_file() {
        // The unix cases above are gated, which left everything about
        // resolution untested on Windows — including the suffix probing that
        // exists only for Windows, where `claude` is `claude.exe`. This is the
        // same assertion with the platform's own naming factored out, so CI
        // exercises the probe on the platform it was written for.
        let root = tempfile::tempdir().expect("tempdir");
        let installed = install_runnable(root.path(), "tool");
        let path_env = root.path().display().to_string();

        assert_eq!(resolve_on_path(Path::new("tool"), &path_env), Some(installed));
        assert_eq!(resolve_on_path(Path::new("tool-missing"), &path_env), None);
    }
}

/// Spawning a real CLI and looking at the environment it actually got. Unix
/// only: the fixture is a shell script.
#[cfg(all(test, unix))]
mod spawned {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// A CLI that prints the PATH it was handed, so a test can see what the
    /// child really received rather than what we meant to send.
    fn path_echoing_cli(tag: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("hl-spawn-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cli = dir.join("fake-agent");
        std::fs::write(&cli, "#!/bin/sh\nprintf '%s\\n' \"$PATH\"\n").unwrap();
        std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o755)).unwrap();
        cli
    }

    /// A stand-in for the user's login shell. It ignores its `-lic` arguments
    /// and answers the way a real one does: startup chatter first, then the
    /// sentinel, then an `env` dump — the shape the parser has to survive.
    fn fake_login_shell(tag: &str, path_line: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("hl-shell-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let shell = dir.join("fake-shell");
        std::fs::write(
            &shell,
            format!(
                "#!/bin/sh\nprintf 'rc chatter\\n'\nprintf '\\n__CLI_STREAM_PATH__\\n'\n\
                 printf 'HOME=/x\\n{path_line}\\nTERM=xterm\\n'\n"
            ),
        )
        .unwrap();
        std::fs::set_permissions(&shell, std::fs::Permissions::from_mode(0o755)).unwrap();
        shell
    }

    /// `SHELL` is process-global, so the two cases below cannot run alongside
    /// each other.
    static SHELL_ENV: Mutex<()> = Mutex::new(());

    #[test]
    fn the_path_comes_from_the_shell_we_asked() {
        // This is the mechanism behind a CLI reading as installed at all in a
        // Finder-launched app, and it was previously covered only down to its
        // parser — the spawn, the sentinel handshake and the `$SHELL` guard had
        // nothing exercising them. A fake shell reaches all three without
        // depending on how this machine's rc happens to be set up.
        let _guard = SHELL_ENV.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let restore = std::env::var("SHELL").ok();

        let shell = fake_login_shell("ok", "PATH=/fake/node/bin:/usr/bin");
        std::env::set_var("SHELL", &shell);
        assert_eq!(
            login_shell_path().as_deref(),
            Some("/fake/node/bin:/usr/bin"),
            "the answer must come from the shell, past its startup chatter",
        );

        // No shell to ask is not an empty PATH — the caller must fall back to
        // the hardcoded list rather than treat "" as the user's real PATH.
        std::env::set_var("SHELL", "");
        assert_eq!(login_shell_path(), None);

        match restore {
            Some(value) => std::env::set_var("SHELL", value),
            None => std::env::remove_var("SHELL"),
        }
    }

    fn run(program: PathBuf, env: Vec<(String, String)>) -> String {
        let lines: Arc<Mutex<Vec<String>>> = Arc::default();
        let sink = Arc::clone(&lines);
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = Arc::clone(&done);
        let spawn = cli_stream::Command::new(program).cwd(std::env::temp_dir()).run_id("t").env(env);
        let _handle = spawn.resolve_cli().stream(move |event| {
            match event {
                cli_stream::Event::Stdout { line, .. } => sink.lock().unwrap().push(line),
                cli_stream::Event::Exited { .. } => flag.store(true, std::sync::atomic::Ordering::SeqCst),
                _ => {}
            }
        })
        .expect("the fixture should spawn");
        let mut finished = false;
        for _ in 0..200 {
            if done.load(std::sync::atomic::Ordering::SeqCst) {
                finished = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        assert!(finished, "the fixture never exited; its output would be whatever arrived in time");
        let out = lines.lock().unwrap().join("\n");
        out
    }

    #[test]
    fn a_spawned_cli_gets_its_own_directory_at_the_front_of_path() {
        // The whole reason this module exists. A Finder-launched .app inherits
        // `/usr/bin:/bin:/usr/sbin:/sbin`, so a CLI installed under nvm cannot
        // see the `node` it was installed beside and exits 127.
        //
        // Nothing in a terminal reproduces that — `open <app>` leaks the
        // launching shell's PATH and passes either way — so this assertion is
        // the only thing standing between the fix and its silent removal.
        let cli = path_echoing_cli("front");
        let parent = cli.parent().unwrap().display().to_string();
        let seen = run(cli.clone(), Vec::new());

        assert!(
            seen.starts_with(&parent),
            "the program's own directory must lead PATH.\n  wanted first: {parent}\n  child saw:    {seen}"
        );
        let _ = std::fs::remove_dir_all(cli.parent().unwrap());
    }

    #[test]
    fn a_path_the_caller_supplies_still_wins() {
        // Documented behaviour: the augmentation is a floor, not a cage. A host
        // that knows exactly which environment it wants gets it.
        let cli = path_echoing_cli("override");
        let seen = run(cli.clone(), vec![("PATH".to_owned(), "/only/this".to_owned())]);
        assert_eq!(seen.trim(), "/only/this", "the caller's PATH is applied last");
        let _ = std::fs::remove_dir_all(cli.parent().unwrap());
    }
}

