//! Instruction / rules files — `AGENTS.md`, `CLAUDE.md` — loaded into the
//! system prompt. Unlike skills (a lazy catalog), these are injected **in
//! full**: they're the conventions the model should always follow.
//!
//! Two rules keep that affordable.
//!
//! **First match wins**, per location. A directory holding both an `AGENTS.md`
//! and a `CLAUDE.md` contributes one of them, not both — the two normally say
//! the same thing, and stacking them charges twice for it. `AGENTS.md` is the
//! cross-tool standard, so it comes first; `CLAUDE.md` is the fallback.
//!
//! **A running byte budget** across every file, so a large global file cannot
//! crowd out the project's own rules and no chain can grow without bound. The
//! budget is spent nearest-first for that reason.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Candidate filenames in a directory, most preferred first. `AGENTS.md` is the
/// standard shared across coding agents; `CLAUDE.md` predates it and is still
/// what many repositories carry. Case variants appear because
/// case-sensitive filesystems do not forgive them.
const FILENAMES: &[&str] = &["AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD"];

/// Default cap on the instruction text taken from disk, matching Codex's
/// `project_doc_max_bytes`. Large enough for real conventions, small enough
/// that a runaway file cannot fill a local model's context.
pub(crate) const DEFAULT_MAX_BYTES: usize = 32 * 1024;

/// Where to look for instructions beyond the working tree.
///
/// `global` is empty by default. A library that reads a user's home directory
/// on its own initiative is guessing, and the guess is wrong the moment the
/// host has its own convention — so the host names the files it wants and
/// nothing outside the working tree is read until it does.
/// [`InstructionSources::discover_global`] supplies the usual suspects for a
/// host that wants them.
#[derive(Clone, Debug)]
pub struct InstructionSources {
    /// Global candidates, most preferred first. The first one that exists is
    /// used; the rest are ignored.
    pub global: Vec<PathBuf>,
    /// Cap on the instruction text taken from disk. Files are read
    /// nearest-first and the remainder is truncated once the budget runs out.
    pub max_bytes: usize,
}

impl Default for InstructionSources {
    fn default() -> Self {
        Self { global: Vec::new(), max_bytes: DEFAULT_MAX_BYTES }
    }
}

impl InstructionSources {
    /// The conventional global instruction files, most preferred first:
    /// `~/.config/AGENTS.md`, then the two agents that ship their own
    /// (`~/.codex/AGENTS.md`, `~/.claude/CLAUDE.md`).
    ///
    /// Opt in by calling this — a host that wants a user's existing global
    /// conventions honoured, wherever that user already keeps them.
    pub fn discover_global() -> Self {
        let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
            return Self::default();
        };
        Self {
            global: vec![
                home.join(".config/AGENTS.md"),
                home.join(".codex/AGENTS.md"),
                home.join(".claude/CLAUDE.md"),
            ],
            ..Self::default()
        }
    }
}

/// The concatenated instruction text visible from `cwd`, or `None` if there is
/// nothing to load.
///
/// Ordering is least- to most-specific, so a nearer file wins on conflict by
/// being read last. The byte budget is spent in the opposite direction —
/// nearest first — so a project's own rules survive a large global file.
pub(crate) fn gather(cwd: &Path, sources: &InstructionSources) -> Option<String> {
    let mut sections = Vec::new();
    let mut remaining = sources.max_bytes;

    for path in resolve(cwd, sources).into_iter().rev() {
        if remaining == 0 {
            break;
        }
        let Ok(content) = std::fs::read_to_string(&path) else { continue };
        let trimmed = content.trim();
        if trimmed.is_empty() {
            continue;
        }
        sections.push(take_within(trimmed, &mut remaining));
    }

    if sections.is_empty() {
        return None;
    }
    sections.reverse(); // back to least- → most-specific
    Some(sections.join("\n\n"))
}

/// Up to `remaining` bytes of `text`, on a character boundary, decrementing the
/// budget by what was taken.
fn take_within(text: &str, remaining: &mut usize) -> String {
    if text.len() <= *remaining {
        *remaining -= text.len();
        return text.to_owned();
    }
    let mut end = *remaining;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    *remaining = 0;
    text[..end].to_owned()
}

/// Existing instruction files, least- to most-specific: the first global
/// candidate that exists, then one file per directory from the project root
/// down to `cwd`.
fn resolve(cwd: &Path, sources: &InstructionSources) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut seen = HashSet::new();

    if let Some(global) = sources.global.iter().find(|path| path.is_file()) {
        files.push(global.clone());
        seen.insert(global.clone());
    }
    for dir in project_dirs(cwd) {
        if let Some(found) = first_in(&dir) {
            if seen.insert(found.clone()) {
                files.push(found);
            }
        }
    }
    files
}

/// The preferred instruction file present in `dir`, if any.
fn first_in(dir: &Path) -> Option<PathBuf> {
    FILENAMES.iter().map(|name| dir.join(name)).find(|path| path.is_file())
}

/// The cwd's directory chain from the git root down to the cwd (so nearer dirs
/// come last). If no `.git` is found walking up, just the cwd — we don't scan
/// the whole filesystem.
fn project_dirs(cwd: &Path) -> Vec<PathBuf> {
    let mut chain = Vec::new();
    let mut cur = cwd;
    loop {
        chain.push(cur.to_path_buf());
        if cur.join(".git").exists() {
            chain.reverse(); // root → cwd
            return chain;
        }
        match cur.parent() {
            Some(parent) => cur = parent,
            None => return vec![cwd.to_path_buf()], // no git root → cwd only
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hl-instr-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Project files only — the default reads nothing outside the working tree.
    fn project_only() -> InstructionSources {
        InstructionSources::default()
    }

    #[test]
    fn gathers_project_files_root_to_cwd() {
        let root = scratch("proj");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join("AGENTS.md"), "root rules").unwrap();
        let sub = root.join("crate-a");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("CLAUDE.md"), "crate rules").unwrap();

        let text = gather(&sub, &project_only()).expect("found instructions");
        assert!(text.contains("root rules") && text.contains("crate rules"));
        // Nearer file comes after the root's (more specific wins on conflict).
        assert!(text.find("root rules") < text.find("crate rules"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_directory_contributes_one_file_not_both() {
        // The common case for a repo that supports several agents: the two
        // files say the same thing, and reading both charged twice for it.
        let root = scratch("both");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join("AGENTS.md"), "the standard").unwrap();
        std::fs::write(root.join("CLAUDE.md"), "the fallback").unwrap();

        let text = gather(&root, &project_only()).expect("found instructions");
        assert!(text.contains("the standard"), "AGENTS.md is preferred: {text}");
        assert!(!text.contains("the fallback"), "CLAUDE.md must not stack: {text}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn nothing_outside_the_working_tree_is_read_by_default() {
        let home = scratch("home");
        std::fs::write(home.join("global.md"), "global rules").unwrap();
        let root = scratch("no-global");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join("AGENTS.md"), "project rules").unwrap();

        let text = gather(&root, &InstructionSources::default()).expect("instructions");
        assert!(!text.contains("global rules"), "default must not reach outside: {text}");

        // ...and does read it once the host asks.
        let opted_in = InstructionSources {
            global: vec![home.join("global.md")],
            ..Default::default()
        };
        let text = gather(&root, &opted_in).expect("instructions");
        assert!(text.contains("global rules"), "host opt-in must be honoured: {text}");
        assert!(text.find("global rules") < text.find("project rules"), "global is least specific");
        let _ = std::fs::remove_dir_all(&home);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_first_global_candidate_that_exists_wins() {
        let home = scratch("globals");
        std::fs::write(home.join("second.md"), "second choice").unwrap();
        std::fs::write(home.join("third.md"), "third choice").unwrap();
        let root = scratch("g-proj");
        std::fs::create_dir_all(root.join(".git")).unwrap();

        let sources = InstructionSources {
            global: vec![home.join("first.md"), home.join("second.md"), home.join("third.md")],
            ..Default::default()
        };
        let text = gather(&root, &sources).expect("instructions");
        assert!(text.contains("second choice"), "first existing candidate: {text}");
        assert!(!text.contains("third choice"), "later candidates are ignored: {text}");
        let _ = std::fs::remove_dir_all(&home);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_budget_truncates_and_spends_it_on_the_nearest_file() {
        // A large global file must not crowd out the project's own rules, so
        // the budget is spent nearest-first even though output stays ordered
        // least- to most-specific.
        let home = scratch("budget-home");
        std::fs::write(home.join("global.md"), "G".repeat(500)).unwrap();
        let root = scratch("budget-proj");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join("AGENTS.md"), "P".repeat(100)).unwrap();

        let sources = InstructionSources { global: vec![home.join("global.md")], max_bytes: 300 };
        let text = gather(&root, &sources).expect("instructions");

        assert_eq!(text.matches('P').count(), 100, "the nearest file is kept whole: {}", text.len());
        assert_eq!(text.matches('G').count(), 200, "the global file takes only what is left");
        assert!(text.len() <= 300 + 2, "budget honoured, plus the joining separator");
        let _ = std::fs::remove_dir_all(&home);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn truncation_never_splits_a_character() {
        let root = scratch("utf8");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join("AGENTS.md"), "é".repeat(10)).unwrap(); // 2 bytes each

        // An odd budget lands mid-character unless the cut is boundary-aware.
        let sources = InstructionSources { global: Vec::new(), max_bytes: 5 };
        let text = gather(&root, &sources).expect("instructions");
        assert_eq!(text, "éé", "cut back to the boundary rather than panicking");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn none_when_absent() {
        let dir = scratch("empty");
        std::fs::create_dir_all(dir.join(".git")).unwrap();
        assert!(gather(&dir, &project_only()).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
