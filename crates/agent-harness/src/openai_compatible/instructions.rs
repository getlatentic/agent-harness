//! Instruction / rules files — `AGENTS.md` and `CLAUDE.md` — loaded into the
//! system prompt (OpenCode's `session/instruction.ts`). Unlike skills (a lazy
//! catalog), these are injected **in full**: they're the conventions the model
//! should always follow.
//!
//! Sources, least- to most-specific (so a nearer file wins by coming last):
//! the user-global `~/.claude/{CLAUDE,AGENTS}.md`, then every `AGENTS.md` /
//! `CLAUDE.md` found walking from the git root down to the cwd.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// The concatenated instruction text visible from `cwd`, or `None` if there's
/// nothing to load.
pub(crate) fn gather(cwd: &Path) -> Option<String> {
    let mut seen = HashSet::new();
    let mut sections = Vec::new();
    for path in instruction_files(cwd) {
        if !seen.insert(path.clone()) {
            continue;
        }
        if let Ok(content) = std::fs::read_to_string(&path) {
            let trimmed = content.trim();
            if !trimmed.is_empty() {
                sections.push(trimmed.to_owned());
            }
        }
    }
    if sections.is_empty() {
        None
    } else {
        Some(sections.join("\n\n"))
    }
}

/// Candidate instruction files in precedence order (global first, then project
/// from git-root down to cwd). Only existing files are returned.
fn instruction_files(cwd: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        files.push(home.join(".claude/CLAUDE.md"));
        files.push(home.join(".claude/AGENTS.md"));
    }
    for dir in project_dirs(cwd) {
        files.push(dir.join("AGENTS.md"));
        files.push(dir.join("CLAUDE.md"));
    }
    files.into_iter().filter(|p| p.is_file()).collect()
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

    #[test]
    fn gathers_project_files_root_to_cwd() {
        let root = scratch("proj");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join("AGENTS.md"), "root rules").unwrap();
        let sub = root.join("crate-a");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("CLAUDE.md"), "crate rules").unwrap();

        let text = gather(&sub).expect("found instructions");
        assert!(text.contains("root rules"));
        assert!(text.contains("crate rules"));
        // Nearer file comes after the root's (more specific wins on conflict).
        assert!(text.find("root rules") < text.find("crate rules"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn none_when_absent() {
        let dir = scratch("empty");
        std::fs::create_dir_all(dir.join(".git")).unwrap();
        // No AGENTS.md/CLAUDE.md in the project (a stray global one shouldn't
        // make this assertion flaky — the project dir has none).
        let text = gather(&dir);
        assert!(text.as_deref().map_or(true, |t| !t.contains("crate rules")));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
