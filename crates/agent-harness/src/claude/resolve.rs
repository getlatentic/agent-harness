//! Locate the `claude` binary on the augmented PATH and classify how it was
//! installed, so [`super::ClaudeHarness::readiness`] can surface the resolved
//! absolute path and an install-kind tag.
//!
//! Why this matters: the npm `@anthropic-ai/claude-code` package is frozen at
//! 1.0.x (the native installer superseded it), and a stale npm-global copy maps
//! the `sonnet` alias to a now-deleted model id → the agent exits 1. Surfacing
//! *which* binary resolves — and whether it's the self-updating native build or
//! a package-manager copy that can go stale — lets the host nudge the user onto
//! the native installer.

use std::path::{Path, PathBuf};

/// How a resolved `claude` binary was installed, classified from its absolute
/// path. The string values are the wire contract for the readiness `details`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InstallKind {
    /// Official native installer — `~/.local/**` (Developer-ID-signed,
    /// self-updating). The vendor-canonical path.
    Native,
    /// A package-manager copy under nvm or an npm global prefix — `**/.nvm/**`
    /// or `**/node_modules/**`. Can go stale (the npm package is frozen).
    NpmGlobal,
    /// A Homebrew copy — `/opt/homebrew/**` or `/usr/local/**`.
    Homebrew,
    /// Shipped inside the host app's resource dir (the bundled runtime).
    Bundled,
    /// Resolves to something none of the above patterns match.
    Unknown,
}

impl InstallKind {
    /// The wire tag attached to readiness `details.install_kind`.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            InstallKind::Native => "native",
            InstallKind::NpmGlobal => "npm-global",
            InstallKind::Homebrew => "homebrew",
            InstallKind::Bundled => "bundled",
            InstallKind::Unknown => "unknown",
        }
    }
}

/// Resolve `program` against PATH the way `Command::new(program)` does — first
/// executable match in PATH order — returning its absolute path. `path` is the
/// augmented PATH the adapter spawns with, so this reports the binary a run
/// would actually launch. `None` when nothing on PATH matches.
pub(crate) fn resolve_on_path(program: &str, path: &str) -> Option<PathBuf> {
    // A program containing a separator is a path, not a PATH lookup (mirrors the
    // OS): resolve it directly.
    if program.contains('/') {
        let candidate = PathBuf::from(program);
        return is_executable_file(&candidate).then_some(candidate);
    }
    path.split(':')
        .filter(|dir| !dir.is_empty())
        .map(|dir| Path::new(dir).join(program))
        .find(|candidate| is_executable_file(candidate))
}

/// Classify a resolved binary path by where it lives. `home` is the user's home
/// dir; `resource_dir` is the host app's bundled-resource dir when one is known
/// (so a bundled copy is told apart from an unknown location).
pub(crate) fn classify(path: &Path, home: &Path, resource_dir: Option<&Path>) -> InstallKind {
    let resolved = canonical_or_self(path);
    let local_bin = home.join(".local");
    if resolved.starts_with(&local_bin) {
        return InstallKind::Native;
    }
    if let Some(resource_dir) = resource_dir {
        if resolved.starts_with(canonical_or_self(resource_dir)) {
            return InstallKind::Bundled;
        }
    }
    if path_contains(&resolved, ".nvm") || path_contains(&resolved, "node_modules") {
        return InstallKind::NpmGlobal;
    }
    if resolved.starts_with("/opt/homebrew") || resolved.starts_with("/usr/local") {
        return InstallKind::Homebrew;
    }
    InstallKind::Unknown
}

/// Resolve symlinks (the native installer's `~/.local/bin/claude` is a shim into
/// `~/.local/share/claude/versions/<v>`; an nvm shim likewise points into a
/// versioned dir), falling back to the original path when canonicalization fails
/// (a broken link still classifies by its declared location).
fn canonical_or_self(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// Whether any path component equals `needle` — matches `**/<needle>/**`
/// regardless of where it sits, so an nvm path under a non-standard home or an
/// npm prefix nested anywhere is caught.
fn path_contains(path: &Path, needle: &str) -> bool {
    path.components().any(|c| c.as_os_str() == needle)
}

#[cfg(unix)]
fn is_executable_file(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    match std::fs::metadata(path) {
        Ok(meta) => meta.is_file() && meta.permissions().mode() & 0o111 != 0,
        Err(_) => false,
    }
}

#[cfg(not(unix))]
fn is_executable_file(path: &Path) -> bool {
    path.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_native_local_bin() {
        let home = Path::new("/Users/dev");
        let path = Path::new("/Users/dev/.local/bin/claude");
        assert_eq!(classify(path, home, None), InstallKind::Native);
    }

    #[test]
    fn classify_nvm_is_npm_global() {
        let home = Path::new("/Users/dev");
        let path = Path::new("/Users/dev/.nvm/versions/node/v22.15.0/bin/claude");
        assert_eq!(classify(path, home, None), InstallKind::NpmGlobal);
    }

    #[test]
    fn classify_node_modules_is_npm_global() {
        let home = Path::new("/Users/dev");
        // A writable npm prefix outside nvm (Compose's bundled prefix points
        // installs at `<data>/runtime/npm/lib/node_modules/...`).
        let path = Path::new("/Users/dev/Library/App/runtime/npm/lib/node_modules/.bin/claude");
        assert_eq!(classify(path, home, None), InstallKind::NpmGlobal);
    }

    #[test]
    fn classify_homebrew() {
        let home = Path::new("/Users/dev");
        assert_eq!(
            classify(Path::new("/opt/homebrew/bin/claude"), home, None),
            InstallKind::Homebrew,
        );
        assert_eq!(
            classify(Path::new("/usr/local/bin/claude"), home, None),
            InstallKind::Homebrew,
        );
    }

    #[test]
    fn classify_bundled_when_under_resource_dir() {
        let home = Path::new("/Users/dev");
        let resource = Path::new("/Applications/Compose.app/Contents/Resources");
        let path = Path::new("/Applications/Compose.app/Contents/Resources/runtime/bin/claude");
        assert_eq!(classify(path, home, Some(resource)), InstallKind::Bundled);
    }

    #[test]
    fn classify_unknown_falls_through() {
        let home = Path::new("/Users/dev");
        assert_eq!(
            classify(Path::new("/some/random/dir/claude"), home, None),
            InstallKind::Unknown,
        );
    }

    #[test]
    fn resolve_picks_first_path_match_in_order() {
        let dir = std::env::temp_dir().join(format!("claude-resolve-{}", std::process::id()));
        let first = dir.join("first");
        let second = dir.join("second");
        std::fs::create_dir_all(&first).unwrap();
        std::fs::create_dir_all(&second).unwrap();
        // Only the second dir has the binary → it must win even though it's later.
        write_executable(&second.join("claude"));
        let path = format!("{}:{}", first.display(), second.display());
        assert_eq!(resolve_on_path("claude", &path), Some(second.join("claude")));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resolve_returns_none_when_absent() {
        let path = format!("{}", std::env::temp_dir().join("definitely-empty-xyz").display());
        assert_eq!(resolve_on_path("claude", &path), None);
    }

    #[cfg(unix)]
    fn write_executable(path: &Path) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::write(path, b"#!/bin/sh\n").unwrap();
        let mut perms = std::fs::metadata(path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(path, perms).unwrap();
    }

    #[cfg(not(unix))]
    fn write_executable(path: &Path) {
        std::fs::write(path, b"").unwrap();
    }
}
