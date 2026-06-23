//! Readiness probe for the bob CLI + its dependencies.
//!
//! Probes `bob --version` + `node`/`npm` with the harness's augmented PATH
//! (cli-stream's login-shell-derived PATH, sentinel-parsed and cached) so
//! nvm/Homebrew-installed binaries resolve even from a Finder-launched GUI app.
//! This matches the claude/codex adapters; a raw `$SHELL -l -c` is fragile
//! without a controlling TTY (it can exit non-zero on shell-startup noise).
//!
//! The returned snapshot is the canonical wire shape served both
//! by the Tauri command and the axum endpoint — keep the field
//! names in sync with `src/lib/ipc/settingsClient.ts`.

use crate::{keychain::auth_source, BOB_MIN_NODE_VERSION};
use serde::Serialize;
use std::process::{Command, Stdio};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BobReadinessSnapshot {
    pub bob: BobProbe,
    pub node: NodeProbe,
    pub npm: NpmProbe,
    pub auth: AuthProbe,
    /// Convenience boolean — true iff bob is usable right now
    /// (CLI installed, key configured). The browser reads this
    /// to drive the "not connected → setup" CTA without
    /// re-implementing the AND.
    pub ready: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BobProbe {
    pub installed: bool,
    pub version: Option<String>,
    pub path: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeProbe {
    pub installed: bool,
    pub version: Option<String>,
    /// True iff the detected version meets bob's minimum.
    pub satisfies_min: bool,
    /// Bob's documented Node floor. Surfaced so the UI doesn't
    /// hard-code the version string.
    pub min_version: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NpmProbe {
    pub installed: bool,
    pub version: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AuthProbe {
    pub configured: bool,
    pub source: Option<String>,
}

/// Run the full readiness probe. All sub-probes execute
/// sequentially because they share the same login-shell startup
/// cost — parallelizing would just multiply the number of bash
/// instances spun up.
pub fn get_readiness() -> BobReadinessSnapshot {
    let bob_version_raw = probe("bob", &["--version"]);
    let bob_version = bob_version_raw.as_ref().and_then(|v| {
        // `bob --version` prints "bob 1.0.4" — take the last token.
        v.split_whitespace().last().map(|s| s.to_owned())
    });
    // bob answering `--version` is the real "installed" signal; resolve its
    // absolute path (for display) only once we know it runs.
    let bob_installed = bob_version.is_some();
    let bob_path = bob_installed
        .then(|| cli_stream::resolve_program(std::path::PathBuf::from("bob")))
        .map(|path| path.to_string_lossy().into_owned());

    let node_raw = probe("node", &["--version"]);
    let node_installed = node_raw.is_some();
    let node_satisfies = node_raw
        .as_deref()
        .map(|v| semver_at_least(v, BOB_MIN_NODE_VERSION))
        .unwrap_or(false);

    let npm_raw = probe("npm", &["--version"]);

    // Important: this is the *boot-time* probe. We deliberately
    // use `auth_source()` (marker-file + env check, no keychain
    // touch) instead of `resolve_api_key()` (which would prompt
    // the user for their macOS login password before they've
    // even seen the app). The full value is fetched later by
    // `run::spawn_bob` when the user actually invokes bob.
    let auth = match auth_source() {
        Some(source) => AuthProbe {
            configured: true,
            source: Some(match source {
                crate::keychain::KeySource::Env => "env".to_owned(),
                crate::keychain::KeySource::Keychain => "keychain".to_owned(),
            }),
        },
        None => AuthProbe { configured: false, source: None },
    };

    BobReadinessSnapshot {
        bob: BobProbe {
            installed: bob_installed,
            path: bob_path,
            version: bob_version,
            error: if bob_installed { None } else { Some("bob CLI not found on PATH".to_owned()) },
        },
        node: NodeProbe {
            installed: node_installed,
            version: node_raw,
            satisfies_min: node_satisfies,
            min_version: BOB_MIN_NODE_VERSION.to_owned(),
        },
        npm: NpmProbe {
            installed: npm_raw.is_some(),
            version: npm_raw,
        },
        ready: bob_installed && node_satisfies && auth.configured,
        auth,
    }
}

/// Run `program args…` with the harness's augmented PATH and return its trimmed
/// stdout, or `None` if it fails or prints nothing. `augmented_node_path()`
/// resolves nvm/Homebrew/etc. via the user's login shell (queried once, cached,
/// bounded) and is robust to shell-startup noise — unlike spawning `$SHELL -l
/// -c`, whose exit status a no-TTY GUI app can't rely on.
fn probe(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program)
        .args(args)
        .env("PATH", cli_stream::augmented_node_path())
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let trimmed = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if trimmed.is_empty() { None } else { Some(trimmed) }
}

/// Compare `vX.Y.Z` (or `X.Y.Z`) strings. Returns true iff
/// `actual >= minimum`. Tolerant of the `v` prefix and missing
/// patch segments. Used both at boot (readiness) and during
/// install (precondition check).
pub(crate) fn semver_at_least(actual: &str, minimum: &str) -> bool {
    fn parse(raw: &str) -> (u32, u32, u32) {
        let clean = raw.trim_start_matches('v').split('-').next().unwrap_or("");
        let mut parts = clean.split('.').map(|seg| seg.parse::<u32>().unwrap_or(0));
        (parts.next().unwrap_or(0), parts.next().unwrap_or(0), parts.next().unwrap_or(0))
    }
    let (a_maj, a_min, a_patch) = parse(actual);
    let (m_maj, m_min, m_patch) = parse(minimum);
    if a_maj != m_maj { return a_maj > m_maj; }
    if a_min != m_min { return a_min > m_min; }
    a_patch >= m_patch
}

#[cfg(test)]
mod tests {
    use super::semver_at_least;

    #[test]
    fn handles_v_prefix() {
        assert!(semver_at_least("v22.15.0", "22.15.0"));
        assert!(semver_at_least("22.15.0", "v22.15.0"));
    }

    #[test]
    fn major_dominates() {
        assert!(semver_at_least("v24.0.0", "22.15.0"));
        assert!(!semver_at_least("v18.99.99", "22.15.0"));
    }

    #[test]
    fn minor_dominates_within_major() {
        assert!(semver_at_least("v22.16.0", "22.15.0"));
        assert!(!semver_at_least("v22.14.999", "22.15.0"));
    }

    #[test]
    fn patch_inclusive() {
        assert!(semver_at_least("v22.15.0", "22.15.0"));
        assert!(!semver_at_least("v22.15.0", "22.15.1"));
    }

    #[test]
    fn ignores_prerelease_suffix() {
        assert!(semver_at_least("v22.15.0-rc.1", "22.15.0"));
    }
}
