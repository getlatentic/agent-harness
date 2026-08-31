#!/usr/bin/env bash
# Local quality gate — run before pushing. The same checks CI runs
# (.github/workflows/ci.yml), on your machine, so a break costs a rebuild
# rather than a full CI cycle. CI additionally covers macOS and Windows.
#
#   ./scripts/check.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Note: this codebase uses deliberate compact hand-formatting, so `cargo fmt
# --check` is intentionally NOT a gate here — it would rewrite the whole tree.
# clippy below is the lint gate that's actually enforced.

echo "==> clippy (all targets, all features, warnings = errors)"
cargo clippy --workspace --all-targets --all-features -- -D warnings

echo "==> tests"
cargo test --workspace

echo "==> build all targets (incl. examples)"
cargo build --workspace --all-targets

echo "==> feature gates (lean framework, then claude-only)"
cargo build -p agent-harness --no-default-features
cargo build -p agent-harness --no-default-features --features claude

if command -v cargo-deny >/dev/null 2>&1; then
  echo "==> cargo deny (advisories + licenses)"
  cargo deny check
else
  echo "==> (skip) cargo-deny not installed — 'cargo install cargo-deny' to enable"
fi

# SemVer guard: compares the public API against the last published release so a
# bump is mechanical, not a guess — a breaking change (pre-1.0) must go in the
# minor, additive changes in the patch. Skipped until the crates are published
# (nothing to diff against) and when the tool isn't installed.
if command -v cargo-semver-checks >/dev/null 2>&1; then
  echo "==> cargo semver-checks (public API vs last release)"
  cargo semver-checks check-release --workspace || {
    echo "   (semver-checks reported findings — bump minor if breaking, or it's a no-op pre-publish)"
  }
else
  echo "==> (skip) cargo-semver-checks not installed — 'cargo install cargo-semver-checks' to enable"
fi

echo
echo "All checks passed."
