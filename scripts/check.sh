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

# `--all-features` because that is what CI runs. Without it the ACP adapter is
# not compiled, and a change to a shared type can pass here and fail there.
echo "==> tests"
cargo test --workspace --all-features

echo "==> build all targets (incl. examples)"
cargo build --workspace --all-targets

echo "==> feature gates (lean framework, then claude-only)"
cargo build -p agent-harness --no-default-features
cargo build -p agent-harness --no-default-features --features claude
# The lean build's tests and examples too: an example or test that names an
# adapter without declaring the feature compiles fine with defaults on and
# only fails here, which is where five of them were found.
cargo test --no-run -p agent-harness --no-default-features

# The oldest compiler `rust-version` promises. Declared in Cargo.toml and
# tested nowhere else, so this is what keeps the number honest. Skipped when
# that toolchain is not installed: `rustup toolchain install 1.88`.
MSRV=$(sed -n 's/^rust-version = "\(.*\)"/\1/p' Cargo.toml)
if rustup run "$MSRV" rustc --version >/dev/null 2>&1; then
  echo "==> MSRV ($MSRV) check"
  cargo "+$MSRV" check --workspace --all-features --all-targets
else
  echo "==> (skip) MSRV toolchain $MSRV not installed — 'rustup toolchain install $MSRV' to enable"
fi

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
