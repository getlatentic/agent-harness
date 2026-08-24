#!/usr/bin/env bash
# Prepare a release. This does not perform one.
#
#   scripts/prepare-release.sh patch     # or minor / major / 1.2.3
#
# Bumps both crates, stamps the CHANGELOG, rewrites the README version pins,
# and commits. Nothing leaves the machine. Publishing happens in CI, caused by
# a tag on `main` — see release.toml for why the order is that way round.
#
# The flags duplicate release.toml on purpose. A release that publishes early
# cannot be undone, so "the config says not to" is not the only thing standing
# between a mistyped command and crates.io.
set -euo pipefail

level="${1:-}"
if [ -z "$level" ]; then
  echo "usage: $0 <patch|minor|major|X.Y.Z>" >&2
  echo "  pre-1.0: a breaking change is 'minor'; additive and fixes are 'patch'." >&2
  exit 2
fi

cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain)" ]; then
  echo "error: working tree is dirty — commit or stash first" >&2
  exit 1
fi

branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" != "main" ]; then
  echo "warning: preparing from '$branch' rather than main" >&2
fi

echo "==> dry run"
cargo release "$level" --no-publish --no-tag --no-push

printf '\n==> apply? [y/N] '
read -r reply
[ "$reply" = "y" ] || { echo "nothing done"; exit 0; }

cargo release "$level" --no-publish --no-tag --no-push --execute --no-confirm

version=$(cargo metadata --no-deps --format-version 1 \
  | python3 -c "import sys,json;print(next(p['version'] for p in json.load(sys.stdin)['packages'] if p['name']=='agent-harness'))")

cat <<NEXT

Prepared agent-harness $version. Nothing has been published.

  git switch -c release/$version && git push -u origin release/$version
  gh pr create --fill

Once CI is green and the PR is merged, tagging the merge commit publishes it:

  git switch main && git pull
  git tag v$version && git push origin v$version

NEXT
