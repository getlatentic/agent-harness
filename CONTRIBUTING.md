# Contributing

Thanks for your interest. This is a small, pre-1.0 workspace of two crates —
`cli-stream` (a generic subprocess engine) ← `agent-harness` (the framework).
The dependency arrow only ever points up.

## Build & check

Before opening a PR, run the local gate:

```sh
./scripts/check.sh
```

It runs format + clippy (`-D warnings`) + tests + the examples + the
feature-gated build — the same things a CI would. There is **no hosted CI**, so
this script is the gate; please make sure it passes.

## Design principles (please keep these)

- **Normalize at the adapter, reinterpret in the consumer.** Each adapter
  translates its CLI's wire format into the neutral `RunEvent`; product-specific
  meaning belongs in the consumer, not the library. This is the keystone.
- **Ground parsers in real output.** A new or changed parser must be grounded in
  the tool's *actual* stdout (capture a real run) and unit-tested — a wrong
  parser is silent data loss, not a loud failure.
- **No panics in library code.** No `unwrap` / `expect` / `panic!` outside
  tests; recover (e.g. a poisoned lock) or return a typed `HarnessError`.
- **Capabilities, not id checks.** Declare what a harness supports via
  `HarnessCapabilities`; never branch on the harness id.
- **`#[non_exhaustive]` the public event enums** so adding a variant stays
  additive.

## Adding a harness

Implement `Harness` — in-tree as a feature-gated module, or out-of-tree in your
own crate — and `Registry::register` it. See the **Extending** section of the
README and [`examples/custom_harness.rs`](crates/agent-harness/examples/custom_harness.rs).

## Commit messages & licensing

Write plain, descriptive commit messages. Contributions are dual-licensed under
**MIT OR Apache-2.0**, matching the crates; by contributing you agree to license
your work under those terms.
