# Fuzz targets

The parsers that eat bytes we did not write: the NDJSON an agent CLI streams on
stdout, and the neutral normalizer behind them.

The contract is **never panics**, not "parses correctly". Classifying a line is
the job, and an unrecognised line is a normal outcome — so there is no expected
output to assert. A panic, on the other hand, takes down the thread pumping a
live run.

Requires nightly (libFuzzer needs sanitizer support):

```bash
cargo +nightly fuzz run claude_line -- -max_total_time=90
```

Targets: `claude_line`, `codex_line`, `raw_line`, `normalize_event`.

`codex_line` feeds the input as a *stream* through one `CodexStreamParser`
rather than line by line, because that parser carries state between lines. The
failures worth finding there — a message opened and never closed, closed twice,
interleaved with something else — need a sequence to reach.

Input is decoded with `from_utf8_lossy`, matching what the streaming layer
hands these parsers. It frames on bytes and decodes lossily, so a parser can be
called with replacement characters in any position, and no input is wasted by
being rejected as invalid UTF-8.

## Seeds

`seeds/<target>/` holds real wire shapes, taken from the formats the parser
tests use. They are committed; the working corpus under `corpus/` is not.
Starting from them saves the fuzzer rediscovering the message envelope:

```bash
cargo +nightly fuzz run claude_line seeds/claude_line -- -max_total_time=90
```

## In CI

A PR replays the committed seeds and exits (`-runs=0`) — deterministic, a few
seconds, and it fails only on an input already known to matter. A PR must never
go red because a fuzzer wandered somewhere new, which is what any time-boxed
run would eventually do.

Discovery is the weekly scheduled run (and `workflow_dispatch`), where each
target gets real time. libFuzzer writes finds into the **first** corpus
directory named and reads the rest, so `corpus/<target> seeds/<target>` keeps
the committed seeds as they are.

## When one finds something

libFuzzer writes the input to `artifacts/<target>/`. **Commit that file** and
add a unit test built from it — the artifact is the reproduction, and a fuzz
target nobody re-runs is not a regression test.
