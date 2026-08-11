//! One line of the Claude CLI's NDJSON.
//!
//! The contract is "never panics", not "parses correctly": the parser's job is
//! to classify a line it did not write, and an unrecognised one is a normal
//! outcome. Bytes are decoded lossily because that is what the streaming layer
//! now hands it — a child's stdout is a byte stream, not a `String`.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    for line in String::from_utf8_lossy(data).lines() {
        let _ = harness::claude::parse_claude_line(line);
    }
});
