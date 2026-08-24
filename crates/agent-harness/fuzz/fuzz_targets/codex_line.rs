//! A whole Codex stdout stream through one parser.
//!
//! `CodexStreamParser` carries state between lines (the agent_message machine),
//! so a single line proves much less than a sequence does: the interesting
//! failures are a message opened and never closed, closed twice, or interleaved
//! with something else. Feeding the input as a stream is what reaches them.

#![no_main]

use harness::codex::CodexStreamParser;
use harness::Event;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut parser = CodexStreamParser::new();
    for line in String::from_utf8_lossy(data).lines() {
        let _ = parser.on_process_event(Event::Stdout {
            run_id: "fuzz".to_owned(),
            line: line.to_owned(),
        });
    }
});
