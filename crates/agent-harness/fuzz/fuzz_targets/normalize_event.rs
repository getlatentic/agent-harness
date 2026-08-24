//! The normalizer every process-backed adapter shares.
//!
//! Fuzzed over the whole `Event` shape rather than stdout alone, because the
//! neutral branches do real work too — stderr is truncated to 240 chars, which
//! is a byte index into text we did not write and so a char-boundary question.

#![no_main]

use harness::{normalize_process_event, Event};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data).into_owned();
    let run_id = "fuzz".to_owned();

    for event in [
        Event::Stdout {
            run_id: run_id.clone(),
            line: text.clone(),
        },
        Event::Stderr {
            run_id: run_id.clone(),
            line: text.clone(),
        },
        Event::Error {
            run_id: run_id.clone(),
            message: text.clone(),
        },
    ] {
        let _ = normalize_process_event(event, harness::claude::parse_claude_line);
    }
});
