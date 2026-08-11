//! The raw passthrough, which every adapter falls back to.
//!
//! It always returns a `Value`, so there is no error path to check — only the
//! promise that no input makes it panic.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    for line in String::from_utf8_lossy(data).lines() {
        let _ = harness::parse_raw_line(line);
    }
});
