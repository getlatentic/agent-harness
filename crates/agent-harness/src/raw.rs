//! The raw passthrough tier — a harness's stdout line decoded as **untyped
//! JSON**, for consumers that want to interpret a harness's output themselves
//! instead of the neutral [`crate::RunEvent`] vocabulary.
//!
//! Harness-agnostic by design: bob, Claude Code, and Codex all emit one JSON
//! object per line, so a single [`parse_raw_line`] serves them all (and any
//! other JSONL CLI). This *is* the consistent raw API across harnesses — there
//! is no per-harness `parse_X_raw`, because the decode carries no
//! harness-specific knowledge.
//!
//! Deliberately **untyped**: one line → one [`serde_json::Value`], no schema
//! imposed. A new field a CLI adds is preserved (a typed struct would silently
//! drop it); a new event `type` just flows through; there is nothing here to
//! keep in sync with any CLI's protocol. Each harness's wire grammar (its
//! `type` discriminator, field names) lives in exactly one place — its
//! *neutral* parser (`parse_bob_line` / `parse_claude_line` /
//! `parse_codex_line`), which interprets it into `RunEvent`. This tier carries
//! none of that.
//!
//! Want the absolute rawest? [`crate::Event::Stdout`] already carries
//! the verbatim line — skip this and parse the string yourself.

use serde_json::Value;

/// Decode one stdout line into an untyped [`serde_json::Value`], losslessly and
/// without interpretation:
///
/// * a JSON value (object, array, scalar) → that value, **verbatim** — every
///   field preserved, nothing reinterpreted;
/// * a non-JSON line → `Value::String(<the raw, untrimmed line>)`;
/// * a blank line → `Value::Null`.
///
/// No special-casing of any harness's events — the consumer reads whatever
/// fields it needs (`v["type"]`, `v["item"]["text"]`, `v["stats"]`, …).
pub fn parse_raw_line(line: &str) -> Value {
    if line.trim().is_empty() {
        return Value::Null;
    }
    // Valid JSON → as-is. Non-JSON → the raw (untrimmed) line, never dropped.
    serde_json::from_str(line.trim()).unwrap_or_else(|_| Value::String(line.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One real line from each built-in harness passes through verbatim — no
    /// field dropped, no `type` interpreted. The same function handles all
    /// three (and anything else that emits JSONL), which is the point.
    #[test]
    fn harness_lines_pass_through_verbatim() {
        // bob: init.
        let v = parse_raw_line(r#"{"type":"init","session_id":"s","model":"premium"}"#);
        assert_eq!(v["type"].as_str(), Some("init"));
        assert_eq!(v["session_id"].as_str(), Some("s"));

        // claude: a streamed text delta (nested).
        let v = parse_raw_line(
            r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}}"#,
        );
        assert_eq!(v["event"]["delta"]["text"].as_str(), Some("hi"));

        // codex: a completed command item (nested object + numbers).
        let v = parse_raw_line(
            r#"{"type":"item.completed","item":{"id":"i2","type":"command_execution","exit_code":0}}"#,
        );
        assert_eq!(v["item"]["type"].as_str(), Some("command_execution"));
        assert_eq!(v["item"]["exit_code"].as_i64(), Some(0));
    }

    #[test]
    fn arrays_and_scalars_pass_through() {
        assert_eq!(parse_raw_line("[1,2,3]")[1].as_i64(), Some(2));
        assert_eq!(parse_raw_line("42").as_i64(), Some(42));
        assert_eq!(parse_raw_line("true").as_bool(), Some(true));
    }

    #[test]
    fn blank_line_is_null() {
        assert_eq!(parse_raw_line("   "), Value::Null);
    }

    #[test]
    fn non_json_line_is_preserved_verbatim() {
        // A CLI occasionally prints prose / stderr-ish lines — kept untrimmed.
        assert_eq!(
            parse_raw_line("  hello world  "),
            Value::String("  hello world  ".to_owned())
        );
    }

    #[test]
    fn unknown_type_passes_through_untouched() {
        // A future/unknown event type isn't special-cased — it's just present.
        let v = parse_raw_line(r#"{"type":"telemetry","foo":42}"#);
        assert_eq!(v["type"].as_str(), Some("telemetry"));
        assert_eq!(v["foo"].as_i64(), Some(42));
    }

    #[test]
    fn unknown_field_on_known_type_is_kept() {
        // The whole point of staying untyped: a NEW field a CLI adds to a known
        // event survives — a typed decode would have dropped it.
        let v = parse_raw_line(r#"{"type":"message","content":"hi","usage":{"out":7}}"#);
        assert_eq!(v["content"].as_str(), Some("hi"));
        assert_eq!(v["usage"]["out"].as_i64(), Some(7)); // future-proof
    }

    #[test]
    fn empty_string_fields_are_kept() {
        let v = parse_raw_line(r#"{"type":"message","content":""}"#);
        assert_eq!(v["content"].as_str(), Some(""));
    }
}
