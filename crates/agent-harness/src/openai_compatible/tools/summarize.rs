//! `summarize` — answer a question about, or summarize, a document too large to
//! read in one context window, with the *harness* driving the chunking loop so a
//! small local model never has to hold the whole file.
//!
//! Code-driven map-reduce (the model only ever sees one small slice at a time):
//! * **chunk** the file in code ([`chunk_text`]) into overlapping windows sized
//!   for a small context;
//! * **map** — one narrow extraction call per chunk over just that chunk;
//! * **collapse** — while the concatenated map notes exceed a budget, batch them
//!   ([`batch_under_budget`]) and re-summarize each batch, looping until they fit
//!   (so the reduce step can never overflow the window);
//! * **reduce** — a final call over the collapsed notes answering the question.
//!
//! The chunking + batching are pure functions (unit-tested without a model); the
//! map/collapse/reduce calls go through [`ModelClient`] ([`ToolCtx::model`]),
//! wired by the run loop over the run's connection config. Read-only → offered in
//! every mode.

use std::sync::atomic::{AtomicBool, Ordering};

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::ToolKind;

use super::{parse_args, schema_for, ModelClient, Tool, ToolCtx, ToolOutcome};

/// Characters per chunk for the map step — a char proxy for ~800–900 tokens,
/// within the 512–1024 range that suits synthesis on a small (~4K) window while
/// leaving room for the prompt + the chunk's own notes.
const CHUNK_CHARS: usize = 3500;
/// Overlap between consecutive chunks, so a fact straddling a boundary still
/// lands whole in at least one chunk.
const CHUNK_OVERLAP: usize = 200;
/// Char budget the collapsed notes must fit before the reduce step — a proxy for
/// ~1.5K tokens, conservative so the reduce input + prompt + question clear a
/// small window. Notes above this are batched and re-summarized until under it.
const COLLAPSE_BUDGET_CHARS: usize = 6000;
/// A guard on the collapse loop so a model that fails to shrink its input can't
/// spin forever; far above the rounds any real document needs.
const MAX_COLLAPSE_ROUNDS: usize = 6;

#[derive(Deserialize, JsonSchema)]
struct SummarizeArgs {
    /// Path to the document, relative to the working directory (absolute is also
    /// accepted).
    path: String,
    /// What to extract or answer about the document. Omit for a general summary.
    #[serde(default)]
    question: Option<String>,
}

pub(super) struct Summarize;
impl Tool for Summarize {
    fn id(&self) -> &str {
        "summarize"
    }
    fn description(&self) -> &str {
        "Summarize or answer a question about a document too large to read in one \
         step. The chunking is handled for you: pass the file path (and an \
         optional question) and get back a synthesized answer over the whole file."
    }
    fn parameters(&self) -> Value {
        schema_for::<SummarizeArgs>()
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }
    fn mutating(&self) -> bool {
        false
    }
    fn truncates_output(&self) -> bool {
        // The reduce output IS the answer (and bounded by construction); the
        // head-cap note would only get in its way.
        false
    }
    fn permission_subject(&self, args: &Value) -> Option<String> {
        args.get("path").and_then(Value::as_str).map(str::to_owned)
    }
    fn execute(&self, args: &Value, ctx: &ToolCtx) -> ToolOutcome {
        let a: SummarizeArgs = match parse_args(args) {
            Ok(a) => a,
            Err(o) => return o,
        };
        let Some(model) = ctx.model else {
            return ToolOutcome::err("summarize: model access is not available in this context");
        };
        let path = if std::path::Path::new(&a.path).is_absolute() {
            std::path::PathBuf::from(&a.path)
        } else {
            ctx.cwd.join(&a.path)
        };
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => return ToolOutcome::err(format!("summarize: reading `{}`: {e}", a.path)),
        };
        let question = a.question.as_deref().map(str::trim).filter(|q| !q.is_empty()).unwrap_or("Summarize this document.");
        match map_reduce(model, &text, question, ctx.cancel) {
            Ok(answer) => ToolOutcome::ok(answer),
            Err(e) => ToolOutcome::err(format!("summarize: {e}")),
        }
    }
}

/// The map prompt over one chunk — deliberately narrow and terse (small models
/// degrade with elaborate instructions), and it pins exact values.
fn map_prompt(question: &str, chunk: &str) -> String {
    format!(
        "Extract the information from this section relevant to: {question}\n\
         Be terse; keep exact names, numbers, and quotes. If nothing here is \
         relevant, reply \"(nothing relevant)\".\n\n\
         Section:\n{chunk}"
    )
}

/// The collapse prompt — fold several section notes into one shorter note,
/// preserving specifics, when the notes don't yet fit the reduce budget.
fn collapse_prompt(question: &str, notes: &str) -> String {
    format!(
        "Combine these section notes into one shorter set of notes, still relevant \
         to: {question}\nKeep exact names, numbers, and quotes; drop redundancy.\n\n\
         Notes:\n{notes}"
    )
}

/// The reduce prompt — the final synthesis answering the question from the
/// collapsed notes.
fn reduce_prompt(question: &str, notes: &str) -> String {
    format!("Using these section notes, answer: {question}\n\nNotes:\n{notes}")
}

/// Drive the map-reduce over `text` for `question`, calling `model` once per
/// chunk (map), as needed to collapse, and once to reduce. The whole file is
/// only ever seen one chunk at a time.
fn map_reduce(model: &dyn ModelClient, text: &str, question: &str, cancel: &AtomicBool) -> Result<String, String> {
    let chunks = chunk_text(text, CHUNK_CHARS, CHUNK_OVERLAP);
    if chunks.is_empty() {
        return Ok("(the document is empty)".to_owned());
    }
    // A single small chunk needs no map/collapse — answer it directly.
    if chunks.len() == 1 {
        check_cancel(cancel)?;
        return model.complete(None, &reduce_prompt(question, &chunks[0]), cancel);
    }
    // MAP: one narrow extraction per chunk, dropping the "(nothing relevant)" ones.
    let mut notes: Vec<String> = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        check_cancel(cancel)?;
        let note = model.complete(None, &map_prompt(question, chunk), cancel)?;
        if !is_irrelevant(&note) {
            notes.push(note.trim().to_owned());
        }
    }
    if notes.is_empty() {
        return Ok(format!("Nothing in the document was relevant to: {question}"));
    }
    // COLLAPSE: re-summarize batches until the notes fit the reduce budget.
    let collapsed = collapse(notes, COLLAPSE_BUDGET_CHARS, |batch| {
        check_cancel(cancel)?;
        model.complete(None, &collapse_prompt(question, batch), cancel)
    })?;
    // REDUCE: one final synthesis over the (now-fitting) notes.
    check_cancel(cancel)?;
    model.complete(None, &reduce_prompt(question, &collapsed.join("\n\n")), cancel)
}

/// Split `text` into overlapping windows of about `chunk_chars` characters, each
/// starting `chunk_chars - overlap` past the previous one so consecutive chunks
/// share `overlap` chars. Operates on chars (not bytes), so multibyte content is
/// never split mid-character. Covers the whole input; the last chunk may be
/// short. `overlap` is clamped below `chunk_chars` so the stride is always
/// positive.
pub(super) fn chunk_text(text: &str, chunk_chars: usize, overlap: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }
    let chunk_chars = chunk_chars.max(1);
    let overlap = overlap.min(chunk_chars - 1);
    let stride = chunk_chars - overlap;
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + chunk_chars).min(chars.len());
        chunks.push(chars[start..end].iter().collect());
        if end == chars.len() {
            break;
        }
        start += stride;
    }
    chunks
}

/// Group consecutive `items` so each group's joined length (with a blank-line
/// separator) stays within `budget` chars. A single item longer than `budget` is
/// its own group (it can't be split here — the model shrinks it). Pure: the
/// grouping decision the collapse loop batches on.
pub(super) fn batch_under_budget(items: &[String], budget: usize) -> Vec<Vec<&str>> {
    const SEP: usize = 2; // "\n\n" between joined notes
    let mut groups: Vec<Vec<&str>> = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    let mut len = 0usize;
    for item in items {
        if !current.is_empty() && len + SEP + item.len() > budget {
            groups.push(std::mem::take(&mut current));
            len = 0;
        }
        len += item.len() + if current.is_empty() { 0 } else { SEP };
        current.push(item);
    }
    if !current.is_empty() {
        groups.push(current);
    }
    groups
}

/// Collapse `notes` until their joined length fits `budget`: while over budget,
/// batch consecutive notes ([`batch_under_budget`]) and replace each batch with
/// `reduce`'s shorter summary of it, repeating. Returns the notes once they fit
/// (or after [`MAX_COLLAPSE_ROUNDS`], so a non-shrinking model can't loop). When
/// already under budget it returns `notes` untouched (no model call). `reduce`
/// is the model-backed re-summarizer in production; tests pass a pure fake.
pub(super) fn collapse(
    mut notes: Vec<String>,
    budget: usize,
    mut reduce: impl FnMut(&str) -> Result<String, String>,
) -> Result<Vec<String>, String> {
    for _ in 0..MAX_COLLAPSE_ROUNDS {
        if joined_len(&notes) <= budget {
            return Ok(notes);
        }
        let groups = batch_under_budget(&notes, budget);
        // A single group that's still over budget can't shrink by re-batching;
        // re-summarize it anyway (the model condenses), but if batching made no
        // structural progress (one group in, one out) and it didn't shrink, stop.
        let mut next = Vec::with_capacity(groups.len());
        for group in &groups {
            if group.len() == 1 && group[0].len() <= budget {
                next.push(group[0].to_owned()); // already fits; don't burn a call
            } else {
                next.push(reduce(&group.join("\n\n"))?.trim().to_owned());
            }
        }
        if next.len() == notes.len() && joined_len(&next) >= joined_len(&notes) {
            return Ok(next); // not shrinking — give the reduce step what we have
        }
        notes = next;
    }
    Ok(notes)
}

/// Joined length of the notes with the blank-line separators the reduce step uses.
fn joined_len(notes: &[String]) -> usize {
    notes.iter().map(String::len).sum::<usize>() + notes.len().saturating_sub(1) * 2
}

/// Whether a map note signals the chunk had nothing relevant (so it's dropped).
fn is_irrelevant(note: &str) -> bool {
    let t = note.trim();
    t.is_empty() || t.eq_ignore_ascii_case("(nothing relevant)") || t.eq_ignore_ascii_case("nothing relevant")
}

fn check_cancel(cancel: &AtomicBool) -> Result<(), String> {
    if cancel.load(Ordering::SeqCst) {
        Err("cancelled".to_owned())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_text_covers_input_with_overlap() {
        let text: String = ('a'..='z').collect(); // 26 chars
        let chunks = chunk_text(&text, 10, 3);
        // stride = 7: [0,10), [7,17), [14,24), [21,26)
        assert_eq!(chunks, vec!["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]);
        // Consecutive chunks share `overlap` chars (3 here).
        assert!(chunks[0].ends_with("hij") && chunks[1].starts_with("hij"));
        // Concatenating with the overlaps removed reconstructs the whole input —
        // i.e. nothing is dropped between chunks.
        let mut rebuilt = chunks[0].clone();
        for c in &chunks[1..] {
            rebuilt.push_str(&c[3..]);
        }
        assert_eq!(rebuilt, text, "chunks cover the entire input");
    }

    #[test]
    fn chunk_text_handles_small_and_empty_and_multibyte() {
        assert!(chunk_text("", 100, 10).is_empty());
        // Smaller than one chunk → a single chunk.
        assert_eq!(chunk_text("short", 100, 10), vec!["short"]);
        // Exactly one chunk, no trailing empty chunk.
        assert_eq!(chunk_text("0123456789", 10, 3), vec!["0123456789"]);
        // Multibyte chars are not split mid-character (each emoji is one char).
        let emoji = "😀😁😂🤣😃😄😅😆"; // 8 chars, 32 bytes
        let chunks = chunk_text(emoji, 3, 1);
        assert!(chunks.iter().all(|c| c.chars().count() <= 3), "char-bounded chunks");
        assert_eq!(chunks.iter().map(|c| c.chars().count()).max().unwrap(), 3);
    }

    #[test]
    fn chunk_text_clamps_overlap_below_chunk_size() {
        // overlap >= chunk_chars would zero/negate the stride; it's clamped so the
        // loop always advances (no infinite loop, no duplicate chunks forever).
        let chunks = chunk_text("abcdefghij", 4, 99);
        assert!(chunks.len() >= 3, "still advances: {chunks:?}");
        assert_eq!(chunks[0], "abcd");
    }

    #[test]
    fn batch_under_budget_groups_within_budget() {
        let items =
            vec!["aaaa".to_owned(), "bbbb".to_owned(), "cccc".to_owned(), "dddd".to_owned()];
        // budget 10: "aaaa\n\nbbbb" = 10 fits; adding cccc would be 16 → new group.
        let groups = batch_under_budget(&items, 10);
        assert_eq!(groups, vec![vec!["aaaa", "bbbb"], vec!["cccc", "dddd"]]);
        // Each group's joined length is within budget.
        for g in &groups {
            assert!(g.join("\n\n").len() <= 10, "group within budget: {g:?}");
        }
    }

    #[test]
    fn batch_under_budget_isolates_an_oversized_item() {
        let items = vec!["small".to_owned(), "x".repeat(50), "tiny".to_owned()];
        let groups = batch_under_budget(&items, 10);
        // The 50-char item can't fit with neighbors → it's alone in its group.
        assert!(groups.iter().any(|g| g.len() == 1 && g[0].len() == 50), "oversized item isolated: {groups:?}");
    }

    #[test]
    fn collapse_noop_when_already_under_budget() {
        let notes = vec!["a".to_owned(), "b".to_owned()];
        let mut calls = 0;
        let out = collapse(notes.clone(), 1000, |_| {
            calls += 1;
            Ok(String::new())
        })
        .unwrap();
        assert_eq!(out, notes, "returned untouched");
        assert_eq!(calls, 0, "no model call when it already fits");
    }

    #[test]
    fn collapse_reduces_many_notes_under_budget() {
        // 20 notes of 100 chars each (~2000 joined) must collapse under a 300-char
        // budget. The fake reducer condenses any batch to a fixed short string —
        // proving the loop converges with a shrinking reducer.
        let notes: Vec<String> = (0..20).map(|i| format!("note {i}: ").to_owned() + &"x".repeat(90)).collect();
        assert!(joined_len(&notes) > 300);
        let mut rounds = 0;
        let out = collapse(notes, 300, |_batch| {
            rounds += 1;
            Ok("condensed".to_owned()) // 9 chars per batch
        })
        .unwrap();
        assert!(joined_len(&out) <= 300, "collapsed under the budget: {} chars", joined_len(&out));
        assert!(rounds > 0, "the reducer was invoked");
    }

    #[test]
    fn collapse_terminates_when_reducer_does_not_shrink() {
        // A reducer that returns its input verbatim can't shrink anything; the
        // loop must still terminate (round guard + no-progress check) rather than
        // spin, returning what it has for the reduce step to use.
        let notes: Vec<String> = (0..5).map(|_| "y".repeat(100)).collect();
        let out = collapse(notes, 50, |batch| Ok(batch.to_owned())).unwrap();
        assert!(!out.is_empty(), "returns the notes rather than looping forever");
    }

    #[test]
    fn is_irrelevant_detects_the_sentinel() {
        assert!(is_irrelevant("(nothing relevant)"));
        assert!(is_irrelevant("  Nothing Relevant  "));
        assert!(is_irrelevant(""));
        assert!(!is_irrelevant("the budget is $5M"));
    }

    /// A fake model: records the prompts it's asked and returns canned replies, so
    /// the map-reduce orchestration is exercised end-to-end without a network.
    struct FakeModel {
        calls: std::cell::RefCell<Vec<String>>,
    }
    impl ModelClient for FakeModel {
        fn complete(&self, _system: Option<&str>, user: &str, _cancel: &AtomicBool) -> Result<String, String> {
            self.calls.borrow_mut().push(user.to_owned());
            // Map calls extract; the reduce call gets the marker so the test can
            // assert which prompt produced the final answer.
            if user.starts_with("Using these section notes") {
                Ok("FINAL ANSWER".to_owned())
            } else {
                Ok("a relevant note".to_owned())
            }
        }
    }

    #[test]
    fn map_reduce_runs_map_then_reduce_over_a_large_doc() {
        // A document spanning several chunks drives one map call per chunk + a
        // reduce. (No collapse needed — the few short notes already fit.)
        let doc = "lorem ipsum ".repeat(1500); // ~18K chars → multiple chunks
        let cancel = AtomicBool::new(false);
        let model = FakeModel { calls: std::cell::RefCell::new(Vec::new()) };
        let out = map_reduce(&model, &doc, "what is this?", &cancel).unwrap();
        assert_eq!(out, "FINAL ANSWER");
        let calls = model.calls.borrow();
        let expected_chunks = chunk_text(&doc, CHUNK_CHARS, CHUNK_OVERLAP).len();
        assert!(expected_chunks > 1, "doc spans multiple chunks");
        let map_calls = calls.iter().filter(|c| c.starts_with("Extract the information")).count();
        let reduce_calls = calls.iter().filter(|c| c.starts_with("Using these section notes")).count();
        assert_eq!(map_calls, expected_chunks, "one map call per chunk");
        assert_eq!(reduce_calls, 1, "exactly one reduce call");
    }

    #[test]
    fn map_reduce_short_doc_answers_directly() {
        // A doc that fits one chunk skips map/collapse — a single reduce call.
        let cancel = AtomicBool::new(false);
        let model = FakeModel { calls: std::cell::RefCell::new(Vec::new()) };
        let out = map_reduce(&model, "a short note", "summarize", &cancel).unwrap();
        assert_eq!(out, "FINAL ANSWER");
        assert_eq!(model.calls.borrow().len(), 1, "one call for a single-chunk doc");
    }

    #[test]
    fn map_reduce_reports_when_nothing_relevant() {
        // Every map call says "(nothing relevant)" → no reduce, a clear message.
        struct Empties;
        impl ModelClient for Empties {
            fn complete(&self, _s: Option<&str>, _u: &str, _c: &AtomicBool) -> Result<String, String> {
                Ok("(nothing relevant)".to_owned())
            }
        }
        let doc = "filler ".repeat(2000); // multiple chunks
        let cancel = AtomicBool::new(false);
        let out = map_reduce(&Empties, &doc, "find the price", &cancel).unwrap();
        assert!(out.contains("Nothing in the document was relevant"), "got: {out}");
    }
}
