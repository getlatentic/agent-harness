//! File tools — `read`, `write`, `edit`. Reimplemented from OpenCode's designs
//! (MIT): line-numbered paged reads, overwrite-with-mkdir writes, and exact
//! string-replacement edits with uniqueness enforcement + a whitespace-tolerant
//! line-match fallback.

use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::{RunMode, ToolKind};

use super::{parse_args, safe_join, schema_for, uses_apply_patch, Tool, ToolCtx, ToolOutcome, MAX_LINE_CHARS, MAX_OUTPUT_BYTES};

/// Default line cap for `read`, mirroring OpenCode's limit.
const DEFAULT_READ_LINES: usize = 2000;

#[derive(Deserialize, JsonSchema)]
struct ReadArgs {
    /// Path to the file, relative to the working directory.
    path: String,
    /// 1-based line number to start at (default 1).
    offset: Option<usize>,
    /// Maximum number of lines to return (default 2000).
    limit: Option<usize>,
}

pub(super) struct Read;
impl Tool for Read {
    fn id(&self) -> &str {
        "read"
    }
    fn description(&self) -> &str {
        "Read a UTF-8 text file (relative to the working directory). Returns \
         line-numbered content; use offset/limit to page through large files."
    }
    fn parameters(&self) -> Value {
        schema_for::<ReadArgs>()
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }
    fn mutating(&self) -> bool {
        false
    }
    fn truncates_output(&self) -> bool {
        // `read` enforces its own byte budget (see `read_file`) with a paging
        // footer ("call read again with offset=…"), so it opts out of the
        // generic head-truncation — whose note would otherwise clobber the
        // more-actionable paging footer.
        false
    }
    fn permission_subject(&self, args: &Value) -> Option<String> {
        args.get("path").and_then(Value::as_str).map(str::to_owned)
    }
    fn execute(&self, args: &Value, ctx: &ToolCtx) -> ToolOutcome {
        let a: ReadArgs = match parse_args(args) {
            Ok(a) => a,
            Err(o) => return o,
        };
        read_file(ctx.cwd, &a.path, a.offset, a.limit)
    }
}

#[derive(Deserialize, JsonSchema)]
struct WriteArgs {
    /// Path to the file, relative to the working directory.
    path: String,
    /// The full new file contents.
    content: String,
}

pub(super) struct Write;
impl Tool for Write {
    fn id(&self) -> &str {
        "write"
    }
    fn description(&self) -> &str {
        "Create or overwrite a UTF-8 text file (relative to the working \
         directory). Parent directories are created as needed."
    }
    fn parameters(&self) -> Value {
        schema_for::<WriteArgs>()
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Write
    }
    fn mutating(&self) -> bool {
        true
    }
    fn offered(&self, _mode: RunMode, model: &str) -> bool {
        // Hidden for gpt-5-class models, which get `apply_patch` instead. Offered
        // in every mode — a write in a read-only run is refused at `execute`.
        !uses_apply_patch(model)
    }
    fn permission_subject(&self, args: &Value) -> Option<String> {
        args.get("path").and_then(Value::as_str).map(str::to_owned)
    }
    fn execute(&self, args: &Value, ctx: &ToolCtx) -> ToolOutcome {
        let a: WriteArgs = match parse_args(args) {
            Ok(a) => a,
            Err(o) => return o,
        };
        write_file(ctx.cwd, &a.path, &a.content)
    }
}

#[derive(Deserialize, JsonSchema)]
struct EditArgs {
    /// Path to the file, relative to the working directory.
    path: String,
    /// Exact text to replace (whitespace and indentation included).
    old_string: String,
    /// Replacement text (must differ from old_string).
    new_string: String,
    /// Replace every occurrence instead of requiring a unique match (default false).
    #[serde(default)]
    replace_all: bool,
}

pub(super) struct Edit;
impl Tool for Edit {
    fn id(&self) -> &str {
        "edit"
    }
    fn description(&self) -> &str {
        "Replace an exact substring in a file. `old_string` must match \
         (whitespace included) and be unique unless `replace_all` is true; a \
         whitespace-tolerant line match is tried if the exact text isn't found."
    }
    fn parameters(&self) -> Value {
        schema_for::<EditArgs>()
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Edit
    }
    fn mutating(&self) -> bool {
        true
    }
    fn offered(&self, _mode: RunMode, model: &str) -> bool {
        // Hidden for gpt-5-class models, which get `apply_patch` instead. Offered
        // in every mode — a write in a read-only run is refused at `execute`.
        !uses_apply_patch(model)
    }
    fn permission_subject(&self, args: &Value) -> Option<String> {
        args.get("path").and_then(Value::as_str).map(str::to_owned)
    }
    fn execute(&self, args: &Value, ctx: &ToolCtx) -> ToolOutcome {
        let a: EditArgs = match parse_args(args) {
            Ok(a) => a,
            Err(o) => return o,
        };
        edit_file(ctx.cwd, &a.path, &a.old_string, &a.new_string, a.replace_all)
    }
}

fn read_file(cwd: &Path, rel: &str, offset: Option<usize>, limit: Option<usize>) -> ToolOutcome {
    // `read` accepts absolute paths (and `..`), like OpenCode's read — and since
    // `bash` can already read anywhere, a cwd sandbox here bought little. It also
    // lets the model read a truncated tool's spill file. Writes/edits stay
    // cwd-scoped (`safe_join`) as the mutation guardrail.
    let path = if Path::new(rel).is_absolute() { PathBuf::from(rel) } else { cwd.join(rel) };
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => return ToolOutcome::err(format!("reading `{rel}`: {e}")),
    };
    let lines: Vec<&str> = content.lines().collect();
    let start = offset.unwrap_or(1).max(1); // 1-based
    let limit = limit.unwrap_or(DEFAULT_READ_LINES);
    let mut out = String::new();
    // A cumulative byte budget alongside the line/char caps: the line cap alone
    // lets 2000 long lines reach megabytes and blow a small context window, so
    // stop once the emitted text nears [`MAX_OUTPUT_BYTES`] and page the rest.
    let mut byte_capped_at: Option<usize> = None;
    let mut last_shown = start - 1; // index past the last emitted line
    for (idx, line) in lines.iter().enumerate().skip(start - 1).take(limit) {
        let n = idx + 1;
        let rendered = if line.chars().count() > MAX_LINE_CHARS {
            let clipped: String = line.chars().take(MAX_LINE_CHARS).collect();
            format!("{n}: {clipped}… (line truncated)\n")
        } else {
            format!("{n}: {line}\n")
        };
        // Stop before exceeding the budget, but always emit at least one line so
        // a single huge line still makes progress rather than looping empty.
        if !out.is_empty() && out.len() + rendered.len() > MAX_OUTPUT_BYTES {
            byte_capped_at = Some(n);
            break;
        }
        out.push_str(&rendered);
        last_shown = idx + 1;
    }
    if let Some(n) = byte_capped_at {
        out.push_str(&format!(
            "… output capped at {} KB at line {}; call read again with offset={}.\n",
            MAX_OUTPUT_BYTES / 1024,
            n - 1,
            n
        ));
    } else if last_shown < lines.len() {
        out.push_str(&format!(
            "… {} more lines; call read again with offset={}.\n",
            lines.len() - last_shown,
            last_shown + 1
        ));
    }
    if out.is_empty() {
        out.push_str("(empty or offset past end of file)");
    }
    ToolOutcome::ok(out)
}

fn write_file(cwd: &Path, rel: &str, content: &str) -> ToolOutcome {
    let Some(path) = safe_join(cwd, rel) else {
        return ToolOutcome::err(format!("path `{rel}` escapes the working directory"));
    };
    if let Some(parent) = path.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        return ToolOutcome::err(format!("creating parent of `{rel}`: {e}"));
    }
    match std::fs::write(&path, content) {
        Ok(()) => ToolOutcome::ok(format!("wrote {} bytes to {rel}", content.len())),
        Err(e) => ToolOutcome::err(format!("writing `{rel}`: {e}")),
    }
}

fn edit_file(cwd: &Path, rel: &str, old: &str, new: &str, replace_all: bool) -> ToolOutcome {
    if old == new {
        return ToolOutcome::err("edit: old_string and new_string are identical");
    }
    let Some(path) = safe_join(cwd, rel) else {
        return ToolOutcome::err(format!("path `{rel}` escapes the working directory"));
    };
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => return ToolOutcome::err(format!("reading `{rel}` to edit: {e}")),
    };

    let exact = content.matches(old).count();
    if exact >= 1 {
        let (updated, n) = if replace_all {
            (content.replace(old, new), exact)
        } else if exact > 1 {
            return ToolOutcome::err(format!(
                "edit: `old_string` is not unique in {rel} ({exact} matches) — add surrounding context, or set replace_all"
            ));
        } else {
            (content.replacen(old, new, 1), 1)
        };
        return write_edit(&path, rel, &updated, n, false);
    }

    // Exact match failed. Try a whitespace-tolerant, line-trimmed match for a
    // single unique span (OpenCode's LineTrimmedReplacer) — the common case
    // where the model's `old_string` has slightly-off indentation/trailing
    // whitespace. `replace_all` stays exact-only (fuzzy multi-replace is unsafe).
    if !replace_all && let Some(updated) = super::replace_line_trimmed(&content, old, new) {
            return write_edit(&path, rel, &updated, 1, true);
        }

    ToolOutcome::err(format!(
        "edit: `old_string` not found in {rel} — it must match exactly (whitespace and indentation included)"
    ))
}

fn write_edit(path: &Path, rel: &str, updated: &str, n: usize, fuzzy: bool) -> ToolOutcome {
    match std::fs::write(path, updated) {
        Ok(()) => ToolOutcome::ok(format!(
            "edited {rel} ({n} replacement{}{})",
            if n == 1 { "" } else { "s" },
            if fuzzy { ", whitespace-tolerant match" } else { "" }
        )),
        Err(e) => ToolOutcome::err(format!("writing edited `{rel}`: {e}")),
    }
}
