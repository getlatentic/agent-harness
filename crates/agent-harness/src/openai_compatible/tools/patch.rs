//! `apply_patch` — apply an OpenAI/OpenCode patch envelope (MIT design). The
//! model emits one `*** Begin Patch … *** End Patch` block describing file
//! adds / updates / deletes (with optional rename); we parse and apply it.
//!
//! Offered **only** to gpt-5-class models (the [`super::uses_apply_patch`] gate),
//! which then get this *instead of* `edit`/`write` — mirroring OpenCode's
//! model-keyed swap. Mutating → Edit mode only.

use std::path::Path;

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::{RunMode, ToolKind};

use super::{parse_args, safe_join, schema_for, uses_apply_patch, Tool, ToolCtx, ToolOutcome};

#[derive(Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct ApplyPatchArgs {
    /// The full patch text: a `*** Begin Patch` / `*** End Patch` block with one
    /// or more `*** Add File:` / `*** Update File:` / `*** Delete File:` sections.
    patch_text: String,
}

pub(super) struct ApplyPatch;
impl Tool for ApplyPatch {
    fn id(&self) -> &str {
        "apply_patch"
    }
    fn description(&self) -> &str {
        "Apply a patch to the working tree. The patch is a `*** Begin Patch` … \
         `*** End Patch` envelope with `*** Add File:`, `*** Update File:` (using \
         `@@` hunks with ` `/`-`/`+` lines, optionally `*** Move to:`), and \
         `*** Delete File:` sections."
    }
    fn parameters(&self) -> Value {
        schema_for::<ApplyPatchArgs>()
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Edit
    }
    fn mutating(&self) -> bool {
        true
    }
    fn offered(&self, _mode: RunMode, model: &str) -> bool {
        // gpt-5-class only; offered in every mode (refused at `execute` if read-only).
        uses_apply_patch(model)
    }
    fn execute(&self, args: &Value, ctx: &ToolCtx) -> ToolOutcome {
        let a: ApplyPatchArgs = match parse_args(args) {
            Ok(a) => a,
            Err(o) => return o,
        };
        match parse_patch(&a.patch_text) {
            Ok(ops) => apply(ops, ctx.cwd),
            Err(e) => ToolOutcome::err(format!("apply_patch: {e}")),
        }
    }
}

/// One file operation parsed from the envelope.
enum Op {
    Add { path: String, content: String },
    Delete { path: String },
    Update { path: String, move_to: Option<String>, hunks: Vec<Hunk> },
}

/// One update hunk: the old block to find and the new block to replace it with
/// (each line's `@@`/` `/`-`/`+` marker already stripped).
struct Hunk {
    old: String,
    new: String,
}

fn parse_patch(text: &str) -> Result<Vec<Op>, String> {
    let mut lines = text.lines().peekable();
    match lines.next() {
        Some(l) if l.trim() == "*** Begin Patch" => {}
        _ => return Err("patch must start with `*** Begin Patch`".to_owned()),
    }
    let mut ops = Vec::new();
    while let Some(&line) = lines.peek() {
        if line.trim() == "*** End Patch" {
            return Ok(ops);
        } else if let Some(path) = line.strip_prefix("*** Add File: ") {
            let path = path.trim().to_owned();
            lines.next();
            let mut content = String::new();
            while let Some(&l) = lines.peek() {
                if l.starts_with("*** ") {
                    break;
                }
                let l = lines.next().unwrap();
                content.push_str(l.strip_prefix('+').unwrap_or(l));
                content.push('\n');
            }
            ops.push(Op::Add { path, content });
        } else if let Some(path) = line.strip_prefix("*** Delete File: ") {
            let path = path.trim().to_owned();
            lines.next();
            ops.push(Op::Delete { path });
        } else if let Some(path) = line.strip_prefix("*** Update File: ") {
            let path = path.trim().to_owned();
            lines.next();
            let move_to = lines
                .peek()
                .and_then(|l| l.strip_prefix("*** Move to: "))
                .map(|d| d.trim().to_owned());
            if move_to.is_some() {
                lines.next();
            }
            let mut hunks = Vec::new();
            let (mut old, mut new) = (String::new(), String::new());
            while let Some(&l) = lines.peek() {
                if l.starts_with("*** ") {
                    break;
                }
                let l = lines.next().unwrap();
                if l.starts_with("@@") {
                    push_hunk(&mut hunks, &mut old, &mut new);
                } else if let Some(rest) = l.strip_prefix('-') {
                    old.push_str(rest);
                    old.push('\n');
                } else if let Some(rest) = l.strip_prefix('+') {
                    new.push_str(rest);
                    new.push('\n');
                } else {
                    let rest = l.strip_prefix(' ').unwrap_or(l);
                    old.push_str(rest);
                    old.push('\n');
                    new.push_str(rest);
                    new.push('\n');
                }
            }
            push_hunk(&mut hunks, &mut old, &mut new);
            ops.push(Op::Update { path, move_to, hunks });
        } else {
            lines.next(); // skip blank/stray lines between sections
        }
    }
    Err("patch is missing its `*** End Patch` line".to_owned())
}

fn push_hunk(hunks: &mut Vec<Hunk>, old: &mut String, new: &mut String) {
    if !old.is_empty() || !new.is_empty() {
        hunks.push(Hunk { old: std::mem::take(old), new: std::mem::take(new) });
    }
}

fn apply(ops: Vec<Op>, cwd: &Path) -> ToolOutcome {
    let mut summary = Vec::new();
    for op in ops {
        match op {
            Op::Add { path, content } => {
                let Some(p) = safe_join(cwd, &path) else {
                    return ToolOutcome::err(format!("path `{path}` escapes the working directory"));
                };
                if let Some(parent) = p.parent() {
                    if let Err(e) = std::fs::create_dir_all(parent) {
                        return ToolOutcome::err(format!("creating parent of `{path}`: {e}"));
                    }
                }
                if let Err(e) = std::fs::write(&p, &content) {
                    return ToolOutcome::err(format!("writing `{path}`: {e}"));
                }
                summary.push(format!("added {path}"));
            }
            Op::Delete { path } => {
                let Some(p) = safe_join(cwd, &path) else {
                    return ToolOutcome::err(format!("path `{path}` escapes the working directory"));
                };
                if let Err(e) = std::fs::remove_file(&p) {
                    return ToolOutcome::err(format!("deleting `{path}`: {e}"));
                }
                summary.push(format!("deleted {path}"));
            }
            Op::Update { path, move_to, hunks } => {
                let Some(p) = safe_join(cwd, &path) else {
                    return ToolOutcome::err(format!("path `{path}` escapes the working directory"));
                };
                let mut content = match std::fs::read_to_string(&p) {
                    Ok(c) => c,
                    Err(e) => return ToolOutcome::err(format!("reading `{path}` to update: {e}")),
                };
                for hunk in &hunks {
                    if content.contains(&hunk.old) {
                        content = content.replacen(&hunk.old, &hunk.new, 1);
                    } else if let Some(updated) = super::replace_line_trimmed(&content, &hunk.old, &hunk.new) {
                        content = updated; // whitespace-tolerant fallback, same as `edit`
                    } else {
                        return ToolOutcome::err(format!(
                            "update to `{path}` did not apply — its context was not found (exact, then whitespace-tolerant)"
                        ));
                    }
                }
                let dst = match &move_to {
                    Some(m) => match safe_join(cwd, m) {
                        Some(d) => d,
                        None => return ToolOutcome::err(format!("move target `{m}` escapes the working directory")),
                    },
                    None => p.clone(),
                };
                if let Some(parent) = dst.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                if let Err(e) = std::fs::write(&dst, &content) {
                    return ToolOutcome::err(format!("writing updated `{path}`: {e}"));
                }
                match move_to {
                    Some(m) => {
                        let _ = std::fs::remove_file(&p); // moved: drop the old path
                        summary.push(format!("updated {path} → {m}"));
                    }
                    None => summary.push(format!("updated {path}")),
                }
            }
        }
    }
    if summary.is_empty() {
        return ToolOutcome::err("patch contained no file operations");
    }
    ToolOutcome::ok(format!("Applied patch:\n{}", summary.join("\n")))
}
