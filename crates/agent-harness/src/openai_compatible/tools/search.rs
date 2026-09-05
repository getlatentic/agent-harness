//! Search tools — `glob` (file matching) and `grep` (content search), both
//! gitignore-aware via ripgrep's own libraries (`ignore` walk + `globset` +
//! `regex`) in-process — no `rg` binary required (OpenCode downloads one;
//! linking the libs avoids both a subprocess and that bootstrap). Read-only, so
//! both are offered in every mode.

use std::path::Path;

use globset::GlobBuilder;
use ignore::WalkBuilder;
use regex::Regex;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::ToolKind;

use super::{parse_args, safe_join, schema_for, Tool, ToolCtx, ToolOutcome, MAX_LINE_CHARS, SEARCH_LIMIT};

#[derive(Deserialize, JsonSchema)]
struct GlobArgs {
    /// Glob pattern, matched against each file's path relative to the working
    /// directory (e.g. `**/*.rs`, `src/*.toml`).
    pattern: String,
    /// Directory to search under, relative to the working directory (default:
    /// the whole tree).
    path: Option<String>,
}

pub(super) struct Glob;
impl Tool for Glob {
    fn id(&self) -> &str {
        "glob"
    }
    fn description(&self) -> &str {
        "Find files by glob pattern (e.g. `**/*.rs`, `src/*.toml`), \
         gitignore-aware. Returns matching paths (max 100)."
    }
    fn parameters(&self) -> Value {
        schema_for::<GlobArgs>()
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Search
    }
    fn mutating(&self) -> bool {
        false
    }
    fn execute(&self, args: &Value, ctx: &ToolCtx) -> ToolOutcome {
        let a: GlobArgs = match parse_args(args) {
            Ok(a) => a,
            Err(o) => return o,
        };
        glob_files(ctx.cwd, &a.pattern, a.path.as_deref())
    }
}

#[derive(Deserialize, JsonSchema)]
struct GrepArgs {
    /// Regular expression to search file contents for.
    pattern: String,
    /// Directory to search under, relative to the working directory (default:
    /// the whole tree).
    path: Option<String>,
    /// Optional glob to restrict which files are searched (e.g. `*.rs`).
    include: Option<String>,
}

pub(super) struct Grep;
impl Tool for Grep {
    fn id(&self) -> &str {
        "grep"
    }
    fn description(&self) -> &str {
        "Search file contents by regular expression, gitignore-aware. Returns \
         `path:line: text` matches (max 100)."
    }
    fn parameters(&self) -> Value {
        schema_for::<GrepArgs>()
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Search
    }
    fn mutating(&self) -> bool {
        false
    }
    fn execute(&self, args: &Value, ctx: &ToolCtx) -> ToolOutcome {
        let a: GrepArgs = match parse_args(args) {
            Ok(a) => a,
            Err(o) => return o,
        };
        grep_files(ctx.cwd, &a.pattern, a.path.as_deref(), a.include.as_deref())
    }
}

#[derive(Deserialize, JsonSchema)]
struct ListArgs {
    /// Directory to list, relative to the working directory (default: the
    /// working directory).
    path: Option<String>,
}

pub(super) struct List;
impl Tool for List {
    fn id(&self) -> &str {
        "list"
    }
    fn description(&self) -> &str {
        "List the immediate entries of a directory (relative to the working \
         directory), gitignore-aware, with directories marked by a trailing `/`. \
         Use it to see what's in a directory; use glob to match files by pattern \
         across the whole tree."
    }
    fn parameters(&self) -> Value {
        schema_for::<ListArgs>()
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Search
    }
    fn mutating(&self) -> bool {
        false
    }
    fn execute(&self, args: &Value, ctx: &ToolCtx) -> ToolOutcome {
        let a: ListArgs = match parse_args(args) {
            Ok(a) => a,
            Err(o) => return o,
        };
        list_dir(ctx.cwd, a.path.as_deref())
    }
}

fn glob_files(cwd: &Path, pattern: &str, path: Option<&str>) -> ToolOutcome {
    let root = match path {
        Some(p) => match safe_join(cwd, p) {
            Some(d) => d,
            None => return ToolOutcome::err(format!("path `{p}` escapes the working directory")),
        },
        None => cwd.to_path_buf(),
    };
    let matcher = match GlobBuilder::new(pattern).build() {
        Ok(g) => g.compile_matcher(),
        Err(e) => return ToolOutcome::err(format!("glob: invalid pattern `{pattern}`: {e}")),
    };
    let mut hits = Vec::new();
    let mut truncated = false;
    // `ignore`'s walker respects .gitignore and skips hidden files by default.
    for entry in WalkBuilder::new(&root).build() {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let rel = entry.path().strip_prefix(cwd).unwrap_or_else(|_| entry.path());
        if matcher.is_match(rel) {
            if hits.len() >= SEARCH_LIMIT {
                truncated = true;
                break;
            }
            hits.push(rel.to_string_lossy().into_owned());
        }
    }
    if hits.is_empty() {
        return ToolOutcome::ok("(no files matched)");
    }
    let mut out = hits.join("\n");
    if truncated {
        out.push_str(&format!("\n… results truncated at {SEARCH_LIMIT}; narrow the pattern."));
    }
    ToolOutcome::ok(out)
}

fn grep_files(cwd: &Path, pattern: &str, path: Option<&str>, include: Option<&str>) -> ToolOutcome {
    let root = match path {
        Some(p) => match safe_join(cwd, p) {
            Some(d) => d,
            None => return ToolOutcome::err(format!("path `{p}` escapes the working directory")),
        },
        None => cwd.to_path_buf(),
    };
    let re = match Regex::new(pattern) {
        Ok(r) => r,
        Err(e) => return ToolOutcome::err(format!("grep: invalid regex `{pattern}`: {e}")),
    };
    let include = match include {
        Some(g) => match GlobBuilder::new(g).build() {
            Ok(gg) => Some(gg.compile_matcher()),
            Err(e) => return ToolOutcome::err(format!("grep: invalid include glob `{g}`: {e}")),
        },
        None => None,
    };
    let mut out = String::new();
    let mut count = 0usize;
    let mut truncated = false;
    'walk: for entry in WalkBuilder::new(&root).build() {
        let Ok(entry) = entry else { continue };
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let rel = entry.path().strip_prefix(cwd).unwrap_or_else(|_| entry.path());
        if let Some(g) = &include
            && !g.is_match(rel)
        {
            continue;
        }
        // Skip binary / non-UTF-8 files (read_to_string errors on them).
        let Ok(content) = std::fs::read_to_string(entry.path()) else { continue };
        for (i, line) in content.lines().enumerate() {
            if re.is_match(line) {
                if count >= SEARCH_LIMIT {
                    truncated = true;
                    break 'walk;
                }
                let shown: String = if line.chars().count() > MAX_LINE_CHARS {
                    line.chars().take(MAX_LINE_CHARS).collect()
                } else {
                    line.to_owned()
                };
                out.push_str(&format!("{}:{}: {shown}\n", rel.to_string_lossy(), i + 1));
                count += 1;
            }
        }
    }
    if count == 0 {
        return ToolOutcome::ok("(no matches)");
    }
    if truncated {
        out.push_str(&format!("… results truncated at {SEARCH_LIMIT}; narrow the pattern or path.\n"));
    }
    ToolOutcome::ok(format!("Found {count} match{}:\n{out}", if count == 1 { "" } else { "es" }))
}

fn list_dir(cwd: &Path, path: Option<&str>) -> ToolOutcome {
    let root = match path {
        Some(p) => match safe_join(cwd, p) {
            Some(d) => d,
            None => return ToolOutcome::err(format!("path `{p}` escapes the working directory")),
        },
        None => cwd.to_path_buf(),
    };
    if !root.is_dir() {
        return ToolOutcome::err(format!("list: `{}` is not a directory", path.unwrap_or(".")));
    }
    // One level deep, gitignore-aware (the same walker as glob): depth 0 is the
    // root itself, which we skip.
    let (mut dirs, mut files) = (Vec::new(), Vec::new());
    let mut truncated = false;
    for entry in WalkBuilder::new(&root).max_depth(Some(1)).build() {
        let Ok(entry) = entry else { continue };
        if entry.depth() == 0 {
            continue;
        }
        if dirs.len() + files.len() >= SEARCH_LIMIT {
            truncated = true;
            break;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry.file_type().is_some_and(|t| t.is_dir()) {
            dirs.push(format!("{name}/"));
        } else {
            files.push(name);
        }
    }
    if dirs.is_empty() && files.is_empty() {
        return ToolOutcome::ok("(empty directory)");
    }
    dirs.sort();
    files.sort();
    let mut out = String::new();
    for entry in dirs.iter().chain(files.iter()) {
        out.push_str(entry);
        out.push('\n');
    }
    if truncated {
        out.push_str(&format!("… truncated at {SEARCH_LIMIT} entries; use glob for a targeted search.\n"));
    }
    ToolOutcome::ok(out.trim_end().to_owned())
}
