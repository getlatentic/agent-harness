//! Skills — discover `SKILL.md` files and surface them to the model.
//!
//! Mirrors OpenCode's design (MIT): the model gets a name+description
//! **catalog** in its system prompt and loads a skill's full body **on demand**
//! with the `skill` tool (progressive disclosure — not preloaded, not a
//! subagent). Discovered from the project's `<cwd>/.claude/skills` and
//! `<cwd>/.agents/skills`, plus any per-user roots the host opts into via
//! [`global_skill_roots`] (first definition of a name wins, global before
//! project). Each skill is a `SKILL.md` with `---` frontmatter
//! (`name` required, `description` optional) and a Markdown body.

use std::collections::HashSet;
use std::iter::Peekable;
use std::path::{Path, PathBuf};

use ignore::WalkBuilder;

/// One discovered skill.
#[derive(Debug, Clone)]
pub(crate) struct Skill {
    pub name: String,
    pub description: Option<String>,
    pub body: String,
}

/// Discover the skills visible from `cwd`, plus any `global_roots` the host
/// supplied. First definition of a given name wins (global roots are scanned
/// before project roots).
///
/// Every catalogued skill costs prompt tokens on every turn, so the same rule
/// as instructions applies: nothing under `$HOME` is scanned unless a host
/// asks. See [`global_skill_roots`].
pub(crate) fn discover(cwd: &Path, global_roots: &[PathBuf]) -> Vec<Skill> {
    discover_in(&roots(cwd, global_roots))
}

/// The conventional per-user skill directories, for a host that wants a user's
/// existing skills honoured. Opt in by passing these as
/// [`OpenHarnessConfig::global_skill_roots`](crate::OpenHarnessConfig::global_skill_roots).
pub fn global_skill_roots() -> Vec<PathBuf> {
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
        return Vec::new();
    };
    vec![home.join(".claude/skills"), home.join(".agents/skills")]
}

/// The roots scanned for `*/SKILL.md`, in precedence order.
fn roots(cwd: &Path, global_roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut roots = global_roots.to_vec();
    roots.push(cwd.join(".claude/skills"));
    roots.push(cwd.join(".agents/skills"));
    roots
}

fn discover_in(roots: &[PathBuf]) -> Vec<Skill> {
    let mut skills = Vec::new();
    let mut seen = HashSet::new();
    for root in roots {
        for path in skill_files(root) {
            if let Some(skill) = read_skill(&path)
                && seen.insert(skill.name.clone())
            {
                skills.push(skill);
            }
        }
    }
    skills
}

/// A name+description catalog for the system prompt, or `None` when there are no
/// skills (so nothing is appended).
pub(crate) fn catalog(skills: &[Skill]) -> Option<String> {
    if skills.is_empty() {
        return None;
    }
    let mut out = String::from(
        "\n\n## Available skills\nEach provides specialized instructions for a kind of task. \
         Call the `skill` tool with a skill's name to load its full instructions when a task matches.\n",
    );
    for skill in skills {
        match &skill.description {
            // One bullet per skill, so a literal block's newlines are flattened
            // rather than splitting the entry across list items.
            Some(d) => out.push_str(&format!("- `{}` — {}\n", skill.name, one_line(d))),
            None => out.push_str(&format!("- `{}`\n", skill.name)),
        }
    }
    Some(out)
}

/// Every `SKILL.md` under `root`. Standard filters are off — skills live in
/// dot-directories (`.claude`), which the default gitignore/hidden filters skip.
///
/// Sorted, and that matters beyond tidiness. The catalog these produce sits in
/// the system prompt ahead of the volatile working-directory block, so it is
/// part of the cacheable prefix every request shares. Directory order is
/// whatever `readdir` returns — stable enough to look fine locally, not
/// guaranteed across machines or after a directory is rewritten — and a single
/// reordered line changes the prompt bytes, missing the KV cache and paying a
/// full re-prefill of everything before it.
fn skill_files(root: &Path) -> Vec<PathBuf> {
    if !root.is_dir() {
        return Vec::new();
    }
    WalkBuilder::new(root)
        .standard_filters(false)
        .sort_by_file_name(std::ffi::OsStr::cmp)
        .build()
        .filter_map(Result::ok)
        .filter(|e| e.file_name() == "SKILL.md")
        .map(|e| e.path().to_path_buf())
        .collect()
}

fn read_skill(path: &Path) -> Option<Skill> {
    let content = std::fs::read_to_string(path).ok()?;
    let (frontmatter, body) = split_frontmatter(&content);
    let name = field(frontmatter, "name")?; // a skill must be named
    Some(Skill { name, description: field(frontmatter, "description"), body: body.trim().to_owned() })
}

/// Split leading `---\n…\n---` frontmatter from the body. Returns
/// `(frontmatter, body)`; if there's no frontmatter block, frontmatter is empty.
fn split_frontmatter(content: &str) -> (&str, &str) {
    let Some(rest) = content.strip_prefix("---\n") else {
        return ("", content);
    };
    match rest.find("\n---") {
        Some(end) => {
            let frontmatter = &rest[..end];
            let after = &rest[end + 1..]; // at the closing `---`
            let body = after
                .strip_prefix("---")
                .map(|b| b.trim_start_matches(['\r', '\n']))
                .unwrap_or(after);
            (frontmatter, body)
        }
        None => ("", content),
    }
}

/// Read a frontmatter field: a flat `key: value`, or a YAML block scalar
/// (`key: >` / `key: |`, with any chomping indicator).
///
/// The block forms matter. A description worth writing is usually longer than
/// one comfortable line, so skill authors fold it — and reading only the marker
/// line yielded a description of `">-"`, which tells the model nothing about
/// when to call the skill. It was silently invisible rather than broken.
fn field(frontmatter: &str, key: &str) -> Option<String> {
    let mut lines = frontmatter.lines().peekable();
    while let Some(line) = lines.next() {
        let Some(rest) = line.trim_start().strip_prefix(key) else { continue };
        let Some(value) = rest.trim_start().strip_prefix(':') else { continue };
        let value = value.trim();
        let text = match block_style(value) {
            Some(style) => read_block(&mut lines, style),
            None => value.trim_matches(['"', '\'']).to_owned(),
        };
        if !text.is_empty() {
            return Some(text);
        }
    }
    None
}

/// Collapse every run of whitespace to a single space.
fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// How a block scalar joins its lines. Chomping indicators (`-`, `+`) only
/// affect trailing newlines, which are trimmed either way.
#[derive(Clone, Copy)]
enum BlockStyle {
    /// `>` — newlines become spaces.
    Folded,
    /// `|` — newlines are kept.
    Literal,
}

fn block_style(value: &str) -> Option<BlockStyle> {
    match value.trim_end_matches(['-', '+']) {
        ">" => Some(BlockStyle::Folded),
        "|" => Some(BlockStyle::Literal),
        _ => None,
    }
}

/// Consume the indented lines belonging to a block scalar, stopping at the
/// first line that is dedented to column zero — the next key.
fn read_block(lines: &mut Peekable<std::str::Lines<'_>>, style: BlockStyle) -> String {
    let mut parts = Vec::new();
    while let Some(line) = lines.peek() {
        if !line.trim().is_empty() && !line.starts_with([' ', '\t']) {
            break;
        }
        parts.push(lines.next().unwrap_or_default().trim());
    }
    match style {
        BlockStyle::Folded => parts.iter().filter(|p| !p.is_empty()).cloned().collect::<Vec<_>>().join(" "),
        BlockStyle::Literal => parts.join("\n").trim().to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hl-skills-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn reads_frontmatter_and_body() {
        let dir = scratch("read");
        let p = dir.join("SKILL.md");
        std::fs::write(&p, "---\nname: deploy\ndescription: How to deploy the app\n---\nStep 1. Do the thing.\n").unwrap();
        let s = read_skill(&p).expect("parsed");
        assert_eq!(s.name, "deploy");
        assert_eq!(s.description.as_deref(), Some("How to deploy the app"));
        assert!(s.body.contains("Step 1"));
        assert!(!s.body.starts_with("---"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_nameless_skill_is_skipped() {
        let dir = scratch("nameless");
        let p = dir.join("SKILL.md");
        std::fs::write(&p, "---\ndescription: no name here\n---\nbody\n").unwrap();
        assert!(read_skill(&p).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn discover_in_dedupes_by_name_first_wins() {
        let dir = scratch("dedupe");
        for (sub, desc) in [("a/deploy", "first"), ("b/deploy", "second")] {
            let d = dir.join(sub);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("SKILL.md"), format!("---\nname: deploy\ndescription: {desc}\n---\nbody\n")).unwrap();
        }
        let skills = discover_in(std::slice::from_ref(&dir));
        assert_eq!(skills.len(), 1, "duplicate names collapse");
        assert_eq!(skills[0].description.as_deref(), Some("first"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn catalog_lists_skills_or_is_none() {
        assert!(catalog(&[]).is_none());
        let skills = vec![Skill { name: "x".into(), description: Some("does x".into()), body: "b".into() }];
        let c = catalog(&skills).unwrap();
        assert!(c.contains("`x`") && c.contains("does x"));
    }

    #[test]
    fn the_catalog_is_byte_identical_across_discoveries() {
        // The catalog is part of the cacheable prompt prefix. If discovery
        // order can drift, the bytes drift with it and every request pays a
        // re-prefill for a list that did not actually change.
        let root = scratch("stable");
        let skills_dir = root.join(".claude/skills");
        for name in ["zebra", "alpha", "middle", "beta"] {
            let dir = skills_dir.join(name);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("SKILL.md"), format!("---\nname: {name}\ndescription: does {name}\n---\nbody"))
                .unwrap();
        }

        let first = catalog(&discover(&root, &[])).expect("catalog");
        for _ in 0..5 {
            assert_eq!(catalog(&discover(&root, &[])).unwrap(), first, "discovery must be stable");
        }
        // Sorted, so the order is a property of the names rather than of the
        // filesystem that happened to hand them over.
        let order: Vec<usize> =
            ["alpha", "beta", "middle", "zebra"].iter().map(|n| first.find(n).expect(n)).collect();
        assert!(order.windows(2).all(|w| w[0] < w[1]), "expected name order, got:\n{first}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_folded_description_is_read_not_left_as_its_marker() {
        // The regression: 7 of 21 real skills wrote `description: >-`, and the
        // flat reader returned ">-" as the description. The skill still showed
        // up in the catalog, so nothing looked broken — the model just had no
        // reason to ever call it.
        let frontmatter = "name: ast-grep\ndescription: >-\n  Structural search and rewrite\n  across files by syntax.\nversion: 1";

        assert_eq!(field(frontmatter, "name").as_deref(), Some("ast-grep"));
        assert_eq!(
            field(frontmatter, "description").as_deref(),
            Some("Structural search and rewrite across files by syntax."),
            "a folded block joins its lines with spaces"
        );
    }

    #[test]
    fn a_literal_block_keeps_its_line_breaks_but_the_catalog_does_not() {
        let frontmatter = "description: |\n  first line\n  second line\n";
        let description = field(frontmatter, "description").expect("description");
        assert_eq!(description, "first line\nsecond line", "`|` keeps newlines");

        let skills = vec![Skill { name: "s".into(), description: Some(description), body: String::new() }];
        let catalog = catalog(&skills).unwrap();
        assert!(
            catalog.contains("- `s` — first line second line\n"),
            "one bullet per skill, so newlines flatten: {catalog}"
        );
    }

    #[test]
    fn a_block_scalar_stops_at_the_next_key() {
        let frontmatter = "description: >\n  wanted text\nname: not-the-description\n";
        assert_eq!(field(frontmatter, "description").as_deref(), Some("wanted text"));
        assert_eq!(field(frontmatter, "name").as_deref(), Some("not-the-description"));
    }

    #[test]
    fn a_flat_quoted_value_still_reads_as_before() {
        assert_eq!(field("description: \"quoted\"\n", "description").as_deref(), Some("quoted"));
        assert_eq!(field("description: plain\n", "description").as_deref(), Some("plain"));
        assert_eq!(field("other: x\n", "description"), None);
    }
}
