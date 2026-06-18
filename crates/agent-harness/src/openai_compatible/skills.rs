//! Skills — discover `SKILL.md` files and surface them to the model.
//!
//! Mirrors OpenCode's design (MIT): the model gets a name+description
//! **catalog** in its system prompt and loads a skill's full body **on demand**
//! with the `skill` tool (progressive disclosure — not preloaded, not a
//! subagent). Discovered from (first definition of a name wins): the global
//! `~/.claude/skills` and `~/.agents/skills`, and the project's
//! `<cwd>/.claude/skills` and `<cwd>/.agents/skills` — so Claude Code skills are
//! picked up directly. Each skill is a `SKILL.md` with `---` frontmatter
//! (`name` required, `description` optional) and a Markdown body.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use ignore::WalkBuilder;

/// One discovered skill.
#[derive(Debug, Clone)]
pub(crate) struct Skill {
    pub name: String,
    pub description: Option<String>,
    pub body: String,
}

/// Discover the skills visible from `cwd`. First definition of a given name
/// wins (global roots are scanned before project roots).
pub(crate) fn discover(cwd: &Path) -> Vec<Skill> {
    discover_in(&roots(cwd))
}

/// The roots scanned for `*/SKILL.md`, in precedence order.
fn roots(cwd: &Path) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        roots.push(home.join(".claude/skills"));
        roots.push(home.join(".agents/skills"));
    }
    roots.push(cwd.join(".claude/skills"));
    roots.push(cwd.join(".agents/skills"));
    roots
}

fn discover_in(roots: &[PathBuf]) -> Vec<Skill> {
    let mut skills = Vec::new();
    let mut seen = HashSet::new();
    for root in roots {
        for path in skill_files(root) {
            if let Some(skill) = read_skill(&path) {
                if seen.insert(skill.name.clone()) {
                    skills.push(skill);
                }
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
            Some(d) => out.push_str(&format!("- `{}` — {d}\n", skill.name)),
            None => out.push_str(&format!("- `{}`\n", skill.name)),
        }
    }
    Some(out)
}

/// Every `SKILL.md` under `root`. Standard filters are off — skills live in
/// dot-directories (`.claude`), which the default gitignore/hidden filters skip.
fn skill_files(root: &Path) -> Vec<PathBuf> {
    if !root.is_dir() {
        return Vec::new();
    }
    WalkBuilder::new(root)
        .standard_filters(false)
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

/// Read a flat `key: value` frontmatter field, stripping surrounding quotes.
fn field(frontmatter: &str, key: &str) -> Option<String> {
    for line in frontmatter.lines() {
        if let Some(rest) = line.trim().strip_prefix(key) {
            if let Some(value) = rest.trim_start().strip_prefix(':') {
                let v = value.trim().trim_matches(['"', '\'']).to_string();
                if !v.is_empty() {
                    return Some(v);
                }
            }
        }
    }
    None
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
}
