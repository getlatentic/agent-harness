//! How much prompt a run can afford, and how much guidance its model needs.
//!
//! These pull in opposite directions. A small local model needs the read-only
//! rules spelled out — a 1B model handed the full tool surface recites the
//! schemas back as prose instead of calling one — while the small context
//! window it usually comes with is exactly what cannot afford that surface.
//!
//! So the two profiles differ in *where* the tokens go, not only in how many.
//! [`PromptProfile::Compact`] withholds the optional tools and spends part of
//! what it saves on plainer instructions; [`PromptProfile::Full`] offers
//! everything and trusts the model to infer the rest.
//!
//! Selection keys on the context window rather than the model's name. The
//! window is a fact we already fetch (Ollama's `/api/show`), and it is the
//! thing that actually breaks; a name is a guess that has to be maintained per
//! vendor.

/// Context windows at or below this get [`PromptProfile::Compact`].
///
/// A full surface costs roughly 1.5k tokens before the conversation starts.
/// At 16k that is affordable; at 8k — Ollama's fallback, and near what a
/// `llama-server` is typically started with — it is most of the budget.
pub const COMPACT_AT_OR_BELOW_TOKENS: u64 = 16_384;

/// Which prompt and tool surface a run gets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PromptProfile {
    /// Decide from the model's context window, falling back to [`Self::Full`]
    /// when the window is unknown.
    #[default]
    Auto,
    /// Every tool, terse instructions.
    Full,
    /// Core tools only, explicit instructions.
    Compact,
}

/// The tools a [`PromptProfile::Compact`] run keeps. Everything else is
/// withheld: each costs schema tokens in every request, and a model small
/// enough to need this profile does worse the more choices it is given.
///
/// `read`/`list`/`glob`/`grep` are how a run finds and inspects files;
/// `write`/`edit`/`bash` are how it changes them (already withheld in
/// [`crate::RunMode::Ask`]). Nothing here is optional to a coding task.
const CORE_TOOLS: &[&str] = &["read", "list", "glob", "grep", "write", "edit", "bash"];

impl PromptProfile {
    /// The profile to actually use, resolving [`Self::Auto`] against the model's
    /// context window. An unknown window resolves to [`Self::Full`] — the
    /// conservative choice for capability, since withholding tools from a model
    /// that could use them silently narrows what a run can do.
    pub fn resolve(self, context_tokens: Option<u64>) -> Self {
        match self {
            Self::Auto => match context_tokens {
                Some(tokens) if tokens <= COMPACT_AT_OR_BELOW_TOKENS => Self::Compact,
                _ => Self::Full,
            },
            explicit => explicit,
        }
    }

    /// Tool ids this profile withholds, on top of whatever the host disabled.
    /// Empty for [`Self::Full`].
    pub(crate) fn withheld_tools(self, all: &[String]) -> Vec<String> {
        match self {
            Self::Compact => {
                all.iter().filter(|id| !CORE_TOOLS.contains(&id.as_str())).cloned().collect()
            }
            _ => Vec::new(),
        }
    }

    /// The base system prompt for this profile.
    pub(crate) fn system_prompt(self) -> &'static str {
        match self {
            Self::Compact => COMPACT_SYSTEM_PROMPT,
            _ => FULL_SYSTEM_PROMPT,
        }
    }
}

/// The default base prompt. Regenerated each run (it is *not* part of the
/// persisted transcript), so it can grow — the skills catalog is appended here.
pub(crate) const FULL_SYSTEM_PROMPT: &str = "You are a careful AI assistant working in the \
    user's files. Do exactly what the user asks — no more, no less — and \
    follow their instructions precisely.\n\
    \n\
    Match the request to the right action:\n\
    - A question, summary, explanation, review, or analysis is a READ-ONLY \
    task: read what you need, then answer directly in your reply. Do NOT \
    create, edit, or overwrite any file for these.\n\
    - Only use a write or edit tool when the user clearly asks you to create \
    or change a file. Then make the smallest change that satisfies the \
    request and keep the user's existing content and style.\n\
    - If the request is ambiguous, ask one brief clarifying question instead \
    of guessing or editing.\n\
    \n\
    Tools (paths are relative to the working directory): `read` to inspect a \
    file; `glob`, `grep`, and `list` to find files and content; `edit` for a \
    targeted change to an existing file; `write` to create or fully replace \
    one; `bash` for builds, tests, and git.\n\
    \n\
    To see what files exist or to find one, call `list` or `glob` first — \
    never guess file names or their contents from memory.\n\
    \n\
    If a write or edit is refused because the run is read-only, do NOT retry \
    it. Tell the user the run is read-only and that they can turn on editing, \
    then answer their request without changing files.\n\
    \n\
    When the task is done, reply with a short, clear final message and make \
    no further tool calls.";

/// The base prompt for a small model on a small context.
///
/// Shorter than [`FULL_SYSTEM_PROMPT`] but not by trimming the rules — the
/// rules are what a weak model gets wrong. What goes is the prose: every line
/// is one imperative, because a model that cannot reliably call a tool also
/// cannot reliably parse a paragraph about when to.
pub(crate) const COMPACT_SYSTEM_PROMPT: &str = "You are a careful coding assistant working in \
    the user's files.\n\
    \n\
    Rules:\n\
    - Do exactly what the user asks. No more.\n\
    - A question, summary, explanation, or review is READ-ONLY. Answer it in \
    your reply. Do NOT write or edit any file.\n\
    - Write or edit ONLY when the user asks you to change a file. Change as \
    little as possible.\n\
    - Never guess a file's name or contents. Call `list` or `glob` first, then \
    `read`.\n\
    - If a write is refused because the run is read-only, do NOT try again. \
    Say so, then answer without changing files.\n\
    - Call one tool at a time and wait for its result.\n\
    - When you have the answer, reply in plain text and stop calling tools.";

#[cfg(test)]
mod tests {
    use super::*;

    fn all_tools() -> Vec<String> {
        ["read", "glob", "grep", "list", "webfetch", "todowrite", "question", "skill", "summarize",
         "websearch", "write", "edit", "bash", "applypatch", "task"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect()
    }

    #[test]
    fn auto_picks_compact_only_for_a_small_window() {
        assert_eq!(PromptProfile::Auto.resolve(Some(4_096)), PromptProfile::Compact);
        assert_eq!(PromptProfile::Auto.resolve(Some(8_192)), PromptProfile::Compact);
        assert_eq!(PromptProfile::Auto.resolve(Some(COMPACT_AT_OR_BELOW_TOKENS)), PromptProfile::Compact);
        assert_eq!(PromptProfile::Auto.resolve(Some(32_768)), PromptProfile::Full);
    }

    #[test]
    fn an_unknown_window_keeps_the_full_surface() {
        // Withholding tools from a model that could use them narrows the run
        // silently, so an absent signal must not trigger the smaller profile.
        assert_eq!(PromptProfile::Auto.resolve(None), PromptProfile::Full);
    }

    #[test]
    fn an_explicit_profile_ignores_the_window() {
        assert_eq!(PromptProfile::Full.resolve(Some(2_048)), PromptProfile::Full);
        assert_eq!(PromptProfile::Compact.resolve(Some(200_000)), PromptProfile::Compact);
    }

    #[test]
    fn compact_keeps_the_core_and_withholds_the_rest() {
        let withheld = PromptProfile::Compact.withheld_tools(&all_tools());

        for core in CORE_TOOLS {
            assert!(!withheld.contains(&(*core).to_owned()), "{core} must survive");
        }
        for optional in ["webfetch", "websearch", "todowrite", "summarize", "task", "skill"] {
            assert!(withheld.contains(&optional.to_owned()), "{optional} must be withheld");
        }
    }

    #[test]
    fn full_withholds_nothing() {
        assert!(PromptProfile::Full.withheld_tools(&all_tools()).is_empty());
    }

    #[test]
    fn the_compact_prompt_is_smaller_but_keeps_every_rule() {
        let full = PromptProfile::Full.system_prompt();
        let compact = PromptProfile::Compact.system_prompt();
        assert!(compact.len() < full.len(), "compact: {} full: {}", compact.len(), full.len());

        // The rules a small model gets wrong are exactly the ones that must
        // survive the cut — this is the whole reason the profile exists.
        assert!(compact.contains("READ-ONLY"));
        assert!(compact.contains("do NOT try again"), "no retry after a read-only refusal");
        assert!(compact.contains("Never guess"), "no inventing file names");
        assert!(compact.contains("stop calling tools"), "must know when to finish");
    }
}
