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
//! Selection keys on facts the backend reports — [`ModelFacts`] — not on the
//! model's name. A name is a guess needing per-vendor upkeep; a window and a
//! parameter count are measured, and between them they answer both halves of
//! the question. Neither alone is enough: `llama3.2:1b` advertises a 131k
//! window, and a 70B model can still be served on a 4k one.

/// Context windows at or below this get [`PromptProfile::Compact`].
///
/// A full surface costs roughly 1.5k tokens before the conversation starts.
/// At 16k that is affordable; at 8k — Ollama's fallback, and near what a
/// `llama-server` is typically started with — it is most of the budget.
pub const COMPACT_AT_OR_BELOW_TOKENS: u64 = 16_384;

/// Models at or below this many billion parameters get [`PromptProfile::Compact`].
///
/// A separate question from the window, and the reason both are needed: context
/// says what a run can *afford* to send, parameters say what the model can be
/// trusted to *do* with it. `llama3.2:1b` advertises a 131k window and would
/// pass the context test comfortably, yet handed eleven tool schemas it recites
/// them back as prose and loops to the turn limit inventing tools. Reliable
/// tool-calling starts around 7B.
pub const COMPACT_AT_OR_BELOW_PARAMS_B: f64 = 7.0;

/// What a backend was able to tell us about a model and how it is served.
///
/// Both measurements are independently optional: no backend reports both, and
/// a bare `llama-server` reports neither. `served_locally` is always known, and
/// is what decides the case where the measurements are absent.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ModelFacts {
    /// Usable context window in tokens. Ollama reports it via `/api/show`,
    /// OpenRouter via `context_length`; a bare `llama-server` does not.
    pub context_tokens: Option<u64>,
    /// Parameter count in billions. Ollama reports it; hosted providers
    /// generally do not.
    pub parameters_b: Option<f64>,
    /// Whether the endpoint is a model served on this machine or the local
    /// network. Decides the profile when neither measurement is available: what
    /// people run locally is small and configured modestly, and the observed
    /// failure there is a hard refusal rather than a slightly narrower run.
    pub served_locally: bool,
}

/// Whether `base_url` points at a locally served model — this machine or the
/// local network.
///
/// Not a security boundary; it only picks a default. A private address is
/// included because "Ollama on the box under the desk" is the same situation as
/// Ollama on this one: a self-hosted model, modestly configured, with no
/// catalog to ask about it.
pub fn is_local_endpoint(base_url: &str) -> bool {
    is_local_host(host_of(base_url))
}

/// The host part of a URL: after the scheme, before the path, without the port
/// or IPv6 brackets.
fn host_of(base_url: &str) -> &str {
    let after_scheme = base_url.split_once("//").map_or(base_url, |(_, rest)| rest);
    let authority = after_scheme.split('/').next().unwrap_or("");
    // An IPv6 literal is bracketed and full of colons, so unwrap it before
    // trying to strip a port.
    match authority.strip_prefix('[') {
        Some(rest) => rest.split(']').next().unwrap_or(rest),
        None => authority.rsplit_once(':').map_or(authority, |(host, _)| host),
    }
}

fn is_local_host(host: &str) -> bool {
    if matches!(host, "localhost" | "::1" | "0.0.0.0") || host.ends_with(".local") {
        return true;
    }
    // Every label must be an octet, or this is a name that merely looks numeric
    // (`172.1.2.3.example.com` is somebody's public host).
    let Some(octets) = host.split('.').map(|label| label.parse::<u8>().ok()).collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    match octets[..] {
        [127, ..] | [10, _, _, _] | [192, 168, _, _] => true,
        [172, second, _, _] => (16..=31).contains(&second),
        _ => false,
    }
}

/// Which prompt and tool surface a run gets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PromptProfile {
    /// Decide from [`ModelFacts`], falling back to [`Self::Full`] when the
    /// backend reported nothing useful.
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
    /// The profile to actually use, resolving [`Self::Auto`] against what the
    /// backend reported.
    ///
    /// Either measurement alone is enough to choose [`Self::Compact`]: a small
    /// window cannot fit the full surface, and a small model cannot use it.
    ///
    /// When neither is available the endpoint decides. A hosted one gets
    /// [`Self::Full`] — withholding tools from a frontier model narrows the run
    /// silently, which is the harder error to notice. A local one gets
    /// [`Self::Compact`], because a self-hosted model is usually small and
    /// started on a modest context, and there the error is loud: `llama-server`
    /// refuses the whole request rather than answering a little worse.
    ///
    /// A host that knows better overrides with an explicit profile.
    pub fn resolve(self, facts: ModelFacts) -> Self {
        let Self::Auto = self else { return self };
        let cramped = facts.context_tokens.is_some_and(|t| t <= COMPACT_AT_OR_BELOW_TOKENS);
        let small = facts.parameters_b.is_some_and(|p| p <= COMPACT_AT_OR_BELOW_PARAMS_B);
        let unmeasured = facts.context_tokens.is_none() && facts.parameters_b.is_none();
        if cramped || small || (unmeasured && facts.served_locally) {
            Self::Compact
        } else {
            Self::Full
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

/// The default base prompt.
///
/// The text lives in a file rather than a `const` with backslash
/// continuations: continuations silently swallow the next line's indentation,
/// which made a stray double space a real and recurring defect, and a diff of
/// the prose is unreadable when every line ends in `\`. Codex and OpenCode both
/// keep their prompts as files for the same reason.
pub(crate) const FULL_SYSTEM_PROMPT: &str = include_str!("prompts/full.md");

/// The base prompt for a small model on a small context.
///
/// Shorter than [`FULL_SYSTEM_PROMPT`] but not by trimming the rules — the
/// rules are what a weak model gets wrong. What goes is the prose: every line
/// is one imperative, because a model that cannot reliably call a tool also
/// cannot reliably parse a paragraph about when to.
pub(crate) const COMPACT_SYSTEM_PROMPT: &str = include_str!("prompts/compact.md");

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

    fn window(tokens: u64) -> ModelFacts {
        ModelFacts { context_tokens: Some(tokens), ..ModelFacts::default() }
    }

    #[test]
    fn a_cramped_window_picks_compact() {
        assert_eq!(PromptProfile::Auto.resolve(window(4_096)), PromptProfile::Compact);
        assert_eq!(PromptProfile::Auto.resolve(window(8_192)), PromptProfile::Compact);
        assert_eq!(
            PromptProfile::Auto.resolve(window(COMPACT_AT_OR_BELOW_TOKENS)),
            PromptProfile::Compact
        );
        assert_eq!(PromptProfile::Auto.resolve(window(32_768)), PromptProfile::Full);
    }

    #[test]
    fn a_small_model_picks_compact_however_large_its_window() {
        // The case the window alone gets wrong, and the reason both facts are
        // read: llama3.2:1b advertises 131k and cannot use eleven tools.
        let tiny_but_roomy =
            ModelFacts { context_tokens: Some(131_072), parameters_b: Some(1.2), ..Default::default() };
        assert_eq!(PromptProfile::Auto.resolve(tiny_but_roomy), PromptProfile::Compact);

        let big_model =
            ModelFacts { context_tokens: Some(131_072), parameters_b: Some(24.0), ..Default::default() };
        assert_eq!(PromptProfile::Auto.resolve(big_model), PromptProfile::Full);
    }

    #[test]
    fn a_big_model_on_a_cramped_window_still_picks_compact() {
        // The mirror case: capable model, no room. Either fact alone decides.
        let squeezed =
            ModelFacts { context_tokens: Some(4_096), parameters_b: Some(70.0), ..Default::default() };
        assert_eq!(PromptProfile::Auto.resolve(squeezed), PromptProfile::Compact);
    }

    #[test]
    fn nothing_reported_from_a_hosted_endpoint_keeps_the_full_surface() {
        // Withholding tools from a frontier model narrows the run silently, so
        // an absent measurement must not by itself trigger the smaller profile.
        assert_eq!(PromptProfile::Auto.resolve(ModelFacts::default()), PromptProfile::Full);
    }

    #[test]
    fn nothing_reported_from_a_local_endpoint_gets_guidance() {
        // The llama-server case: it reports no window, and guessing Full there
        // produced a 400 for the whole request rather than a worse answer.
        let local = ModelFacts { served_locally: true, ..Default::default() };
        assert_eq!(PromptProfile::Auto.resolve(local), PromptProfile::Compact);

        // A measurement still wins over the location.
        let roomy_local =
            ModelFacts { context_tokens: Some(131_072), parameters_b: Some(24.0), served_locally: true };
        assert_eq!(PromptProfile::Auto.resolve(roomy_local), PromptProfile::Full);
    }

    #[test]
    fn local_endpoints_are_recognised_by_address() {
        for local in [
            "http://localhost:11434",
            "http://127.0.0.1:8080",
            "http://[::1]:8080",
            "http://192.168.1.14:11434",
            "http://10.0.0.5:8080",
            "http://172.16.4.2:8080",
            "http://studio.local:1234",
        ] {
            assert!(is_local_endpoint(local), "{local} should read as local");
        }
        for hosted in [
            "https://openrouter.ai/api",
            "https://api.deepseek.com",
            "https://172.1.2.3.example.com",
            "https://api.together.xyz/v1",
        ] {
            assert!(!is_local_endpoint(hosted), "{hosted} should read as hosted");
        }
    }

    #[test]
    fn an_explicit_profile_ignores_every_fact() {
        assert_eq!(PromptProfile::Full.resolve(window(2_048)), PromptProfile::Full);
        assert_eq!(PromptProfile::Compact.resolve(window(200_000)), PromptProfile::Compact);
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

