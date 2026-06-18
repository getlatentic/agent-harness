//! models.dev catalog lookup for [`Harness::list_models`].
//!
//! [models.dev](https://models.dev) is the open catalog of model specs (the same
//! one opencode draws from). Its `api.json` is one GET, keyed by provider, so a
//! CLI adapter tied to a provider (Claude → `anthropic`, Codex → `openai`) can
//! offer a *live* model list instead of a hardcoded one — via [`provider_models`].
//!
//! The network call + HTTP client are gated behind the **`models-dev`** feature
//! (off by default, keeping the neutral core HTTP-free). With the feature off,
//! [`provider_models`] returns an empty list, so adapters fall back to their
//! static models. With it on, the ~2 MB catalog is fetched **once per process**
//! and cached; a provider's models are filtered to the agent-usable ones
//! (`tool_call: true`, which drops embeddings / tts / image models) and mapped to
//! [`HarnessModel`].
//!
//! [`Harness::list_models`]: crate::Harness::list_models

use crate::HarnessModel;

/// The agent-usable models a provider serves per models.dev, mapped to
/// [`HarnessModel`] and sorted by id for a stable picker order. Empty when the
/// `models-dev` feature is off, the catalog can't be fetched, or the provider is
/// unknown — so a caller can fall back to its own static list.
pub fn provider_models(provider: &str) -> Vec<HarnessModel> {
    #[cfg(feature = "models-dev")]
    {
        imp::provider_models(provider)
    }
    #[cfg(not(feature = "models-dev"))]
    {
        let _ = provider;
        Vec::new()
    }
}

#[cfg(feature = "models-dev")]
mod imp {
    use std::collections::HashMap;
    use std::sync::OnceLock;
    use std::time::Duration;

    use serde::Deserialize;

    use crate::HarnessModel;

    const API_URL: &str = "https://models.dev/api.json";

    /// models.dev combined catalog: `{ <providerId>: { models: { <id>: Model } } }`.
    #[derive(Deserialize)]
    struct Catalog(HashMap<String, Provider>);

    #[derive(Deserialize)]
    struct Provider {
        #[serde(default)]
        models: HashMap<String, Model>,
    }

    #[derive(Deserialize)]
    struct Model {
        /// Id passed to the CLI (`--model`).
        id: String,
        /// Human label; falls back to the id.
        #[serde(default)]
        name: Option<String>,
        /// Supports tool calls — our proxy for "agent-usable" (text-only
        /// embeddings / tts share the text modality but have `tool_call: false`).
        #[serde(default)]
        tool_call: bool,
    }

    /// The catalog, fetched once and cached for the process lifetime (it's ~2 MB;
    /// refetching per `list_models` call would be wasteful). A failed fetch caches
    /// `None`, so callers fall back without retrying every call.
    fn catalog() -> Option<&'static Catalog> {
        static CACHE: OnceLock<Option<Catalog>> = OnceLock::new();
        CACHE.get_or_init(fetch).as_ref()
    }

    fn fetch() -> Option<Catalog> {
        let body = ureq::get(API_URL)
            .timeout(Duration::from_secs(15))
            .call()
            .ok()?
            .into_string()
            .ok()?;
        serde_json::from_str(&body).ok()
    }

    pub fn provider_models(provider: &str) -> Vec<HarnessModel> {
        catalog().map(|c| select(c, provider)).unwrap_or_default()
    }

    /// Pure filter+map (no network), so the selection logic is unit-testable.
    fn select(catalog: &Catalog, provider: &str) -> Vec<HarnessModel> {
        let Some(p) = catalog.0.get(provider) else {
            return Vec::new();
        };
        let mut models: Vec<HarnessModel> = p
            .models
            .values()
            .filter(|m| m.tool_call)
            .map(|m| HarnessModel {
                value: m.id.clone(),
                label: m.name.clone().unwrap_or_else(|| m.id.clone()),
            })
            .collect();
        models.sort_by(|a, b| a.value.cmp(&b.value));
        models
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn select_keeps_only_tool_call_models_and_maps_name() {
            let json = r#"{
              "anthropic": { "models": {
                "claude-x": { "id": "claude-x", "name": "Claude X", "tool_call": true },
                "embed-x":  { "id": "embed-x",  "name": "Embed X",  "tool_call": false }
              }},
              "openai": { "models": {
                "o9": { "id": "o9", "tool_call": true }
              }}
            }"#;
            let catalog: Catalog = serde_json::from_str(json).expect("parse catalog");

            // anthropic: only the tool_call model survives; `name` → label.
            let a = select(&catalog, "anthropic");
            assert_eq!(a, vec![HarnessModel { value: "claude-x".into(), label: "Claude X".into() }]);

            // openai: no `name` → label falls back to the id.
            let o = select(&catalog, "openai");
            assert_eq!(o, vec![HarnessModel { value: "o9".into(), label: "o9".into() }]);

            // unknown provider → empty (caller falls back to its static list).
            assert!(select(&catalog, "nope").is_empty());
        }

        // A network smoke test against the real catalog — ignored by default so
        // CI / offline runs never flake. Run with
        // `cargo test -p agent-harness --features models-dev -- --ignored`.
        #[test]
        #[ignore = "network: fetches https://models.dev/api.json"]
        fn live_catalog_has_anthropic_and_openai_models() {
            assert!(!provider_models("anthropic").is_empty(), "anthropic should list models");
            assert!(!provider_models("openai").is_empty(), "openai should list models");
            assert!(provider_models("totally-unknown-xyz").is_empty());
        }
    }
}
