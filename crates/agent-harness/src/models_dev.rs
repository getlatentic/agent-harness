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
//! static models. With it on, the ~2 MB catalog is fetched once and cached **on
//! disk** (under `AGENT_HARNESS_CACHE_DIR`, when the host app sets it): later
//! launches load the cache instantly — so the picker works offline — and refresh
//! it in the background. A provider's models are filtered to the agent-usable
//! ones (`tool_call: true`, which drops embeddings / tts / image models), mapped
//! to [`HarnessModel`], and ordered newest first.
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
    use std::path::PathBuf;
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
        /// ISO release date (`YYYY-MM-DD`) where models.dev has it — used to put
        /// newer models first in the picker.
        #[serde(default)]
        release_date: Option<String>,
    }

    /// The catalog for the process. Prefers the on-disk cache — instant and
    /// works offline — and refreshes it in the background; on a cold first run
    /// with no cache it fetches once and persists it. A miss caches `None`, so
    /// callers fall back without retrying every call.
    fn catalog() -> Option<&'static Catalog> {
        static CACHE: OnceLock<Option<Catalog>> = OnceLock::new();
        CACHE
            .get_or_init(|| {
                if let Some(cached) = load_cached() {
                    // The catalog changes slowly — refresh at most once a day.
                    if cache_is_stale() {
                        std::thread::spawn(refresh_cache);
                    }
                    return Some(cached);
                }
                let body = fetch_remote()?;
                write_cache(&body);
                serde_json::from_str(&body).ok()
            })
            .as_ref()
    }

    /// Where the catalog is cached, when the host app names a cache dir via
    /// `AGENT_HARNESS_CACHE_DIR`; `None` → no disk cache (fetch-only).
    fn cache_path() -> Option<PathBuf> {
        let dir = std::env::var_os("AGENT_HARNESS_CACHE_DIR")?;
        Some(PathBuf::from(dir).join("models_dev.json"))
    }

    fn load_cached() -> Option<Catalog> {
        let body = std::fs::read_to_string(cache_path()?).ok()?;
        serde_json::from_str(&body).ok()
    }

    fn write_cache(body: &str) {
        let Some(path) = cache_path() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(path, body);
    }

    fn fetch_remote() -> Option<String> {
        ureq::get(API_URL)
            .timeout(Duration::from_secs(8))
            .call()
            .ok()?
            .into_string()
            .ok()
    }

    /// Refetch and rewrite the disk cache so the next launch is current.
    fn refresh_cache() {
        if let Some(body) = fetch_remote() {
            write_cache(&body);
        }
    }

    /// Whether the cache file is at least a day old — the only time the
    /// background refresh fires, so we re-fetch the ~2 MB catalog at most daily.
    fn cache_is_stale() -> bool {
        const MAX_AGE: Duration = Duration::from_secs(24 * 60 * 60);
        let Some(path) = cache_path() else {
            return false;
        };
        match std::fs::metadata(&path).and_then(|meta| meta.modified()) {
            Ok(modified) => modified.elapsed().map(|age| age >= MAX_AGE).unwrap_or(true),
            Err(_) => true,
        }
    }

    pub fn provider_models(provider: &str) -> Vec<HarnessModel> {
        catalog().map(|c| select(c, provider)).unwrap_or_default()
    }

    /// Pure filter+map (no network), so the selection logic is unit-testable.
    fn select(catalog: &Catalog, provider: &str) -> Vec<HarnessModel> {
        let Some(p) = catalog.0.get(provider) else {
            return Vec::new();
        };
        let mut models: Vec<&Model> = p.models.values().filter(|m| m.tool_call).collect();
        // Newest first: models.dev `release_date` is ISO (`YYYY-MM-DD`), so a
        // reverse string compare orders chronologically; undated models sort to
        // the bottom, ties broken by id for a stable order.
        models.sort_by(|a, b| {
            b.release_date
                .cmp(&a.release_date)
                .then_with(|| a.id.cmp(&b.id))
        });
        models
            .into_iter()
            .map(|m| HarnessModel {
                value: m.id.clone(),
                label: m.name.clone().unwrap_or_else(|| m.id.clone()),
            })
            .collect()
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

        #[test]
        fn select_orders_newest_release_first() {
            let json = r#"{
              "anthropic": { "models": {
                "old":     { "id": "old",     "tool_call": true, "release_date": "2023-03-01" },
                "new":     { "id": "new",     "tool_call": true, "release_date": "2024-10-01" },
                "mid":     { "id": "mid",     "tool_call": true, "release_date": "2024-02-01" },
                "undated": { "id": "undated", "tool_call": true }
              }}
            }"#;
            let catalog: Catalog = serde_json::from_str(json).expect("parse catalog");
            let ids: Vec<String> =
                select(&catalog, "anthropic").into_iter().map(|m| m.value).collect();
            assert_eq!(ids, ["new", "mid", "old", "undated"], "newest first, undated last");
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
