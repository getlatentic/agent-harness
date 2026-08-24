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
//! to [`ModelChoice`], and ordered newest first.
//!
//! [`Harness::list_models`]: crate::Harness::list_models

use crate::ModelChoice;

/// The agent-usable models a provider serves per models.dev, mapped to
/// [`ModelChoice`] and sorted by id for a stable picker order. Empty when the
/// `models-dev` feature is off, the catalog can't be fetched, or the provider is
/// unknown — so a caller can fall back to its own static list.
/// A model's context window from the catalog, when it lists one.
///
/// The only cross-provider source: a hosted endpoint publishes its window in a
/// shape of its own or not at all, so without this a hosted run has no window
/// and profile selection has one less fact to work with.
pub fn context_limit(provider: &str, model: &str) -> Option<u64> {
    #[cfg(feature = "models-dev")]
    {
        imp::context_limit(provider, model)
    }
    #[cfg(not(feature = "models-dev"))]
    {
        let _ = (provider, model);
        None
    }
}

pub fn provider_models(provider: &str) -> Vec<ModelChoice> {
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
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;
    use std::time::Duration;

    use serde::Deserialize;

    use crate::ModelChoice;

    const API_URL: &str = "https://models.dev/api.json";

    /// models.dev combined catalog: `{ <providerId>: { models: { <id>: Model } } }`.
    #[derive(Deserialize)]
    struct Catalog(HashMap<String, Provider>);

    #[derive(Deserialize)]
    struct Provider {
        #[serde(default, deserialize_with = "models_skipping_unreadable")]
        models: HashMap<String, Model>,
    }

    /// Deserialize the model map one entry at a time, dropping any that does
    /// not parse.
    ///
    /// This is a ~4 MB file published by someone else, and `serde` fails a
    /// whole document on one bad field. Derived normally, a single model
    /// missing its `id` would cost **every provider** its entire list — and
    /// refetching returns the same bytes, so it stays broken until upstream
    /// fixes it. One unreadable model should cost that model.
    fn models_skipping_unreadable<'de, D>(de: D) -> Result<HashMap<String, Model>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = HashMap::<String, serde_json::Value>::deserialize(de)?;
        Ok(raw
            .into_iter()
            .filter_map(|(key, value)| Some((key, serde_json::from_value(value).ok()?)))
            .collect())
    }

    #[derive(Deserialize)]
    struct Limit {
        /// Context window in tokens.
        #[serde(default)]
        context: Option<u64>,
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
        /// Context and output ceilings. The catalog is the only source of these
        /// for a hosted provider: OpenRouter publishes a `context_length` on its
        /// own models list, but nothing does across providers.
        #[serde(default)]
        limit: Option<Limit>,
    }

    /// The catalog for the process. Prefers the on-disk cache — instant and
    /// works offline — and refreshes it in the background; on a cold first run
    /// with no cache it fetches once and persists it. A miss caches `None`, so
    /// callers fall back without retrying every call.
    fn catalog() -> Option<&'static Catalog> {
        static CACHE: OnceLock<Option<Catalog>> = OnceLock::new();
        CACHE.get_or_init(|| load_or_fetch(fetch_remote)).as_ref()
    }

    /// Prefer the on-disk cache — instant, and works offline — refreshing it in
    /// the background once it is a day old; on a cold run with no cache, fetch
    /// once and persist.
    ///
    /// `fetch` is a parameter so this can be exercised without the network, and
    /// without the process-wide `OnceLock` in [`catalog`] fixing the outcome
    /// for every later test in the binary.
    ///
    /// A body is parsed before it is written: caching one we could not read
    /// would spend the disk on something the next launch has to discard.
    fn load_or_fetch(fetch: impl FnOnce() -> Option<String>) -> Option<Catalog> {
        if let Some(cached) = load_cached() {
            // The catalog changes slowly — refresh at most once a day.
            if cache_is_stale() {
                std::thread::spawn(refresh_cache);
            }
            return Some(cached);
        }
        let body = fetch()?;
        let parsed = serde_json::from_str(&body).ok()?;
        write_cache(&body);
        Some(parsed)
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
        refresh_from(fetch_remote);
    }

    /// The rewrite itself, with the fetch as a parameter.
    ///
    /// A failed fetch leaves the existing cache untouched — this runs in the
    /// background on a launch that already has a working catalog, so a network
    /// blip must not trade it for nothing.
    fn refresh_from(fetch: impl FnOnce() -> Option<String>) {
        if let Some(body) = fetch() {
            write_cache(&body);
        }
    }

    /// Whether the cache file is at least a day old — the only time the
    /// background refresh fires, so we re-fetch the ~2 MB catalog at most daily.
    ///
    /// A host that named no cache directory has nothing to refresh.
    fn cache_is_stale() -> bool {
        cache_path().is_some_and(|path| stale(&path))
    }

    /// How old is too old, given a path. Separate from [`cache_is_stale`] so the
    /// rule can be checked against a real file without a process-wide
    /// environment variable deciding where that file lives.
    fn stale(path: &Path) -> bool {
        const MAX_AGE: Duration = Duration::from_secs(24 * 60 * 60);
        match std::fs::metadata(path).and_then(|meta| meta.modified()) {
            Ok(modified) => modified.elapsed().map(|age| age >= MAX_AGE).unwrap_or(true),
            // Absent or unreadable: fetching is the way to find out, and the
            // alternative is a picker that stays empty forever.
            Err(_) => true,
        }
    }

    pub fn context_limit(provider: &str, model: &str) -> Option<u64> {
        catalog()?
            .0
            .get(provider)?
            .models
            .values()
            .find(|entry| entry.id == model)?
            .limit
            .as_ref()?
            .context
    }

    pub fn provider_models(provider: &str) -> Vec<ModelChoice> {
        catalog().map(|c| select(c, provider)).unwrap_or_default()
    }

    /// Pure filter+map (no network), so the selection logic is unit-testable.
    fn select(catalog: &Catalog, provider: &str) -> Vec<ModelChoice> {
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
            .map(|m| ModelChoice {
                value: m.id.clone(),
                label: m.name.clone().unwrap_or_else(|| m.id.clone()),
            })
            .collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// `AGENT_HARNESS_CACHE_DIR` is process-global, so these cannot run
        /// beside each other.
        static CACHE_ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());

        fn with_cache_dir<T>(tag: &str, body: impl FnOnce(&Path) -> T) -> T {
            let _guard = CACHE_ENV.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let restore = std::env::var_os("AGENT_HARNESS_CACHE_DIR");
            let dir = std::env::temp_dir().join(format!("hl-cache-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::env::set_var("AGENT_HARNESS_CACHE_DIR", &dir);

            let out = body(&dir);

            match restore {
                Some(value) => std::env::set_var("AGENT_HARNESS_CACHE_DIR", value),
                None => std::env::remove_var("AGENT_HARNESS_CACHE_DIR"),
            }
            let _ = std::fs::remove_dir_all(&dir);
            out
        }

        const SAMPLE: &str = r#"{"anthropic":{"models":{"claude-x":{"id":"claude-x","name":"Claude X","tool_call":true,"limit":{"context":200000}}}}}"#;

        #[test]
        fn one_unreadable_model_costs_that_model_and_nothing_else() {
            // models.dev is a ~4 MB file published by someone else, and serde
            // fails a whole document on one bad field. Derived normally, a
            // single model missing its `id` dropped every provider's list —
            // and a refetch returns the same bytes, so "no models anywhere"
            // persisted until upstream fixed it.
            let mixed = r#"{
                "anthropic":{"models":{"good":{"id":"good","tool_call":true}}},
                "openai":{"models":{
                    "bad":{"tool_call":true},
                    "fine":{"id":"fine","tool_call":true}
                }}
            }"#;
            let catalog: Catalog =
                serde_json::from_str(mixed).expect("one bad model must not fail the document");

            assert_eq!(select(&catalog, "anthropic").len(), 1, "an unrelated provider is untouched");
            let openai = select(&catalog, "openai");
            assert_eq!(openai.len(), 1, "the readable sibling survives");
            assert_eq!(openai[0].value, "fine");
        }

        #[test]
        fn a_failed_refresh_leaves_the_working_cache_alone() {
            // This runs in the background on a launch that already loaded a
            // catalog. Rewriting unconditionally would trade a working cache
            // for whatever a network blip returned, and the next launch would
            // start from nothing.
            with_cache_dir("refresh", |dir| {
                write_cache(SAMPLE);
                refresh_from(|| None);
                assert_eq!(
                    std::fs::read_to_string(dir.join("models_dev.json")).unwrap(),
                    SAMPLE,
                    "a failed refresh is a no-op",
                );

                refresh_from(|| Some("{}".to_owned()));
                assert_eq!(
                    std::fs::read_to_string(dir.join("models_dev.json")).unwrap(),
                    "{}",
                    "and a successful one replaces it",
                );
            });
        }

        #[test]
        fn a_cold_start_fetches_once_and_keeps_what_it_got() {
            // The whole point of the disk cache: pay ~4 MB once, not per
            // launch. If the fetched body were not persisted, every start would
            // pay it again and nothing would fail.
            with_cache_dir("cold", |dir| {
                let catalog = load_or_fetch(|| Some(SAMPLE.to_owned()))
                    .expect("a cold start uses what it fetched");
                assert_eq!(select(&catalog, "anthropic").len(), 1);
                assert!(dir.join("models_dev.json").is_file(), "and writes it down");
            });
        }

        #[test]
        fn a_warm_start_does_not_reach_the_network_at_all() {
            // Reading the cache is what makes an offline launch work. A cold
            // path that ran anyway would still *look* right — it returns a
            // catalog either way — so the assertion has to be that the fetch
            // was never called.
            with_cache_dir("warm", |_| {
                write_cache(SAMPLE);
                let catalog = load_or_fetch(|| panic!("the disk cache must be preferred"))
                    .expect("the cached catalog");
                assert_eq!(select(&catalog, "anthropic").len(), 1);
            });
        }

        #[test]
        fn a_body_we_cannot_read_is_not_cached() {
            // Writing first and parsing second would spend the disk on
            // something the next launch has to discard, and turn one bad
            // response into a file someone has to delete by hand.
            with_cache_dir("garbage", |dir| {
                assert!(load_or_fetch(|| Some("<html>not json</html>".to_owned())).is_none());
                assert!(!dir.join("models_dev.json").exists(), "nothing was kept");
            });
        }

        #[test]
        fn an_unreachable_catalog_is_absent_rather_than_empty() {
            // No cache and no network is "we do not know", which lets a caller
            // fall back. An empty catalog would instead read as "this provider
            // has no models" — a wrong answer rather than a missing one.
            with_cache_dir("offline", |_| {
                assert!(load_or_fetch(|| None).is_none());
            });
        }

        #[test]
        fn what_is_written_to_the_cache_is_what_comes_back() {
            // The catalog is ~4 MB over the network. A cache that writes but
            // cannot read itself back is silent and costs that on every launch,
            // so the round trip is the property, not either half alone.
            with_cache_dir("roundtrip", |dir| {
                assert!(load_cached().is_none(), "nothing cached yet");

                write_cache(SAMPLE);
                assert!(dir.join("models_dev.json").is_file(), "the parent dir is created");

                let loaded = load_cached().expect("what was just written must load");
                let models = select(&loaded, "anthropic");
                assert_eq!(models.len(), 1);
                assert_eq!(models[0].value, "claude-x");
            });
        }

        #[test]
        fn a_damaged_cache_is_ignored_rather_than_believed() {
            // A half-written file (a crash mid-write, a full disk) must send us
            // back to the network, not surface as an empty model list — an
            // empty catalog reads to the caller as "this provider has no
            // models", which is a wrong answer rather than a missing one.
            with_cache_dir("damaged", |_| {
                write_cache(&SAMPLE[..SAMPLE.len() / 2]);
                assert!(load_cached().is_none(), "a truncated cache is not a catalog");

                write_cache("");
                assert!(load_cached().is_none(), "nor is an empty one");
            });
        }

        #[test]
        fn a_stale_cache_on_disk_is_what_triggers_a_refresh() {
            // `cache_is_stale` is the wiring between the rule and the configured
            // directory; with it stuck on false the catalog is fetched once and
            // never updated again, which is a model list that is wrong until
            // someone deletes a file by hand.
            with_cache_dir("stale", |dir| {
                assert!(cache_is_stale(), "no cache yet, so fetching is how we find out");

                write_cache(SAMPLE);
                assert!(!cache_is_stale(), "just written");

                let path = dir.join("models_dev.json");
                let long_ago = std::time::SystemTime::now() - Duration::from_secs(25 * 60 * 60);
                let file = std::fs::File::options().write(true).open(&path).unwrap();
                file.set_times(std::fs::FileTimes::new().set_modified(long_ago)).unwrap();
                assert!(cache_is_stale(), "a day-old cache is refreshed");
            });
        }

        #[test]
        fn a_host_that_named_no_cache_dir_writes_nothing_anywhere() {
            // `cache_path` returning None means fetch-only. Writing to some
            // default location instead would put a 4 MB file somewhere the host
            // never agreed to.
            let _guard = CACHE_ENV.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let restore = std::env::var_os("AGENT_HARNESS_CACHE_DIR");
            std::env::remove_var("AGENT_HARNESS_CACHE_DIR");

            assert!(cache_path().is_none());
            write_cache(SAMPLE); // must not panic, must not write
            assert!(load_cached().is_none());
            assert!(!cache_is_stale(), "nothing to refresh is not a stale cache");

            if let Some(value) = restore {
                std::env::set_var("AGENT_HARNESS_CACHE_DIR", value);
            }
        }

        #[test]
        fn a_cache_is_refetched_daily_and_a_missing_one_immediately() {
            // The catalog is ~2 MB, so refreshing it on every launch is the
            // thing this rule exists to prevent — but never refreshing means a
            // model list that is wrong until someone clears a file by hand.
            let dir = std::env::temp_dir().join(format!("hl-catalog-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("models_dev.json");

            assert!(stale(&path), "nothing cached yet, so fetching is how we find out");

            std::fs::write(&path, "{}").unwrap();
            assert!(!stale(&path), "just written is not a day old");

            // Six hours is the case that pins the interval to a *day*. A
            // just-written file and a 25-hour-old one read the same either side
            // of almost any threshold, so on their own they say only that some
            // rule exists: `24 * 60 * 60` could become `24 + 60 + 60` (144
            // seconds) and both would still pass. Six hours rather than one
            // because `24 + 60 * 60` is 3,624 seconds — just over an hour, and
            // an hour-old file cannot tell that from a day either.
            let earlier = std::time::SystemTime::now() - Duration::from_secs(6 * 60 * 60);
            let file = std::fs::File::options().write(true).open(&path).unwrap();
            file.set_times(std::fs::FileTimes::new().set_modified(earlier)).unwrap();
            assert!(!stale(&path), "six hours is not a day");

            // Backdate it past the threshold.
            let file = std::fs::File::options().write(true).open(&path).unwrap();
            let long_ago = std::time::SystemTime::now() - Duration::from_secs(25 * 60 * 60);
            file.set_times(std::fs::FileTimes::new().set_modified(long_ago)).unwrap();
            assert!(stale(&path), "a day-old catalog is refetched");

            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn no_cache_directory_means_nothing_to_refresh() {
            // The library does not pick a cache location: a host names one or
            // there is no disk cache at all. Inventing a path under $HOME is
            // exactly what this crate stopped doing for instruction files.
            assert!(cache_path().is_none() || std::env::var_os("AGENT_HARNESS_CACHE_DIR").is_some());
        }

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
            assert_eq!(a, vec![ModelChoice { value: "claude-x".into(), label: "Claude X".into() }]);

            // openai: no `name` → label falls back to the id.
            let o = select(&catalog, "openai");
            assert_eq!(o, vec![ModelChoice { value: "o9".into(), label: "o9".into() }]);

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

#[cfg(all(test, feature = "models-dev"))]
mod limit_tests {
    #[test]
    fn a_hosted_window_comes_from_the_catalog_or_is_absent() {
        // Network- and cache-dependent, so this asserts the shape rather than a
        // number: a known provider/model either yields a plausible window or
        // nothing (offline, no cache), and an unknown one always yields nothing.
        if let Some(window) = super::context_limit("openrouter", "openai/gpt-oss-120b") {
            assert!(window >= 8_192, "a real model's window should be sane, got {window}");
        }
        assert_eq!(super::context_limit("openrouter", "no-such-model"), None);
        assert_eq!(super::context_limit("no-such-provider", "openai/gpt-oss-120b"), None);
    }
}
