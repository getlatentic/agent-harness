//! The harness registry — an **open** builder so consumers compose their
//! own set of harnesses (the built-ins *and/or* their own custom
//! `impl Harness`), plus convenience constructors over the built-in
//! adapters for hosts that just want "all of them".
//!
//! This is the extensibility seam: a third party adds a provider by
//! implementing [`Harness`] in their own crate and calling
//! [`Registry::register`] — no fork of this crate required.

use serde::Serialize;

use crate::{Features, Harness, Info, Readiness};
#[cfg(feature = "claude")]
use crate::Claude;
#[cfg(feature = "codex")]
use crate::Codex;

/// One row of a picker: who a harness is, and what it can do.
///
/// These are separate questions to the [`Harness`] trait — identity is asked
/// once, capability constantly — but a UI needs both at the same moment, and
/// this is the shape that crosses to it. Serializable and `camelCase`, so a
/// host hands it to a frontend unchanged.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Listing {
    pub manifest: Info,
    pub capabilities: Features,
}

/// The identifier used when the caller doesn't pick one. (A literal so it's
/// available even in builds compiled without the `claude` feature; hosts
/// override as needed.)
pub const DEFAULT_HARNESS_ID: &str = "claude";

/// An open set of harnesses. Build it with the ones you want — the
/// built-ins (`Claude`/`Codex`) and/or your own:
///
/// ```no_run
/// use harness::Registry;
/// let reg = Registry::new()
///     .register(harness::Claude::new());
///     // .register(MyCustomHarness::new())   // your own impl Harness
/// assert!(reg.by_id("claude").is_some());
/// ```
#[derive(Default)]
pub struct Registry {
    harnesses: Vec<Box<dyn Harness>>,
}

impl Registry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a harness. Chainable. Registration order is preserved (it's the
    /// UI display order; the first registered is the conventional default).
    pub fn register(mut self, harness: impl Harness + 'static) -> Self {
        self.harnesses.push(Box::new(harness));
        self
    }

    /// Add an already-boxed harness — like [`register`](Registry::register) but
    /// for a `Box<dyn Harness>` a host built behind the trait object (e.g. its
    /// configured providers). Chainable.
    pub fn register_boxed(mut self, harness: Box<dyn Harness>) -> Self {
        self.harnesses.push(harness);
        self
    }

    /// Resolve a harness by its [`Info::id`].
    pub fn by_id(&self, id: &str) -> Option<&dyn Harness> {
        self.harnesses
            .iter()
            .map(Box::as_ref)
            .find(|h| h.info().id == id)
    }

    /// Resolve a harness by id, taking ownership of its box out of the registry —
    /// for a host that needs an owned `Box<dyn Harness>` to hold across a run,
    /// rather than the borrow [`by_id`](Registry::by_id) returns.
    pub fn into_by_id(self, id: &str) -> Option<Box<dyn Harness>> {
        self.harnesses.into_iter().find(|h| h.info().id == id)
    }

    /// Probe readiness of every registered harness, in registration order — the
    /// "what's actually on this machine" discovery a picker renders. Each probe
    /// may shell out; treat as blocking and run it off the UI thread.
    pub fn discover(&self) -> Vec<Readiness> {
        self.harnesses.iter().map(|h| h.readiness()).collect()
    }

    /// Every registered harness, in registration order, as a picker renders it:
    /// who it is and what it supports.
    pub fn catalog(&self) -> Vec<Listing> {
        self.harnesses
            .iter()
            .map(|h| Listing { manifest: h.info(), capabilities: h.features() })
            .collect()
    }

    /// The ids of every registered harness, in registration order.
    pub fn ids(&self) -> Vec<String> {
        self.harnesses.iter().map(|h| h.info().id).collect()
    }
}

/// A [`Registry`] of the built-in adapters compiled into this build
/// (claude / codex), in display order.
pub fn default_registry() -> Registry {
    #[allow(unused_mut)]
    let mut reg = Registry::new();
    #[cfg(feature = "claude")]
    {
        reg = reg.register(Claude::new());
    }
    #[cfg(feature = "codex")]
    {
        reg = reg.register(Codex::new());
    }
    reg
}

/// Resolve a *built-in* harness by id, as an owned box — convenience for
/// hosts that look one up per call. Returns `None` for an unknown id.
pub fn harness_by_id(id: &str) -> Option<Box<dyn Harness>> {
    let _ = id;
    #[cfg(feature = "claude")]
    {
        if id == crate::CLAUDE_HARNESS_ID {
            return Some(Box::new(Claude::new()));
        }
    }
    #[cfg(feature = "codex")]
    {
        if id == crate::CODEX_HARNESS_ID {
            return Some(Box::new(Codex::new()));
        }
    }
    None
}

/// Metadata for every built-in harness — the payload the UI picker renders.
pub fn harness_catalog() -> Vec<Listing> {
    default_registry().catalog()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CredentialSpec, Features, Readiness, RunCallback,
        RunHandle, RunRequest,
    };

    #[test]
    fn default_registry_lists_claude_codex_in_order() {
        assert_eq!(default_registry().ids(), vec!["claude", "codex"]);
        assert_eq!(default_registry().catalog()[0].manifest.id, DEFAULT_HARNESS_ID);
    }

    #[test]
    fn harness_by_id_resolves_builtins_and_rejects_unknown() {
        assert!(harness_by_id("claude").is_some());
        assert!(harness_by_id("codex").is_some());
        assert!(harness_by_id("nope").is_none());
    }

    /// The flag and the behaviour have to agree, or the flag is worse than
    /// nothing: a host reads `withheld_tools: false` and picks another adapter,
    /// or reads `true` and trusts a sandbox. An adapter that advertises it
    /// cannot withhold tools must actually refuse to try.
    /// One name per guarantee the crate makes — the contract on
    /// [`crate::Harness`] says a guarantee ships as a [`crate::Features`] flag
    /// *and* enforcement here, so this list and the assertions below have to
    /// move together.
    ///
    /// `Features` is destructured exhaustively on purpose, with no `..`: a new
    /// flag fails to compile until someone says whether it is a guarantee and,
    /// if so, adds the arm that holds adapters to it. The question gets asked
    /// at the only moment anyone will think about it.
    fn guarantees() -> Vec<&'static str> {
        let crate::Features {
            credential_required: _,
            previews_edits: _,
            models: _,
            custom_model: _,
            effort: _,
            max_turns: _,
            withheld_tools: _,
            login: _,
            custom_instructions: _,
        } = crate::Features::default();
        vec!["withheld_tools"]
    }

    #[test]
    fn an_adapter_that_cannot_withhold_tools_refuses_to_pretend() {
        for name in guarantees() {
            assert_eq!(name, "withheld_tools", "a new guarantee needs its own arm below");
        }
        for listing in default_registry().catalog() {
            let id = &listing.manifest.id;
            if listing.capabilities.withheld_tools {
                continue;
            }
            let harness = harness_by_id(id).expect("catalogued");
            let refused = harness
                .start(
                    crate::RunRequest {
                        tools: crate::ToolAccess::None,
                        ..crate::RunRequest::default()
                    },
                    std::sync::Arc::new(|_| {}),
                )
                .is_err();
            assert!(refused, "{id} advertises no withholding but accepted the request anyway");
        }
    }

    #[test]
    fn capabilities_match_each_adapter_and_back_credential_required() {
        let caps = |id: &str| harness_by_id(id).unwrap().features();

        let claude = caps("claude");
        assert!(!claude.credential_required && !claude.previews_edits);
        assert!(!claude.models.is_empty() && !claude.custom_model);
        assert!(claude.max_turns && !claude.effort);

        let codex = caps("codex");
        assert!(!codex.credential_required && !codex.previews_edits);
        assert!(codex.custom_model && codex.effort && !codex.max_turns);

        assert!(claude.login && codex.login);
        // Both adapters map `extra_instructions` onto their CLI; the picker
        // shows the field only where this is true.
        assert!(claude.custom_instructions && codex.custom_instructions);
    }

    // A third-party / custom provider — proves the registry is open: this
    // type lives "outside" the built-ins yet registers + resolves the same.
    struct Acme;
    impl Harness for Acme {
        fn info(&self) -> Info {
            Info {
                id: "acme".to_owned(),
                display_name: "Acme".to_owned(),
                description: "A custom third-party harness.".to_owned(),
                install_hint: None,
            }
        }

        fn features(&self) -> Features {
            Features { custom_model: true, ..Default::default() }
        }
        fn readiness(&self) -> Readiness {
            Readiness {
                harness_id: "acme".to_owned(),
                ready: true,
                installed: true,
                version: None,
                auth_configured: true,
                error: None,
                details: serde_json::Value::Null,
            }
        }
        fn start(
            &self,
            _req: RunRequest,
            _on_event: RunCallback,
        ) -> Result<RunHandle, crate::Error> {
            // A real API-backed harness would call its HTTP endpoint here and
            // emit RunEvents through `on_event`; the dummy never runs.
            Err(crate::Error::Other(
                "acme: run not implemented in test".to_owned(),
            ))
        }
        fn credential(&self) -> CredentialSpec {
            CredentialSpec {
                label: "Acme key".to_owned(),
                keychain_service: "acme".to_owned(),
                keychain_account: "ACME_API_KEY".to_owned(),
                required: false,
            }
        }
    }

    #[test]
    fn custom_harness_registers_and_resolves_alongside_builtins() {
        let reg = Registry::new().register(Claude::new()).register(Acme);
        assert!(reg.by_id("claude").is_some());
        assert!(reg.by_id("acme").is_some(), "custom harness must resolve");
        assert_eq!(reg.ids(), vec!["claude", "acme"]);
    }

    #[test]
    fn register_boxed_then_into_by_id_returns_an_owned_box() {
        let reg = Registry::new().register_boxed(Box::new(Acme));
        assert_eq!(reg.ids(), vec!["acme"]);
        let owned: Option<Box<dyn Harness>> = reg.into_by_id("acme");
        assert!(owned.is_some(), "into_by_id must hand back the owned box");
    }

    #[test]
    fn discover_probes_readiness_of_every_registered_harness() {
        let readiness = Registry::new().register_boxed(Box::new(Acme)).discover();
        assert_eq!(readiness.len(), 1);
        assert_eq!(readiness[0].harness_id, "acme");
        assert!(readiness[0].ready);
    }
}
