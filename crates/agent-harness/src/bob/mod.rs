//! `bob` CLI as a [`Harness`].
//!
//! The bob adapter: wraps the standalone [`bob_rs`] SDK (detection,
//! install, keychain, spawn) behind the neutral [`crate::Harness`]
//! trait, and parses bob's `--output-format stream-json` stdout into the
//! shared [`crate::RunEvent`] vocabulary via the [`parser`] module.
//!
//! Auth: Compose stores bob's API key (in the OS keychain via `bob_rs`),
//! so `credential().required` is `true` and `supports_login` is `false` —
//! unlike the Claude/Codex adapters, which own their CLI's login.

use std::sync::{Arc, Mutex};

use bob_rs::{
    get_readiness, install_bob, spawn_bob, BobApprovalMode, BobChatMode, RunBobOptions,
    KEYCHAIN_ACCOUNT, KEYCHAIN_SERVICE,
};
use crate::{
    normalize_process_event, CredentialSpec, Harness, HarnessCapabilities, HarnessError,
    HarnessInfo, HarnessReadiness, InstallCallback, RunCallback, RunHandle, RunMode, RunRequest,
};

pub mod parser;

pub use parser::{bob_tool_kind, normalize_bob_event, parse_bob_line, BobStreamParser};

/// Registry id for the bob harness.
pub const BOB_HARNESS_ID: &str = "bob";

/// `bob` CLI as a [`Harness`]. Delegates to the [`bob_rs`] SDK;
/// this is just the neutral face over it.
#[derive(Debug, Default, Clone)]
pub struct BobHarness;

impl BobHarness {
    pub fn new() -> Self {
        Self
    }
}

impl Harness for BobHarness {
    fn info(&self) -> HarnessInfo {
        HarnessInfo {
            id: BOB_HARNESS_ID.to_owned(),
            display_name: "Bob".to_owned(),
            description: "IBM's bob agent CLI. Runs locally via Node.js.".to_owned(),
            requires_install: true,
            capabilities: HarnessCapabilities {
                // Compose stores bob's API key. bob runs edit-capable in
                // `auto_edit` (see `run`), so it writes files directly like
                // Claude/Codex — the host reviews via its edit gate, not an
                // in-stream preview. It exposes no model/effort/turn-cap knobs.
                credential_required: true,
                previews_edits: false,
                models: Vec::new(),
                allows_custom_model: false,
                supports_effort: false,
                supports_max_turns: false,
                supports_login: false,
                supports_custom_instructions: false,
            },
        }
    }

    fn readiness(&self) -> HarnessReadiness {
        let snapshot = get_readiness();
        // Preserve the rich bob probe for the UI while presenting a
        // neutral top-level shape. Serialization can't realistically
        // fail for this owned struct; fall back to null if it does.
        let details = serde_json::to_value(&snapshot).unwrap_or(serde_json::Value::Null);
        HarnessReadiness {
            harness_id: BOB_HARNESS_ID.to_owned(),
            // Tool readiness only; the host owns the API key.
            ready: snapshot.bob.installed && snapshot.node.satisfies_min,
            installed: snapshot.bob.installed,
            version: snapshot.bob.version.clone(),
            auth_configured: snapshot.auth.configured,
            error: snapshot.bob.error.clone(),
            details,
        }
    }

    fn install(&self, on_event: InstallCallback) -> Result<(), HarnessError> {
        // The closure captures only the `Arc` (Clone + Send + Sync +
        // 'static), so it satisfies `install_bob`'s `F: FnMut + Send
        // + Sync + Clone + 'static` bound. bob-rs reports failures as a typed
        // `BobError`; carry it as the install error's source.
        install_bob(move |event| (*on_event)(event)).map_err(HarnessError::install)
    }

    fn run(&self, request: RunRequest, on_event: RunCallback) -> Result<RunHandle, HarnessError> {
        let opts = RunBobOptions {
            prompt: request.prompt,
            chat_mode: match request.mode {
                RunMode::Ask => BobChatMode::Ask,
                // "Edit" maps onto bob's code mode — the one that
                // proposes file changes.
                RunMode::Edit => BobChatMode::Code,
            },
            // Edit mode auto-approves edit tools (bob's `auto_edit`, the
            // analogue of Claude's `acceptEdits`) so the agent writes directly
            // and the host reviews via its edit gate. Ask is read-only, so
            // `default` is fine. A host wanting full bypass passes
            // `--approval-mode yolo` through `extra_args` (last-wins).
            approval_mode: match request.mode {
                RunMode::Edit => BobApprovalMode::AutoEdit,
                RunMode::Ask => BobApprovalMode::Default,
            },
            max_coins: 30,
            cwd: request.cwd,
            bob_executable: request.tuning.binary_path.clone(),
            // Host passthrough — the same RunTuning.extra_args the claude/codex
            // adapters honor, so a client applies a flag uniformly across all
            // three harnesses (bob appends it to its own argv).
            extra_args: request.tuning.extra_args,
            // Continue a prior session instead of replaying history — the same
            // RunRequest.resume the claude/codex adapters honor (`bob -r <id>`).
            resume: request.resume,
        };
        // bob emits its own process events (lifecycle + raw stream-json
        // stdout lines). Normalize each into zero or more harness-neutral
        // `RunEvent`s here, so the consumer only ever sees the normalized
        // shape — the keystone of the abstraction. bob streams its
        // reasoning inline as `<thinking>…</thinking>` and its answer via
        // the `attempt_completion` tool, across many lines — so parsing is
        // stateful. Hold one `BobStreamParser` for the whole run; the
        // stdout reader thread drives it sequentially, the `Mutex` just
        // satisfies the `Fn + Send + Sync` callback bound.
        // In Edit mode bob runs `auto_edit` (above), so it *applies* edits
        // rather than proposing them. Drop the parser's SuggestedEdits in that
        // mode: an applied write is not a proposal, so it must not surface as
        // an accept/reject suggestion (that would double with the host's edit
        // gate / diff). The same `write_to_file`'s ToolStart/ToolEnd (Write) is
        // retained, so it still shows as a file-op — consistent with how
        // Claude/Codex applied writes appear. (Ask mode is read-only — no edits
        // anyway — so the flag only ever matters in Edit mode.)
        let edits_are_applied = matches!(request.mode, RunMode::Edit);
        let parser = Arc::new(Mutex::new(BobStreamParser::default()));
        let handle = spawn_bob(opts, request.run_id, move |event| {
            // Recover a poisoned lock rather than panic on the reader thread —
            // parsing is total, so the held parser is never mid-corruption.
            let mut parser = parser.lock().unwrap_or_else(|p| p.into_inner());
            for normalized in normalize_process_event(event, |line| {
                let mut parsed = parser.parse_line(line);
                if edits_are_applied {
                    parsed.edits.clear();
                }
                parsed
            }) {
                (*on_event)(normalized);
            }
        })
        .map_err(HarnessError::spawn)?;
        Ok(Box::new(handle))
    }

    fn credential(&self) -> CredentialSpec {
        CredentialSpec {
            label: "Bob API key".to_owned(),
            keychain_service: KEYCHAIN_SERVICE.to_owned(),
            keychain_account: KEYCHAIN_ACCOUNT.to_owned(),
            required: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bob_info_requires_install() {
        let info = BobHarness::new().info();
        assert_eq!(info.id, BOB_HARNESS_ID);
        assert!(info.requires_install);
    }

    #[test]
    fn bob_credential_points_at_the_shared_keychain_slot() {
        let cred = BobHarness::new().credential();
        assert_eq!(cred.keychain_service, KEYCHAIN_SERVICE);
        assert_eq!(cred.keychain_account, KEYCHAIN_ACCOUNT);
        assert!(cred.required);
        // `credential_required` capability must agree with the spec — the
        // frontend gates its preflight on the capability, so they can't drift.
        assert_eq!(
            BobHarness::new().info().capabilities.credential_required,
            cred.required
        );
    }

    #[test]
    fn bob_default_login_is_unsupported() {
        // bob authenticates via its stored API key, not an interactive
        // CLI sign-in, so the default `login` stays unsupported.
        let cb: InstallCallback = Arc::new(|_| {});
        assert!(BobHarness::new().login(cb).is_err());
        assert!(!BobHarness::new().info().capabilities.supports_login);
    }
}
