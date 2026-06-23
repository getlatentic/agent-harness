//! Generic streaming subprocess engine.
//!
//! Spawn a child CLI, stream its stdout/stderr line-by-line through a
//! callback as [`ProcessEvent`]s, cancel it (SIGTERM → SIGKILL), and
//! augment `PATH` so Node-based CLIs resolve even from a Finder-launched
//! `.app`. No agent / harness *protocol* knowledge — it parses no CLI's
//! output and knows no agent's wire format. The one node-specific concession
//! is the best-effort PATH resolver (`augmented_node_path`): every consumer in
//! this family drives a Node-based CLI, and as the shared leaf this is the one
//! place `bob-rs` and `agent-harness` can both reuse it without a cycle.
//! Otherwise it's purely subprocess streaming, useful to anyone driving a CLI.
//!
//! [`InstallEvent`] is the sibling shape for streamed install/login output.
//!
//! This is a deliberate *leaf* crate: both `bob-rs` (the bob SDK) and
//! `agent-harness` (the framework) depend on it, which is what lets
//! `bob-rs` stay standalone without a dependency cycle.

pub mod error;
pub mod install;
pub mod process;

pub use error::StreamError;
pub use install::InstallEvent;
pub use process::{
    augmented_node_path, resolve_program, spawn_streaming, ProcessEvent, ProcessHandle,
};
