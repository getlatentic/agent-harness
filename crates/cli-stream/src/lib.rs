//! Streaming subprocess control: spawn a child, read its **stdout** and
//! **stderr** line-by-line, write to its **stdin**, and cancel it
//! (SIGTERM → SIGKILL on unix, `TerminateProcess` elsewhere). No console
//! window is flashed on Windows.
//!
//! It knows no CLI's output format and no agent's protocol — it moves lines.
//!
//! ```no_run
//! use cli_stream::{spawn_streaming, ProcessEvent, Spawn, Stdin};
//!
//! # fn main() -> Result<(), cli_stream::StreamError> {
//! // stdout and stderr are always streamed; stdin is opt-in.
//! let spawn = Spawn::new("some-server", ".", "run-1")
//!     .args(vec!["--stdio".to_owned()])
//!     .env(vec![("RUST_LOG".to_owned(), "info".to_owned())])
//!     .stdin(Stdin::Piped);
//!
//! let handle = spawn_streaming(spawn, |event| match event {
//!     ProcessEvent::Stdout { line, .. } => println!("out: {line}"),
//!     ProcessEvent::Stderr { line, .. } => eprintln!("err: {line}"),
//!     ProcessEvent::Exited { exit_code, .. } => eprintln!("exit {exit_code:?}"),
//!     _ => {}
//! })?;
//!
//! // …and talk back to it.
//! handle.write_stdin_line(r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#)?;
//! handle.cancel()?;
//! # Ok(())
//! # }
//! ```
//!
//! [`InstallEvent`] is the sibling shape for streamed install/login output.
//!
//! A deliberate *leaf*: it depends on nothing of ours, so anything driving a
//! CLI can use it. Finding a CLI a user installed — resolving a bare name,
//! locating the `node` it was installed beside — is a different question, and
//! lives with the caller that needs it (`agent_harness::node_cli`).

pub mod error;
pub mod install;
pub mod process;

pub use error::StreamError;
pub use install::InstallEvent;
pub use process::{hidden_command, spawn_streaming, ProcessEvent, ProcessHandle, Spawn, Stdin};
