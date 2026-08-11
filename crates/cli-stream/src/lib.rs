//! Streaming subprocess control: spawn a child, read its **stdout** and
//! **stderr** line-by-line, write to its **stdin**, and cancel it
//! (SIGTERM → SIGKILL on unix, `TerminateProcess` elsewhere). No console
//! window is flashed on Windows.
//!
//! It knows no CLI's output format and no agent's protocol — it moves lines.
//!
//! ```no_run
//! use cli_stream::{spawn, ProcessEvent, Spawn};
//!
//! # fn main() -> Result<(), cli_stream::StreamError> {
//! // Read the events off a channel — stdout and stderr are always streamed.
//! let (handle, events) = spawn(Spawn::new("some-cli").args(["--verbose"]))?;
//! for event in events {
//!     match event {
//!         ProcessEvent::Stdout { line, .. } => println!("out: {line}"),
//!         ProcessEvent::Stderr { line, .. } => eprintln!("err: {line}"),
//!         _ => {}
//!     }
//! }
//! # let _ = handle;
//! # Ok(())
//! # }
//! ```
//!
//! Stdin is opt-in, because a child that inherits a terminal's stdin can block
//! forever waiting for input nobody is typing. Ask for it to answer a child
//! that asks something — the handle is `Clone`, so a reply can be written from
//! inside the reader:
//!
//! ```no_run
//! use cli_stream::{spawn_streaming, ProcessEvent, Spawn, Stdin};
//!
//! # fn main() -> Result<(), cli_stream::StreamError> {
//! let spawn = Spawn::new("some-server").args(["--stdio"]).stdin(Stdin::Piped);
//!
//! let replier = std::sync::Arc::new(std::sync::Mutex::new(None));
//! let slot = std::sync::Arc::clone(&replier);
//! let handle = spawn_streaming(spawn, move |event| {
//!     if let ProcessEvent::Stdout { line, .. } = event {
//!         if line.contains("ready") {
//!             if let Some(h) = slot.lock().unwrap().as_ref() {
//!                 let _: &cli_stream::ProcessHandle = h;
//!                 let _ = h.write_stdin_line(r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#);
//!             }
//!         }
//!     }
//! })?;
//! *replier.lock().unwrap() = Some(handle.clone());
//!
//! // …or just write to it directly, whenever you like.
//! handle.write_stdin_line("hello")?;
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
pub use process::{hidden_command, spawn, spawn_streaming, ProcessEvent, ProcessHandle, Spawn, Stdin};
