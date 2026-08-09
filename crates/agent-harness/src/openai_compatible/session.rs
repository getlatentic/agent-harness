//! Session persistence — what makes a run *stateful* and `resume`-able.
//!
//! The model is OpenCode's (MIT): a session is a metadata **record** plus an
//! ordered **transcript**, addressed by id under a host-provided data root. We
//! diverge in granularity — one JSON file holds the whole transcript (the model
//! only needs the message list replayed; the UI consumes the live `RunEvent`
//! stream), rather than OpenCode's per-message/part files.
//!
//! Persistence is **opt-in**: a harness with no session dir runs ephemerally
//! (no disk writes); one configured via `OpenHarness::with_session_dir`
//! persists here and can resume by id. Layout under the root:
//! `sessions/<id>.json` (metadata, cheap to list) + `messages/<id>.json` (the
//! transcript — everything after the regenerated system prompt).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::wire::ChatMessage;

/// A persisted session's metadata — cheap to list without loading the
/// transcript, so a sessions view can render titles/models without reading
/// every message file. Forward-compatible: new fields carry `#[serde(default)]`
/// so an older on-disk record still deserializes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    pub id: String,
    /// A short display title (derived from the opening prompt; host-renamable).
    #[serde(default)]
    pub title: Option<String>,
    /// The model the session was started with.
    #[serde(default)]
    pub model: Option<String>,
    /// The working directory the session ran in (for project grouping/display).
    #[serde(default)]
    pub cwd: Option<String>,
    /// The parent session, set when this is a `task` subagent's child session
    /// (`None` for a top-level session). Lets a host show a session tree.
    #[serde(default)]
    pub parent_id: Option<String>,
    /// Epoch milliseconds at creation and at the last update.
    pub created_at: u64,
    pub updated_at: u64,
}

/// Current time in epoch milliseconds (saturates to 0 before the epoch, which
/// can't happen in practice).
pub(crate) fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Mint a new, time-ordered session id — `ses_<epoch_nanos>_<counter>`. Nanos
/// give ordering + practical uniqueness; the process-local counter breaks ties
/// within the same nanosecond. Dep-free (no `uuid`/`getrandom`).
pub(crate) fn new_session_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("ses_{nanos:x}_{n:x}")
}

/// A short session title from the opening prompt — its first non-empty line,
/// capped at 60 chars.
pub(crate) fn title_from_prompt(prompt: &str) -> String {
    let first = prompt.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
    let title: String = first.chars().take(60).collect();
    if title.is_empty() {
        "New session".to_owned()
    } else {
        title
    }
}

/// A JSON-file session store rooted at a host-provided directory. Used only
/// when the harness has a session dir configured.
#[derive(Debug, Clone)]
pub(crate) struct FileStore {
    root: PathBuf,
}

impl FileStore {
    pub(crate) fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn session_path(&self, id: &str) -> PathBuf {
        self.root.join("sessions").join(format!("{id}.json"))
    }
    fn messages_path(&self, id: &str) -> PathBuf {
        self.root.join("messages").join(format!("{id}.json"))
    }

    /// Write a session's metadata record (create or overwrite).
    pub(crate) fn put_record(&self, record: &SessionRecord) -> Result<(), String> {
        write_json(&self.session_path(&record.id), record)
    }

    /// Read a session's metadata, or `None` if it doesn't exist.
    pub(crate) fn get_record(&self, id: &str) -> Result<Option<SessionRecord>, String> {
        read_json_opt(&self.session_path(id))
    }

    /// All known sessions, newest-updated first. An unreadable record is
    /// skipped rather than failing the whole list (a half-written file
    /// shouldn't hide every other session).
    pub(crate) fn list_records(&self) -> Result<Vec<SessionRecord>, String> {
        let dir = self.root.join("sessions");
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(format!("listing sessions in {}: {e}", dir.display())),
        };
        let mut out: Vec<SessionRecord> = entries
            .flatten()
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("json"))
            .filter_map(|e| read_json_opt::<SessionRecord>(&e.path()).ok().flatten())
            .collect();
        out.sort_by_key(|r| std::cmp::Reverse(r.updated_at));
        Ok(out)
    }

    /// Bump a session's `updated_at`. No-op if the record is absent. Best-effort
    /// metadata, so callers ignore the result.
    pub(crate) fn touch(&self, id: &str, updated_at: u64) -> Result<(), String> {
        if let Some(mut rec) = self.get_record(id)? {
            rec.updated_at = updated_at;
            self.put_record(&rec)?;
        }
        Ok(())
    }

    /// Load a session's transcript (the messages after the system prompt), or an
    /// empty vec if it has none yet.
    pub(crate) fn load_messages(&self, id: &str) -> Result<Vec<ChatMessage>, String> {
        Ok(read_json_opt(&self.messages_path(id))?.unwrap_or_default())
    }

    /// Persist a session's transcript (overwrites — the loop hands over the full
    /// message list each turn).
    pub(crate) fn save_messages(&self, id: &str, messages: &[ChatMessage]) -> Result<(), String> {
        write_json(&self.messages_path(id), &messages)
    }
}

/// Write JSON atomically: a sibling temp file, then a rename.
///
/// A transcript is rewritten in full after every turn, and a plain truncating
/// write is only whole between the truncate and the last byte. A crash, a
/// SIGKILL or a full disk inside that window leaves a partial file — and since
/// the document has to parse as one value, the loss is the entire conversation
/// rather than the turn in flight. Rename is atomic on POSIX and on Windows for
/// a same-directory replace, so a reader sees the old file or the new one.
fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("creating {}: {e}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(value).map_err(|e| format!("serializing {}: {e}", path.display()))?;

    // Same directory, so the rename cannot cross a filesystem boundary. The pid
    // keeps two processes writing the same session from colliding on the temp.
    let temp = path.with_extension(format!("{}.tmp", std::process::id()));
    std::fs::write(&temp, json).map_err(|e| format!("writing {}: {e}", temp.display()))?;
    std::fs::rename(&temp, path).map_err(|e| {
        let _ = std::fs::remove_file(&temp);
        format!("replacing {}: {e}", path.display())
    })
}

fn read_json_opt<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Option<T>, String> {
    match std::fs::read_to_string(path) {
        Ok(s) => serde_json::from_str(&s).map(Some).map_err(|e| format!("parsing {}: {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("reading {}: {e}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hl-session-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn record_roundtrip_and_list_is_newest_first() {
        let dir = scratch("rec");
        let store = FileStore::new(&dir);
        assert!(store.get_record("nope").unwrap().is_none());
        assert!(store.list_records().unwrap().is_empty());

        let a = SessionRecord { id: "a".into(), title: Some("first".into()), model: Some("m".into()), cwd: None, parent_id: None, created_at: 100, updated_at: 100 };
        let b = SessionRecord { id: "b".into(), title: None, model: None, cwd: None, parent_id: None, created_at: 200, updated_at: 200 };
        store.put_record(&a).unwrap();
        store.put_record(&b).unwrap();
        assert_eq!(store.get_record("a").unwrap().unwrap().title.as_deref(), Some("first"));
        let ids: Vec<String> = store.list_records().unwrap().into_iter().map(|r| r.id).collect();
        assert_eq!(ids, ["b", "a"], "newest updated_at first");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn messages_roundtrip_and_touch() {
        let dir = scratch("msg");
        let store = FileStore::new(&dir);
        assert!(store.load_messages("s1").unwrap().is_empty());

        let msgs = vec![ChatMessage::user("hi"), ChatMessage::tool_result("c1", "done")];
        store.save_messages("s1", &msgs).unwrap();
        let loaded = store.load_messages("s1").unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].content.as_deref(), Some("hi"));

        // touch is a no-op without a record, then bumps updated_at once present.
        store.touch("s1", 999).unwrap();
        assert!(store.get_record("s1").unwrap().is_none());
        store.put_record(&SessionRecord { id: "s1".into(), title: None, model: None, cwd: None, parent_id: None, created_at: 1, updated_at: 1 }).unwrap();
        store.touch("s1", 999).unwrap();
        assert_eq!(store.get_record("s1").unwrap().unwrap().updated_at, 999);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn new_session_id_is_unique_and_prefixed() {
        let a = new_session_id();
        let b = new_session_id();
        assert_ne!(a, b);
        assert!(a.starts_with("ses_"));
    }

    #[test]
    fn title_is_first_line_capped() {
        assert_eq!(title_from_prompt("Fix the parser\nand tests"), "Fix the parser");
        assert_eq!(title_from_prompt("  \n  hello  "), "hello");
        assert_eq!(title_from_prompt(""), "New session");
        assert_eq!(title_from_prompt(&"x".repeat(100)).chars().count(), 60);
    }
}
