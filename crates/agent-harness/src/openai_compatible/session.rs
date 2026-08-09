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
//! `sessions/<id>.json` (metadata, cheap to list) + `messages/<id>.jsonl` (the
//! transcript — everything after the regenerated system prompt, one message per
//! line, appended).
//!
//! JSONL for the transcript, matching Codex and Claude Code. A whole-array file
//! has to be rewritten to record one exchange, which is O(n) per turn and, more
//! importantly, only valid as a complete document: a process killed mid-write
//! left JSON that no longer parsed, losing the conversation rather than the
//! turn. A torn append costs the final line. `<id>.json` is still read so
//! sessions written by an older build resume.

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
    /// The transcript log — one JSON message per line, appended.
    fn messages_path(&self, id: &str) -> PathBuf {
        self.root.join("messages").join(format!("{id}.jsonl"))
    }

    /// The pre-JSONL whole-array file, still read so existing sessions resume.
    fn legacy_messages_path(&self, id: &str) -> PathBuf {
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
    /// A session's transcript. Reads the JSONL log, falling back to the
    /// pre-JSONL whole-array file so a session written by an older build still
    /// resumes.
    ///
    /// A trailing partial line is dropped rather than failing the read: it is
    /// the signature of a process killed mid-append, and the turns before it
    /// are intact and worth keeping.
    pub(crate) fn load_messages(&self, id: &str) -> Result<Vec<ChatMessage>, String> {
        let path = self.messages_path(id);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(read_json_opt(&self.legacy_messages_path(id))?.unwrap_or_default())
            }
            Err(e) => return Err(format!("reading {}: {e}", path.display())),
        };
        Ok(text.lines().filter_map(|line| serde_json::from_str(line).ok()).collect())
    }

    /// Append the messages added since the last save.
    ///
    /// Append rather than rewrite, which is how Codex and Claude Code store
    /// theirs: a full rewrite costs O(n) per turn and re-serialises the whole
    /// conversation to record one exchange. It is also the safer failure —
    /// a kill mid-append truncates the last line, where a kill mid-rewrite
    /// used to leave a JSON document that no longer parsed at all.
    pub(crate) fn append_messages(&self, id: &str, messages: &[ChatMessage]) -> Result<(), String> {
        if messages.is_empty() {
            return Ok(());
        }
        let path = self.messages_path(id);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("creating {}: {e}", parent.display()))?;
        }
        let mut encoded = String::new();
        for message in messages {
            let line = serde_json::to_string(message)
                .map_err(|e| format!("serializing a message for {}: {e}", path.display()))?;
            encoded.push_str(&line);
            encoded.push('\n');
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("opening {}: {e}", path.display()))?;
        std::io::Write::write_all(&mut file, encoded.as_bytes())
            .map_err(|e| format!("appending to {}: {e}", path.display()))
    }

    /// Replace a session's transcript wholesale — for compaction, which
    /// rewrites history rather than extending it.
    pub(crate) fn replace_messages(&self, id: &str, messages: &[ChatMessage]) -> Result<(), String> {
        let path = self.messages_path(id);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("creating {}: {e}", parent.display()))?;
        }
        let mut encoded = String::new();
        for message in messages {
            let line = serde_json::to_string(message)
                .map_err(|e| format!("serializing a message for {}: {e}", path.display()))?;
            encoded.push_str(&line);
            encoded.push('\n');
        }
        let temp = path.with_extension(format!("{}.tmp", std::process::id()));
        std::fs::write(&temp, encoded).map_err(|e| format!("writing {}: {e}", temp.display()))?;
        std::fs::rename(&temp, &path).map_err(|e| {
            let _ = std::fs::remove_file(&temp);
            format!("replacing {}: {e}", path.display())
        })
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
    fn a_truncated_last_line_costs_only_that_turn() {
        // The failure a full rewrite could not survive: a process killed
        // mid-write. Appending makes the damage the final line, and the turns
        // before it still load.
        let dir = scratch("torn");
        let store = FileStore::new(&dir);
        store
            .append_messages("s1", &[ChatMessage::user("first"), ChatMessage::user("second")])
            .unwrap();

        let path = dir.join("messages").join("s1.jsonl");
        let mut raw = std::fs::read_to_string(&path).unwrap();
        raw.push_str("{\"role\":\"user\",\"cont"); // killed mid-append
        std::fs::write(&path, raw).unwrap();

        let loaded = store.load_messages("s1").unwrap();
        assert_eq!(loaded.len(), 2, "the intact turns survive a torn tail");
        assert_eq!(loaded[0].content.as_deref(), Some("first"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_session_written_by_an_older_build_still_loads() {
        // Pre-JSONL sessions are a whole-array `.json`. Dropping them would
        // silently lose every conversation a user already had.
        let dir = scratch("legacy");
        let store = FileStore::new(&dir);
        let legacy = dir.join("messages").join("s1.json");
        std::fs::create_dir_all(legacy.parent().unwrap()).unwrap();
        std::fs::write(&legacy, serde_json::to_string(&vec![ChatMessage::user("from before")]).unwrap())
            .unwrap();

        let loaded = store.load_messages("s1").unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content.as_deref(), Some("from before"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn appending_extends_rather_than_replacing() {
        let dir = scratch("append");
        let store = FileStore::new(&dir);
        store.append_messages("s1", &[ChatMessage::user("one")]).unwrap();
        store.append_messages("s1", &[ChatMessage::user("two")]).unwrap();
        assert_eq!(store.load_messages("s1").unwrap().len(), 2);

        // Compaction is the one caller that must shorten the log.
        store.replace_messages("s1", &[ChatMessage::user("summary")]).unwrap();
        let loaded = store.load_messages("s1").unwrap();
        assert_eq!(loaded.len(), 1, "replace truncates, it does not extend");
        assert_eq!(loaded[0].content.as_deref(), Some("summary"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn messages_roundtrip_and_touch() {
        let dir = scratch("msg");
        let store = FileStore::new(&dir);
        assert!(store.load_messages("s1").unwrap().is_empty());

        let msgs = vec![ChatMessage::user("hi"), ChatMessage::tool_result("c1", "done")];
        store.append_messages("s1", &msgs).unwrap();
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
