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
//! `sessions/<id>.jsonl` — one file per session: a `session` header line, then
//! one `message` line per turn, appended. Pi stores sessions this way; Codex
//! and Claude Code likewise keep a session to one JSONL file.
//!
//! Two properties come from that shape. Appending is O(1) in the length of the
//! conversation, where rewriting a whole-array file to record one exchange is
//! O(n) — and a whole-array file is only meaningful complete, so a process
//! killed mid-write lost the conversation rather than the turn. A torn append
//! costs the final line. And with the metadata in the same file as the
//! transcript, the two cannot disagree; as separate writes they could, if the
//! process died between them.
//!
//! Listing reads only each file's first line, so it never parses transcripts.
//! (Codex goes further with one global index — a single read for any number of
//! sessions — which is the better answer at large session counts.)
//!
//! The earlier two-file layout (`sessions/<id>.json` + `messages/<id>.{jsonl,json}`)
//! is still read, so sessions written by an older build resume.

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

    /// A session is one file: a header record, then one message per line.
    fn log_path(&self, id: &str) -> PathBuf {
        self.root.join("sessions").join(format!("{id}.jsonl"))
    }

    /// The two-file layout this replaced — a metadata `.json` beside a
    /// transcript under `messages/`. Read so existing sessions still resume.
    fn legacy_record_path(&self, id: &str) -> PathBuf {
        self.root.join("sessions").join(format!("{id}.json"))
    }
    fn legacy_messages_paths(&self, id: &str) -> [PathBuf; 2] {
        let dir = self.root.join("messages");
        [dir.join(format!("{id}.jsonl")), dir.join(format!("{id}.json"))]
    }

    /// Write a session's header. Creates the file, or rewrites the header in
    /// place when a host renames a session — rare enough that the rewrite does
    /// not matter, unlike the per-turn append.
    pub(crate) fn put_record(&self, record: &SessionRecord) -> Result<(), String> {
        let messages = self.load_messages(&record.id)?;
        self.write_log(&record.id, record, &messages)
    }

    /// Read a session's metadata: the header line, or the legacy file.
    pub(crate) fn get_record(&self, id: &str) -> Result<Option<SessionRecord>, String> {
        match read_header(&self.log_path(id)) {
            Some(record) => Ok(Some(record)),
            None => read_json_opt(&self.legacy_record_path(id)),
        }
    }

    /// All known sessions, newest-updated first. An unreadable header is
    /// skipped rather than failing the whole list — a half-written file
    /// shouldn't hide every other session.
    pub(crate) fn list_records(&self) -> Result<Vec<SessionRecord>, String> {
        let dir = self.root.join("sessions");
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(format!("listing sessions in {}: {e}", dir.display())),
        };
        let mut out: Vec<SessionRecord> = entries
            .flatten()
            .filter_map(|entry| {
                let path = entry.path();
                match path.extension().and_then(|x| x.to_str()) {
                    // Only the header is read, so listing does not parse
                    // transcripts however long they are.
                    Some("jsonl") => read_header(&path),
                    Some("json") => read_json_opt::<SessionRecord>(&path).ok().flatten(),
                    _ => None,
                }
            })
            .collect();
        out.sort_by_key(|r| std::cmp::Reverse(r.updated_at));
        Ok(out)
    }

    /// Bump a session's `updated_at`. No-op if the session is absent.
    pub(crate) fn touch(&self, id: &str, updated_at: u64) -> Result<(), String> {
        if let Some(mut record) = self.get_record(id)? {
            record.updated_at = updated_at;
            self.put_record(&record)?;
        }
        Ok(())
    }

    /// A session's transcript, from the header file or the legacy layout.
    ///
    /// A trailing partial line is dropped rather than failing the read: it is
    /// the signature of a process killed mid-append, and the turns before it
    /// are intact and worth keeping.
    pub(crate) fn load_messages(&self, id: &str) -> Result<Vec<ChatMessage>, String> {
        if let Some(text) = read_to_string_opt(&self.log_path(id))? {
            return Ok(text.lines().filter_map(|line| parse_line(line).message()).collect());
        }
        let [jsonl, json] = self.legacy_messages_paths(id);
        if let Some(text) = read_to_string_opt(&jsonl)? {
            return Ok(text.lines().filter_map(|line| serde_json::from_str(line).ok()).collect());
        }
        Ok(read_json_opt(&json)?.unwrap_or_default())
    }

    /// Append the messages added since the last save — O(1) in the length of
    /// the conversation, where rewriting the whole transcript to record one
    /// exchange was O(n), and where a kill mid-append costs the final line
    /// rather than a document that no longer parses.
    pub(crate) fn append_messages(&self, id: &str, messages: &[ChatMessage]) -> Result<(), String> {
        if messages.is_empty() {
            return Ok(());
        }
        let path = self.log_path(id);
        ensure_parent(&path)?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("opening {}: {e}", path.display()))?;
        std::io::Write::write_all(&mut file, encode_messages(messages)?.as_bytes())
            .map_err(|e| format!("appending to {}: {e}", path.display()))
    }

    /// Replace a session's transcript wholesale, keeping its header — for
    /// compaction, which replaces old turns with a summary rather than
    /// extending them.
    pub(crate) fn replace_messages(&self, id: &str, messages: &[ChatMessage]) -> Result<(), String> {
        let header = self.get_record(id)?;
        match header {
            Some(record) => self.write_log(id, &record, messages),
            // No header yet (a subagent's transcript is written before one
            // exists): the messages alone are still worth keeping.
            None => {
                let path = self.log_path(id);
                ensure_parent(&path)?;
                write_atomic(&path, &encode_messages(messages)?)
            }
        }
    }

    /// Rewrite a session file as header + messages.
    fn write_log(&self, id: &str, record: &SessionRecord, messages: &[ChatMessage]) -> Result<(), String> {
        let path = self.log_path(id);
        ensure_parent(&path)?;
        let header = serde_json::to_string(&LogLine::Session(record.clone()))
            .map_err(|e| format!("serializing the header for {}: {e}", path.display()))?;
        write_atomic(&path, &format!("{header}\n{}", encode_messages(messages)?))
    }
}

/// One line of a session file, tagged so the header and the messages can share
/// it. Pi's sessions are shaped this way; keeping metadata in the same file as
/// the transcript is what makes it impossible for the two to disagree after a
/// crash between two separate writes.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum LogLine {
    Session(SessionRecord),
    Message(ChatMessage),
}

impl LogLine {
    fn message(self) -> Option<ChatMessage> {
        match self {
            Self::Message(message) => Some(message),
            Self::Session(_) => None,
        }
    }
}

fn parse_line(line: &str) -> LogLine {
    serde_json::from_str(line).unwrap_or(LogLine::Session(SessionRecord {
        id: String::new(),
        title: None,
        model: None,
        cwd: None,
        parent_id: None,
        created_at: 0,
        updated_at: 0,
    }))
}

/// The header of a session file, without reading past the first line.
fn read_header(path: &Path) -> Option<SessionRecord> {
    let file = std::fs::File::open(path).ok()?;
    let mut first = String::new();
    std::io::BufRead::read_line(&mut std::io::BufReader::new(file), &mut first).ok()?;
    match serde_json::from_str(&first).ok()? {
        LogLine::Session(record) if !record.id.is_empty() => Some(record),
        _ => None,
    }
}

fn encode_messages(messages: &[ChatMessage]) -> Result<String, String> {
    let mut out = String::new();
    for message in messages {
        let line = serde_json::to_string(&LogLine::Message(message.clone()))
            .map_err(|e| format!("serializing a message: {e}"))?;
        out.push_str(&line);
        out.push('\n');
    }
    Ok(out)
}

fn ensure_parent(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("creating {}: {e}", parent.display()))?;
    }
    Ok(())
}

fn read_to_string_opt(path: &Path) -> Result<Option<String>, String> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("reading {}: {e}", path.display())),
    }
}

/// Replace a file's contents atomically: sibling temp, then rename.
fn write_atomic(path: &Path, contents: &str) -> Result<(), String> {
    let temp = path.with_extension(format!("{}.tmp", std::process::id()));
    std::fs::write(&temp, contents).map_err(|e| format!("writing {}: {e}", temp.display()))?;
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
    fn the_header_and_the_transcript_live_in_one_file() {
        // The reason for the layout: two files written separately can disagree
        // if the process dies between them. One file makes that impossible,
        // and listing still only reads the first line.
        let dir = scratch("onefile");
        let store = FileStore::new(&dir);
        let record = SessionRecord {
            id: "s1".to_owned(),
            title: Some("a chat".to_owned()),
            model: Some("m".to_owned()),
            cwd: None,
            parent_id: None,
            created_at: 1,
            updated_at: 2,
        };
        store.put_record(&record).unwrap();
        store.append_messages("s1", &[ChatMessage::user("hello")]).unwrap();

        assert!(dir.join("sessions").join("s1.jsonl").is_file());
        assert!(!dir.join("messages").exists(), "no second file to fall out of step");

        let listed = store.list_records().unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].title.as_deref(), Some("a chat"));
        assert_eq!(store.load_messages("s1").unwrap().len(), 1, "the header is not a message");
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

        let path = dir.join("sessions").join("s1.jsonl");
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
        // The two-file layout: a whole-array transcript under `messages/`.
        // Dropping it would silently lose every conversation a user had.
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
    fn an_unreadable_session_reads_as_an_error_and_not_as_an_absent_one() {
        // "Missing" and "unreadable" must not collapse into one answer. A
        // sessions directory that cannot be read presenting as "you have no
        // sessions" is how a conversation disappears with nothing failing.
        let dir = scratch("unreadable");
        let store = FileStore::new(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // A file where the sessions directory should be.
        std::fs::write(dir.join("sessions"), "not a directory").unwrap();
        assert!(store.list_records().is_err(), "an unlistable directory is not an empty one");
        std::fs::remove_file(dir.join("sessions")).unwrap();

        // A directory where a transcript should be.
        std::fs::create_dir_all(dir.join("sessions").join("s1.jsonl")).unwrap();
        assert!(store.load_messages("s1").is_err(), "an unreadable transcript is not an empty one");

        // A directory where a legacy record should be.
        std::fs::create_dir_all(dir.join("sessions").join("s2.json")).unwrap();
        assert!(store.get_record("s2").is_err(), "an unreadable record is not an absent one");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_record_written_by_an_older_build_still_loads_and_lists() {
        // The legacy layout kept metadata in its own `sessions/<id>.json`.
        // Reading only the transcript would resume the conversation while
        // losing its title and model, so the session comes back nameless.
        let dir = scratch("legacyrec");
        let store = FileStore::new(&dir);
        let path = dir.join("sessions").join("s1.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let record = SessionRecord {
            id: "s1".to_owned(),
            title: Some("an older chat".to_owned()),
            model: Some("m".to_owned()),
            cwd: None,
            parent_id: None,
            created_at: 5,
            updated_at: 5,
        };
        std::fs::write(&path, serde_json::to_string(&record).unwrap()).unwrap();

        assert_eq!(store.get_record("s1").unwrap().unwrap().title.as_deref(), Some("an older chat"));
        let listed = store.list_records().unwrap();
        assert_eq!(listed.len(), 1, "a legacy record lists alongside current ones");
        assert_eq!(listed[0].id, "s1");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_header_without_an_id_is_not_a_session() {
        // A first line that parses does not make it a header: an id is what a
        // session is addressed by, so listing one puts an entry in a sessions
        // view that cannot be opened.
        let dir = scratch("noid");
        let store = FileStore::new(&dir);
        let path = dir.join("sessions").join("s1.jsonl");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "{\"type\":\"session\",\"id\":\"\",\"created_at\":0,\"updated_at\":0}\n").unwrap();

        assert!(store.get_record("s1").unwrap().is_none());
        assert!(store.list_records().unwrap().is_empty(), "an id-less header is skipped, not listed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_timestamps_come_from_a_real_clock() {
        // `updated_at` is what orders a sessions list. A constant clock leaves
        // every session equal and the order arbitrary, which a round trip
        // through the store cannot notice.
        let now = now_millis();
        assert!(now > 1_700_000_000_000, "epoch milliseconds, not seconds and not a constant: {now}");
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
