//! Translate ACP `session/update` notifications into the neutral
//! [`crate::RunEvent`] stream — the payoff of aligning `RunEvent`'s schema to
//! ACP's `SessionUpdate` (so this mapping is near 1:1 and lossless).
//!
//! Pure (no async, no I/O), so it's unit-tested without a live ACP agent.

use agent_client_protocol::schema as acp;

use crate::{PlanEntry, PlanEntryPriority, PlanEntryStatus, RunEvent, ToolKind, ToolLocation};

/// One ACP `SessionUpdate` → zero or more `RunEvent`s for `run_id`.
pub(crate) fn session_update_to_events(run_id: &str, update: acp::SessionUpdate) -> Vec<RunEvent> {
    let rid = || run_id.to_owned();
    match update {
        // Streamed assistant text / reasoning — the exact-match cases.
        acp::SessionUpdate::AgentMessageChunk(chunk) => match text_of(&chunk.content) {
            Some(delta) => vec![RunEvent::Text { run_id: rid(), delta }],
            None => vec![],
        },
        acp::SessionUpdate::AgentThoughtChunk(chunk) => match text_of(&chunk.content) {
            Some(delta) => vec![RunEvent::Thinking { run_id: rid(), delta }],
            None => vec![],
        },
        // A tool call is announced → ToolStart; its completion arrives as a
        // ToolCallUpdate with a terminal status → ToolEnd.
        acp::SessionUpdate::ToolCall(call) => vec![RunEvent::ToolStart {
            run_id: rid(),
            tool_call_id: call.tool_call_id.0.to_string(),
            title: call.title,
            tool_kind: map_kind(call.kind),
            locations: map_locations(&call.locations),
            raw_input: call.raw_input.as_ref().map(ToString::to_string),
        }],
        acp::SessionUpdate::ToolCallUpdate(update) => match update.fields.status {
            // Terminal status → ToolEnd carrying the result content + raw output
            // + the files it touched.
            Some(status @ (acp::ToolCallStatus::Completed | acp::ToolCallStatus::Failed)) => {
                vec![RunEvent::ToolEnd {
                    run_id: rid(),
                    tool_call_id: update.tool_call_id.0.to_string(),
                    ok: matches!(status, acp::ToolCallStatus::Completed),
                    content: tool_output(&update.fields),
                    raw_output: update.fields.raw_output.as_ref().map(ToString::to_string),
                    locations: update.fields.locations.as_deref().map(map_locations).unwrap_or_default(),
                }]
            }
            // pending / in_progress / no status change → no terminal event yet.
            _ => vec![],
        },
        acp::SessionUpdate::Plan(plan) => {
            let entries = plan
                .entries
                .into_iter()
                .map(|e| PlanEntry {
                    content: e.content,
                    status: map_plan_status(e.status),
                    priority: Some(map_plan_priority(e.priority)),
                })
                .collect();
            vec![RunEvent::Plan { run_id: rid(), entries }]
        }
        // user echo / available-commands / current-mode / usage / etc. have no
        // place in our stream (and `SessionUpdate` is #[non_exhaustive]).
        _ => vec![],
    }
}

/// Extract plain text from an ACP content block (only `text` blocks carry it).
fn text_of(block: &acp::ContentBlock) -> Option<String> {
    match block {
        acp::ContentBlock::Text(t) => Some(t.text.clone()),
        _ => None,
    }
}

/// Flatten an ACP tool call's result — its `content` blocks (text), else the
/// `raw_output` JSON — into `ToolEnd.output`, so the host can show what the tool
/// returned (the result/diff), not just that it finished.
fn tool_output(fields: &acp::ToolCallUpdateFields) -> Option<String> {
    if let Some(blocks) = &fields.content {
        let text = blocks.iter().filter_map(tool_content_text).collect::<Vec<_>>().join("\n");
        if !text.trim().is_empty() {
            return Some(text);
        }
    }
    fields.raw_output.as_ref().map(ToString::to_string)
}

/// Text from one tool-call content block (text content verbatim; a diff/other is
/// noted by kind).
fn tool_content_text(content: &acp::ToolCallContent) -> Option<String> {
    match content {
        acp::ToolCallContent::Content(c) => text_of(&c.content),
        acp::ToolCallContent::Diff(_) => Some("[diff]".to_owned()),
        _ => None,
    }
}

/// ACP `ToolCallLocation`s → our neutral [`ToolLocation`]s (the files the call
/// touches), so the host can show the subject + offer follow-along.
fn map_locations(locs: &[acp::ToolCallLocation]) -> Vec<ToolLocation> {
    locs.iter()
        .map(|l| ToolLocation { path: l.path.to_string_lossy().into_owned(), line: l.line })
        .collect()
}

/// ACP tool `kind` → our neutral [`ToolKind`] (we extended `ToolKind` to mirror
/// ACP's set for exactly this; `think`/`switch_mode`/unknown → `Other`).
fn map_kind(kind: acp::ToolKind) -> ToolKind {
    match kind {
        acp::ToolKind::Read => ToolKind::Read,
        acp::ToolKind::Edit => ToolKind::Edit,
        acp::ToolKind::Delete => ToolKind::Delete,
        acp::ToolKind::Move => ToolKind::Move,
        acp::ToolKind::Search => ToolKind::Search,
        acp::ToolKind::Execute => ToolKind::Execute,
        acp::ToolKind::Fetch => ToolKind::Fetch,
        _ => ToolKind::Other,
    }
}

fn map_plan_status(status: acp::PlanEntryStatus) -> PlanEntryStatus {
    match status {
        acp::PlanEntryStatus::Pending => PlanEntryStatus::Pending,
        acp::PlanEntryStatus::InProgress => PlanEntryStatus::InProgress,
        acp::PlanEntryStatus::Completed => PlanEntryStatus::Completed,
        _ => PlanEntryStatus::Pending,
    }
}

fn map_plan_priority(priority: acp::PlanEntryPriority) -> PlanEntryPriority {
    match priority {
        acp::PlanEntryPriority::High => PlanEntryPriority::High,
        acp::PlanEntryPriority::Medium => PlanEntryPriority::Medium,
        acp::PlanEntryPriority::Low => PlanEntryPriority::Low,
        _ => PlanEntryPriority::Medium,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(text: &str) -> acp::ContentChunk {
        acp::ContentChunk::new(acp::ContentBlock::from(text))
    }

    #[test]
    fn message_vs_thought_chunks_split_correctly() {
        let msg = session_update_to_events("r1", acp::SessionUpdate::AgentMessageChunk(chunk("hello")));
        assert!(matches!(msg.as_slice(), [RunEvent::Text { delta, .. }] if delta == "hello"));

        let thought = session_update_to_events("r1", acp::SessionUpdate::AgentThoughtChunk(chunk("hmm")));
        assert!(matches!(thought.as_slice(), [RunEvent::Thinking { delta, .. }] if delta == "hmm"));

        // The user-message echo has no place in our stream.
        let user = session_update_to_events("r1", acp::SessionUpdate::UserMessageChunk(chunk("u")));
        assert!(user.is_empty());
    }

    /// Build the update from its wire form rather than its Rust constructors.
    /// The input to this module is JSON an agent sent, so deserializing pins the
    /// field names too — a rename upstream should fail here, not in the field.
    fn update(payload: serde_json::Value) -> acp::SessionUpdate {
        serde_json::from_value(payload).expect("a valid session/update payload")
    }

    fn events(payload: serde_json::Value) -> Vec<RunEvent> {
        session_update_to_events("r1", update(payload))
    }

    #[test]
    fn an_announced_tool_call_carries_its_subject_and_kind() {
        // `locations` is what lets a host follow along in the file the agent is
        // working in; `kind` is what picks the icon and the read/write framing.
        let out = events(serde_json::json!({
            "sessionUpdate": "tool_call",
            "toolCallId": "c1",
            "title": "Read config",
            "kind": "read",
            "status": "pending",
            "locations": [{ "path": "/tmp/a.toml", "line": 12 }],
            "rawInput": { "path": "/tmp/a.toml" }
        }));

        let [RunEvent::ToolStart { tool_call_id, title, tool_kind, locations, raw_input, .. }] = out.as_slice() else {
            panic!("expected one ToolStart, got {out:?}");
        };
        assert_eq!(tool_call_id, "c1");
        assert_eq!(title, "Read config");
        assert_eq!(*tool_kind, ToolKind::Read);
        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0].path, "/tmp/a.toml");
        assert_eq!(locations[0].line, Some(12));
        assert!(raw_input.as_ref().is_some_and(|raw| raw.contains("a.toml")));
    }

    #[test]
    fn only_a_finished_tool_call_ends_it() {
        // An in-progress update is a status change, not a result. Emitting
        // ToolEnd for it would close the card while the tool is still running.
        for status in ["pending", "in_progress"] {
            let out = events(serde_json::json!({
                "sessionUpdate": "tool_call_update", "toolCallId": "c1", "status": status
            }));
            assert!(out.is_empty(), "{status} is not terminal, got {out:?}");
        }

        let done = events(serde_json::json!({
            "sessionUpdate": "tool_call_update",
            "toolCallId": "c1",
            "status": "completed",
            "content": [{ "type": "content", "content": { "type": "text", "text": "42 lines" } }]
        }));
        let [RunEvent::ToolEnd { ok, content, .. }] = done.as_slice() else {
            panic!("expected one ToolEnd, got {done:?}");
        };
        assert!(ok);
        assert_eq!(content.as_deref(), Some("42 lines"));

        let failed = events(serde_json::json!({
            "sessionUpdate": "tool_call_update", "toolCallId": "c1", "status": "failed"
        }));
        assert!(matches!(failed.as_slice(), [RunEvent::ToolEnd { ok: false, .. }]), "{failed:?}");
    }

    #[test]
    fn a_result_falls_back_to_raw_output_when_there_is_no_text() {
        // A tool whose result is structured rather than prose still has
        // something worth showing; an empty card is worse than raw JSON.
        let out = events(serde_json::json!({
            "sessionUpdate": "tool_call_update",
            "toolCallId": "c1",
            "status": "completed",
            "content": [],
            "rawOutput": { "count": 3 }
        }));
        let [RunEvent::ToolEnd { content, .. }] = out.as_slice() else { panic!("{out:?}") };
        assert!(content.as_ref().is_some_and(|c| c.contains("count")), "got {content:?}");
    }

    #[test]
    fn a_diff_result_is_named_rather_than_rendered() {
        let out = events(serde_json::json!({
            "sessionUpdate": "tool_call_update",
            "toolCallId": "c1",
            "status": "completed",
            "content": [{ "type": "diff", "path": "/a.txt", "oldText": "a", "newText": "b" }]
        }));
        let [RunEvent::ToolEnd { content, .. }] = out.as_slice() else { panic!("{out:?}") };
        assert_eq!(content.as_deref(), Some("[diff]"));
    }

    #[test]
    fn a_plan_keeps_the_order_status_and_priority_of_every_entry() {
        // The whole plan is replaced on each update, so dropping or reordering
        // an entry silently rewrites what the user is watching.
        let out = events(serde_json::json!({
            "sessionUpdate": "plan",
            "entries": [
                { "content": "first", "priority": "high", "status": "completed" },
                { "content": "second", "priority": "low", "status": "in_progress" },
                { "content": "third", "priority": "medium", "status": "pending" }
            ]
        }));
        let [RunEvent::Plan { entries, .. }] = out.as_slice() else { panic!("{out:?}") };
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].content, "first");
        assert_eq!(entries[0].status, PlanEntryStatus::Completed);
        assert_eq!(entries[0].priority, Some(PlanEntryPriority::High));
        assert_eq!(entries[1].status, PlanEntryStatus::InProgress);
        assert_eq!(entries[1].priority, Some(PlanEntryPriority::Low));
        assert_eq!(entries[2].status, PlanEntryStatus::Pending);
        assert_eq!(entries[2].priority, Some(PlanEntryPriority::Medium));
    }

    #[test]
    fn every_acp_tool_kind_maps_to_one_of_ours() {
        // `ToolKind` was extended to mirror ACP's set precisely so this is 1:1.
        // A kind quietly collapsing to Other is a card losing its meaning.
        for (wire, expected) in [
            ("read", ToolKind::Read),
            ("edit", ToolKind::Edit),
            ("delete", ToolKind::Delete),
            ("move", ToolKind::Move),
            ("search", ToolKind::Search),
            ("execute", ToolKind::Execute),
            ("fetch", ToolKind::Fetch),
            ("think", ToolKind::Other),
            ("other", ToolKind::Other),
        ] {
            let out = events(serde_json::json!({
                "sessionUpdate": "tool_call", "toolCallId": "c", "title": "t", "kind": wire
            }));
            let [RunEvent::ToolStart { tool_kind, .. }] = out.as_slice() else { panic!("{wire}: {out:?}") };
            assert_eq!(*tool_kind, expected, "kind {wire}");
        }
    }

    #[test]
    fn an_update_we_do_not_model_is_dropped_rather_than_guessed_at() {
        // `SessionUpdate` is #[non_exhaustive] and grows upstream. Anything new
        // must pass through silently instead of becoming a misleading event.
        assert!(events(serde_json::json!({
            "sessionUpdate": "available_commands_update", "availableCommands": []
        }))
        .is_empty());
    }
}
