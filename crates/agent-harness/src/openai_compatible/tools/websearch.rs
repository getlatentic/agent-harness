//! `websearch` — real-time web search. Ported from OpenCode's design (MIT): it
//! speaks MCP `tools/call` to a hosted search endpoint (Exa or Parallel),
//! selected by which API key is set in the environment. (OpenCode also has a
//! free "opencode gateway" path that we don't have, so here a key is required.)
//!
//! Offered only when a key is configured — otherwise the tool isn't shown to the
//! model at all. Read-only. We discover the endpoint's search tool via
//! `tools/list` rather than hard-coding a per-provider tool name.

use std::time::Duration;

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::{RunMode, ToolKind};

use super::{parse_args, schema_for, Tool, ToolCtx, ToolOutcome};

#[derive(Deserialize, JsonSchema)]
struct SearchArgs {
    /// The search query.
    query: String,
    /// Maximum number of results to return (default 8).
    num_results: Option<u32>,
}

pub(super) struct WebSearch;
impl Tool for WebSearch {
    fn id(&self) -> &str {
        "websearch"
    }
    fn description(&self) -> &str {
        "Search the web in real time and return relevant results. Use it for \
         current information beyond the model's training data; prefer the \
         current year in time-sensitive queries."
    }
    fn parameters(&self) -> Value {
        schema_for::<SearchArgs>()
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Search
    }
    fn mutating(&self) -> bool {
        false
    }
    fn offered(&self, _mode: RunMode, _model: &str) -> bool {
        provider().is_some()
    }
    fn execute(&self, args: &Value, _ctx: &ToolCtx) -> ToolOutcome {
        let a: SearchArgs = match parse_args(args) {
            Ok(a) => a,
            Err(o) => return o,
        };
        let Some((url, bearer)) = provider() else {
            return ToolOutcome::err("websearch is unavailable — set EXA_API_KEY or PARALLEL_API_KEY");
        };
        search(&url, bearer.as_deref(), &a.query, a.num_results.unwrap_or(8))
    }
}

fn search(url: &str, bearer: Option<&str>, query: &str, num_results: u32) -> ToolOutcome {
    // Discover the endpoint's search tool (avoids hard-coding a per-provider name).
    let list = match mcp_call(url, bearer, "tools/list", json!({})) {
        Ok(v) => v,
        Err(e) => return ToolOutcome::err(e),
    };
    let Some(tool_name) = search_tool_name(&list) else {
        return ToolOutcome::err("websearch: the endpoint advertised no search tool");
    };

    let params = json!({ "name": tool_name, "arguments": { "query": query, "numResults": num_results } });
    let result = match mcp_call(url, bearer, "tools/call", params) {
        Ok(v) => v,
        Err(e) => return ToolOutcome::err(e),
    };
    let text = results_text(&result);
    if text.trim().is_empty() {
        ToolOutcome::ok("(no results)".to_owned())
    } else {
        ToolOutcome::ok(text)
    }
}

/// Which advertised tool to call. Named tools differ per provider, so the one
/// whose name mentions searching wins and the first is the fallback — a search
/// endpoint's first tool is the search.
///
/// Separate from [`search`] because it is a choice, not a request: given the
/// wrong answer the run calls some other tool and reports its output as search
/// results.
fn search_tool_name(list: &Value) -> Option<String> {
    let tools = list.get("tools").and_then(Value::as_array)?;
    tools
        .iter()
        .filter_map(|t| t.get("name").and_then(Value::as_str))
        .find(|name| name.contains("search"))
        .or_else(|| tools.first().and_then(|t| t.get("name").and_then(Value::as_str)))
        .map(str::to_owned)
}

/// The text of an MCP tool result, one content block per line. Non-text blocks
/// are dropped: a search result that came back as an image has nothing the
/// model can read.
fn results_text(result: &Value) -> String {
    result
        .get("content")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

/// POST a JSON-RPC request to the MCP endpoint and return its `result` (or an
/// error). Handles both a plain-JSON body and an SSE (`data:`-prefixed) body.
fn mcp_call(url: &str, bearer: Option<&str>, method: &str, params: Value) -> Result<Value, String> {
    let body = json!({ "jsonrpc": "2.0", "id": 1, "method": method, "params": params });
    let mut req = ureq::post(url)
        .timeout(Duration::from_secs(30))
        .set("Content-Type", "application/json")
        .set("Accept", "application/json, text/event-stream");
    if let Some(b) = bearer {
        req = req.set("Authorization", &format!("Bearer {b}"));
    }
    let resp = req.send_json(body).map_err(|e| format!("websearch: {method} request failed: {e}"))?;
    let text = resp.into_string().map_err(|e| format!("websearch: reading {method} response: {e}"))?;
    parse_jsonrpc(&text)
}

fn parse_jsonrpc(text: &str) -> Result<Value, String> {
    let json: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        // SSE transport: take the last `data:` line that parses as JSON.
        Err(_) => text
            .lines()
            .filter_map(|l| l.strip_prefix("data:").map(str::trim))
            .filter_map(|d| serde_json::from_str::<Value>(d).ok())
            .next_back()
            .ok_or_else(|| "websearch: no JSON in the endpoint response".to_owned())?,
    };
    if let Some(err) = json.get("error") {
        return Err(format!("websearch: endpoint error: {err}"));
    }
    json.get("result").cloned().ok_or_else(|| "websearch: response had no result".to_owned())
}

/// The configured search endpoint `(url, bearer_token)`, by env key. Exa carries
/// its key in the URL; Parallel uses a bearer header.
fn provider() -> Option<(String, Option<String>)> {
    if let Some(key) = env_key("EXA_API_KEY") {
        return Some((format!("https://mcp.exa.ai/mcp?exaApiKey={}", percent_encode(&key)), None));
    }
    if let Some(key) = env_key("PARALLEL_API_KEY") {
        return Some(("https://search.parallel.ai/mcp".to_owned(), Some(key)));
    }
    None
}

fn env_key(name: &str) -> Option<String> {
    std::env::var(name).ok().map(|v| v.trim().to_owned()).filter(|v| !v.is_empty())
}

/// Minimal percent-encoding for putting an API key in a query parameter.
fn percent_encode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_api_key_is_encoded_before_it_goes_in_a_url() {
        // Exa carries the key as a query parameter, so anything not
        // URL-unreserved has to be escaped — a key containing `&` or `#`
        // otherwise truncates itself and the request fails as "unauthorized"
        // rather than as "your key was mangled".
        assert_eq!(percent_encode("abcXYZ019-_.~"), "abcXYZ019-_.~", "unreserved passes through");
        assert_eq!(percent_encode("a&b=c#d e/f"), "a%26b%3Dc%23d%20e%2Ff");
        assert_eq!(percent_encode("café"), "caf%C3%A9", "encoded per UTF-8 byte");
        assert_eq!(percent_encode(""), "");
    }

    #[test]
    fn a_jsonrpc_reply_is_read_from_plain_json_or_from_a_stream() {
        // The same endpoint answers either way, so understanding only one is a
        // provider that works until it doesn't.
        assert_eq!(parse_jsonrpc(r#"{"result":{"ok":true}}"#).unwrap(), json!({ "ok": true }));

        let sse = "event: message\ndata: {\"result\":{\"n\":1}}\n\ndata: {\"result\":{\"n\":2}}\n";
        assert_eq!(parse_jsonrpc(sse).unwrap(), json!({ "n": 2 }), "the last frame is the answer");
    }

    #[test]
    fn an_endpoint_error_is_surfaced_rather_than_read_as_an_empty_result() {
        let err = parse_jsonrpc(r#"{"error":{"code":-32601,"message":"no such tool"}}"#).unwrap_err();
        assert!(err.contains("no such tool"), "got {err}");

        assert!(parse_jsonrpc(r#"{"jsonrpc":"2.0"}"#).unwrap_err().contains("no result"));
        assert!(parse_jsonrpc("<html>not json at all</html>").unwrap_err().contains("no JSON"));
    }

    #[test]
    fn the_tool_that_searches_is_preferred_and_the_first_is_the_fallback() {
        let listed = |names: &[&str]| {
            json!({ "tools": names.iter().map(|n| json!({ "name": n })).collect::<Vec<_>>() })
        };
        assert_eq!(search_tool_name(&listed(&["crawl", "web_search", "map"])).as_deref(), Some("web_search"));
        assert_eq!(
            search_tool_name(&listed(&["find_pages", "crawl"])).as_deref(),
            Some("find_pages"),
            "no name mentions searching, so the first is the search"
        );
        assert!(search_tool_name(&listed(&[])).is_none(), "nothing to call");
        assert!(search_tool_name(&json!({})).is_none(), "an endpoint that advertised nothing");
    }

    #[test]
    fn results_are_joined_and_unreadable_blocks_are_dropped() {
        let result = json!({ "content": [
            { "type": "text", "text": "first hit" },
            { "type": "image", "data": "…" },
            { "type": "text", "text": "second hit" }
        ]});
        assert_eq!(results_text(&result), "first hit\nsecond hit");
        assert_eq!(results_text(&json!({ "content": [] })), "");
        assert_eq!(results_text(&json!({})), "", "a shape we did not expect reads as nothing found");
    }
}
