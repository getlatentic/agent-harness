//! `webfetch` — fetch a URL and return its content as text, markdown, or html.
//! Ported from OpenCode's webfetch (MIT): HTTP→HTTPS upgrade, format options, a
//! 5 MB response cap, and a per-call timeout (default 30 s, max 120 s).
//!
//! `markdown` (the default) converts HTML to Markdown via the `htmd` crate
//! (Turndown-inspired); `text` strips tags to readable plain text; `html`
//! returns the raw body. Non-HTML bodies (JSON, plain text) pass through
//! unchanged. Read-only → offered in both modes.

use std::io::Read;
use std::time::Duration;

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::ToolKind;

use super::{parse_args, schema_for, Tool, ToolCtx, ToolOutcome};

const MAX_RESPONSE_BYTES: u64 = 5 * 1024 * 1024;
const DEFAULT_TIMEOUT_SECS: u64 = 30;
const MAX_TIMEOUT_SECS: u64 = 120;

#[derive(Deserialize, JsonSchema)]
struct FetchArgs {
    /// The URL to fetch (an `http://` URL is upgraded to `https://`).
    url: String,
    /// Output format: "text", "markdown" (default), or "html".
    format: Option<String>,
    /// Timeout in seconds (default 30, max 120).
    timeout: Option<u64>,
}

pub(super) struct WebFetch;
impl Tool for WebFetch {
    fn id(&self) -> &str {
        "webfetch"
    }
    fn description(&self) -> &str {
        "Fetch the contents of a URL and return it as text, markdown, or html. \
         Use it to retrieve and analyze web content."
    }
    fn parameters(&self) -> Value {
        schema_for::<FetchArgs>()
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Fetch
    }
    fn mutating(&self) -> bool {
        false
    }
    fn execute(&self, args: &Value, _ctx: &ToolCtx) -> ToolOutcome {
        let a: FetchArgs = match parse_args(args) {
            Ok(a) => a,
            Err(o) => return o,
        };
        fetch(&a.url, a.format.as_deref().unwrap_or("markdown"), a.timeout)
    }
}

fn fetch(url: &str, format: &str, timeout: Option<u64>) -> ToolOutcome {
    let url = match url.strip_prefix("http://") {
        Some(rest) => format!("https://{rest}"),
        None => url.to_owned(),
    };
    if !url.starts_with("https://") {
        return ToolOutcome::err(format!("webfetch: `{url}` is not a valid http(s) URL"));
    }
    let secs = timeout.filter(|&t| t > 0).unwrap_or(DEFAULT_TIMEOUT_SECS).min(MAX_TIMEOUT_SECS);
    let resp = match ureq::get(&url).timeout(Duration::from_secs(secs)).call() {
        Ok(r) => r,
        Err(e) => return ToolOutcome::err(format!("webfetch: request to {url} failed: {e}")),
    };
    // Reject early on a declared content length over the cap.
    if let Some(len) = resp.header("Content-Length").and_then(|l| l.parse::<u64>().ok()) {
        if len > MAX_RESPONSE_BYTES {
            return ToolOutcome::err("webfetch: response too large (exceeds 5MB limit)".to_owned());
        }
    }
    let content_type = resp.header("Content-Type").unwrap_or("").to_owned();
    // Read at most the cap + 1 byte so an over-limit body is detected.
    let mut body = String::new();
    if let Err(e) = resp.into_reader().take(MAX_RESPONSE_BYTES + 1).read_to_string(&mut body) {
        return ToolOutcome::err(format!("webfetch: reading {url}: {e}"));
    }
    if body.len() as u64 > MAX_RESPONSE_BYTES {
        return ToolOutcome::err("webfetch: response too large (exceeds 5MB limit)".to_owned());
    }
    let is_html = content_type.contains("html") || looks_like_html(&body);
    ToolOutcome::ok(render(format, body, is_html))
}

/// Convert a fetched body to the requested format: `html` raw, `text`
/// tag-stripped, `markdown` (the default) real HTML→Markdown via `htmd`
/// (falling back to plain text if conversion fails). Non-HTML bodies pass
/// through unchanged.
fn render(format: &str, body: String, is_html: bool) -> String {
    match format {
        "html" => body,
        "text" if is_html => html_to_text(&body),
        _ if is_html => htmd::convert(&body).unwrap_or_else(|_| html_to_text(&body)),
        _ => body,
    }
}

fn looks_like_html(s: &str) -> bool {
    let head = s.trim_start();
    head.starts_with("<!") || head.starts_with("<html") || head.contains("<body") || head.contains("<div")
}

/// A pragmatic HTML → readable-text extraction: drop `<script>`/`<style>`
/// spans, strip remaining tags, decode the common entities, and collapse blank
/// lines. Not a full Markdown conversion.
fn html_to_text(html: &str) -> String {
    let stripped = strip_span(html, "<script", "</script>");
    let stripped = strip_span(&stripped, "<style", "</style>");
    let mut text = String::with_capacity(stripped.len());
    let mut in_tag = false;
    for ch in stripped.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => text.push(ch),
            _ => {}
        }
    }
    let text = text
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'");
    // Collapse runs of blank lines and trim trailing whitespace per line.
    let mut out = String::with_capacity(text.len());
    let mut blanks = 0;
    for line in text.lines() {
        let trimmed = line.trim_end();
        if trimmed.trim().is_empty() {
            blanks += 1;
            if blanks <= 1 {
                out.push('\n');
            }
        } else {
            blanks = 0;
            out.push_str(trimmed);
            out.push('\n');
        }
    }
    out.trim().to_owned()
}

/// Remove every `<open … close>` span (case-sensitive; HTML tags are lowercase
/// in practice). Operates on the original string so byte offsets stay valid.
fn strip_span(s: &str, open: &str, close: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(start) = rest.find(open) {
        out.push_str(&rest[..start]);
        let after = &rest[start..];
        match after.find(close) {
            Some(end) => rest = &after[end + close.len()..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_converts_html_per_format() {
        let html = "<h1>Title</h1><p>Hello <strong>world</strong></p>".to_string();

        // markdown → real structured conversion via htmd
        let md = render("markdown", html.clone(), true);
        assert!(md.contains("# Title"), "heading became markdown: {md:?}");
        assert!(md.contains("**world**"), "bold preserved: {md:?}");

        // text → tags stripped, content kept
        let txt = render("text", html.clone(), true);
        assert!(!txt.contains('<'), "tags stripped: {txt:?}");
        assert!(txt.contains("Title") && txt.contains("world"));

        // html → raw, unchanged
        assert_eq!(render("html", html.clone(), true), html);

        // non-HTML body passes through even for markdown
        let json = "{\"a\":1}".to_string();
        assert_eq!(render("markdown", json.clone(), false), json);
    }
}
