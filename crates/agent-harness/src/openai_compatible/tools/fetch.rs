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

/// The URL actually requested. `http://` is upgraded rather than refused —
/// models paste plain-http links constantly and the page is nearly always
/// served over TLS anyway — and every other scheme is refused outright.
///
/// The refusal is the point: a fetch tool that followed `file://` would read
/// the host's disk on the model's say-so, and this tool is offered even in
/// read-only runs because reading the *web* is not reading the machine.
fn resolve_url(url: &str) -> Result<String, String> {
    let url = match url.strip_prefix("http://") {
        Some(rest) => format!("https://{rest}"),
        None => url.to_owned(),
    };
    if url.starts_with("https://") {
        Ok(url)
    } else {
        Err(format!("webfetch: `{url}` is not a valid http(s) URL"))
    }
}

/// How long to wait. A host may ask for less, never for more: an unbounded
/// wait is a run that never finishes and a turn budget spent on one URL.
/// Zero reads as "unset" rather than "give up immediately".
fn timeout_secs(requested: Option<u64>) -> u64 {
    requested.filter(|&t| t > 0).unwrap_or(DEFAULT_TIMEOUT_SECS).min(MAX_TIMEOUT_SECS)
}

fn fetch(url: &str, format: &str, timeout: Option<u64>) -> ToolOutcome {
    match resolve_url(url) {
        Ok(url) => transfer(&url, format, timeout),
        Err(message) => ToolOutcome::err(message),
    }
}

/// Fetch, cap, and render a URL that has **already passed [`resolve_url`]`.
///
/// Split from [`fetch`] because which URLs are allowed is a policy question
/// and moving the bytes is a mechanism, and composing them left the mechanism
/// — including the response-size cap, which is a real guard — unreachable from
/// any test: `resolve_url` upgrades `http` to `https`, so a local stand-in
/// server could never be the thing fetched.
///
/// Private, and it is the caller's job to have resolved first: this does not
/// re-check the scheme.
fn transfer(url: &str, format: &str, timeout: Option<u64>) -> ToolOutcome {
    let secs = timeout_secs(timeout);
    let resp = match ureq::get(url).timeout(Duration::from_secs(secs)).call() {
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

    /// A server answering once with the given status, headers and body.
    fn serving(body: Vec<u8>, content_type: &str, declared_len: Option<u64>) -> String {
        let server = tiny_http::Server::http("127.0.0.1:0").expect("bind ephemeral port");
        let url = format!("http://{}", server.server_addr());
        let content_type = content_type.to_owned();
        std::thread::spawn(move || {
            while let Ok(request) = server.recv() {
                let mut response = tiny_http::Response::from_data(body.clone()).with_header(
                    tiny_http::Header::from_bytes(&b"Content-Type"[..], content_type.as_bytes())
                        .expect("header"),
                );
                if let Some(len) = declared_len {
                    response = response.with_header(
                        tiny_http::Header::from_bytes(
                            &b"Content-Length"[..],
                            len.to_string().as_bytes(),
                        )
                        .expect("header"),
                    );
                }
                let _ = request.respond(response);
            }
        });
        url
    }

    #[test]
    fn an_oversized_body_is_refused_even_when_the_server_never_said_how_big_it_was() {
        // The cheap check reads Content-Length, and a server is free to omit it
        // or lie. If that were the only check, the cap would hold exactly when
        // it was not needed — so the read is capped at the limit plus one byte
        // and the overflow is caught after the fact.
        let body = vec![b'a'; (MAX_RESPONSE_BYTES + 1) as usize];
        let url = serving(body, "text/plain", None);

        let outcome = transfer(&url, "text", None);
        assert!(!outcome.ok);
        assert!(outcome.output.contains("too large"), "got {:?}", outcome.output);
    }

    /// A raw listener, because the point of the next test is a response whose
    /// declared length does not match its body — and a real HTTP server
    /// helpfully corrects that for you.
    fn serving_raw(response: &'static str) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let url = format!("http://{}", listener.local_addr().expect("addr"));
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                use std::io::{BufRead, BufReader, Write};
                let mut reader = BufReader::new(&stream);
                let mut line = String::new();
                // Drain the request head so the client is not writing into a
                // closed socket before it reads.
                while reader.read_line(&mut line).unwrap_or(0) > 0 {
                    if line.ends_with("\r\n\r\n") || line == "\r\n" {
                        break;
                    }
                    line.clear();
                }
                let mut stream = &stream;
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        url
    }

    #[test]
    fn a_declared_oversize_is_refused_on_the_header_alone() {
        // The early exit: a `Content-Length` over the cap is refused before the
        // body is pulled, so we never move 5MB we are about to discard. The
        // body here is four bytes, so only the header can be what refused it.
        let url = serving_raw(
            "HTTP/1.1 200 OK\r\n\
             Content-Type: text/plain\r\n\
             Content-Length: 6000000\r\n\
             \r\n\
             tiny",
        );
        let outcome = transfer(&url, "text", Some(5));
        assert!(!outcome.ok, "got {:?}", outcome.output);
        assert!(outcome.output.contains("too large"), "got {:?}", outcome.output);
    }

    #[test]
    fn a_body_at_the_limit_is_still_delivered() {
        // The boundary the two checks share — off by one here rejects a page
        // that is exactly allowed.
        let body = vec![b'a'; MAX_RESPONSE_BYTES as usize];
        let url = serving(body, "text/plain", None);
        let outcome = transfer(&url, "text", None);
        assert!(outcome.ok, "got {:?}", outcome.output);
        assert_eq!(outcome.output.len(), MAX_RESPONSE_BYTES as usize);
    }

    #[test]
    fn html_is_converted_whether_or_not_the_header_admits_it() {
        // Plenty of servers send HTML as text/plain or with no type at all.
        // Handing raw markup to the model wastes the context the conversion
        // exists to save.
        let markup = b"<html><body><h1>Title</h1><p>Words here.</p></body></html>".to_vec();

        let declared = serving(markup.clone(), "text/html; charset=utf-8", None);
        let outcome = transfer(&declared, "markdown", None);
        assert!(outcome.ok);
        assert!(outcome.output.contains("Title"), "got {:?}", outcome.output);
        assert!(!outcome.output.contains("<h1>"), "markup should be gone: {:?}", outcome.output);

        let lying = serving(markup, "text/plain", None);
        let outcome = transfer(&lying, "markdown", None);
        assert!(!outcome.output.contains("<h1>"), "sniffed, not trusted: {:?}", outcome.output);
    }

    #[test]
    fn asking_for_html_returns_it_verbatim() {
        // The escape hatch: a caller that wants the markup gets the markup.
        let markup = b"<html><body><h1>Title</h1></body></html>".to_vec();
        let url = serving(markup, "text/html", None);
        let outcome = transfer(&url, "html", None);
        assert!(outcome.output.contains("<h1>Title</h1>"), "got {:?}", outcome.output);
    }

    #[test]
    fn an_unreachable_url_is_an_error_naming_it() {
        // A dead endpoint must not read as a page with nothing on it.
        let outcome = transfer("http://127.0.0.1:1/nothing-here", "text", Some(1));
        assert!(!outcome.ok);
        assert!(outcome.output.contains("127.0.0.1:1"), "got {:?}", outcome.output);
    }

    #[test]
    fn a_stripped_script_takes_its_closing_tag_with_it() {
        // The offset that resumes after `</script>` is what decides whether the
        // tag itself is removed or leaks into the text. Content *after* the
        // span is what shows the difference — a page ending at the closing tag
        // reads the same either way.
        let text = html_to_text("<p>before</p><script>var x = 1;</script><p>after</p>");
        assert!(text.contains("before") && text.contains("after"), "got {text:?}");
        assert!(!text.contains("var x"), "the script body is gone: {text:?}");
        assert!(!text.contains('/'), "and so is the closing tag: {text:?}");

        // Two spans in a row: the resume point has to be right each time, not
        // just the first.
        let text = html_to_text("a<style>p{color:red}</style>b<style>i{}</style>c");
        assert_eq!(text, "abc", "got {text:?}");
    }

    #[test]
    fn an_unterminated_script_does_not_leak_the_rest_of_the_page() {
        // Real pages are truncated mid-download. Without a closing tag there is
        // no safe resume point, so everything after the opener is dropped
        // rather than emitted as script source.
        let text = html_to_text("<p>keep</p><script>var x = 1; // and then nothing");
        assert!(text.contains("keep"));
        assert!(!text.contains("var x"), "got {text:?}");
    }

    #[test]
    fn runs_of_blank_lines_collapse_to_one() {
        // Whitespace is the cheapest thing to waste a context window on, and
        // generated markup is full of it. One blank line survives a run so
        // paragraphs still read apart.
        let text = html_to_text("<p>one</p>\n\n\n\n\n<p>two</p>");
        assert!(!text.contains("\n\n\n"), "no run longer than one blank: {text:?}");
        assert!(text.contains("one") && text.contains("two"));
    }

    #[test]
    fn only_the_web_is_fetchable_and_plain_http_is_upgraded() {
        // `webfetch` is offered in read-only runs on the grounds that reading
        // the web is not reading the machine. A scheme that reaches the disk or
        // the local network would quietly make that untrue.
        assert_eq!(resolve_url("http://example.test/a").unwrap(), "https://example.test/a");
        assert_eq!(resolve_url("https://example.test/a").unwrap(), "https://example.test/a");

        for refused in ["file:///etc/passwd", "ftp://example.test/x", "/etc/passwd", "example.test"] {
            let err = resolve_url(refused).unwrap_err();
            assert!(err.contains("not a valid http(s) URL"), "{refused} → {err}");
        }
    }

    #[test]
    fn a_fetch_waits_for_a_bounded_time_the_host_can_shorten_but_not_extend() {
        // An unbounded wait is a run that never finishes; a zero one would make
        // every fetch fail on a slow page.
        assert_eq!(timeout_secs(None), DEFAULT_TIMEOUT_SECS);
        assert_eq!(timeout_secs(Some(0)), DEFAULT_TIMEOUT_SECS, "zero reads as unset, not as give up now");
        assert_eq!(timeout_secs(Some(5)), 5, "a shorter wait is the host's to choose");
        assert_eq!(timeout_secs(Some(9_999)), MAX_TIMEOUT_SECS, "a longer one is not");
    }

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
