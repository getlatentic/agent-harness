//! Shared helpers for the examples. Not an example itself — Cargo only builds
//! `examples/*.rs` and `examples/*/main.rs`, so this directory is ignored.

use harness::{Harness, Error, OpenHarness};

/// The installed model with the most parameters.
///
/// Largest, not first: the agent loop offers the model nine tool schemas, and a
/// very small model answers by reciting them back as prose instead of calling
/// one. Ollama's `capabilities` cannot tell the two apart — a 1B model reports
/// `tools` because its chat template has the syntax, not because it can use it
/// — so parameter count is the signal available.
///
/// Falls back to on-disk size when a backend reports no parameter count, so a
/// model with missing metadata still ranks somewhere sensible.
pub fn largest_installed(harness: &OpenHarness) -> Result<Option<String>, Error> {
    let mut installed = harness.list_installed_models()?;
    installed.sort_by(|a, b| {
        billions(&a.parameter_size)
            .total_cmp(&billions(&b.parameter_size))
            .then(a.size.cmp(&b.size))
    });
    Ok(installed.pop().map(|model| model.name))
}

/// `"7.6B"` → `7.6`, `"800M"` → `0.8`. Unreported or unparseable sorts last.
fn billions(parameter_size: &Option<String>) -> f64 {
    let Some(raw) = parameter_size else { return 0.0 };
    let text = raw.trim();
    let (digits, scale) = match text.chars().last() {
        Some('B' | 'b') => (&text[..text.len() - 1], 1.0),
        Some('M' | 'm') => (&text[..text.len() - 1], 0.001),
        _ => (text, 1.0),
    };
    digits.trim().parse::<f64>().unwrap_or(0.0) * scale
}
