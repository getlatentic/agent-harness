//! The OpenAI `/v1/models` endpoint — the model list every OpenAI-compatible
//! server is expected to serve.
//!
//! Ollama, LM Studio, llama.cpp, vLLM and the hosted gateways all implement it,
//! which is what makes it the right default for an endpoint nobody has written
//! an adapter for: a host can point at a server it has never heard of and still
//! offer a real model picker instead of a free-text box.

use std::time::Duration;

use serde_json::Value;

use crate::ModelChoice;

/// Give up rather than hang a picker. A local server answers in milliseconds
/// and a remote one in well under this.
const TIMEOUT: Duration = Duration::from_secs(5);

/// The models `base_url` serves, newest-agnostic and sorted for a stable list.
///
/// `data[].id` is the only field the spec guarantees, so it is the only one read
/// — a server that adds its own is neither required nor trusted to.
pub fn list_models(base_url: &str, api_key: Option<&str>) -> Result<Vec<ModelChoice>, String> {
    let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
    let mut request = ureq::get(&url).timeout(TIMEOUT);
    if let Some(key) = api_key {
        request = request.set("Authorization", &format!("Bearer {key}"));
    }
    let response = request
        .call()
        .map_err(|error| format!("model list from {url} failed: {error}"))?;
    let body: Value = response
        .into_json()
        .map_err(|error| format!("decoding model list from {url}: {error}"))?;
    Ok(choices(&body))
}

/// Map a `/v1/models` body to picker entries. Split from the request so the
/// parsing is testable without a server, which is where the shape surprises
/// live: a gateway may return an empty `data`, entries without an `id`, or
/// duplicates across two upstreams.
fn choices(body: &Value) -> Vec<ModelChoice> {
    let mut ids: Vec<String> = body
        .get("data")
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(|entry| entry.get("id").and_then(Value::as_str))
                .map(str::to_owned)
                .filter(|id| !id.trim().is_empty())
                .collect()
        })
        .unwrap_or_default();
    ids.sort();
    ids.dedup();
    ids.into_iter()
        .map(|id| ModelChoice { value: id.clone(), label: id })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reads_the_ids_and_sorts_them() {
        let body = json!({"data": [{"id": "qwen3:8b"}, {"id": "granite4:micro"}]});
        let models = choices(&body);
        assert_eq!(
            models.iter().map(|m| m.value.as_str()).collect::<Vec<_>>(),
            ["granite4:micro", "qwen3:8b"]
        );
        // The label is the id: there is no display name in the payload, and
        // inventing one would misname somebody's fine-tune.
        assert_eq!(models[0].label, "granite4:micro");
    }

    #[test]
    fn drops_entries_that_cannot_name_a_model() {
        let body = json!({"data": [{"id": "a"}, {"object": "model"}, {"id": ""}, {"id": "   "}]});
        assert_eq!(choices(&body).len(), 1);
    }

    #[test]
    fn dedupes_ids_served_twice() {
        // A gateway fronting two upstreams can list the same model from both.
        let body = json!({"data": [{"id": "gpt-4o"}, {"id": "gpt-4o"}]});
        assert_eq!(choices(&body).len(), 1);
    }

    #[test]
    fn an_unexpected_body_is_an_empty_list_not_a_panic() {
        // Something answered on the URL; it just was not a model list.
        for body in [json!({}), json!({"data": null}), json!({"data": []}), json!([])] {
            assert!(choices(&body).is_empty(), "{body} should yield nothing");
        }
    }
}
