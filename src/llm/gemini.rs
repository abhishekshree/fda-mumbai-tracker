use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::models::LlmAction;

use super::openrouter::{escalate, remap};
use super::triage::Job;
use super::{strip_code_fences, system_prompt, truncate};

const MAX_ATTEMPTS: usize = 6;

pub(crate) fn gemini_url(model: &str, api_key: &str) -> String {
    format!("https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}")
}

pub(crate) fn retry_delay(attempt: usize, retry_after: Option<u64>) -> Duration {
    let secs = retry_after
        .map(|s| s.min(120))
        .unwrap_or_else(|| (15u64 * (1u64 << attempt.min(4))).min(120));
    #[cfg(test)]
    return Duration::from_secs(secs);
    #[cfg(not(test))]
    {
        // full jitter so concurrent batches don't re-hit the API in lockstep;
        // nanos are enough entropy for sequential workers
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|t| t.subsec_nanos() as u64)
            .unwrap_or(0);
        Duration::from_secs(secs - nanos % (secs / 2 + 1))
    }
}

pub(crate) fn retry_after(resp: &reqwest::Response) -> Option<u64> {
    resp.headers()
        .get("retry-after")?
        .to_str()
        .ok()?
        .parse()
        .ok()
}

pub(crate) async fn post(
    url: &str,
    key: Option<&str>,
    payload: &Value,
    calls: &mut usize,
) -> Result<reqwest::Response> {
    *calls += 1;
    let mut req = crate::http_client().post(url).json(payload);
    if let Some(key) = key {
        req = req.bearer_auth(key);
    }
    req.send()
        .await
        .map_err(|e| anyhow::anyhow!("request error: {e}"))
}

/// One Gemini round-trip. Transport failures yield None (the caller backs off
/// and retries); anything the server answered yields status/body/wait.
async fn gemini_round(
    url: &str,
    payload: &Value,
    calls: &mut usize,
) -> Option<(reqwest::StatusCode, String, Option<u64>)> {
    let resp = match post(url, None, payload, calls).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("gemini {e:#}");
            return None;
        }
    };
    let wait = retry_after(&resp);
    let status = resp.status();
    let text = resp
        .text()
        .await
        .unwrap_or_else(|e| format!("<unreadable body: {e}>"));
    Some((status, text, wait))
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn extract_batch(
    api_key: &str,
    model: &str,
    chunk: &[Job<'_>],
    calls: &mut usize,
    delivery: bool,
    openrouter_key: Option<&str>,
    openrouter_model: &str,
    openrouter_fallbacks: &[String],
) -> Result<Vec<LlmAction>> {
    let payload = json!({
        "system_instruction": {"parts": [{"text": system_prompt(delivery)}]},
        "contents": [{"parts": [{"text": serde_json::to_string(
            &json!({ "items": chunk.iter().map(|j| j.item).collect::<Vec<_>>() })
        ).context("serialize news batch")?}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "maxOutputTokens": 8192
        }
    });
    let url = gemini_url(model, api_key);

    for attempt in 0..MAX_ATTEMPTS {
        let Some((status, text, wait)) = gemini_round(&url, &payload, calls).await else {
            tokio::time::sleep(retry_delay(attempt, None)).await;
            continue;
        };
        let retryable = status.as_u16() == 429 || status.is_server_error();
        if !status.is_success() && !retryable {
            return Err(anyhow::anyhow!("gemini http {status}: {text}"));
        }
        if !status.is_success() && is_quota_exhausted(&text) {
            eprintln!("gemini http {status}, attempt {attempt}; {text}");
            return escalate(
                openrouter_key,
                chunk,
                calls,
                delivery,
                openrouter_model,
                openrouter_fallbacks,
            )
            .await;
        }
        if !status.is_success() {
            eprintln!("gemini http {status}, attempt {attempt}; {text}");
            let wait = wait.or_else(|| retry_delay_from_body(&text));
            tokio::time::sleep(retry_delay(attempt, wait)).await;
            continue;
        }
        let body: Value = serde_json::from_str(&text).context("gemini json")?;
        let text = response_text(&body);
        if text.trim().is_empty() {
            // ponytail: thinking models can burn the token budget and return
            // 200 with zero text; retry, then OpenRouter via the normal path
            let finish = body["candidates"][0]["finishReason"]
                .as_str()
                .unwrap_or("unknown");
            eprintln!("gemini empty response (finishReason={finish}), attempt {attempt}; retrying");
            tokio::time::sleep(retry_delay(attempt, None)).await;
            continue;
        }
        return Ok(remap(parse_llm_text(&text)?, chunk));
    }
    eprintln!("gemini API failed after {MAX_ATTEMPTS} attempts; falling back to openrouter");
    escalate(
        openrouter_key,
        chunk,
        calls,
        delivery,
        openrouter_model,
        openrouter_fallbacks,
    )
    .await
}

fn is_quota_exhausted(text: &str) -> bool {
    text.contains("Quota exceeded") || text.contains("RESOURCE_EXHAUSTED")
}

// Gemini sends the backoff in the error body
// (google.rpc.RetryInfo.retryDelay, e.g. "58s"), not the retry-after header
// the old code read. Honor it so retries stop escalating 5/min into 20/day.
fn retry_delay_from_body(text: &str) -> Option<u64> {
    let body: Value = serde_json::from_str(text).ok()?;
    let details = body.get("error")?.get("details")?.as_array()?;
    details.iter().find_map(|d| {
        let s = d.get("retryDelay")?.as_str()?;
        let num: String = s
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        num.parse::<f64>()
            .ok()
            .map(|secs| secs.ceil().max(1.0) as u64)
    })
}

pub(crate) fn response_text(body: &Value) -> String {
    body.pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default()
}

pub(crate) fn parse_llm_text(text: &str) -> Result<Vec<LlmAction>> {
    let text = strip_code_fences(text);
    let parsed: Value = serde_json::from_str(&text).map_err(|e| {
        anyhow::anyhow!(
            "LLM returned invalid JSON: {e}; body: {}",
            truncate(text.as_str(), 300)
        )
    })?;

    let raw = match parsed {
        Value::Array(arr) => arr,
        Value::Object(map) => map
            .get("actions")
            .and_then(|v| v.as_array())
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!("expected JSON array or object with \"actions\" array")
            })?,
        _ => return Err(anyhow::anyhow!("unexpected LLM response shape")),
    };

    let mut actions = Vec::with_capacity(raw.len());
    for v in raw {
        match serde_json::from_value::<LlmAction>(v.clone()) {
            Ok(a) => actions.push(a),
            Err(e) => eprintln!(
                "dropping invalid LLM record: {e}: {}",
                truncate(&v.to_string(), 200)
            ),
        }
    }

    Ok(actions)
}

pub(crate) fn sanitize_action(mut a: LlmAction) -> LlmAction {
    a.establishment = clamp(a.establishment, 200);
    clamp_opt(&mut a.area, 120);
    clamp_opt(&mut a.city, 120);
    clamp_opt(&mut a.brand, 120);
    clamp_opt(&mut a.operator, 200);
    clamp_opt(&mut a.fssai_number, 64);
    clamp_opt(&mut a.details, 2000);
    a.violations = a
        .violations
        .into_iter()
        .map(|v| clamp(v, 300))
        .filter(|v| !v.is_empty())
        .collect();
    a.violations.truncate(5);
    a.platforms = a
        .platforms
        .into_iter()
        .map(|p| p.trim().to_lowercase())
        .filter(|p| !p.is_empty())
        .collect();
    a.platforms.truncate(6);
    a
}

fn clamp(s: String, max: usize) -> String {
    let trimmed = s.trim().to_string();
    if trimmed.chars().count() > max {
        trimmed.chars().take(max).collect()
    } else {
        trimmed
    }
}

fn clamp_opt(field: &mut Option<String>, max: usize) {
    if let Some(v) = field.take() {
        *field = Some(clamp(v, max));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ActionType;
    use serde_json::json;

    #[test]
    fn strips_fences() {
        assert_eq!(strip_code_fences("```json\n[1,2]\n```"), "[1,2]");
        assert_eq!(strip_code_fences("[1,2]"), "[1,2]");
    }

    #[test]
    fn parses_minimal_action() {
        let body = json!({
            "candidates": [{
                "content": {"parts": [{"text": "[{\"establishment\":\"Domino's\",\"actionType\":\"licence_suspension\",\"sourceIndex\":0}]"}]}
            }]
        });
        let actions = parse_llm_text(&response_text(&body)).unwrap();
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].establishment, "Domino's");
        assert_eq!(actions[0].action_type, ActionType::LicenceSuspension);
    }

    #[test]
    fn accepts_snake_case_keys() {
        let body = json!({
            "candidates": [{
                "content": {"parts": [{"text": "[{\"establishment\":\"X\",\"action_type\":\"inspection\",\"source_index\":0}]"}]}
            }]
        });
        let actions = parse_llm_text(&response_text(&body)).unwrap();
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].action_type, ActionType::Inspection);
    }

    #[test]
    fn tolerates_null_arrays() {
        let body = json!({
            "candidates": [{
                "content": {"parts": [{"text": "[{\"establishment\":\"X\",\"actionType\":\"sealing\",\"violations\":null,\"platforms\":null,\"source_index\":0}]"}]}
            }]
        });
        let actions = parse_llm_text(&response_text(&body)).unwrap();
        assert_eq!(actions.len(), 1);
        assert!(actions[0].violations.is_empty());
        assert!(actions[0].platforms.is_empty());
    }

    #[test]
    fn drops_unknown_action_type() {
        let body = json!({
            "candidates": [{
                "content": {"parts": [{"text": "[{\"establishment\":\"X\",\"actionType\":\"bogus\",\"sourceIndex\":0}]"}]}
            }]
        });
        assert_eq!(parse_llm_text(&response_text(&body)).unwrap().len(), 0);
    }

    #[test]
    fn empty_parts_yield_empty_text() {
        let body = json!({
            "candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": []}}]
        });
        assert_eq!(response_text(&body), "");
        assert!(response_text(&json!({})).is_empty());
    }

    #[test]
    fn retry_delay_backs_off_and_caps() {
        assert_eq!(retry_delay(0, None).as_secs(), 15, "first backoff is 15s");
        assert_eq!(retry_delay(1, None).as_secs(), 30, "second backoff is 30s");
        assert_eq!(retry_delay(2, None).as_secs(), 60, "third backoff is 60s");
        assert_eq!(retry_delay(3, None).as_secs(), 120, "backoff caps at 120s");
        assert_eq!(
            retry_delay(5, None).as_secs(),
            120,
            "cap holds past attempt 4"
        );
        assert_eq!(
            retry_delay(0, Some(5)).as_secs(),
            5,
            "explicit retry-after wins"
        );
        assert_eq!(
            retry_delay(0, Some(300)).as_secs(),
            120,
            "retry-after caps at 120s"
        );
    }

    #[test]
    fn detects_quota_exhaustion() {
        assert!(is_quota_exhausted("Quota exceeded for metric: ..."));
        assert!(is_quota_exhausted("\"status\": \"RESOURCE_EXHAUSTED\""));
        assert!(!is_quota_exhausted("high demand, try again later"));
    }

    #[test]
    fn parses_fenced_text() {
        let text = "```json\n[{\"establishment\":\"X\",\"actionType\":\"inspection\",\"source_index\":0}]\n```";
        let actions = parse_llm_text(text).unwrap();
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].establishment, "X");
    }

    #[test]
    fn retry_delay_reads_gemini_retry_info() {
        let body = r#"{"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "58s"}]}}"#;
        assert_eq!(retry_delay_from_body(body), Some(58));
        let frac = r#"{"error": {"details": [{"retryDelay": "6.13s"}]}}"#;
        assert_eq!(retry_delay_from_body(frac), Some(7));
        assert_eq!(retry_delay_from_body("not json"), None);
    }
}
