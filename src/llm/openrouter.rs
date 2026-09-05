use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::models::LlmAction;

use super::gemini::{parse_llm_text, post, retry_after, retry_delay};
use super::system_prompt;
use super::triage::Job;

const OPENROUTER_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
/// Verified live against the OpenRouter /models API. lfm-2.5 is built for
/// data extraction with structured-output support; z-ai stays last as the
/// privacy-safe resort under no-training settings.
pub(crate) const DEFAULT_OPENROUTER_MODEL: &str = "liquid/lfm-2.5-2.6b:free";
pub(crate) const OPENROUTER_FALLBACKS: &[&str] = &[
    "cohere/north-mini-code:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "z-ai/glm-5.2:free",
];
const OPENROUTER_MAX_ATTEMPTS: usize = 3;

/// Single OpenRouter escalation path: fall back, then readdress positions
/// to the caller's item indices; orphan indices the LLM invented are dropped.
pub(crate) async fn escalate(
    openrouter_key: Option<&str>,
    chunk: &[Job<'_>],
    calls: &mut usize,
    delivery: bool,
    openrouter_model: &str,
    openrouter_fallbacks: &[String],
) -> Result<Vec<LlmAction>> {
    Ok(remap(
        fallback(
            openrouter_key,
            chunk,
            calls,
            delivery,
            openrouter_model,
            openrouter_fallbacks,
        )
        .await?,
        chunk,
    ))
}

pub(crate) fn remap(actions: Vec<LlmAction>, chunk: &[Job<'_>]) -> Vec<LlmAction> {
    actions
        .into_iter()
        .filter_map(|mut a| {
            let job = chunk.get(a.source_index)?;
            a.source_index = job.orig;
            Some(a)
        })
        .collect()
}

async fn fallback(
    openrouter_key: Option<&str>,
    chunk: &[Job<'_>],
    calls: &mut usize,
    delivery: bool,
    openrouter_model: &str,
    openrouter_fallbacks: &[String],
) -> Result<Vec<LlmAction>> {
    let Some(key) = openrouter_key else {
        return Err(anyhow::anyhow!(
            "gemini API failed and no OPENROUTER_API_KEY fallback configured"
        ));
    };
    openrouter_with_model(
        key,
        openrouter_model,
        chunk,
        calls,
        delivery,
        openrouter_fallbacks,
    )
    .await
}

/// OpenRouter 404s/4xx on guardrails never succeed on retry — fail fast
/// instead of burning backoffs per batch. Returns the error to abort with.
fn openrouter_fatal(status: u16, text: &str) -> Option<anyhow::Error> {
    match status {
        400 | 401 | 403 | 404 => Some(anyhow::anyhow!(
            "openrouter http {status} (non-retryable): {text}"
        )),
        402 if text.contains("Insufficient credits")
            || text.contains("never purchased credits") =>
        {
            Some(anyhow::anyhow!(
                "openrouter 402 insufficient credits: {text}"
            ))
        }
        _ => None,
    }
}

/// One OpenRouter round-trip. Transport failures yield None (the caller backs
/// off and retries); anything the server answered yields status/body/wait.
async fn openrouter_round(
    api_key: &str,
    payload: &Value,
    calls: &mut usize,
) -> Option<(reqwest::StatusCode, String, Option<u64>)> {
    let resp = match post(OPENROUTER_URL, Some(api_key), payload, calls).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("openrouter request error: {e}");
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

/// Parse a 200 OpenRouter body into actions.
fn openrouter_success(text: &str, model: &str) -> Result<Vec<LlmAction>> {
    let body: Value = serde_json::from_str(text).context("openrouter json")?;
    let responded = body["model"].as_str().unwrap_or(model);
    let citations = body["citations"].as_array().map_or(0, Vec::len);
    eprintln!("openrouter ok: model={responded}, web_searches={citations}");
    let text = body["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("no text in openrouter response"))?;
    parse_llm_text(text)
}

/// Handle an OpenRouter 402 by halving the token budget. Non-402 statuses
/// yield false so the caller falls through to the normal backoff. Bails when
/// the budget is already at the floor.
fn shrink_budget_on_402(status: u16, text: &str, payload: &mut Value) -> Result<bool> {
    if status != 402 {
        return Ok(false);
    }
    let cur = payload["max_tokens"].as_u64().unwrap_or(32768);
    if cur <= 8192 {
        return Err(anyhow::anyhow!("openrouter low balance (402): {text}"));
    }
    let lower = cur / 2;
    payload["max_tokens"] = json!(lower);
    eprintln!("openrouter 402 (low balance); retrying with max_tokens={lower}");
    Ok(true)
}

async fn openrouter_with_model(
    api_key: &str,
    model: &str,
    chunk: &[Job<'_>],
    calls: &mut usize,
    delivery: bool,
    openrouter_fallbacks: &[String],
) -> Result<Vec<LlmAction>> {
    eprintln!("openrouter attempt: model={model}, items={}", chunk.len());
    let fallbacks: Vec<String> = openrouter_fallbacks
        .iter()
        .filter(|m| m.as_str() != model)
        .cloned()
        .collect();
    let mut payload = json!({
        "model": model,
        "models": fallbacks,
        "messages": [
            {"role": "system", "content": system_prompt(delivery)},
            {"role": "user", "content": serde_json::to_string(
                &json!({ "items": chunk.iter().map(|j| j.item).collect::<Vec<_>>() })
            ).context("serialize news batch")?}
        ],
        "tools": [{"type": "openrouter:web_search"}],
        "response_format": {"type": "json_object"},
        "plugins": [{"id": "response-healing"}],
        "temperature": 0.0,
        "max_tokens": 32768
    });
    for attempt in 0..OPENROUTER_MAX_ATTEMPTS {
        let Some((status, text, wait)) = openrouter_round(api_key, &payload, calls).await else {
            tokio::time::sleep(retry_delay(attempt, None)).await;
            continue;
        };
        if status.is_success() {
            return openrouter_success(&text, model);
        }
        eprintln!("openrouter http {status}, attempt {attempt}; {text}");
        if let Some(e) = openrouter_fatal(status.as_u16(), &text) {
            return Err(e);
        }
        if shrink_budget_on_402(status.as_u16(), &text, &mut payload)? {
            continue;
        }
        tokio::time::sleep(retry_delay(attempt, wait)).await;
    }
    Err(anyhow::anyhow!(
        "openrouter {model} failed after {OPENROUTER_MAX_ATTEMPTS} attempts"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn fatal_statuses_abort_without_retry() {
        assert!(openrouter_fatal(404, "no such model").is_some());
        assert!(openrouter_fatal(401, "bad key").is_some());
        assert!(openrouter_fatal(429, "slow down").is_none());
        assert!(openrouter_fatal(500, "boom").is_none());
    }

    #[test]
    fn budget_halves_on_402_until_floor() {
        let mut payload = json!({"max_tokens": 32768});
        assert!(shrink_budget_on_402(402, "low", &mut payload).unwrap());
        assert_eq!(payload["max_tokens"], json!(16384));
        assert!(!shrink_budget_on_402(429, "slow", &mut payload).unwrap());
        let mut floor = json!({"max_tokens": 8192});
        assert!(shrink_budget_on_402(402, "low", &mut floor).is_err());
    }

    #[test]
    fn success_parses_first_choice_content() {
        let body = json!({
            "model": "m",
            "choices": [{"message": {"content": "[{\"establishment\":\"X\",\"actionType\":\"inspection\",\"source_index\":0}]"}}]
        });
        let actions = openrouter_success(&body.to_string(), "m").unwrap();
        assert_eq!(actions.len(), 1, "one action parsed");
        let _ = parse_llm_text("[]").unwrap();
    }
}
