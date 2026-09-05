use std::borrow::Cow;
use std::collections::HashSet;

use anyhow::{anyhow, Result};

use crate::db::{ActionInsert, RecentEvent};
use crate::models::{LlmAction, NewsItem};

pub const SYSTEM_PROMPT: &str = include_str!("../prompts/system.txt");

const DELIVERY_MODE: &str = include_str!("../prompts/delivery.txt");

pub(crate) fn system_prompt(delivery: bool) -> Cow<'static, str> {
    if delivery {
        format!("{SYSTEM_PROMPT}{DELIVERY_MODE}").into()
    } else {
        SYSTEM_PROMPT.into()
    }
}

pub(crate) const BATCH_SIZE: usize = 20;
// ponytail: Gemini free tier allows ~5 requests/min — concurrent batches
// hit 429s together, so batches run sequentially with backoff, no semaphore.

/// Latest Flash alias, not a pinned version: pinned 3.5-flash extracted
/// worse, so we track latest until a pinned version beats it.
pub const DEFAULT_GEMINI_MODEL: &str = "gemini-flash-latest";

mod dedupe;
mod gemini;
mod openrouter;
mod triage;

pub async fn extract(
    api_key: &str,
    model: &str,
    items: &[NewsItem],
    delivery: bool,
) -> Result<(Vec<LlmAction>, usize)> {
    let mut jobs = Vec::new();
    for (orig, item) in items.iter().enumerate() {
        if triage::triage(&triage::haystack(item)).is_some() {
            jobs.push(triage::Job { orig, item });
        }
    }
    eprintln!(
        "llm pre-filter: {}/{} items relevant",
        jobs.len(),
        items.len()
    );
    if jobs.is_empty() {
        return Ok((Vec::new(), 0));
    }

    // ponytail: resolve OpenRouter config once at the boundary; deeper fns
    // take it as params so retry/fallback paths stay pure of env reads.
    let openrouter_key = std::env::var("OPENROUTER_API_KEY").ok();
    let openrouter_model = std::env::var("OPENROUTER_MODEL")
        .unwrap_or_else(|_| openrouter::DEFAULT_OPENROUTER_MODEL.to_string());
    let openrouter_fallbacks: Vec<String> = std::env::var("OPENROUTER_FALLBACKS")
        .map(|s| {
            s.split(',')
                .map(str::trim)
                .filter(|m| !m.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_else(|_| {
            openrouter::OPENROUTER_FALLBACKS
                .iter()
                .map(|m| m.to_string())
                .collect()
        });

    let mut calls = 0;
    let mut actions = Vec::new();
    let mut failed = 0;
    for (batch_no, chunk) in jobs.chunks(BATCH_SIZE).enumerate() {
        match gemini::extract_batch(
            api_key,
            model,
            chunk,
            &mut calls,
            delivery,
            openrouter_key.as_deref(),
            &openrouter_model,
            &openrouter_fallbacks,
        )
        .await
        {
            Ok(batch) => actions.extend(batch),
            Err(e) => actions.extend(salvaged(batch_no, &e, chunk, &mut failed)),
        }
    }
    let batch_count = jobs.len().div_ceil(BATCH_SIZE);
    if failed == batch_count {
        return Err(anyhow!("all {batch_count} llm batches failed"));
    }

    let mut seen = HashSet::new();
    actions = actions
        .into_iter()
        .filter(|a| !a.establishment.trim().is_empty())
        .filter(|a| {
            seen.insert((
                a.source_index,
                a.establishment.to_lowercase(),
                a.action_type,
            ))
        })
        .map(gemini::sanitize_action)
        .collect();

    eprintln!(
        "llm: {calls} calls, {failed} failed batches, {} extracted actions",
        actions.len()
    );
    for a in &actions {
        eprintln!(
            "  record: {} | {} | {} | {} | violations={} | details={}",
            a.establishment,
            a.city.as_deref().unwrap_or("-"),
            a.action_type,
            a.action_date.as_deref().unwrap_or("-"),
            if a.violations.is_empty() {
                "-".to_string()
            } else {
                a.violations.join("; ")
            },
            a.details.as_deref().unwrap_or("-"),
        );
    }

    Ok((actions, calls))
}

/// Rule fallback for a failed batch. An empty vec means the whole batch
/// failed; the caller counts it and moves on.
fn salvaged(
    batch_no: usize,
    err: &anyhow::Error,
    chunk: &[triage::Job<'_>],
    failed: &mut usize,
) -> Vec<LlmAction> {
    let kept = triage::rule_extract(chunk);
    if kept.is_empty() {
        *failed += 1;
        eprintln!("llm batch {batch_no} failed: {err:#}");
    } else {
        eprintln!(
            "llm batch {batch_no} failed ({err:#}); rule fallback kept {}",
            kept.len()
        );
    }
    kept
}

/// Ask Gemini which of today's records describe an event already covered by
/// another record or a recent DB row. Returns indices into `rows` to drop and
/// the number of API calls used. Best effort: on any failure returns no drops
/// so the name-heuristic dedup in db::upsert_actions stays the safety net.
pub async fn collapse_dupes(
    api_key: &str,
    model: &str,
    rows: &[ActionInsert],
    recent: &[RecentEvent],
) -> (Vec<usize>, usize) {
    let mut calls = 0;
    match dedupe::collapse_once(api_key, model, rows, recent, &mut calls).await {
        Ok(drops) => (drops, calls),
        Err(e) => {
            eprintln!("dupe-collapse llm failed ({e:#}); using name heuristics only");
            (Vec::new(), calls)
        }
    }
}

pub(crate) fn truncate(s: &str, max: usize) -> String {
    let out: String = s.chars().take(max).collect();
    if s.chars().count() > max {
        format!("{out}…")
    } else {
        out
    }
}

pub(crate) fn strip_code_fences(s: &str) -> String {
    let s = s.trim();
    let Some(body) = s.strip_prefix("```") else {
        return s.to_string();
    };
    let body = body.trim_end_matches('`').trim();
    match body.split_once('\n') {
        Some((_lang, rest)) => rest.trim().to_string(),
        None => body.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::extract;

    #[tokio::test]
    async fn empty_items_skip_llm() {
        let (actions, calls) = extract("key", "model", &[], false).await.unwrap();
        assert!(actions.is_empty());
        assert_eq!(calls, 0);
    }
}
