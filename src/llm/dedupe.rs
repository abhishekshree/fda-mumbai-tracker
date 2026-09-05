use std::collections::BTreeSet;

use anyhow::{Context, Result};
use chrono::NaiveDate;
use serde_json::{json, Value};

use crate::db::{ActionInsert, RecentEvent};

use super::gemini::{gemini_url, post, response_text};
use super::{strip_code_fences, truncate};

const DEDUPE_PROMPT: &str = include_str!("../prompts/dedupe.txt");

pub(crate) async fn collapse_once(
    api_key: &str,
    model: &str,
    rows: &[ActionInsert],
    recent: &[RecentEvent],
    calls: &mut usize,
) -> Result<Vec<usize>> {
    let record = |id_prefix: &str,
                  establishment: &str,
                  action_type: &str,
                  action_date: NaiveDate,
                  city: &Option<String>,
                  area: &Option<String>| {
        json!({
            "id": id_prefix.to_string(),
            "establishment": establishment,
            "actionType": action_type,
            "actionDate": action_date.to_string(),
            "city": city,
            "area": area,
        })
    };
    let new_items: Vec<Value> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            record(
                &format!("N{i}"),
                &r.establishment,
                &r.action_type,
                r.action_date,
                &r.city,
                &r.area,
            )
        })
        .collect();
    let known_items: Vec<Value> = recent
        .iter()
        .enumerate()
        .map(|(i, e)| {
            record(
                &format!("K{i}"),
                &e.establishment,
                &e.action_type,
                e.action_date,
                &e.city,
                &e.area,
            )
        })
        .collect();

    let payload = json!({
        "system_instruction": {"parts": [{"text": DEDUPE_PROMPT}]},
        "contents": [{"parts": [{"text": serde_json::to_string(
            &json!({"new": new_items, "known": known_items})
        ).context("serialize dupe payload")?}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "maxOutputTokens": 2048
        }
    });
    let resp = post(&gemini_url(model, api_key), None, &payload, calls).await?;
    let status = resp.status();
    let body: Value = resp.json().await.context("gemini dupe json body")?;
    if !status.is_success() {
        anyhow::bail!("gemini http {status}");
    }
    let text = response_text(&body);
    if text.trim().is_empty() {
        anyhow::bail!("gemini returned empty response");
    }
    Ok(drops_from_groups(&parse_groups(&text)?, rows.len()))
}

pub(crate) fn parse_groups(text: &str) -> Result<Vec<Vec<String>>> {
    let stripped = strip_code_fences(text.trim());
    let parsed: Value = serde_json::from_str(&stripped).map_err(|e| {
        anyhow::anyhow!(
            "dupe response invalid JSON: {e}; body: {}",
            truncate(text, 300)
        )
    })?;
    let groups = parsed
        .get("groups")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("dupe response missing \"groups\" array"))?;
    Ok(groups
        .iter()
        .filter_map(|g| g.as_array())
        .map(|ids| {
            ids.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
        .filter(|g| !g.is_empty())
        .collect())
}

/// A group means "these ids are one event". If it touches a known row, every
/// new id in it is a re-report and gets dropped; otherwise the lowest-indexed
/// new id survives as the canonical record. Unknown or out-of-range ids are
/// ignored.
pub(crate) fn drops_from_groups(groups: &[Vec<String>], n_new: usize) -> Vec<usize> {
    let mut drop = BTreeSet::new();
    for group in groups {
        let mut news: Vec<usize> = group
            .iter()
            .filter_map(|id| id.strip_prefix('N').and_then(|n| n.parse().ok()))
            .filter(|i| *i < n_new)
            .collect();
        news.sort_unstable();
        news.dedup();
        if news.is_empty() {
            continue;
        }
        let touches_known = group.iter().any(|id| id.starts_with('K'));
        for i in news.iter().skip(usize::from(!touches_known)) {
            drop.insert(*i);
        }
    }
    drop.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_groups_accepts_valid_and_fenced_bodies() {
        assert_eq!(
            parse_groups(r#"{"groups": [["N0","N2"], ["K1","N5"]]}"#).unwrap(),
            vec![
                vec!["N0".to_string(), "N2".to_string()],
                vec!["K1".to_string(), "N5".to_string()]
            ]
        );
        assert_eq!(
            parse_groups("```json\n{\"groups\": []}\n```").unwrap(),
            Vec::<Vec<String>>::new()
        );
    }

    #[test]
    fn parse_groups_rejects_nonconforming_bodies() {
        assert!(parse_groups("no json").is_err());
        assert!(parse_groups(r#"{"actions": []}"#).is_err());
    }

    #[test]
    fn drops_new_ids_but_keeps_lowest_per_event() {
        assert_eq!(
            drops_from_groups(&[vec!["N3".into(), "N1".into(), "N7".into()]], 10),
            vec![3, 7]
        );
    }

    #[test]
    fn drops_every_new_id_touching_a_known_row() {
        assert_eq!(
            drops_from_groups(&[vec!["K4".into(), "N2".into()]], 10),
            vec![2]
        );
    }

    #[test]
    fn ignores_junk_and_out_of_range_ids() {
        assert!(drops_from_groups(&[vec!["N9".into(), "bogus".into()]], 3).is_empty());
        assert!(drops_from_groups(&[], 3).is_empty());
    }
}
