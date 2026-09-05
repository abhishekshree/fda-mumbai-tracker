use anyhow::Result;
use chrono::{Duration as ChronoDuration, Utc};
use serde_json::json;

use crate::db::{self, ActionInsert};
use crate::llm;
use crate::models::{canonical_outlet_type, coerce_action_date, nonempty, LlmAction, NewsItem};
use crate::news;

#[derive(Debug, Clone)]
pub struct ScrapeReport {
    pub articles_seen: usize,
    pub articles_new: usize,
    pub actions_upserted: usize,
    pub llm_calls: usize,
}

pub async fn run_scrape(gemini_key: &str, model: &str) -> Result<(i64, ScrapeReport)> {
    run_with_window(gemini_key, model, "when:1d", 14, news::MAX_ITEMS, false).await
}

pub async fn run_with_window(
    gemini_key: &str,
    model: &str,
    window: &str,
    seen_days: i64,
    max_items: usize,
    delivery: bool,
) -> Result<(i64, ScrapeReport)> {
    let pool = db::pool().await?;
    let run_id = db::begin_run(pool).await?;

    match scrape_once(
        pool, gemini_key, model, window, seen_days, max_items, delivery,
    )
    .await
    {
        Ok(report) => {
            db::finish_run(
                pool,
                run_id,
                report.articles_seen,
                report.articles_new,
                report.actions_upserted,
                report.llm_calls,
                &json!({"model": model, "window": window, "delivery": delivery}),
            )
            .await?;
            Ok((run_id, report))
        }
        Err(e) => {
            db::fail_run(pool, run_id, &format!("{e:#}")).await?;
            Err(e)
        }
    }
}

async fn scrape_once(
    pool: &sqlx::PgPool,
    gemini_key: &str,
    model: &str,
    window: &str,
    seen_days: i64,
    max_items: usize,
    delivery: bool,
) -> Result<ScrapeReport> {
    let client = crate::http_client();
    let seen = Utc::now() - ChronoDuration::days(seen_days);

    let items = news::fetch_items(client, window).await?;
    let already_seen = db::seen_urls(pool, seen).await?;
    let articles_seen = items.len();
    let fresh = news::enrich(client, items, &already_seen, max_items).await;

    let (actions, mut llm_calls) = llm::extract(gemini_key, model, &fresh, delivery).await?;
    let rows = build_rows(&fresh, &actions, delivery);
    let (rows, collapse_calls) = collapse_dupes(pool, gemini_key, model, rows).await;
    llm_calls += collapse_calls;
    let upserted = db::upsert_actions(pool, &rows).await?;

    Ok(ScrapeReport {
        articles_seen,
        articles_new: fresh.len(),
        actions_upserted: upserted,
        llm_calls,
    })
}

/// LLM pass that collapses re-reported events before insert; falls back to
/// the db.rs name heuristics alone when the DB lookup or the call fails.
async fn collapse_dupes(
    pool: &sqlx::PgPool,
    gemini_key: &str,
    model: &str,
    rows: Vec<ActionInsert>,
) -> (Vec<ActionInsert>, usize) {
    if rows.is_empty() {
        return (rows, 0);
    }
    let recent = match db::recent_events(pool, 10).await {
        Ok(recent) => recent,
        Err(e) => {
            eprintln!("recent_events unavailable ({e:#}); skipping dupe-collapse pass");
            return (rows, 0);
        }
    };
    let (drops, calls) = llm::collapse_dupes(gemini_key, model, &rows, &recent).await;
    if drops.is_empty() {
        return (rows, calls);
    }
    eprintln!(
        "dupe-collapse dropping {} record(s): {drops:?}",
        drops.len()
    );
    let drop_set: std::collections::HashSet<usize> = drops.into_iter().collect();
    let kept = rows
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !drop_set.contains(i))
        .map(|(_, r)| r)
        .collect();
    (kept, calls)
}

pub fn build_rows(items: &[NewsItem], actions: &[LlmAction], delivery: bool) -> Vec<ActionInsert> {
    actions
        .iter()
        .filter_map(|a| {
            if delivery && a.platforms.is_empty() {
                return None;
            }
            let item = items.get(a.source_index)?;
            let establishment = nonempty(Some(a.establishment.as_str()))?;
            Some(ActionInsert {
                establishment,
                area: nonempty(a.area.as_deref()),
                city: nonempty(a.city.as_deref()),
                brand: nonempty(a.brand.as_deref()),
                operator: nonempty(a.operator.as_deref()),
                outlet_type: a.outlet_type.as_deref().map(canonical_outlet_type),
                action_type: a.action_type.to_string(),
                action_date: coerce_action_date(a.action_date.as_deref(), item.published),
                violations: a.violations.clone(),
                compliance_score: a.compliance_score.filter(|s| (0..=100).contains(s)),
                fssai_number: nonempty(a.fssai_number.as_deref()),
                details: nonempty(a.details.as_deref()),
                platforms: a.platforms.clone(),
                source_url: item.url.clone(),
                source_publisher: nonempty(item.source.as_deref()),
                source_headline: Some(item.title.clone()),
                published_at: item.published,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ActionType;
    use chrono::Utc;

    fn item(title: &str, url: &str) -> NewsItem {
        NewsItem {
            title: title.into(),
            url: url.into(),
            source: Some("Test Press".into()),
            published: Some(Utc::now()),
            snippet: None,
        }
    }

    #[test]
    fn builds_rows_from_sources() {
        let items = vec![item(
            "Domino's licence suspended in Mumbai",
            "https://a.test/1",
        )];
        let actions = vec![LlmAction {
            establishment: "Domino's Vile Parle".into(),
            area: Some("Vile Parle West".into()),
            city: Some("Mumbai".into()),
            brand: Some("Domino's".into()),
            outlet_type: Some("restaurant".into()),
            action_type: ActionType::LicenceSuspension,
            action_date: Some("2026-08-11".into()),
            violations: vec!["pest control lapses".into()],
            compliance_score: Some(54),
            platforms: vec!["zomato".into()],
            source_index: 0,
            ..Default::default()
        }];
        let rows = build_rows(&items, &actions, false);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].brand.as_deref(), Some("Domino's"));
        assert_eq!(rows[0].action_type, "licence_suspension");
        assert_eq!(rows[0].outlet_type.as_deref(), Some("restaurant"));
        assert_eq!(rows[0].source_url, "https://a.test/1");
    }

    #[test]
    fn unknown_outlet_type_maps_to_other() {
        let items = vec![item("a", "b")];
        let action = LlmAction {
            establishment: "X".into(),
            outlet_type: Some("pavement stand".into()),
            action_type: ActionType::Inspection,
            source_index: 0,
            ..Default::default()
        };
        let rows = build_rows(&items, &[action], false);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].outlet_type.as_deref(), Some("other"));
    }

    #[test]
    fn drops_orphan_source_index() {
        let items = vec![item("a", "b")];
        let action = LlmAction {
            establishment: "X".into(),
            action_type: ActionType::Inspection,
            source_index: 5,
            ..Default::default()
        };
        assert!(build_rows(&items, &[action], false).is_empty());
    }

    #[test]
    fn delivery_requires_platform_listing() {
        let items = vec![item("a", "b")];
        let action = LlmAction {
            establishment: "X".into(),
            city: Some("Mumbai".into()),
            outlet_type: Some("restaurant".into()),
            action_type: ActionType::Inspection,
            source_index: 0,
            ..Default::default()
        };
        assert!(
            build_rows(&items, &[action], true).is_empty(),
            "delivery mode drops platform-less outlets"
        );
    }

    #[test]
    fn delivery_keeps_listed_outlets() {
        let items = vec![item("a", "b")];
        let action = LlmAction {
            establishment: "X".into(),
            city: Some("Mumbai".into()),
            outlet_type: Some("restaurant".into()),
            action_type: ActionType::Inspection,
            platforms: vec!["zomato".into(), "swiggy".into()],
            source_index: 0,
            ..Default::default()
        };
        assert_eq!(
            build_rows(&items, &[action], true).len(),
            1,
            "listed outlet survives delivery filter"
        );
    }

    fn scored(score: Option<i32>) -> LlmAction {
        LlmAction {
            establishment: "X".into(),
            action_type: ActionType::Inspection,
            compliance_score: score,
            source_index: 0,
            ..Default::default()
        }
    }

    #[test]
    fn compliance_score_in_range_passes_through() {
        let items = vec![item("a", "b")];
        assert_eq!(
            build_rows(&items, &[scored(Some(54))], false)[0].compliance_score,
            Some(54)
        );
    }

    #[test]
    fn compliance_score_out_of_range_becomes_none() {
        let items = vec![item("a", "b")];
        for score in [Some(999), Some(-5), Some(101), None] {
            assert_eq!(
                build_rows(&items, &[scored(score)], false)[0].compliance_score,
                None,
                "{score:?} becomes None"
            );
        }
    }
}
