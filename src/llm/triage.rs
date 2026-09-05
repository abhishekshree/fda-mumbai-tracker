use crate::models::{ActionType, LlmAction, NewsItem};

/// A relevance-filtered batch item; `orig` indexes the caller's `items`
/// slice so `source_index` still addresses it for `build_rows`.
pub(crate) struct Job<'a> {
    pub(crate) orig: usize,
    pub(crate) item: &'a NewsItem,
}

pub(crate) fn haystack(it: &NewsItem) -> String {
    format!(
        "{} {} {}",
        it.title,
        it.snippet.as_deref().unwrap_or(""),
        it.source.as_deref().unwrap_or("")
    )
    .to_lowercase()
}

/// Keyword rules as data, not branches: each entry is alternative
/// conjunctions + the action they signal. First match in table order wins,
/// so entries run most- to least-specific. Adding a keyword edits this
/// table, never the control flow below it.
struct Rule {
    needs: &'static [&'static [&'static str]],
    action: ActionType,
}

const RULES: &[Rule] = &[
    Rule {
        needs: &[&["improvement notice"]],
        action: ActionType::ImprovementNotice,
    },
    Rule {
        needs: &[&["licence", "suspend"], &["license", "suspend"]],
        action: ActionType::LicenceSuspension,
    },
    Rule {
        needs: &[
            &["stop business"],
            &["closure", "order"],
            &["shut down", "fda"],
        ],
        action: ActionType::StopBusiness,
    },
    Rule {
        needs: &[&["seal"]],
        action: ActionType::Sealing,
    },
    Rule {
        needs: &[&["seiz"]],
        action: ActionType::Seizure,
    },
    Rule {
        needs: &[&["reopen"]],
        action: ActionType::Reopened,
    },
    Rule {
        needs: &[&["raid"], &["inspect"], &["fda"]],
        action: ActionType::Inspection,
    },
];

/// Validated keyword signal: prose in, typed action out. Single-use plain
/// fn — the old Signal/TryFrom/From ceremony paid for reuse that never came.
fn signal(hay: &str) -> Option<ActionType> {
    RULES.iter().find_map(|rule| {
        rule.needs
            .iter()
            .any(|needles| needles.iter().all(|n| hay.contains(n)))
            .then_some(rule.action)
    })
}

// Only these corroborate a generic-English trigger: "fda" and
// licence/license are unambiguous enough. Deliberately excludes
// seal/seiz/raid/inspect — those ARE the ambiguous words, so they
// can't corroborate themselves.
const CORROBORATION: &[&str] = &["fda", "food safety", "licence", "license"];

pub(crate) fn triage(hay: &str) -> Option<ActionType> {
    let action = signal(hay)?;
    let needs_corroboration = matches!(
        action,
        ActionType::Sealing | ActionType::Seizure | ActionType::Inspection
    );
    (!needs_corroboration || CORROBORATION.iter().any(|c| hay.contains(c))).then_some(action)
}

// Deterministic safety net: when a batch's LLM path fails, keep one minimal
// record per signal-bearing item so the run degrades instead of zeroing out.
// Fields the rules can't know (area, violations, dates) stay empty —
// build_rows + coerce_action_date fill date defaults downstream.
pub(crate) fn rule_extract(jobs: &[Job<'_>]) -> Vec<LlmAction> {
    jobs.iter()
        .filter_map(|j| {
            let action_type = triage(&haystack(j.item))?;
            let name: String = j
                .item
                .title
                .split(['|', '-', ':'])
                .next()?
                .trim()
                .chars()
                .take(120)
                .collect();
            let name = name.trim();
            (!name.is_empty()).then(|| {
                LlmAction::minimal(
                    name.to_string(),
                    action_type,
                    j.orig,
                    j.item.snippet.clone(),
                )
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ActionType;

    #[test]
    fn triage_needs_corroboration_only_for_ambiguous_types() {
        assert_eq!(
            triage("fda raid seals eatery"),
            Some(ActionType::Sealing),
            "sealing with fda context passes"
        );
        assert_eq!(
            triage("shop sealed after fire"),
            None,
            "sealing without corroboration is dropped"
        );
        assert_eq!(
            triage("outlet served improvement notice"),
            Some(ActionType::ImprovementNotice),
            "unambiguous improvement notice passes"
        );
        assert_eq!(
            triage("eatery reopened"),
            Some(ActionType::Reopened),
            "reopened passes without corroboration"
        );
        assert_eq!(
            triage("licence suspended over pests"),
            Some(ActionType::LicenceSuspension),
            "licence suspension passes without corroboration"
        );
    }

    #[test]
    fn rule_fallback_keeps_signal_items_only() {
        let licenced = NewsItem {
            title: "Domino's licence suspended in Mumbai over pests".into(),
            url: "https://x.test/1".into(),
            source: None,
            published: None,
            snippet: None,
        };
        let cricket = NewsItem {
            title: "cricket highlights and match report".into(),
            url: "https://x.test/1".into(),
            source: None,
            published: None,
            snippet: None,
        };
        let jobs = vec![
            Job {
                orig: 4,
                item: &licenced,
            },
            Job {
                orig: 7,
                item: &cricket,
            },
        ];
        assert!(
            triage(&haystack(jobs[1].item)).is_none(),
            "cricket headline carries no signal"
        );
        let kept = rule_extract(&jobs);
        assert_eq!(kept.len(), 1, "only the signal item is kept");
        assert_eq!(kept[0].source_index, 4, "orig index survives");
        assert_eq!(
            kept[0].action_type,
            ActionType::LicenceSuspension,
            "licence suspension detected"
        );
    }
}
