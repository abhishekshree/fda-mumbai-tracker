use crate::models::LlmAction;

/// Words that mark an establishment as a pharma product, never a food outlet.
const PRODUCT_WORDS: &[&str] = &["tablet", "syrup", "medicine", "drug"];

/// Phrasing that marks a story as "action pending", never a concrete action.
/// Deliberately specific: bare "await" would nuke genuine suspensions
/// "awaiting reinspection".
const NO_ACTION_YET: &[&str] = &[
    "awaits repl",
    "awaiting respon",
    "reply awaited",
    "response awaited",
    "seeks explanation",
    "seeks reply",
    "sought explanation",
    "yet to respond",
];

/// Longest genuine outlet name in history is 9 words
/// ("IIT Bombay Hostel Canteen (Hostels 12, 13 & 14)").
const MAX_ESTABLISHMENT_WORDS: usize = 9;
/// Shortest genuine details in history is 43 chars; floor at 20.
const MIN_DETAILS_CHARS: usize = 20;

/// Post-LLM quality gate: better no record than a wrong one. Every rule is
/// calibrated against the pre-Sept-2026 history (67/67 rows pass) and each
/// rejects a shape the 6 Sept 2026 junk took.
pub(crate) fn check(a: &LlmAction) -> Result<(), &'static str> {
    let name = a.establishment.trim();
    if name.is_empty() {
        return Err("empty establishment");
    }
    if name.starts_with(['\'', '"', '‘', '“']) {
        return Err("quote-led establishment");
    }
    let lower = name.to_lowercase();
    let details = a.details.as_deref().unwrap_or("").trim();
    let hay = format!("{lower} {}", details.to_lowercase());
    if NO_ACTION_YET.iter().any(|w| hay.contains(w)) {
        return Err("no concrete action yet");
    }
    if PRODUCT_WORDS.iter().any(|w| lower.contains(w)) {
        return Err("establishment is a product, not an outlet");
    }
    if lower.contains("fda") {
        return Err("establishment is the regulator, not an outlet");
    }
    if name.split_whitespace().count() > MAX_ESTABLISHMENT_WORDS {
        return Err("establishment reads like a headline fragment");
    }
    if details.chars().count() < MIN_DETAILS_CHARS {
        return Err("no substantive details");
    }
    if a.city.as_deref().map(str::trim).unwrap_or("").is_empty() {
        return Err("no city");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ActionType;

    fn action(est: &str, details: Option<&str>, city: Option<&str>) -> LlmAction {
        LlmAction {
            establishment: est.into(),
            action_type: ActionType::LicenceSuspension,
            violations: vec!["pest infestation".into()],
            details: details.map(str::to_string),
            city: city.map(str::to_string),
            source_index: 0,
            ..Default::default()
        }
    }

    fn good(est: &str) -> LlmAction {
        action(
            est,
            Some("licence suspended over cockroach infestation found during inspection"),
            Some("Mumbai"),
        )
    }

    #[test]
    fn keeps_full_good_record() {
        assert!(check(&good("Domino's Vile Parle")).is_ok());
    }

    #[test]
    fn keeps_inspection_without_violations_when_detailed() {
        // History rows 14/32: inspections with no cited violations but real details.
        let mut a = good("Bombay High Court Canteen");
        a.action_type = ActionType::Inspection;
        a.violations.clear();
        a.details = Some(
            "FDA team inspected canteens inside the Bombay High Court premises following queries from the court."
                .into(),
        );
        assert!(check(&a).is_ok());
    }

    #[test]
    fn keeps_longest_genuine_outlet_name() {
        assert!(check(&good("IIT Bombay Hostel Canteen (Hostels 12, 13 & 14)")).is_ok());
    }

    #[test]
    fn keeps_details_at_exact_floor() {
        let mut a = good("Domino's Vile Parle");
        a.details = Some("12345678901234567890".into());
        assert!(check(&a).is_ok());
    }

    #[test]
    fn drops_quote_establishment() {
        assert_eq!(
            check(&good("'Popularity was never my goal'")),
            Err("quote-led establishment")
        );
    }

    #[test]
    fn drops_product_establishment() {
        assert_eq!(
            check(&good(
                "Popular Re 1 Ayurvedic Digestive Tablet Fails Maharashtra FDA Quality Test"
            )),
            Err("establishment is a product, not an outlet")
        );
    }

    #[test]
    fn drops_regulator_as_establishment() {
        assert_eq!(
            check(&good("Maharashtra FDA rolls out year")),
            Err("establishment is the regulator, not an outlet")
        );
    }

    #[test]
    fn drops_awaiting_reply_establishments() {
        for est in [
            "FDA awaiting response from restaurants at Mumbai Cricket Association premises",
            "FDA awaits replies from restaurants at MCA premises",
            "Maharashtra FDA awaits replies from five MCA restaurants over licensing violations",
        ] {
            assert_eq!(check(&good(est)), Err("no concrete action yet"), "{est}");
        }
    }

    #[test]
    fn drops_headline_fragment_establishment() {
        assert_eq!(
            check(&good(
                "Popular Re 1 Digestive Item Fails Maharashtra Quality Test Today Here"
            )),
            Err("establishment reads like a headline fragment")
        );
    }

    #[test]
    fn drops_content_free_record() {
        let mut a = good("Domino's Vile Parle");
        a.details = None;
        a.violations.clear();
        assert_eq!(check(&a), Err("no substantive details"));
    }

    #[test]
    fn drops_short_details() {
        let mut a = good("Domino's Vile Parle");
        a.details = Some("1234567890123456789".into());
        assert_eq!(check(&a), Err("no substantive details"));
    }

    #[test]
    fn drops_cityless_record() {
        let mut a = good("Domino's Vile Parle");
        a.city = None;
        assert_eq!(check(&a), Err("no city"));
    }

    #[test]
    fn drops_seeks_explanation_details() {
        let mut a = good("MCA BKC Club");
        a.details = Some(
            "FDA has issued a notice and seeks explanation from the club within seven days".into(),
        );
        assert_eq!(check(&a), Err("no concrete action yet"));
    }

    #[test]
    fn drops_empty_establishment() {
        assert_eq!(check(&good("   ")), Err("empty establishment"));
    }
}
