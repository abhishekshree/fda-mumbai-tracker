use std::fmt;
use std::str::FromStr;

use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsItem {
    pub title: String,
    pub url: String,
    pub source: Option<String>,
    pub published: Option<DateTime<Utc>>,
    pub snippet: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionType {
    LicenceSuspension,
    StopBusiness,
    ImprovementNotice,
    Sealing,
    Seizure,
    #[default]
    Inspection,
    Reopened,
}

impl ActionType {
    const TABLE: [(Self, &'static str); 7] = [
        (Self::LicenceSuspension, "licence_suspension"),
        (Self::StopBusiness, "stop_business"),
        (Self::ImprovementNotice, "improvement_notice"),
        (Self::Sealing, "sealing"),
        (Self::Seizure, "seizure"),
        (Self::Inspection, "inspection"),
        (Self::Reopened, "reopened"),
    ];
    pub const ALL: [Self; 7] = [
        Self::TABLE[0].0,
        Self::TABLE[1].0,
        Self::TABLE[2].0,
        Self::TABLE[3].0,
        Self::TABLE[4].0,
        Self::TABLE[5].0,
        Self::TABLE[6].0,
    ];

    fn as_code(self) -> &'static str {
        Self::TABLE
            .iter()
            .find(|(t, _)| *t == self)
            .map(|(_, code)| *code)
            .unwrap_or("unknown")
    }
}

impl FromStr for ActionType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        Self::TABLE
            .iter()
            .find(|(_, code)| *code == s)
            .map(|(t, _)| *t)
            .ok_or_else(|| format!("unknown action type: {s}"))
    }
}

impl fmt::Display for ActionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_code())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutletType {
    Restaurant,
    CloudKitchen,
    QuickCommerce,
    Warehouse,
    Dhaba,
    Hotel,
    Bakery,
    Club,
    Mess,
    Dairy,
    StreetVendor,
    Other,
}

impl From<&str> for OutletType {
    fn from(s: &str) -> Self {
        match s.trim() {
            "restaurant" => Self::Restaurant,
            "cloud_kitchen" => Self::CloudKitchen,
            "quick_commerce" => Self::QuickCommerce,
            "warehouse" => Self::Warehouse,
            "dhaba" => Self::Dhaba,
            "hotel" => Self::Hotel,
            "bakery" => Self::Bakery,
            "club" => Self::Club,
            "mess" => Self::Mess,
            "dairy" => Self::Dairy,
            "street_vendor" => Self::StreetVendor,
            _ => Self::Other,
        }
    }
}

impl fmt::Display for OutletType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Restaurant => "restaurant",
            Self::CloudKitchen => "cloud_kitchen",
            Self::QuickCommerce => "quick_commerce",
            Self::Warehouse => "warehouse",
            Self::Dhaba => "dhaba",
            Self::Hotel => "hotel",
            Self::Bakery => "bakery",
            Self::Club => "club",
            Self::Mess => "mess",
            Self::Dairy => "dairy",
            Self::StreetVendor => "street_vendor",
            Self::Other => "other",
        })
    }
}

pub fn canonical_outlet_type(s: &str) -> String {
    OutletType::from(s).to_string()
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmAction {
    pub establishment: String,
    #[serde(default)]
    pub area: Option<String>,
    #[serde(default)]
    pub city: Option<String>,
    #[serde(default)]
    pub brand: Option<String>,
    #[serde(default)]
    pub operator: Option<String>,
    #[serde(default, alias = "outlet_type")]
    pub outlet_type: Option<String>,
    #[serde(alias = "action_type")]
    pub action_type: ActionType,
    #[serde(default, alias = "action_date")]
    pub action_date: Option<String>,
    #[serde(default, deserialize_with = "deserialize_string_vec")]
    pub violations: Vec<String>,
    #[serde(default)]
    pub compliance_score: Option<i32>,
    #[serde(default, alias = "fssai_number")]
    pub fssai_number: Option<String>,
    #[serde(default)]
    pub details: Option<String>,
    #[serde(default, deserialize_with = "deserialize_string_vec")]
    pub platforms: Vec<String>,
    #[serde(default, alias = "source_index")]
    pub source_index: usize,
}

fn deserialize_string_vec<'de, D>(de: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<Vec<String>>::deserialize(de)?.unwrap_or_default())
}

impl LlmAction {
    /// Minimal record for the deterministic rule fallback: only what keyword
    /// rules can know. Everything else stays empty for downstream defaults.
    pub fn minimal(
        establishment: String,
        action_type: ActionType,
        source_index: usize,
        details: Option<String>,
    ) -> Self {
        Self {
            establishment,
            action_type,
            source_index,
            details,
            ..Default::default()
        }
    }
}

pub fn nonempty(v: Option<&str>) -> Option<String> {
    v.map(str::trim)
        .filter(|s| !s.is_empty() && !s.eq_ignore_ascii_case("null"))
        .map(str::to_string)
}

pub fn coerce_action_date(raw: Option<&str>, published: Option<DateTime<Utc>>) -> NaiveDate {
    if let Some(s) = raw.and_then(|s| NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d").ok()) {
        return s;
    }
    published
        .map(|d| d.date_naive())
        .unwrap_or_else(|| Utc::now().date_naive())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_type_round_trips() {
        assert_eq!(
            "licence_suspension".parse::<ActionType>().unwrap(),
            ActionType::LicenceSuspension,
            "licence_suspension round-trips"
        );
        assert_eq!(
            "stop_business".parse::<ActionType>().unwrap(),
            ActionType::StopBusiness,
            "stop_business round-trips"
        );
        assert_eq!(
            "improvement_notice".parse::<ActionType>().unwrap(),
            ActionType::ImprovementNotice,
            "improvement_notice round-trips"
        );
        assert_eq!(
            "sealing".parse::<ActionType>().unwrap(),
            ActionType::Sealing,
            "sealing round-trips"
        );
        assert_eq!(
            "seizure".parse::<ActionType>().unwrap(),
            ActionType::Seizure,
            "seizure round-trips"
        );
        assert_eq!(
            "inspection".parse::<ActionType>().unwrap(),
            ActionType::Inspection,
            "inspection round-trips"
        );
        assert_eq!(
            "reopened".parse::<ActionType>().unwrap(),
            ActionType::Reopened,
            "reopened round-trips"
        );
        assert!(
            "suspended".parse::<ActionType>().is_err(),
            "unknown code is rejected"
        );
    }

    #[test]
    fn outlet_type_unknown_maps_to_other() {
        assert_eq!(canonical_outlet_type("restaurant"), "restaurant");
        assert_eq!(canonical_outlet_type("stand"), "other");
        assert_eq!(canonical_outlet_type(""), "other");
    }

    #[test]
    fn date_trusts_reported_value() {
        assert_eq!(
            coerce_action_date(Some("2026-08-11"), None),
            NaiveDate::from_ymd_opt(2026, 8, 11).unwrap()
        );
        assert_eq!(coerce_action_date(None, None), Utc::now().date_naive());
        assert_eq!(
            coerce_action_date(Some("not a date"), Some(Utc::now())),
            Utc::now().date_naive()
        );
    }
}
