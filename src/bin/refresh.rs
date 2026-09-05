use std::collections::HashSet;

use anyhow::Result;
use chrono::{Duration as ChronoDuration, Utc};

use fda_mumbai_tracker::{db, llm, news, scrape};

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let (gemini_key, model) = fda_mumbai_tracker::load_config()?;
    let window = std::env::args().nth(1).unwrap_or_else(|| "when:1d".into());
    let days = std::env::args()
        .nth(2)
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(1);

    let pool = db::pool().await?;
    let client = fda_mumbai_tracker::http_client();

    let items = news::fetch_items(client, &window).await?;
    println!("fetched {} items", items.len());
    let fresh = news::enrich(client, items, &HashSet::new(), news::MAX_ITEMS).await;
    println!("fresh {} items", fresh.len());

    let (actions, calls) = llm::extract(&gemini_key, &model, &fresh, false).await?;
    let rows = scrape::build_rows(&fresh, &actions, false);
    let since = Utc::now() - ChronoDuration::days(days);
    let (deleted, inserted) = db::replace_actions(pool, since, &rows).await?;
    println!("refresh ok: llm_calls={calls} deleted={deleted} inserted={inserted}");
    Ok(())
}
