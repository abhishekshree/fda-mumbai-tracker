pub mod db;
pub mod llm;
pub mod models;
pub mod news;
pub mod scrape;

use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::Client;

static HTTP_CLIENT: OnceLock<Client> = OnceLock::new();

pub fn http_client() -> &'static Client {
    HTTP_CLIENT.get_or_init(|| {
        Client::builder()
            .user_agent(news::USER_AGENT)
            .timeout(Duration::from_secs(60))
            .build()
            .expect("build shared reqwest client")
    })
}

pub fn load_config() -> Result<(String, String)> {
    let key = std::env::var("GEMINI_API_KEY").context("GEMINI_API_KEY is not set")?;
    let model =
        std::env::var("GEMINI_MODEL").unwrap_or_else(|_| llm::DEFAULT_GEMINI_MODEL.to_string());
    Ok((key, model))
}
