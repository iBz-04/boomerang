//! Gemini text expansion: turns a user search prompt into visual retrieval queries.

use boomerang_core::error::CoreError;
use serde::{Deserialize, Serialize};
use tracing::info;

const GEMINI_GENERATE_URL: &str =
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent";
const MAX_QUERIES: usize = 4;

const SYSTEM_INSTRUCTION: &str = "You expand user prompts for video semantic search. Return JSON only: {\"queries\":[\"...\"]}. Produce 1 to 4 short visual descriptions (objects, people, actions, setting, colors, camera view). The first query must repeat the user prompt verbatim. No markdown.";

#[derive(Debug, Serialize)]
struct GenerateRequest {
    contents: Vec<Content>,
    generation_config: GenerationConfig,
    system_instruction: Content,
}

#[derive(Debug, Serialize)]
struct GenerationConfig {
    temperature: f32,
    response_mime_type: String,
    response_schema: ResponseSchema,
}

#[derive(Debug, Serialize)]
struct ResponseSchema {
    #[serde(rename = "type")]
    schema_type: String,
    properties: serde_json::Value,
    required: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Debug, Serialize)]
struct Part {
    text: String,
}

#[derive(Debug, Deserialize)]
struct GenerateResponse {
    candidates: Option<Vec<Candidate>>,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    content: Option<CandidateContent>,
}

#[derive(Debug, Deserialize)]
struct CandidateContent {
    parts: Option<Vec<CandidatePart>>,
}

#[derive(Debug, Deserialize)]
struct CandidatePart {
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ExpandedPayload {
    queries: Vec<String>,
}

/// Distinct visual search strings derived from the user's prompt.
pub async fn expand_search_queries(user_query: &str) -> Result<Vec<String>, CoreError> {
    let trimmed = user_query.trim();
    if trimmed.is_empty() {
        return Err(CoreError::Config("search query must not be empty".into()));
    }

    let api_key = std::env::var("GEMINI_API_KEY")
        .map_err(|_| CoreError::MissingApiKey("GEMINI_API_KEY not set".into()))?;

    let client = reqwest::Client::new();
    let url = format!("{GEMINI_GENERATE_URL}?key={api_key}");

    let body = GenerateRequest {
        contents: vec![Content {
            parts: vec![Part {
                text: format!("User prompt:\n{trimmed}"),
            }],
        }],
        generation_config: GenerationConfig {
            temperature: 0.2,
            response_mime_type: "application/json".to_string(),
            response_schema: ResponseSchema {
                schema_type: "OBJECT".to_string(),
                properties: serde_json::json!({
                    "queries": {
                        "type": "ARRAY",
                        "items": { "type": "STRING" }
                    }
                }),
                required: vec!["queries".to_string()],
            },
        },
        system_instruction: Content {
            parts: vec![Part {
                text: SYSTEM_INSTRUCTION.to_string(),
            }],
        },
    };

    let response = client
        .post(url)
        .json(&body)
        .send()
        .await
        .map_err(|e| CoreError::EmbeddingApi(format!("query expand HTTP failed: {e}")))?;

    let status = response.status();
    let raw = response
        .text()
        .await
        .map_err(|e| CoreError::EmbeddingApi(format!("query expand read body failed: {e}")))?;

    if !status.is_success() {
        return Err(CoreError::EmbeddingApi(format!(
            "query expand failed ({}): {raw}",
            status.as_u16()
        )));
    }

    let parsed: GenerateResponse = serde_json::from_str(&raw)
        .map_err(|e| CoreError::EmbeddingApi(format!("query expand parse response: {e}")))?;

    let text = parsed
        .candidates
        .and_then(|mut c| c.pop())
        .and_then(|c| c.content)
        .and_then(|c| c.parts)
        .and_then(|mut p| p.pop())
        .and_then(|p| p.text)
        .ok_or_else(|| CoreError::EmbeddingApi("query expand returned no text".into()))?;

    let payload: ExpandedPayload = serde_json::from_str(&text)
        .map_err(|e| CoreError::EmbeddingApi(format!("query expand parse JSON payload: {e}")))?;

    let mut queries = dedupe_queries(payload.queries);
    if queries.is_empty() {
        return Err(CoreError::EmbeddingApi(
            "query expand returned no usable queries".into(),
        ));
    }
    queries.truncate(MAX_QUERIES);

    info!(count = queries.len(), "expanded search queries");
    Ok(queries)
}

fn dedupe_queries(mut candidates: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();

    for candidate in std::mem::take(&mut candidates) {
        let trimmed = candidate.trim();
        if trimmed.is_empty() {
            continue;
        }
        let key = normalize_key(trimmed);
        if out
            .iter()
            .any(|existing: &String| normalize_key(existing) == key)
        {
            continue;
        }
        out.push(trimmed.to_string());
    }

    out
}

fn normalize_key(value: &str) -> String {
    value
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dedupe_queries_removes_duplicates_preserving_order() {
        let queries = dedupe_queries(vec![
            "red car stops".into(),
            "red car stops".into(),
            "vehicle braking at intersection".into(),
        ]);
        assert_eq!(queries.len(), 2);
        assert_eq!(queries[0], "red car stops");
        assert_eq!(queries[1], "vehicle braking at intersection");
    }

    #[test]
    fn test_dedupe_queries_returns_empty_when_all_blank() {
        let queries = dedupe_queries(vec!["   ".into(), "".into()]);
        assert!(queries.is_empty());
    }
}
