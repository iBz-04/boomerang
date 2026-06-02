// Endpoint for retrieving anomalous highlights from the indexed footage (POST /highlights).

use axum::{extract::State, response::IntoResponse, Json};
use boomerang_core::search::HighlightConfig;
use boomerang_core::types::{EmbeddingSpace, ScoringMethod};
use serde::Deserialize;
use std::path::Path;

use crate::error::ApiError;
use crate::routes::match_result::{build_match_results, SearchResponse};
use crate::state::AppState;

/// Highlights request parameters.
#[derive(Deserialize)]
pub struct HighlightsRequest {
    pub count: usize,
    pub method: String,
    pub source_file: Option<String>,
}

/// Helper to resolve the active indexed space or fall back to defaults.
async fn resolve_space(backend: &str, model: Option<&str>) -> Result<EmbeddingSpace, ApiError> {
    if let Some(space) = vector_store::detect_space("qdrant").await? {
        Ok(space)
    } else {
        let embedder = semantic_embed::create_embedder(backend, model)?;
        let space = embedder.embedding_space()?;
        Ok(space)
    }
}

/// Handle anomaly highlight retrieval.
pub async fn highlights_handler(
    State(state): State<AppState>,
    Json(req): Json<HighlightsRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let embedding_space = resolve_space(&state.backend, state.model.as_deref()).await?;
    let store = vector_store::create_store("qdrant", &embedding_space).await?;
    let source_file = normalize_optional_source_file(req.source_file).await?;

    if req.count == 0 {
        return Err(ApiError::BadRequest(
            "count must be greater than zero".to_string(),
        ));
    }

    let method_str = req.method.trim();
    let method = match method_str {
        "centroid" => ScoringMethod::Centroid,
        "knn" => ScoringMethod::Knn,
        "lof" => ScoringMethod::Lof,
        _ => {
            return Err(ApiError::BadRequest(format!(
                "unknown scoring method: {}. Use centroid, knn, or lof.",
                method_str
            )))
        }
    };

    let config = HighlightConfig {
        count: req.count,
        method,
        neighbors: 10,
        dedupe_threshold: 0.9,
        exclude_baseline: false,
        source_file,
    };

    let results = footage_search::rank_highlights(store.as_ref(), &config).await?;

    if results.is_empty() {
        return Ok(Json(SearchResponse {
            results: vec![],
            rewritten_query: None,
            search_queries: None,
        }));
    }

    let match_results = build_match_results(results, req.count)?;

    Ok(Json(SearchResponse {
        results: match_results,
        rewritten_query: None,
        search_queries: None,
    }))
}

async fn normalize_optional_source_file(
    source_file: Option<String>,
) -> Result<Option<String>, ApiError> {
    match source_file {
        Some(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                return Err(ApiError::BadRequest(
                    "source_file must not be blank when provided".to_string(),
                ));
            }
            let path = Path::new(trimmed);
            match tokio::fs::canonicalize(path).await {
                Ok(canonical) => Ok(Some(canonical.to_string_lossy().to_string())),
                Err(_) => Ok(Some(trimmed.to_string())),
            }
        }
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_normalize_optional_source_file_rejects_blank_values() {
        assert!(normalize_optional_source_file(Some("".to_string()))
            .await
            .is_err());
    }
}
