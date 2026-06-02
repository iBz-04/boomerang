// Endpoint for searching indexed video footage by text query (POST /search).

use axum::{extract::State, response::IntoResponse, Json};
use boomerang_core::search::SearchConfig;
use boomerang_core::types::EmbeddingSpace;
use serde::Deserialize;

use crate::error::ApiError;
use crate::routes::match_result::{build_match_results, SearchResponse};
use crate::state::AppState;

/// Search request parameters.
#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub results: Option<usize>,
    pub threshold: Option<f64>,
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

/// Handle semantic search over indexed footage.
pub async fn search_handler(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let embedding_space = resolve_space(&state.backend, state.model.as_deref()).await?;
    let embedder = semantic_embed::create_embedder(
        embedding_space.backend.as_str(),
        embedding_space.model.as_deref(),
    )?;
    let store = vector_store::create_store("qdrant", &embedding_space).await?;

    let query_embedding = embedder.embed_query(&req.query).await?;

    let search_config = SearchConfig {
        max_results: req.results.unwrap_or(5),
        threshold: req.threshold.unwrap_or(0.41),
        dedupe_threshold: None,
    };

    let results =
        footage_search::search_by_text(store.as_ref(), query_embedding.as_slice(), &search_config)
            .await?;

    if results.is_empty() {
        return Ok(Json(SearchResponse { results: vec![] }));
    }

    let limit = req.results.unwrap_or(5);

    let match_results = build_match_results(results, limit)?;

    Ok(Json(SearchResponse {
        results: match_results,
    }))
}
