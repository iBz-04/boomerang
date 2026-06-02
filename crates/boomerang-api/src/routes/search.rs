// Endpoint for searching indexed video footage by text query (POST /search).

use std::path::Path;
use axum::{
    extract::State,
    response::IntoResponse,
    Json,
};
use boomerang_core::search::SearchConfig;
use boomerang_core::types::EmbeddingSpace;
use serde::{Deserialize, Serialize};

use crate::error::ApiError;
use crate::state::AppState;

/// Search request parameters.
#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub results: Option<usize>,
    pub threshold: Option<f64>,
}

/// A single matched clip result.
#[derive(Serialize)]
pub struct ClipResult {
    pub file: String,
    pub start: f64,
    pub end: f64,
    pub score: f64,
    pub clip_url: String,
}

/// Search response envelope.
#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<ClipResult>,
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

    let results = footage_search::search_by_text(
        store.as_ref(),
        query_embedding.as_slice(),
        &search_config,
    )
    .await?;

    if results.is_empty() {
        return Ok(Json(SearchResponse { results: vec![] }));
    }

    let limit = req.results.unwrap_or(5);
    let clips = clip_trim::trim_top_results(&results, &state.clips_dir, limit).await?;

    let clips_count = clips.len();
    let clip_results = results
        .into_iter()
        .enumerate()
        .filter(|(i, _)| *i < clips_count)
        .map(|(i, r)| {
            let clip_path = &clips[i];
            let filename = clip_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            // Svelte client page expects the full source filename/path for displaying
            let file_display = Path::new(&r.source_file)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            ClipResult {
                file: file_display,
                start: r.start_time,
                end: r.end_time,
                score: r.similarity_score,
                clip_url: format!("/clips/{}", filename),
            }
        })
        .collect();

    Ok(Json(SearchResponse {
        results: clip_results,
    }))
}
