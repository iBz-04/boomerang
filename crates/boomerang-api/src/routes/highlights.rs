// Endpoint for retrieving anomalous highlights from the indexed footage (POST /highlights).

use std::path::Path;
use axum::{
    extract::State,
    response::IntoResponse,
    Json,
};
use boomerang_core::search::HighlightConfig;
use boomerang_core::types::{EmbeddingSpace, ScoringMethod};
use serde::Deserialize;

use crate::error::ApiError;
use crate::routes::search::{ClipResult, SearchResponse};
use crate::state::AppState;

/// Highlights request parameters.
#[derive(Deserialize)]
pub struct HighlightsRequest {
    pub count: Option<usize>,
    pub method: Option<String>,
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

    let count = req.count.unwrap_or(5);
    let method_str = req.method.as_deref().unwrap_or("knn");
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
        count,
        method,
        neighbors: 10,
        dedupe_threshold: 0.9,
        exclude_baseline: false,
    };

    let results = footage_search::rank_highlights(store.as_ref(), &config).await?;

    if results.is_empty() {
        return Ok(Json(SearchResponse { results: vec![] }));
    }

    let clips = clip_trim::trim_top_results(&results, &state.clips_dir, count).await?;

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
