// Endpoint for searching indexed video footage by text query (POST /search).

use axum::{extract::State, response::IntoResponse, Json};
use boomerang_core::embedding::Embedding;
use boomerang_core::search::SearchConfig;
use boomerang_core::types::EmbeddingSpace;
use serde::Deserialize;
use std::path::Path;
use tracing::{debug, info, info_span, Instrument};

use crate::error::ApiError;
use crate::routes::match_result::{build_match_results, SearchResponse};
use crate::routes::temporal_refine::refine_search_results;
use crate::state::AppState;

const MAX_SEARCH_RESULTS: usize = 10;

#[derive(Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub results: usize,
    pub threshold: f64,
    pub dedupe_threshold: Option<f64>,
    pub source_file: Option<String>,
}

async fn resolve_space(backend: &str, model: Option<&str>) -> Result<EmbeddingSpace, ApiError> {
    if let Some(space) = vector_store::detect_space("qdrant").await? {
        Ok(space)
    } else {
        let embedder = semantic_embed::create_embedder(backend, model)?;
        let space = embedder.embedding_space()?;
        Ok(space)
    }
}

pub async fn search_handler(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let request_id = state.next_request_id();
    let span = info_span!(
        "api_search",
        request_id,
        query_len = req.query.chars().count(),
        results = req.results,
        threshold = req.threshold,
        dedupe_threshold = ?req.dedupe_threshold,
        source_file = ?req.source_file
    );

    search_handler_inner(state, req).instrument(span).await
}

async fn search_handler_inner(
    state: AppState,
    req: SearchRequest,
) -> Result<impl IntoResponse, ApiError> {
    let trimmed = req.query.trim();
    if trimmed.is_empty() {
        return Err(ApiError::BadRequest("query must not be empty".to_string()));
    }

    let limit = validate_result_limit(req.results)?;
    let threshold = validate_threshold(req.threshold)?;
    let dedupe_threshold = validate_optional_dedupe(req.dedupe_threshold)?;
    let source_file = normalize_optional_source_file(req.source_file).await?;

    let embedding_space = resolve_space(&state.backend, state.model.as_deref()).await?;
    let embedder = semantic_embed::create_embedder(
        embedding_space.backend.as_str(),
        embedding_space.model.as_deref(),
    )?;
    let store = vector_store::create_store("qdrant", &embedding_space).await?;

    let search_queries = semantic_embed::query_expand::expand_search_queries(trimmed).await?;
    info!(
        original_query = %trimmed,
        expanded_count = search_queries.len(),
        ?search_queries,
        "expanded search queries"
    );

    let mut embeddings: Vec<Embedding> = Vec::with_capacity(search_queries.len());
    for query in &search_queries {
        let embedding = embedder.embed_query(query).await?;
        log_embedding_summary(query, &embedding);
        embeddings.push(embedding);
    }

    let embedding_refs: Vec<&[f32]> = embeddings.iter().map(|e| e.as_slice()).collect();
    let search_config = SearchConfig {
        max_results: limit,
        threshold,
        dedupe_threshold,
        source_file: source_file.clone(),
    };

    let results =
        footage_search::search_by_embeddings(store.as_ref(), &embedding_refs, &search_config)
            .await?;
    let results = refine_search_results(results, &embeddings, embedder.as_ref()).await?;

    let rewritten_query = search_queries.join(" · ");
    let response_queries = search_queries;

    if results.is_empty() {
        return Ok(Json(SearchResponse {
            results: vec![],
            rewritten_query: Some(rewritten_query),
            search_queries: Some(response_queries),
        }));
    }

    let match_results = build_match_results(results, limit)?;
    for (rank, result) in match_results.iter().enumerate() {
        info!(
            rank = rank + 1,
            file = %result.file,
            start = result.start,
            end = result.end,
            score = result.score,
            "search response match"
        );
    }

    Ok(Json(SearchResponse {
        results: match_results,
        rewritten_query: Some(rewritten_query),
        search_queries: Some(response_queries),
    }))
}

fn log_embedding_summary(label: &str, embedding: &Embedding) {
    let finite_values = embedding
        .data
        .iter()
        .filter(|value| value.is_finite())
        .count();
    let l2_norm = embedding
        .data
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    let sample: Vec<f32> = embedding.data.iter().copied().take(8).collect();

    info!(
        label,
        dimensions = embedding.dimensions,
        finite_values,
        l2_norm,
        "embedding produced"
    );
    debug!(
        label,
        dimensions = embedding.dimensions,
        sample = ?sample,
        "embedding sample"
    );
}

fn validate_result_limit(limit: usize) -> Result<usize, ApiError> {
    if limit == 0 {
        return Err(ApiError::BadRequest(
            "results must be greater than zero".to_string(),
        ));
    }
    if limit > MAX_SEARCH_RESULTS {
        return Err(ApiError::BadRequest(format!(
            "results must be less than or equal to {MAX_SEARCH_RESULTS}"
        )));
    }
    Ok(limit)
}

fn validate_threshold(threshold: f64) -> Result<f64, ApiError> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(ApiError::BadRequest(
            "threshold must be between 0.0 and 1.0".to_string(),
        ));
    }
    Ok(threshold)
}

fn validate_optional_dedupe(threshold: Option<f64>) -> Result<Option<f64>, ApiError> {
    if let Some(value) = threshold {
        if !(0.0..=1.0).contains(&value) {
            return Err(ApiError::BadRequest(
                "dedupe_threshold must be between 0.0 and 1.0".to_string(),
            ));
        }
    }
    Ok(threshold)
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

    #[test]
    fn test_validate_result_limit_rejects_zero() {
        assert!(validate_result_limit(0).is_err());
    }

    #[test]
    fn test_validate_threshold_rejects_out_of_range_values() {
        assert!(validate_threshold(-0.1).is_err());
        assert!(validate_threshold(1.1).is_err());
    }

    #[test]
    fn test_validate_optional_dedupe_rejects_out_of_range_values() {
        assert!(validate_optional_dedupe(Some(-0.1)).is_err());
        assert!(validate_optional_dedupe(Some(1.1)).is_err());
        assert!(validate_optional_dedupe(None).is_ok());
    }

    #[tokio::test]
    async fn test_normalize_optional_source_file_rejects_blank_values() {
        assert!(normalize_optional_source_file(Some("   ".to_string()))
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_normalize_optional_source_file_canonicalizes_existing_paths() {
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("video.mp4");
        tokio::fs::write(&file_path, b"test").await.unwrap();

        let normalized =
            normalize_optional_source_file(Some(file_path.to_string_lossy().to_string()))
                .await
                .unwrap();

        assert_eq!(
            normalized,
            Some(
                file_path
                    .canonicalize()
                    .unwrap()
                    .to_string_lossy()
                    .to_string()
            )
        );
    }
}
