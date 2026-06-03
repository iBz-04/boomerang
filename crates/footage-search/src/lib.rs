//! Semantic search and anomaly-based highlight ranking over indexed footage.
//!
//! Provides text-based and image-based search against stored video embeddings,
//! plus highlight detection using centroid distance, KNN, and LOF methods.

pub mod highlights;
pub mod merge;
pub mod search;

use boomerang_core::error::CoreError;
use boomerang_core::search::{HighlightConfig, SearchConfig, SearchResult};
use boomerang_core::store::VectorStore;

/// Search indexed footage with a text query.
pub async fn search_by_text(
    store: &dyn VectorStore,
    query_embedding: &[f32],
    config: &SearchConfig,
) -> Result<Vec<SearchResult>, CoreError> {
    search::search_with_embedding(store, query_embedding, config).await
}

/// Search with multiple query embeddings and rank by fused retrieval evidence.
pub async fn search_by_embeddings(
    store: &dyn VectorStore,
    query_embeddings: &[&[f32]],
    config: &SearchConfig,
) -> Result<Vec<SearchResult>, CoreError> {
    merge::search_with_embeddings(store, query_embeddings, config).await
}

/// Search indexed footage with an image query.
pub async fn search_by_image(
    store: &dyn VectorStore,
    image_embedding: &[f32],
    config: &SearchConfig,
) -> Result<Vec<SearchResult>, CoreError> {
    search::search_with_embedding(store, image_embedding, config).await
}

/// Rank the most anomalous clips in the index.
pub async fn rank_highlights(
    store: &dyn VectorStore,
    config: &HighlightConfig,
) -> Result<Vec<SearchResult>, CoreError> {
    highlights::rank_highlights(store, config).await
}
