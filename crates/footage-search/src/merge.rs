//! Merge and rank hits from multiple query embeddings.

use boomerang_core::search::{SearchConfig, SearchResult};
use boomerang_core::store::VectorStore;
use boomerang_core::{embedding::Embedding, error::CoreError};

use crate::search::{candidate_limit, deduplicate_results};

/// Search with several query vectors and keep the best score per chunk.
pub async fn search_with_embeddings(
    store: &dyn VectorStore,
    query_embeddings: &[&[f32]],
    config: &SearchConfig,
) -> Result<Vec<SearchResult>, CoreError> {
    if query_embeddings.is_empty() {
        return Err(CoreError::Config(
            "at least one query embedding is required".into(),
        ));
    }

    let per_query_candidates = candidate_limit(config.max_results.saturating_mul(2));
    let mut merged: Vec<SearchResult> = Vec::new();

    for query_embedding in query_embeddings {
        let query = Embedding::new((*query_embedding).to_vec());
        let mut hits = if let Some(source_file) = config.source_file.as_deref() {
            store
                .search_by_source_file(&query, per_query_candidates, source_file)
                .await?
        } else {
            store.search(&query, per_query_candidates).await?
        };
        merge_hits(&mut merged, hits.drain(..));
    }

    merged.sort_by(|a, b| b.similarity_score.total_cmp(&a.similarity_score));
    merged.retain(|hit| hit.similarity_score >= config.threshold);

    if let Some(threshold) = config.dedupe_threshold {
        if merged.len() > 1 {
            merged = deduplicate_results(merged, threshold);
        }
    }

    merged.truncate(config.max_results);
    Ok(merged)
}

fn merge_hits(merged: &mut Vec<SearchResult>, incoming: impl Iterator<Item = SearchResult>) {
    for hit in incoming {
        if let Some(existing) = merged.iter_mut().find(|r| same_chunk(r, &hit)) {
            if hit.similarity_score > existing.similarity_score {
                *existing = hit;
            }
        } else {
            merged.push(hit);
        }
    }
}

fn same_chunk(a: &SearchResult, b: &SearchResult) -> bool {
    a.source_file == b.source_file
        && (a.start_time - b.start_time).abs() < 0.05
        && (a.end_time - b.end_time).abs() < 0.05
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(source: &str, start: f64, end: f64, score: f64) -> SearchResult {
        SearchResult::new(source.into(), start, end, score)
    }

    #[test]
    fn test_merge_hits_keeps_highest_score() {
        let mut merged = vec![hit("/v/a.mp4", 0.0, 8.0, 0.5)];
        merge_hits(&mut merged, [hit("/v/a.mp4", 0.0, 8.0, 0.72)].into_iter());
        assert_eq!(merged.len(), 1);
        assert!((merged[0].similarity_score - 0.72).abs() < f64::EPSILON);
    }

    #[test]
    fn test_merge_hits_keeps_distinct_chunks() {
        let mut merged = vec![hit("/v/a.mp4", 0.0, 8.0, 0.6)];
        merge_hits(&mut merged, [hit("/v/a.mp4", 8.0, 16.0, 0.55)].into_iter());
        assert_eq!(merged.len(), 2);
    }
}
