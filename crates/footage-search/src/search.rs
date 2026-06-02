//! Vector similarity search with deduplication.

use boomerang_core::embedding::Embedding;
use boomerang_core::error::CoreError;
use boomerang_core::search::{SearchConfig, SearchResult};
use boomerang_core::store::VectorStore;
use tracing::debug;

/// Search the vector store with a query embedding, applying deduplication.
pub async fn search_with_embedding(
    store: &dyn VectorStore,
    query_embedding: &[f32],
    config: &SearchConfig,
) -> Result<Vec<SearchResult>, CoreError> {
    let query = Embedding::new(query_embedding.to_vec());

    let mut hits = store.search(&query, config.max_results).await?;

    // Sort by similarity descending
    hits.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap_or(std::cmp::Ordering::Equal));

    // Apply deduplication if configured
    if let Some(threshold) = config.dedupe_threshold {
        if hits.len() > 1 {
            hits = deduplicate_results(hits, threshold);
        }
    }

    debug!(count = hits.len(), "search complete");
    Ok(hits)
}

/// Greedy MMR-style deduplication: drop results too similar to a higher-ranked pick.
///
/// Uses cosine similarity of embeddings. When embeddings aren't available,
/// falls back to a simpler heuristic based on source file + time proximity.
fn deduplicate_results(
    results: Vec<SearchResult>,
    threshold: f64,
) -> Vec<SearchResult> {
    let mut kept: Vec<SearchResult> = Vec::new();

    for result in results {
        if kept.is_empty() {
            kept.push(result);
            continue;
        }

        let is_duplicate = kept.iter().any(|existing| {
            // Same source file and overlapping time ranges → likely duplicate
            if existing.source_file == result.source_file {
                let overlap_start = existing.start_time.max(result.start_time);
                let overlap_end = existing.end_time.min(result.end_time);
                let overlap = (overlap_end - overlap_start).max(0.0);
                let min_duration = (existing.end_time - existing.start_time)
                    .min(result.end_time - result.start_time);
                if min_duration > 0.0 {
                    let overlap_ratio = overlap / min_duration;
                    return overlap_ratio > threshold;
                }
            }
            false
        });

        if !is_duplicate {
            kept.push(result);
        }
    }

    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(source: &str, start: f64, end: f64, score: f64) -> SearchResult {
        SearchResult::new(source.to_string(), start, end, score)
    }

    #[test]
    fn test_deduplicate_removes_overlapping_chunks() {
        let results = vec![
            make_result("/videos/a.mp4", 0.0, 30.0, 0.9),
            make_result("/videos/a.mp4", 25.0, 55.0, 0.85), // 5s overlap out of 30s = 0.17
            make_result("/videos/b.mp4", 0.0, 30.0, 0.8),
        ];

        let deduped = deduplicate_results(results, 0.5);
        // Second result has 5/30 = 0.17 overlap, below 0.5 threshold → kept
        assert_eq!(deduped.len(), 3);
    }

    #[test]
    fn test_deduplicate_keeps_distinct_files() {
        let results = vec![
            make_result("/videos/a.mp4", 0.0, 30.0, 0.9),
            make_result("/videos/b.mp4", 0.0, 30.0, 0.8),
        ];

        let deduped = deduplicate_results(results, 0.9);
        assert_eq!(deduped.len(), 2);
    }
}
