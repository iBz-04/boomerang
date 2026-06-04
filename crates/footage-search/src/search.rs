//! Vector similarity search with deduplication.

use boomerang_core::embedding::Embedding;
use boomerang_core::error::CoreError;
use boomerang_core::search::{SearchConfig, SearchResult};
use boomerang_core::store::VectorStore;
use tracing::{debug, info};

const CANDIDATE_MULTIPLIER: usize = 4;
const MIN_CANDIDATES: usize = 10;

/// Search the vector store with a query embedding, applying deduplication.
pub async fn search_with_embedding(
    store: &dyn VectorStore,
    query_embedding: &[f32],
    config: &SearchConfig,
) -> Result<Vec<SearchResult>, CoreError> {
    info!(
        query_dimensions = query_embedding.len(),
        max_results = config.max_results,
        threshold = config.threshold,
        dedupe_threshold = ?config.dedupe_threshold,
        "search engine received query embedding"
    );

    let query = Embedding::new(query_embedding.to_vec())?;
    let candidate_limit = candidate_limit(config.max_results);
    info!(
        candidate_limit,
        max_results = config.max_results,
        "requesting vector-store candidates"
    );

    let mut hits = if let Some(source_file) = config.source_file.as_deref() {
        store
            .search_by_source_file(&query, candidate_limit, source_file)
            .await?
    } else {
        store.search(&query, candidate_limit).await?
    };
    let raw_candidate_count = hits.len();
    info!(
        raw_candidate_count,
        candidate_limit, "vector-store candidates returned"
    );

    hits.sort_by(|a, b| b.similarity_score.total_cmp(&a.similarity_score));
    for (rank, hit) in hits.iter_mut().enumerate() {
        hit.ranking_score = hit.similarity_score;
        hit.support_count = 1;
        hit.best_rank = rank;
    }

    let below_threshold_count = hits
        .iter()
        .filter(|hit| hit.similarity_score < config.threshold)
        .count();
    hits.retain(|hit| hit.similarity_score >= config.threshold);
    info!(
        threshold = config.threshold,
        before_threshold = raw_candidate_count,
        below_threshold_count,
        after_threshold = hits.len(),
        "threshold filter applied"
    );

    if let Some(threshold) = config.dedupe_threshold {
        if hits.len() > 1 {
            let before_dedupe = hits.len();
            hits = deduplicate_results(hits, threshold);
            info!(
                dedupe_threshold = threshold,
                before_dedupe,
                removed_by_dedupe = before_dedupe.saturating_sub(hits.len()),
                after_dedupe = hits.len(),
                "dedupe filter applied"
            );
        } else {
            info!(
                dedupe_threshold = threshold,
                candidates = hits.len(),
                "dedupe filter skipped"
            );
        }
    } else {
        info!("dedupe filter disabled");
    }

    let before_truncate = hits.len();
    hits.truncate(config.max_results);
    info!(
        max_results = config.max_results,
        before_truncate,
        returned_results = hits.len(),
        "search results truncated"
    );

    for (rank, hit) in hits.iter().enumerate() {
        info!(
            rank = rank + 1,
            file = %hit.source_file,
            start = hit.start_time,
            end = hit.end_time,
            score = hit.similarity_score,
            "search hit retained"
        );
    }

    debug!(count = hits.len(), "search complete");
    Ok(hits)
}

pub(crate) fn candidate_limit(max_results: usize) -> usize {
    max_results
        .saturating_mul(CANDIDATE_MULTIPLIER)
        .max(MIN_CANDIDATES)
}

/// Drop later results that overlap a higher-ranked result from the same source file.
pub(crate) fn deduplicate_results(results: Vec<SearchResult>, threshold: f64) -> Vec<SearchResult> {
    let mut kept: Vec<SearchResult> = Vec::new();

    for result in results {
        if kept.is_empty() {
            kept.push(result);
            continue;
        }

        let is_duplicate = kept.iter().any(|existing| {
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
        } else {
            debug!(
                file = %result.source_file,
                start = result.start_time,
                end = result.end_time,
                score = result.similarity_score,
                dedupe_threshold = threshold,
                "search hit removed by dedupe"
            );
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

    #[test]
    fn test_threshold_filter_removes_low_confidence_results() {
        let mut results = vec![
            make_result("/videos/a.mp4", 0.0, 30.0, 0.9),
            make_result("/videos/b.mp4", 0.0, 30.0, 0.2),
        ];
        results.retain(|hit| hit.similarity_score >= 0.5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source_file, "/videos/a.mp4");
    }

    #[test]
    fn test_candidate_limit_overfetches_for_thresholding() {
        assert_eq!(candidate_limit(1), 10);
        assert_eq!(candidate_limit(5), 20);
    }
}
