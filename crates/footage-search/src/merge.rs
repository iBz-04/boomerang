//! Merge and rank hits from multiple query embeddings.

use std::collections::HashSet;

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
    let mut merged: Vec<MergedHit> = Vec::new();

    for query_embedding in query_embeddings {
        let query = Embedding::new((*query_embedding).to_vec())?;
        let mut hits = if let Some(source_file) = config.source_file.as_deref() {
            store
                .search_by_source_file(&query, per_query_candidates, source_file)
                .await?
        } else {
            store.search(&query, per_query_candidates).await?
        };
        hits.sort_by(|a, b| b.similarity_score.total_cmp(&a.similarity_score));
        merge_hits_with_k(&mut merged, hits.drain(..), config.rank_fusion_k);
    }

    merged.retain(|hit| hit.best_similarity >= config.threshold);

    if let Some(threshold) = config.dedupe_threshold {
        if merged.len() > 1 {
            merged.sort_by(compare_merged_hits);
            let ranked_hits: Vec<SearchResult> = merged
                .iter()
                .cloned()
                .map(|merged_hit| merged_hit.into_search_result())
                .collect();
            let deduped_hits = deduplicate_results(ranked_hits, threshold);
            let deduped_signatures: HashSet<(String, u64, u64)> =
                deduped_hits.iter().map(hit_signature).collect();
            merged.retain(|merged_hit| {
                deduped_signatures.contains(&hit_signature(&merged_hit.result))
            });
        }
    }

    merged.sort_by(compare_merged_hits);
    let mut ranked_hits: Vec<SearchResult> = merged
        .into_iter()
        .map(|merged_hit| merged_hit.into_search_result())
        .collect();
    ranked_hits.truncate(config.max_results);
    Ok(ranked_hits)
}

#[cfg(test)]
fn merge_hits(merged: &mut Vec<MergedHit>, incoming: impl Iterator<Item = SearchResult>) {
    merge_hits_with_k(merged, incoming, 60.0);
}

fn merge_hits_with_k(
    merged: &mut Vec<MergedHit>,
    incoming: impl Iterator<Item = SearchResult>,
    rank_fusion_k: f64,
) {
    for (rank, hit) in incoming.enumerate() {
        if let Some(existing) = merged
            .iter_mut()
            .find(|merged_hit| same_chunk(&merged_hit.result, &hit))
        {
            if hit.similarity_score > existing.best_similarity {
                existing.best_similarity = hit.similarity_score;
                existing.best_rank = existing.best_rank.min(rank);
                existing.result = hit;
            } else {
                existing.best_rank = existing.best_rank.min(rank);
            }
            existing.rrf_score += reciprocal_rank(rank, rank_fusion_k);
            existing.support_count += 1;
        } else {
            merged.push(MergedHit {
                best_similarity: hit.similarity_score,
                rrf_score: reciprocal_rank(rank, rank_fusion_k),
                support_count: 1,
                best_rank: rank,
                result: hit,
            });
        }
    }
}

fn reciprocal_rank(rank: usize, rank_fusion_k: f64) -> f64 {
    let k = rank_fusion_k.max(1.0);
    1.0 / (k + rank as f64 + 1.0)
}

fn compare_merged_hits(left: &MergedHit, right: &MergedHit) -> std::cmp::Ordering {
    right
        .rrf_score
        .total_cmp(&left.rrf_score)
        .then_with(|| right.support_count.cmp(&left.support_count))
        .then_with(|| right.best_similarity.total_cmp(&left.best_similarity))
}

fn hit_signature(hit: &SearchResult) -> (String, u64, u64) {
    (
        hit.source_file.clone(),
        hit.start_time.to_bits(),
        hit.end_time.to_bits(),
    )
}

fn same_chunk(a: &SearchResult, b: &SearchResult) -> bool {
    a.source_file == b.source_file
        && (a.start_time - b.start_time).abs() < 0.05
        && (a.end_time - b.end_time).abs() < 0.05
}

#[derive(Clone)]
struct MergedHit {
    result: SearchResult,
    best_similarity: f64,
    rrf_score: f64,
    support_count: usize,
    best_rank: usize,
}

impl MergedHit {
    fn into_search_result(mut self) -> SearchResult {
        self.result.similarity_score = self.best_similarity;
        self.result.ranking_score = self.rrf_score;
        self.result.support_count = self.support_count;
        self.result.best_rank = self.best_rank;
        self.result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(source: &str, start: f64, end: f64, score: f64) -> SearchResult {
        SearchResult::new(source.into(), start, end, score)
    }

    #[test]
    fn test_merge_hits_keeps_highest_score() {
        let mut merged = vec![MergedHit {
            result: hit("/v/a.mp4", 0.0, 8.0, 0.5),
            best_similarity: 0.5,
            rrf_score: reciprocal_rank(0, 60.0),
            support_count: 1,
            best_rank: 0,
        }];
        merge_hits(&mut merged, [hit("/v/a.mp4", 0.0, 8.0, 0.72)].into_iter());
        assert_eq!(merged.len(), 1);
        assert!((merged[0].best_similarity - 0.72).abs() < f64::EPSILON);
        assert_eq!(merged[0].support_count, 2);
        assert_eq!(merged[0].best_rank, 0);
    }

    #[test]
    fn test_merge_hits_keeps_distinct_chunks() {
        let mut merged = vec![MergedHit {
            result: hit("/v/a.mp4", 0.0, 8.0, 0.6),
            best_similarity: 0.6,
            rrf_score: reciprocal_rank(0, 60.0),
            support_count: 1,
            best_rank: 0,
        }];
        merge_hits(&mut merged, [hit("/v/a.mp4", 8.0, 16.0, 0.55)].into_iter());
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_compare_merged_hits_prefers_consensus_over_single_best_score() {
        let consensus = MergedHit {
            result: hit("/v/a.mp4", 0.0, 8.0, 0.62),
            best_similarity: 0.62,
            rrf_score: reciprocal_rank(1, 60.0) + reciprocal_rank(1, 60.0),
            support_count: 2,
            best_rank: 1,
        };
        let singleton = MergedHit {
            result: hit("/v/b.mp4", 8.0, 16.0, 0.91),
            best_similarity: 0.91,
            rrf_score: reciprocal_rank(0, 60.0),
            support_count: 1,
            best_rank: 0,
        };

        let mut hits = vec![singleton, consensus];
        hits.sort_by(compare_merged_hits);

        assert_eq!(hits[0].result.source_file, "/v/a.mp4");
    }

    #[test]
    fn test_into_search_result_preserves_similarity_and_sets_ranking_fields() {
        let merged = MergedHit {
            result: hit("/v/a.mp4", 0.0, 8.0, 0.7),
            best_similarity: 0.82,
            rrf_score: 0.04,
            support_count: 3,
            best_rank: 1,
        };

        let result = merged.into_search_result();

        assert_eq!(result.similarity_score, 0.82);
        assert_eq!(result.ranking_score, 0.04);
        assert_eq!(result.support_count, 3);
        assert_eq!(result.best_rank, 1);
    }
}
