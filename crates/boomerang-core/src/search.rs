//! Search result types.

use serde::{Deserialize, Serialize};

/// A single search result from the vector store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Absolute path to the source video file.
    pub source_file: String,
    /// Start time of the matching chunk in seconds.
    pub start_time: f64,
    /// End time of the matching chunk in seconds.
    pub end_time: f64,
    /// Best raw similarity score reported by the vector store.
    pub similarity_score: f64,
    /// Ranking score used for final ordering after fusion or reranking.
    pub ranking_score: f64,
    /// Number of query embeddings that retrieved this result.
    pub support_count: usize,
    /// Best zero-based rank this result reached in an individual retrieval.
    pub best_rank: usize,
}

impl SearchResult {
    pub fn new(source_file: String, start_time: f64, end_time: f64, similarity_score: f64) -> Self {
        Self {
            source_file,
            start_time,
            end_time,
            similarity_score,
            ranking_score: similarity_score,
            support_count: 1,
            best_rank: 0,
        }
    }

    pub fn with_ranking(
        mut self,
        ranking_score: f64,
        support_count: usize,
        best_rank: usize,
    ) -> Self {
        self.ranking_score = ranking_score;
        self.support_count = support_count;
        self.best_rank = best_rank;
        self
    }
}

/// Parameters for a search operation.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Maximum number of results to return.
    pub max_results: usize,
    /// Minimum similarity threshold (0.0–1.0).
    pub threshold: f64,
    /// Deduplication similarity ceiling (None = disabled).
    pub dedupe_threshold: Option<f64>,
    /// Optional source file scope for in-video retrieval.
    pub source_file: Option<String>,
    /// Smoothing constant for reciprocal-rank fusion.
    pub rank_fusion_k: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_results: 5,
            threshold: 0.41,
            dedupe_threshold: None,
            source_file: None,
            rank_fusion_k: 60.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_result_new_uses_similarity_as_initial_ranking_score() {
        let result = SearchResult::new("/video/a.mp4".to_string(), 1.0, 5.0, 0.72);

        assert_eq!(result.similarity_score, 0.72);
        assert_eq!(result.ranking_score, 0.72);
        assert_eq!(result.support_count, 1);
        assert_eq!(result.best_rank, 0);
    }

    #[test]
    fn test_search_result_with_ranking_sets_ranking_metadata() {
        let result =
            SearchResult::new("/video/a.mp4".to_string(), 1.0, 5.0, 0.72).with_ranking(0.04, 3, 1);

        assert_eq!(result.similarity_score, 0.72);
        assert_eq!(result.ranking_score, 0.04);
        assert_eq!(result.support_count, 3);
        assert_eq!(result.best_rank, 1);
    }
}

/// Parameters for highlight/anomaly ranking.
#[derive(Debug, Clone)]
pub struct HighlightConfig {
    /// Number of highlights to return.
    pub count: usize,
    /// Scoring method.
    pub method: crate::types::ScoringMethod,
    /// Number of neighbors for KNN/LOF.
    pub neighbors: usize,
    /// Deduplication threshold.
    pub dedupe_threshold: f64,
    /// Whether to exclude the baseline (half nearest centroid).
    pub exclude_baseline: bool,
    /// Optional source file scope for per-video highlight search.
    pub source_file: Option<String>,
}

impl Default for HighlightConfig {
    fn default() -> Self {
        Self {
            count: 5,
            method: crate::types::ScoringMethod::Knn,
            neighbors: 10,
            dedupe_threshold: 0.9,
            exclude_baseline: false,
            source_file: None,
        }
    }
}
