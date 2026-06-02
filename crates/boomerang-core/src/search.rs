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
    /// Cosine similarity score (0.0–1.0, higher is better).
    pub similarity_score: f64,
}

impl SearchResult {
    pub fn new(source_file: String, start_time: f64, end_time: f64, similarity_score: f64) -> Self {
        Self {
            source_file,
            start_time,
            end_time,
            similarity_score,
        }
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
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_results: 5,
            threshold: 0.41,
            dedupe_threshold: None,
        }
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
}

impl Default for HighlightConfig {
    fn default() -> Self {
        Self {
            count: 5,
            method: crate::types::ScoringMethod::Knn,
            neighbors: 10,
            dedupe_threshold: 0.9,
            exclude_baseline: false,
        }
    }
}
