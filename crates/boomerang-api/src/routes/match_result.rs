// Shared match-result mapping for API search and highlight responses.

use std::path::Path;

use boomerang_core::search::SearchResult;
use serde::Serialize;

use crate::error::ApiError;

/// A single matched time range in the indexed source video.
#[derive(Serialize)]
pub struct MatchResult {
    pub file: String,
    pub source_file: String,
    pub start: f64,
    pub end: f64,
    pub score: f64,
}

/// Search response envelope.
#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<MatchResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rewritten_query: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_queries: Option<Vec<String>>,
}

/// Build API results with source metadata and seekable time ranges.
pub fn build_match_results(
    results: Vec<SearchResult>,
    limit: usize,
) -> Result<Vec<MatchResult>, ApiError> {
    let mut match_results = Vec::with_capacity(limit.min(results.len()));

    for result in results.into_iter().take(limit) {
        match_results.push(build_match_result(result)?);
    }

    Ok(match_results)
}

fn build_match_result(result: SearchResult) -> Result<MatchResult, ApiError> {
    let file_display = Path::new(&result.source_file)
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            ApiError::Internal(anyhow::anyhow!(
                "source file '{}' is missing a valid file name",
                result.source_file
            ))
        })?
        .to_string();

    Ok(MatchResult {
        file: file_display,
        source_file: result.source_file,
        start: result.start_time,
        end: result.end_time,
        score: result.similarity_score,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_match_results_keeps_seek_time_and_score() {
        let result = SearchResult::new("/tmp/camera/front.mp4".to_string(), 12.5, 42.0, 0.87);
        let matches = build_match_results(vec![result], 1).expect("match result should build");

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].file, "front.mp4");
        assert_eq!(matches[0].source_file, "/tmp/camera/front.mp4");
        assert_eq!(matches[0].start, 12.5);
        assert_eq!(matches[0].end, 42.0);
        assert_eq!(matches[0].score, 0.87);
    }

    #[test]
    fn test_build_match_results_respects_limit() {
        let first = SearchResult::new("/tmp/first.mp4".to_string(), 1.0, 2.0, 0.9);
        let second = SearchResult::new("/tmp/second.mp4".to_string(), 3.0, 4.0, 0.8);
        let matches =
            build_match_results(vec![first, second], 1).expect("match results should build");

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].file, "first.mp4");
    }
}
