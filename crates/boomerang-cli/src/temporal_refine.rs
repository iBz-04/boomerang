// Temporal refinement narrows coarse retrieval spans to short clip windows.

use std::path::Path;

use anyhow::Result;
use boomerang_core::chunk::ChunkingConfig;
use boomerang_core::embedding::{Embedder, Embedding};
use boomerang_core::search::SearchResult;

const MAX_RESULT_DURATION_SECONDS: f64 = 6.0;
const REFINE_TOP_RESULTS: usize = 3;
const REFINE_STRIDE_SECONDS: f64 = 3.0;
const REFINE_RANK_FUSION_K: f64 = 20.0;

pub async fn refine_search_results(
    results: Vec<SearchResult>,
    query_embeddings: &[Embedding],
    embedder: &dyn Embedder,
) -> Result<Vec<SearchResult>> {
    if results.is_empty() || query_embeddings.is_empty() {
        return Ok(results);
    }

    let temp_dir = tempfile::tempdir()?;
    let mut refined = Vec::with_capacity(results.len());

    for (index, result) in results.into_iter().enumerate() {
        if index >= REFINE_TOP_RESULTS || result_duration(&result) <= MAX_RESULT_DURATION_SECONDS {
            refined.push(clamp_result_duration(result, MAX_RESULT_DURATION_SECONDS));
            continue;
        }

        refined.push(refine_result(result, query_embeddings, embedder, temp_dir.path()).await?);
    }

    Ok(refined)
}

#[cfg(test)]
fn max_result_duration_seconds() -> f64 {
    MAX_RESULT_DURATION_SECONDS
}

async fn refine_result(
    result: SearchResult,
    query_embeddings: &[Embedding],
    embedder: &dyn Embedder,
    temp_dir: &Path,
) -> Result<SearchResult> {
    let windows = candidate_windows(
        result.start_time,
        result.end_time,
        MAX_RESULT_DURATION_SECONDS,
    );
    if windows.len() <= 1 {
        return Ok(clamp_result_duration(result, MAX_RESULT_DURATION_SECONDS));
    }

    let mut candidates = Vec::with_capacity(windows.len());
    for (index, (start_time, end_time)) in windows.iter().copied().enumerate() {
        let clip_path = temp_dir.join(format!(
            "refine_{}_{}_{}.mp4",
            sanitize_source_file(&result.source_file),
            index,
            start_time.to_bits()
        ));
        let trimmed = clip_trim::trim_clip(
            Path::new(&result.source_file),
            start_time,
            end_time,
            &clip_path,
            0.0,
        )
        .await?;
        let processed =
            video_chunking::chunker::preprocess_chunk(&trimmed, &refine_chunking_config()).await?;
        let embedding = embedder.embed_video(&processed.to_string_lossy()).await?;
        candidates.push(WindowCandidate {
            start_time,
            end_time,
            embedding,
            fused_rank_score: 0.0,
            best_similarity: f64::NEG_INFINITY,
            support_count: 0,
            best_rank: usize::MAX,
        });
    }

    for query_embedding in query_embeddings {
        let mut ranking: Vec<(usize, f64)> = candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| {
                (
                    index,
                    cosine_similarity(query_embedding.as_slice(), candidate.embedding.as_slice()),
                )
            })
            .collect();
        ranking.sort_by(|left, right| right.1.total_cmp(&left.1));

        for (rank, (candidate_index, similarity)) in ranking.into_iter().enumerate() {
            let candidate = &mut candidates[candidate_index];
            candidate.fused_rank_score += reciprocal_rank(rank);
            candidate.best_similarity = candidate.best_similarity.max(similarity);
            candidate.support_count += 1;
            candidate.best_rank = candidate.best_rank.min(rank);
        }
    }

    candidates.sort_by(compare_candidates);
    let best = candidates
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("temporal refinement produced no candidates"))?;

    let mut refined = result;
    refined.start_time = best.start_time;
    refined.end_time = best.end_time;
    refined.similarity_score = best.best_similarity;
    Ok(refined.with_ranking(best.fused_rank_score, best.support_count, best.best_rank))
}

fn clamp_result_duration(mut result: SearchResult, max_duration_seconds: f64) -> SearchResult {
    if result_duration(&result) > max_duration_seconds {
        result.end_time = result.start_time + max_duration_seconds;
    }
    result
}

fn result_duration(result: &SearchResult) -> f64 {
    result.end_time - result.start_time
}

fn candidate_windows(start_time: f64, end_time: f64, window_duration: f64) -> Vec<(f64, f64)> {
    let duration = end_time - start_time;
    if duration <= window_duration {
        return vec![(start_time, end_time)];
    }

    let mut windows = Vec::new();
    let mut window_start = start_time;
    while window_start + window_duration < end_time {
        windows.push((window_start, window_start + window_duration));
        window_start += REFINE_STRIDE_SECONDS.min(window_duration);
    }
    windows.push((end_time - window_duration, end_time));
    windows.dedup_by(|left, right| {
        left.0.to_bits() == right.0.to_bits() && left.1.to_bits() == right.1.to_bits()
    });
    windows
}

fn refine_chunking_config() -> ChunkingConfig {
    ChunkingConfig::default()
}

fn reciprocal_rank(rank: usize) -> f64 {
    1.0 / (REFINE_RANK_FUSION_K + rank as f64 + 1.0)
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;

    for (left_value, right_value) in left.iter().zip(right.iter()) {
        let left_value = f64::from(*left_value);
        let right_value = f64::from(*right_value);
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }

    if left_norm <= f64::EPSILON || right_norm <= f64::EPSILON {
        return f64::NEG_INFINITY;
    }

    dot / (left_norm.sqrt() * right_norm.sqrt())
}

fn compare_candidates(left: &WindowCandidate, right: &WindowCandidate) -> std::cmp::Ordering {
    right
        .fused_rank_score
        .total_cmp(&left.fused_rank_score)
        .then_with(|| right.best_similarity.total_cmp(&left.best_similarity))
        .then_with(|| left.start_time.total_cmp(&right.start_time))
}

fn sanitize_source_file(source_file: &str) -> String {
    Path::new(source_file)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("clip")
        .chars()
        .map(|value| {
            if value.is_ascii_alphanumeric() || value == '-' || value == '_' {
                value
            } else {
                '_'
            }
        })
        .collect()
}

struct WindowCandidate {
    start_time: f64,
    end_time: f64,
    embedding: Embedding,
    fused_rank_score: f64,
    best_similarity: f64,
    support_count: usize,
    best_rank: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidate_windows_limits_results_to_six_second_spans() {
        let windows = candidate_windows(0.0, 30.0, max_result_duration_seconds());

        assert!(windows.iter().all(|(start, end)| end - start <= 6.0));
        assert_eq!(windows.first().copied(), Some((0.0, 6.0)));
        assert_eq!(windows.last().copied(), Some((24.0, 30.0)));
    }

    #[test]
    fn test_clamp_result_duration_shortens_long_results() {
        let result = SearchResult::new("/tmp/a.mp4".to_string(), 0.0, 30.0, 0.8);
        let clamped = clamp_result_duration(result, 6.0);

        assert_eq!(clamped.start_time, 0.0);
        assert_eq!(clamped.end_time, 6.0);
    }
}
