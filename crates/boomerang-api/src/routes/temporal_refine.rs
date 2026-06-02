// Lightweight second-stage temporal refinement for coarse search hits.

use std::path::Path;

use boomerang_core::embedding::{Embedder, Embedding};
use boomerang_core::search::SearchResult;

const REFINE_TOP_RESULTS: usize = 3;
const REFINE_WINDOW_SECONDS: f64 = 3.0;
const REFINE_WINDOW_COUNT: usize = 3;
const REFINE_TARGET_RESOLUTION: u32 = 480;
const REFINE_TARGET_FPS: u32 = 4;

pub async fn refine_search_results(
    results: Vec<SearchResult>,
    query_embeddings: &[Embedding],
    embedder: &dyn Embedder,
) -> anyhow::Result<Vec<SearchResult>> {
    if results.is_empty() || query_embeddings.is_empty() {
        return Ok(results);
    }

    let temp_dir = tempfile::tempdir()?;
    let mut refined = Vec::with_capacity(results.len());

    for (index, result) in results.into_iter().enumerate() {
        if index >= REFINE_TOP_RESULTS {
            refined.push(result);
            continue;
        }

        refined.push(refine_result(&result, query_embeddings, embedder, temp_dir.path()).await?);
    }

    Ok(refined)
}

async fn refine_result(
    result: &SearchResult,
    query_embeddings: &[Embedding],
    embedder: &dyn Embedder,
    temp_dir: &Path,
) -> anyhow::Result<SearchResult> {
    let spans = refinement_spans(result.start_time, result.end_time);
    if spans.len() == 1 && spans[0] == (result.start_time, result.end_time) {
        return Ok(result.clone());
    }

    let mut best_span = (result.start_time, result.end_time);
    let mut best_score = f64::NEG_INFINITY;

    for (span_index, (start_time, end_time)) in spans.into_iter().enumerate() {
        let clip_path = temp_dir.join(format!(
            "refine_{}_{}_{}.mp4",
            sanitize_segment(&result.source_file),
            span_index,
            start_time.to_bits()
        ));
        let trimmed = clip_trim::trim_clip(
            Path::new(&result.source_file),
            start_time,
            end_time,
            &clip_path,
            0.0,
        )
        .await
        .map_err(|error| anyhow::anyhow!("failed to trim refinement clip: {error}"))?;
        let processed =
            video_chunking::chunker::preprocess_chunk(&trimmed, &refinement_chunking_config())
                .await
                .map_err(|error| {
                    anyhow::anyhow!("failed to preprocess refinement clip: {error}")
                })?;
        let clip_embedding = embedder
            .embed_video(&processed.to_string_lossy())
            .await
            .map_err(|error| anyhow::anyhow!("failed to embed refinement clip: {error}"))?;
        let score = query_embeddings
            .iter()
            .map(|query_embedding| {
                cosine_similarity(query_embedding.as_slice(), clip_embedding.as_slice())
            })
            .fold(f64::NEG_INFINITY, f64::max);

        if score > best_score {
            best_score = score;
            best_span = (start_time, end_time);
        }
    }

    Ok(SearchResult::new(
        result.source_file.clone(),
        best_span.0,
        best_span.1,
        result.similarity_score,
    ))
}

fn refinement_chunking_config() -> boomerang_core::chunk::ChunkingConfig {
    boomerang_core::chunk::ChunkingConfig {
        chunk_duration: 1,
        overlap: 0,
        target_resolution: REFINE_TARGET_RESOLUTION,
        target_fps: REFINE_TARGET_FPS,
        skip_preprocess: false,
        skip_still_detection: true,
    }
}

fn refinement_spans(start_time: f64, end_time: f64) -> Vec<(f64, f64)> {
    let duration = end_time - start_time;
    if duration <= 0.0 {
        return vec![(start_time, end_time)];
    }

    let window = duration.min(REFINE_WINDOW_SECONDS);
    let slack = (duration - window).max(0.0);
    if slack <= f64::EPSILON {
        return vec![(start_time, end_time)];
    }

    (0..REFINE_WINDOW_COUNT)
        .map(|index| {
            let progress = if REFINE_WINDOW_COUNT <= 1 {
                0.0
            } else {
                index as f64 / (REFINE_WINDOW_COUNT as f64 - 1.0)
            };
            let span_start = start_time + slack * progress;
            (span_start, span_start + window)
        })
        .collect()
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    if left.len() != right.len() || left.is_empty() {
        return f64::NEG_INFINITY;
    }

    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;

    for (left_value, right_value) in left.iter().zip(right.iter()) {
        let left_f64 = f64::from(*left_value);
        let right_f64 = f64::from(*right_value);
        dot += left_f64 * right_f64;
        left_norm += left_f64 * left_f64;
        right_norm += right_f64 * right_f64;
    }

    if left_norm <= f64::EPSILON || right_norm <= f64::EPSILON {
        return f64::NEG_INFINITY;
    }

    dot / (left_norm.sqrt() * right_norm.sqrt())
}

fn sanitize_segment(source_file: &str) -> String {
    Path::new(source_file)
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("source")
        .chars()
        .map(|value| {
            if value.is_ascii_alphanumeric() || value == '_' || value == '-' {
                value
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refinement_spans_cover_beginning_middle_and_end() {
        let spans = refinement_spans(10.0, 18.0);

        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0], (10.0, 13.0));
        assert_eq!(spans[1], (12.5, 15.5));
        assert_eq!(spans[2], (15.0, 18.0));
    }

    #[test]
    fn test_refinement_spans_keep_short_windows_intact() {
        let spans = refinement_spans(4.0, 6.0);

        assert_eq!(spans, vec![(4.0, 6.0)]);
    }

    #[test]
    fn test_cosine_similarity_prefers_aligned_vectors() {
        let aligned = cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]);
        let opposed = cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]);

        assert!(aligned > opposed);
    }
}
