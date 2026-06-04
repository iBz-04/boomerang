//! Exact-moment reranking trims coarse search hits down to stable onset spans.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use boomerang_core::embedding::{Embedder, Embedding};
use boomerang_core::error::CoreError;
use boomerang_core::search::SearchResult;
use tempfile::TempDir;
use tokio::process::Command;

const EXACT_WINDOW_SECONDS: f64 = 1.0;
const EXACT_WINDOW_STRIDE_SECONDS: f64 = 0.5;
const EXACT_RIGHT_PADDING_SECONDS: f64 = 1.0;
const EXACT_MAX_DURATION_SECONDS: f64 = 3.0;
const EXACT_THRESHOLD_MARGIN: f64 = 0.03;
const EXACT_TOP_RESULTS: usize = 3;
const STABLE_LOOKAHEAD_WINDOWS: usize = 3;
const STABLE_ACTIVE_WINDOWS: usize = 2;
const TARGET_RESOLUTION: u32 = 480;
const TARGET_FPS: u32 = 4;

pub async fn refine_search_results_exact(
    results: Vec<SearchResult>,
    query_embeddings: &[Embedding],
    embedder: &dyn Embedder,
    threshold: f64,
) -> Result<Vec<SearchResult>, CoreError> {
    if results.is_empty() || query_embeddings.is_empty() {
        return Ok(results);
    }

    let temp_dir = tempfile::tempdir()?;
    let mut refined = Vec::with_capacity(results.len());

    for (index, result) in results.into_iter().enumerate() {
        if index >= EXACT_TOP_RESULTS {
            refined.push(result);
            continue;
        }

        refined.push(
            refine_result_exact(
                result,
                index,
                query_embeddings,
                embedder,
                threshold,
                &temp_dir,
            )
            .await?,
        );
    }

    Ok(refined)
}

async fn refine_result_exact(
    mut result: SearchResult,
    result_index: usize,
    query_embeddings: &[Embedding],
    embedder: &dyn Embedder,
    threshold: f64,
    temp_dir: &TempDir,
) -> Result<SearchResult, CoreError> {
    let candidate_start = result.start_time.max(0.0);
    let candidate_end =
        (result.end_time + EXACT_RIGHT_PADDING_SECONDS).max(candidate_start + EXACT_WINDOW_SECONDS);
    let windows = build_window_scores(
        &result.source_file,
        result_index,
        candidate_start,
        candidate_end,
        query_embeddings,
        embedder,
        temp_dir.path(),
    )
    .await?;

    if windows.is_empty() {
        return Ok(result);
    }

    let exact_interval = choose_exact_interval(&windows, threshold);
    result.start_time = exact_interval.start_time;
    result.end_time = exact_interval.end_time;
    result.similarity_score = exact_interval.peak_similarity;
    Ok(result)
}

async fn build_window_scores(
    source_file: &str,
    result_index: usize,
    candidate_start: f64,
    candidate_end: f64,
    query_embeddings: &[Embedding],
    embedder: &dyn Embedder,
    output_dir: &Path,
) -> Result<Vec<ScoredWindow>, CoreError> {
    let spans = micro_window_spans(candidate_start, candidate_end);
    let mut windows = Vec::with_capacity(spans.len());

    for (window_index, span) in spans.into_iter().enumerate() {
        let window_path = output_dir.join(format!("exact_{result_index}_{window_index}.mp4"));
        extract_window(source_file, span.start_time, span.end_time, &window_path).await?;
        let embedding = embedder
            .embed_video(window_path.to_string_lossy().as_ref())
            .await?;
        windows.push(ScoredWindow {
            start_time: span.start_time,
            end_time: span.end_time,
            similarity: fused_similarity(query_embeddings, &embedding),
        });
    }

    Ok(windows)
}

async fn extract_window(
    source_file: &str,
    start_time: f64,
    end_time: f64,
    output_path: &Path,
) -> Result<(), CoreError> {
    let ffmpeg_path = find_ffmpeg().await?;
    let duration = (end_time - start_time).max(0.1);
    let vf = format!("scale=-2:{TARGET_RESOLUTION},fps={TARGET_FPS}");
    let output = Command::new(&ffmpeg_path)
        .arg("-y")
        .arg("-ss")
        .arg(start_time.to_string())
        .arg("-i")
        .arg(source_file)
        .arg("-t")
        .arg(duration.to_string())
        .arg("-vf")
        .arg(vf)
        .arg("-c:v")
        .arg("libx264")
        .arg("-crf")
        .arg("28")
        .arg("-preset")
        .arg("veryfast")
        .arg("-c:a")
        .arg("aac")
        .arg("-b:a")
        .arg("64k")
        .arg(output_path)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(CoreError::Io)?;

    if output.status.success() {
        return Ok(());
    }

    Err(CoreError::Ffmpeg(
        String::from_utf8_lossy(&output.stderr).to_string(),
    ))
}

async fn find_ffmpeg() -> Result<String, CoreError> {
    let output = Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .await;
    if matches!(output, Ok(ref result) if result.status.success()) {
        return Ok("ffmpeg".to_string());
    }

    for candidate in [
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
    ] {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return Ok(candidate.to_string());
        }
    }

    Err(CoreError::Ffmpeg(
        "ffmpeg not found on PATH. Install ffmpeg to continue.".into(),
    ))
}

fn micro_window_spans(start_time: f64, end_time: f64) -> Vec<WindowSpan> {
    let duration = (end_time - start_time).max(0.0);
    if duration <= EXACT_WINDOW_SECONDS {
        return vec![WindowSpan {
            start_time,
            end_time,
        }];
    }

    let mut spans = Vec::new();
    let mut current_start = start_time;

    while current_start < end_time {
        let current_end = (current_start + EXACT_WINDOW_SECONDS).min(end_time);
        spans.push(WindowSpan {
            start_time: current_start,
            end_time: current_end,
        });
        if current_end >= end_time {
            break;
        }
        current_start += EXACT_WINDOW_STRIDE_SECONDS;
    }

    spans
}

fn choose_exact_interval(windows: &[ScoredWindow], threshold: f64) -> ExactInterval {
    let active_threshold = (threshold - EXACT_THRESHOLD_MARGIN).max(0.0);
    if let Some(onset_index) = detect_stable_onset(windows, active_threshold) {
        let end_index = extend_active_run(windows, onset_index, active_threshold);
        return ExactInterval {
            start_time: windows[onset_index].start_time,
            end_time: windows[end_index].end_time,
            peak_similarity: windows[onset_index..=end_index]
                .iter()
                .map(|window| window.similarity)
                .max_by(|left, right| left.total_cmp(right))
                .unwrap_or(windows[onset_index].similarity),
        };
    }

    let best_index = windows
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.similarity.total_cmp(&right.similarity))
        .map(|(index, _)| index)
        .unwrap_or(0);
    ExactInterval {
        start_time: windows[best_index].start_time,
        end_time: windows[best_index].end_time,
        peak_similarity: windows[best_index].similarity,
    }
}

fn detect_stable_onset(windows: &[ScoredWindow], active_threshold: f64) -> Option<usize> {
    windows.iter().enumerate().find_map(|(index, _)| {
        let upper_bound = (index + STABLE_LOOKAHEAD_WINDOWS).min(windows.len());
        let active_windows = windows[index..upper_bound]
            .iter()
            .filter(|window| window.similarity >= active_threshold)
            .count();
        if active_windows >= STABLE_ACTIVE_WINDOWS && windows[index].similarity >= active_threshold
        {
            Some(index)
        } else {
            None
        }
    })
}

fn extend_active_run(windows: &[ScoredWindow], onset_index: usize, active_threshold: f64) -> usize {
    let start_time = windows[onset_index].start_time;
    let mut last_index = onset_index;

    for (index, window) in windows.iter().enumerate().skip(onset_index + 1) {
        if window.start_time - start_time >= EXACT_MAX_DURATION_SECONDS {
            break;
        }
        if window.similarity < active_threshold {
            break;
        }
        last_index = index;
    }

    last_index
}

fn fused_similarity(query_embeddings: &[Embedding], candidate: &Embedding) -> f64 {
    let mut best = f64::NEG_INFINITY;
    let mut total = 0.0;

    for query_embedding in query_embeddings {
        let similarity = cosine_similarity(query_embedding.as_slice(), candidate.as_slice());
        best = best.max(similarity);
        total += similarity;
    }

    let mean = total / query_embeddings.len() as f64;
    0.75 * best + 0.25 * mean
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(left_value, right_value)| f64::from(*left_value) * f64::from(*right_value))
        .sum()
}

struct WindowSpan {
    start_time: f64,
    end_time: f64,
}

struct ScoredWindow {
    start_time: f64,
    end_time: f64,
    similarity: f64,
}

struct ExactInterval {
    start_time: f64,
    end_time: f64,
    peak_similarity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scored_window(start_time: f64, end_time: f64, similarity: f64) -> ScoredWindow {
        ScoredWindow {
            start_time,
            end_time,
            similarity,
        }
    }

    #[test]
    fn test_choose_exact_interval_picks_earliest_stable_onset() {
        let interval = choose_exact_interval(
            &[
                scored_window(0.0, 1.0, 0.21),
                scored_window(0.5, 1.5, 0.32),
                scored_window(1.0, 2.0, 0.44),
                scored_window(1.5, 2.5, 0.51),
                scored_window(2.0, 3.0, 0.54),
            ],
            0.41,
        );

        assert_eq!(interval.start_time, 1.0);
        assert_eq!(interval.end_time, 3.0);
        assert!((interval.peak_similarity - 0.54).abs() < f64::EPSILON);
    }

    #[test]
    fn test_choose_exact_interval_falls_back_to_peak_window_without_stable_run() {
        let interval = choose_exact_interval(
            &[
                scored_window(0.0, 1.0, 0.29),
                scored_window(0.5, 1.5, 0.38),
                scored_window(1.0, 2.0, 0.36),
            ],
            0.41,
        );

        assert_eq!(interval.start_time, 0.5);
        assert_eq!(interval.end_time, 1.5);
        assert!((interval.peak_similarity - 0.38).abs() < f64::EPSILON);
    }

    #[test]
    fn test_micro_window_spans_cover_tail_of_candidate_range() {
        let spans = micro_window_spans(4.0, 6.2);

        assert_eq!(spans.first().map(|span| span.start_time), Some(4.0));
        assert_eq!(spans.last().map(|span| span.end_time), Some(6.2));
        assert!(spans.len() >= 4);
    }
}
