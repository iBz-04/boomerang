//! ffmpeg-based video chunking and preprocessing.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use boomerang_core::chunk::{ChunkingConfig, VideoChunk};
use boomerang_core::types::{ChunkId, TimeRange};
use thiserror::Error;
use tokio::process::Command;
use tracing::{debug, warn};

/// Errors specific to video chunking operations.
#[derive(Error, Debug)]
pub enum ChunkingError {
    #[error("ffmpeg not found: {0}")]
    FfmpegNotFound(String),

    #[error("failed to get video duration: {0}")]
    DurationParse(String),

    #[error("ffmpeg chunk extraction failed: {0}")]
    ExtractionFailed(String),

    #[error("invalid chunking parameters: overlap ({overlap}s) must be less than chunk_duration ({chunk_duration}s)")]
    InvalidOverlap { overlap: u32, chunk_duration: u32 },
}

/// Compute expected chunk time spans without actually splitting the video.
pub fn expected_chunk_spans(
    duration_secs: f64,
    chunk_duration: u32,
    overlap: u32,
) -> Result<Vec<TimeRange>, ChunkingError> {
    if overlap >= chunk_duration {
        return Err(ChunkingError::InvalidOverlap {
            overlap,
            chunk_duration,
        });
    }

    let chunk_dur = chunk_duration as f64;
    let overlap_f = overlap as f64;

    if duration_secs <= chunk_dur {
        return Ok(vec![TimeRange::new(0.0, duration_secs)]);
    }

    let step = chunk_dur - overlap_f;
    let mut spans = Vec::new();
    let mut start = 0.0;

    while start < duration_secs {
        let end = (start + chunk_dur).min(duration_secs);
        spans.push(TimeRange::new(start, end));
        start += step;
        if start + overlap_f >= duration_secs {
            break;
        }
    }

    Ok(spans)
}

/// Split a video into overlapping chunks using ffmpeg stream copy.
pub async fn chunk_video(
    video_path: &Path,
    config: &ChunkingConfig,
    output_dir: &Path,
) -> Result<Vec<VideoChunk>, ChunkingError> {
    let video_path = video_path.canonicalize().map_err(|e| {
        ChunkingError::ExtractionFailed(format!("cannot resolve path: {e}"))
    })?;

    let ffmpeg_path = ffmpeg::find_ffmpeg().await?;
    let duration = ffmpeg::get_duration(&video_path, &ffmpeg_path).await?;

    let spans = expected_chunk_spans(duration, config.chunk_duration, config.overlap)?;
    let source_file = video_path.to_string_lossy().to_string();

    let mut chunks = Vec::with_capacity(spans.len());

    for (idx, span) in spans.iter().enumerate() {
        let chunk_filename = format!("chunk_{:03}.mp4", idx);
        let chunk_path = output_dir.join(&chunk_filename);

        let length = span.end - span.start;

        let output = Command::new(&ffmpeg_path)
            .arg("-y")
            .arg("-ss")
            .arg(span.start.to_string())
            .arg("-i")
            .arg(video_path.as_os_str())
            .arg("-t")
            .arg(length.to_string())
            .arg("-c")
            .arg("copy")
            .arg(&chunk_path)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| ChunkingError::ExtractionFailed(format!("ffmpeg spawn failed: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(ChunkingError::ExtractionFailed(stderr.to_string()));
        }

        let chunk_id = make_chunk_id(&source_file, span.start);
        debug!(
            chunk = %chunk_id,
            start = span.start,
            end = span.end,
            "extracted chunk"
        );

        chunks.push(VideoChunk::new(
            chunk_id,
            source_file.clone(),
            span.start,
            span.end,
            chunk_path.to_string_lossy().to_string(),
        ));
    }

    Ok(chunks)
}

/// Preprocess a video chunk: downscale and reduce frame rate.
pub async fn preprocess_chunk(
    chunk_path: &Path,
    config: &ChunkingConfig,
) -> Result<PathBuf, ChunkingError> {
    if config.skip_preprocess {
        return Ok(chunk_path.to_path_buf());
    }

    let ffmpeg_path = ffmpeg::find_ffmpeg().await?;

    let stem = chunk_path.file_stem().unwrap_or_default().to_string_lossy();
    let parent = chunk_path.parent().unwrap_or(Path::new("."));
    let out_path = parent.join(format!("{}_preprocessed.mp4", stem));

    let vf = format!(
        "scale=-2:{},fps={}",
        config.target_resolution, config.target_fps
    );

    let output = Command::new(&ffmpeg_path)
        .arg("-y")
        .arg("-i")
        .arg(chunk_path.as_os_str())
        .arg("-vf")
        .arg(&vf)
        .arg("-c:v")
        .arg("libx264")
        .arg("-crf")
        .arg("28")
        .arg("-c:a")
        .arg("aac")
        .arg("-b:a")
        .arg("64k")
        .arg(&out_path)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| ChunkingError::ExtractionFailed(format!("preprocess spawn failed: {e}")))?;

    if !output.status.success() {
        warn!("preprocessing failed, using original chunk");
        return Ok(chunk_path.to_path_buf());
    }

    debug!(original = %chunk_path.display(), preprocessed = %out_path.display(), "preprocessed chunk");
    Ok(out_path)
}

/// Generate a deterministic chunk ID from source file + start time.
///
/// Produces a UUIDv5 (namespace + name) so the ID is stable across runs and
/// is accepted by vector stores that require UUID or integer point IDs.
pub fn make_chunk_id(source_file: &str, start_time: f64) -> ChunkId {
    let name = format!("{source_file}@{}", start_time.to_bits());
    let uuid = uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_URL, name.as_bytes());
    ChunkId(uuid.to_string())
}

/// Internal ffmpeg utilities.
pub(crate) mod ffmpeg {
    use std::path::Path;
    use std::process::Stdio;
    use tokio::process::Command;

    use super::ChunkingError;

    /// Find a working ffmpeg binary on the system.
    pub async fn find_ffmpeg() -> Result<String, ChunkingError> {
        // Try system ffmpeg first
        let output = Command::new("ffmpeg")
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .await;

        match output {
            Ok(o) if o.status.success() => return Ok("ffmpeg".to_string()),
            _ => {}
        }

        // Try common paths
        for candidate in &["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg"] {
            if Path::new(candidate).exists() {
                return Ok(candidate.to_string());
            }
        }

        Err(ChunkingError::FfmpegNotFound(
            "ffmpeg not found on PATH. Install ffmpeg to continue.".to_string(),
        ))
    }

    /// Get video duration in seconds using ffprobe or ffmpeg.
    pub async fn get_duration(video_path: &Path, ffmpeg_path: &str) -> Result<f64, ChunkingError> {
        // Try ffprobe first
        let ffprobe = ffmpeg_path.replace("ffmpeg", "ffprobe");
        if Path::new(&ffprobe).exists() {
            if let Ok(dur) = probe_duration(video_path, &ffprobe).await {
                return Ok(dur);
            }
        }

        // Fall back to ffmpeg stderr parsing
        parse_duration_from_ffmpeg(video_path, ffmpeg_path).await
    }

    async fn probe_duration(video_path: &Path, ffprobe_path: &str) -> Result<f64, ChunkingError> {
        let output = Command::new(ffprobe_path)
            .arg("-v")
            .arg("quiet")
            .arg("-print_format")
            .arg("json")
            .arg("-show_format")
            .arg(video_path.as_os_str())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .await
            .map_err(|e: std::io::Error| ChunkingError::DurationParse(e.to_string()))?;

        let info: serde_json::Value = serde_json::from_slice(&output.stdout)
            .map_err(|e| ChunkingError::DurationParse(e.to_string()))?;

        info["format"]["duration"]
            .as_str()
            .and_then(|s| s.parse::<f64>().ok())
            .ok_or_else(|| ChunkingError::DurationParse("no duration field".to_string()))
    }

    async fn parse_duration_from_ffmpeg(
        video_path: &Path,
        ffmpeg_path: &str,
    ) -> Result<f64, ChunkingError> {
        let output = Command::new(ffmpeg_path)
            .arg("-i")
            .arg(video_path.as_os_str())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|e| ChunkingError::DurationParse(e.to_string()))?;

        let stderr = String::from_utf8_lossy(&output.stderr);
        let duration = super::parse_duration_line(&stderr)
            .ok_or_else(|| ChunkingError::DurationParse("no duration in ffmpeg output".to_string()))?;
        Ok(duration)
    }
}

/// Parse "Duration: HH:MM:SS.xx" from ffmpeg stderr.
fn parse_duration_line(stderr: &str) -> Option<f64> {
    let prefix = "Duration:";
    let pos = stderr.find(prefix)?;
    let rest = stderr[pos + prefix.len()..].trim_start();
    let parts: Vec<&str> = rest.splitn(3, ':').collect();
    if parts.len() < 3 {
        return None;
    }
    let h: f64 = parts[0].trim().parse().ok()?;
    let m: f64 = parts[1].trim().parse().ok()?;
    let s: f64 = parts[2]
        .split(|c: char| !c.is_ascii_digit() && c != '.')
        .next()?
        .parse()
        .ok()?;
    Some(h * 3600.0 + m * 60.0 + s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expected_chunk_spans_short_video() {
        let spans = expected_chunk_spans(10.0, 30, 5).unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].start, 0.0);
        assert_eq!(spans[0].end, 10.0);
    }

    #[test]
    fn test_expected_chunk_spans_exact() {
        let spans = expected_chunk_spans(30.0, 30, 5).unwrap();
        assert_eq!(spans.len(), 1);
    }

    #[test]
    fn test_expected_chunk_spans_long_video() {
        let spans = expected_chunk_spans(100.0, 30, 5).unwrap();
        assert!(spans.len() >= 3);
        // First chunk starts at 0
        assert_eq!(spans[0].start, 0.0);
        assert_eq!(spans[0].end, 30.0);
        // Overlap means second chunk starts at 25
        assert_eq!(spans[1].start, 25.0);
    }

    #[test]
    fn test_invalid_overlap_rejected() {
        let result = expected_chunk_spans(100.0, 30, 30);
        assert!(result.is_err());
    }

    #[test]
    fn test_make_chunk_id_deterministic() {
        let id1 = make_chunk_id("/videos/test.mp4", 10.0);
        let id2 = make_chunk_id("/videos/test.mp4", 10.0);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_make_chunk_id_different_for_different_inputs() {
        let id1 = make_chunk_id("/videos/test.mp4", 10.0);
        let id2 = make_chunk_id("/videos/test.mp4", 20.0);
        assert_ne!(id1, id2);
    }
}
