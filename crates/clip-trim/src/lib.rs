//! ffmpeg clip extraction with padding and re-encode fallback.
//!
//! Extracts segments from source videos using ffmpeg, trying stream-copy
//! first for speed, then falling back to re-encoding if needed.

use std::path::{Path, PathBuf};
use std::process::Stdio;

use boomerang_core::search::SearchResult;
use thiserror::Error;
use tokio::process::Command;
use tracing::{debug, warn};

/// Errors specific to clip trimming.
#[derive(Error, Debug)]
pub enum TrimError {
    #[error("invalid time range: end ({end}) must be after start ({start})")]
    InvalidRange { start: f64, end: f64 },

    #[error("ffmpeg not found")]
    FfmpegNotFound,

    #[error("ffmpeg trim failed: {0}")]
    FfmpegFailed(String),

    #[error("cannot write to output directory: {0}")]
    CannotWrite(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Extract a segment from a source video using ffmpeg.
///
/// Adds `padding` seconds before and after the match window, clamped to
/// file boundaries. Tries stream-copy first, falls back to re-encoding.
pub async fn trim_clip(
    source_file: &Path,
    start_time: f64,
    end_time: f64,
    output_path: &Path,
    padding: f64,
) -> Result<PathBuf, TrimError> {
    if end_time <= start_time {
        return Err(TrimError::InvalidRange {
            start: start_time,
            end: end_time,
        });
    }

    let ffmpeg_path = find_ffmpeg().await?;
    let duration = get_duration(source_file, &ffmpeg_path).await?;

    let padded_start = (start_time - padding).max(0.0);
    let padded_end = (end_time + padding).min(duration);
    let length = padded_end - padded_start;

    // Ensure output directory exists
    if let Some(parent) = output_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    // Attempt 1: stream-copy (fast, no quality loss)
    let _result = Command::new(&ffmpeg_path)
        .arg("-y")
        .arg("-ss")
        .arg(padded_start.to_string())
        .arg("-i")
        .arg(source_file.as_os_str())
        .arg("-t")
        .arg(length.to_string())
        .arg("-c")
        .arg("copy")
        .arg(output_path.as_os_str())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| TrimError::FfmpegFailed(e.to_string()))?;

    if output_path.exists() {
        let meta = tokio::fs::metadata(output_path).await?;
        if meta.len() > 1024 {
            debug!(path = %output_path.display(), "clip trimmed via stream-copy");
            return Ok(output_path.to_path_buf());
        }
    }

    // Attempt 2: re-encode (more compatible)
    warn!("stream-copy failed, falling back to re-encode");
    let result = Command::new(&ffmpeg_path)
        .arg("-y")
        .arg("-i")
        .arg(source_file.as_os_str())
        .arg("-ss")
        .arg(padded_start.to_string())
        .arg("-t")
        .arg(length.to_string())
        .arg("-c:v")
        .arg("mpeg4")
        .arg("-q:v")
        .arg("5")
        .arg("-c:a")
        .arg("aac")
        .arg("-b:a")
        .arg("128k")
        .arg(output_path.as_os_str())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| TrimError::FfmpegFailed(e.to_string()))?;

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return Err(TrimError::FfmpegFailed(stderr.to_string()));
    }

    debug!(path = %output_path.display(), "clip trimmed via re-encode");
    Ok(output_path.to_path_buf())
}

/// Trim the top N search results and save them to an output directory.
pub async fn trim_top_results(
    results: &[SearchResult],
    output_dir: &Path,
    count: usize,
) -> Result<Vec<PathBuf>, TrimError> {
    if results.is_empty() {
        return Ok(vec![]);
    }

    let mut paths = Vec::with_capacity(count.min(results.len()));

    for result in results.iter().take(count) {
        let filename = safe_filename(&result.source_file, result.start_time, result.end_time);
        let output_path = output_dir.join(&filename);

        let clip = trim_clip(
            Path::new(&result.source_file),
            result.start_time,
            result.end_time,
            &output_path,
            2.0,
        )
        .await?;

        paths.push(clip);
    }

    Ok(paths)
}

/// Format seconds as e.g. "02m15s".
fn fmt_time(seconds: f64) -> String {
    let total = seconds as u64;
    let m = total / 60;
    let s = total % 60;
    format!("{m:02}m{s:02}s")
}

/// Build a filesystem-safe descriptive filename.
fn safe_filename(source_file: &str, start: f64, end: f64) -> String {
    let base = Path::new(source_file)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("clip");

    let safe_base: String = base
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();

    format!("match_{}_{}-{}.mp4", safe_base, fmt_time(start), fmt_time(end))
}

/// Find a working ffmpeg binary.
async fn find_ffmpeg() -> Result<String, TrimError> {
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

    for candidate in &["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg"] {
        if Path::new(candidate).exists() {
            return Ok(candidate.to_string());
        }
    }

    Err(TrimError::FfmpegNotFound)
}

/// Get video duration in seconds.
async fn get_duration(video_path: &Path, ffmpeg_path: &str) -> Result<f64, TrimError> {
    let output = Command::new(ffmpeg_path)
        .arg("-i")
        .arg(video_path.as_os_str())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| TrimError::FfmpegFailed(e.to_string()))?;

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Parse "Duration: HH:MM:SS.xx"
    for line in stderr.lines() {
        if let Some(rest) = line.trim().strip_prefix("Duration:") {
            let parts: Vec<&str> = rest.trim().splitn(3, ':').collect();
            if parts.len() == 3 {
                let h: f64 = parts[0].trim().parse().unwrap_or(0.0);
                let m: f64 = parts[1].trim().parse().unwrap_or(0.0);
                let s: f64 = parts[2]
                    .split(|c: char| !c.is_ascii_digit() && c != '.')
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                return Ok(h * 3600.0 + m * 60.0 + s);
            }
        }
    }

    Err(TrimError::FfmpegFailed(
        "could not parse duration from ffmpeg output".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fmt_time() {
        assert_eq!(fmt_time(0.0), "00m00s");
        assert_eq!(fmt_time(65.0), "01m05s");
        assert_eq!(fmt_time(125.0), "02m05s");
    }

    #[test]
    fn test_safe_filename() {
        let name = safe_filename("/videos/front_2024-01-15_14-30.mp4", 135.0, 165.0);
        assert!(name.starts_with("match_"));
        assert!(name.ends_with(".mp4"));
        assert!(name.contains("02m15s"));
        assert!(name.contains("02m45s"));
    }
}
