//! Still-frame detection for video chunks.
//!
//! Detects chunks with no meaningful visual change (e.g., parked car)
//! by comparing JPEG file sizes across sampled frames.

use std::path::Path;
use std::process::Stdio;

use tokio::process::Command;
use tracing::debug;

use crate::chunker::ChunkingError;

/// Check if a video chunk contains mostly still frames.
///
/// Extracts 3 evenly-spaced frames as JPEG and compares file sizes.
/// Similar JPEG sizes indicate similar visual content (still scene).
pub async fn is_still_frame(chunk_path: &Path, threshold: f64) -> Result<bool, ChunkingError> {
    let ffmpeg_path = super::chunker::ffmpeg::find_ffmpeg().await?;

    // Get total frame count
    let output = Command::new(&ffmpeg_path)
        .arg("-i")
        .arg(chunk_path.as_os_str())
        .arg("-map")
        .arg("0:v:0")
        .arg("-c")
        .arg("copy")
        .arg("-f")
        .arg("null")
        .arg("-")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| ChunkingError::ExtractionFailed(e.to_string()))?;

    let stderr = String::from_utf8_lossy(&output.stderr);

    let total_frames = parse_frame_count(&stderr);
    if total_frames < 3 {
        return Ok(false);
    }

    let f1 = total_frames / 3;
    let f2 = 2 * total_frames / 3;

    // Extract 3 frames as JPEG to a temp dir
    let tmp_dir =
        tempfile::tempdir().map_err(|e| ChunkingError::ExtractionFailed(e.to_string()))?;
    let out_pattern = tmp_dir.path().join("frame_%03d.jpg");

    let vf = format!("select=eq(n\\,0)+eq(n\\,{})+eq(n\\,{})", f1, f2);

    let _ = Command::new(&ffmpeg_path)
        .arg("-y")
        .arg("-i")
        .arg(chunk_path.as_os_str())
        .arg("-vf")
        .arg(&vf)
        .arg("-vsync")
        .arg("vfr")
        .arg(&out_pattern)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .await;

    // Collect frame file sizes
    let mut sizes: Vec<u64> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(tmp_dir.path()) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                sizes.push(meta.len());
            }
        }
    }

    if sizes.len() < 2 {
        return Ok(false);
    }

    let min_size = *sizes.iter().min().unwrap_or(&0);
    let max_size = *sizes.iter().max().unwrap_or(&0);

    if max_size == 0 {
        return Ok(false);
    }

    let ratio = min_size as f64 / max_size as f64;
    debug!(min = min_size, max = max_size, ratio, "still-frame check");

    Ok(ratio >= threshold)
}

fn parse_frame_count(stderr: &str) -> u32 {
    // Try "frame= NNN" pattern first
    for line in stderr.lines() {
        if let Some(rest) = line.strip_prefix("frame=") {
            if let Ok(n) = rest.split_whitespace().next().unwrap_or("0").parse::<u32>() {
                return n;
            }
        }
    }

    // Fall back to duration * fps estimation
    let fps = stderr
        .lines()
        .find_map(|line| {
            let line = line.trim();
            if line.contains("fps") {
                line.split_whitespace()
                    .find(|w| w.parse::<f64>().is_ok())
                    .and_then(|w| w.parse::<f64>().ok())
            } else {
                None
            }
        })
        .unwrap_or(30.0);

    // Parse duration
    let duration = stderr
        .lines()
        .find_map(|line| {
            let line = line.trim();
            if line.starts_with("Duration:") {
                let parts: Vec<&str> = line
                    .strip_prefix("Duration:")
                    .unwrap_or("")
                    .trim()
                    .splitn(3, ':')
                    .collect();
                if parts.len() == 3 {
                    let h: f64 = parts[0].trim().parse().ok()?;
                    let m: f64 = parts[1].trim().parse().ok()?;
                    let s: f64 = parts[2]
                        .split(|c: char| !c.is_ascii_digit() && c != '.')
                        .next()?
                        .parse()
                        .ok()?;
                    Some(h * 3600.0 + m * 60.0 + s)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .unwrap_or(0.0);

    (duration * fps) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_frame_count_from_frame_line() {
        let stderr = "frame=  150 fps=30 q=-1.0 Lsize=N/A time=00:00:05.00 bitrate=N/A speed=10x";
        assert_eq!(parse_frame_count(stderr), 150);
    }

    #[test]
    fn test_parse_frame_count_fallback() {
        let stderr = "  Stream #0:0: Video: h264, 30 fps\n  Duration: 00:00:05.00";
        let count = parse_frame_count(stderr);
        assert!(count > 0);
    }
}
