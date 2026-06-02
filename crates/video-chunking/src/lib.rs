//! Video chunking, still-frame detection, and preprocessing via ffmpeg.
//!
//! Splits video files into overlapping segments, detects static scenes,
//! and preprocesses chunks (downscale, reduce fps) for efficient embedding.

pub mod chunker;
pub mod scanner;
pub mod still_frame;

use std::path::Path;

/// Supported video file extensions.
pub const SUPPORTED_EXTENSIONS: &[&str] = &["mp4", "mov"];

/// Check if a file path has a supported video extension.
pub fn is_supported_video(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| SUPPORTED_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}
