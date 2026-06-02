//! Video chunk domain model.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::types::{ChunkId, EmbeddingBackend, TimeRange};

/// A segment of video extracted from a source file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoChunk {
    /// Unique identifier derived from source file + start time.
    pub id: ChunkId,
    /// Absolute path to the source video file.
    pub source_file: String,
    /// Time range within the source file.
    pub time_range: TimeRange,
    /// Path to the temporary chunk file on disk.
    pub chunk_path: String,
}

impl VideoChunk {
    pub fn new(
        id: ChunkId,
        source_file: String,
        start_time: f64,
        end_time: f64,
        chunk_path: String,
    ) -> Self {
        Self {
            id,
            source_file,
            time_range: TimeRange::new(start_time, end_time),
            chunk_path,
        }
    }
}

/// Metadata stored alongside an embedding in the vector store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    pub source_file: String,
    pub start_time: f64,
    pub end_time: f64,
    pub indexed_at: DateTime<Utc>,
    pub backend: EmbeddingBackend,
    pub model: Option<String>,
    pub dimensions: usize,
}

/// A chunk with its embedding vector, ready for storage.
#[derive(Debug, Clone)]
pub struct IndexedChunk {
    pub chunk: VideoChunk,
    pub embedding: Vec<f32>,
    pub metadata: ChunkMetadata,
}

/// Parameters controlling video chunking behavior.
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    /// Duration of each chunk in seconds.
    pub chunk_duration: u32,
    /// Overlap between consecutive chunks in seconds.
    pub overlap: u32,
    /// Target height for preprocessing (width scales to maintain aspect ratio).
    pub target_resolution: u32,
    /// Target frames per second for preprocessing.
    pub target_fps: u32,
    /// Whether to skip preprocessing entirely.
    pub skip_preprocess: bool,
    /// Whether to skip still-frame detection.
    pub skip_still_detection: bool,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_duration: 30,
            overlap: 5,
            target_resolution: 480,
            target_fps: 5,
            skip_preprocess: false,
            skip_still_detection: false,
        }
    }
}
