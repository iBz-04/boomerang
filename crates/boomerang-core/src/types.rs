//! Newtype wrappers for domain identifiers and dimensions.

use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Unique identifier for a video chunk, derived from source file + start time.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkId(pub String);

impl ChunkId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ChunkId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Video resolution in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resolution {
    pub width: u32,
    pub height: u32,
}

impl Resolution {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

/// Duration in seconds, stored as f64 for precision.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DurationSecs(pub f64);

impl DurationSecs {
    pub const fn from_secs(secs: f64) -> Self {
        Self(secs)
    }

    pub fn as_secs_f64(&self) -> f64 {
        self.0
    }
}

/// Time range within a video, in seconds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: f64,
    pub end: f64,
}

impl TimeRange {
    pub fn new(start: f64, end: f64) -> Self {
        Self { start, end }
    }

    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

/// Backend type for embedding generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingBackend {
    Gemini,
    Local,
    QwenCloud,
}

impl EmbeddingBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Gemini => "gemini",
            Self::Local => "local",
            Self::QwenCloud => "qwen-cloud",
        }
    }
}

impl std::fmt::Display for EmbeddingBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for EmbeddingBackend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "gemini" => Ok(Self::Gemini),
            "local" => Ok(Self::Local),
            "qwen-cloud" => Ok(Self::QwenCloud),
            _ => Err(format!("unknown embedding backend: {s}")),
        }
    }
}

/// A fully qualified embedding space for one backend/model combination.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingSpace {
    pub backend: EmbeddingBackend,
    pub model: Option<String>,
    pub dimensions: usize,
}

impl EmbeddingSpace {
    pub fn new(backend: EmbeddingBackend, model: Option<String>, dimensions: usize) -> Self {
        Self {
            backend,
            model,
            dimensions,
        }
    }
}

/// Scoring method for highlight/anomaly detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringMethod {
    Centroid,
    Knn,
    Lof,
}

impl ScoringMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Centroid => "centroid",
            Self::Knn => "knn",
            Self::Lof => "lof",
        }
    }
}

/// Mode for query-constrained anomaly detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgainstMode {
    /// Rank anomalies only among top matches of the query.
    Within,
    /// Score over full index, weighted by query similarity.
    Global,
}
