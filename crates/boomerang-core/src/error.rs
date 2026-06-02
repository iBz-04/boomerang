//! Error types for the boomerang workspace.

use thiserror::Error;

/// Errors that can occur across boomerang crates.
#[derive(Error, Debug)]
pub enum CoreError {
    #[error("video file not found: {0}")]
    FileNotFound(String),

    #[error("ffmpeg error: {0}")]
    Ffmpeg(String),

    #[error("embedding API error: {0}")]
    EmbeddingApi(String),

    #[error("API key not configured for backend: {0}")]
    MissingApiKey(String),

    #[error("API quota exceeded: {0}")]
    QuotaExceeded(String),

    #[error("backend mismatch: index was built with {indexed}, but {requested} was requested")]
    BackendMismatch {
        indexed: String,
        requested: String,
    },

    #[error("vector store error: {0}")]
    Store(String),

    #[error("invalid configuration: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("{0}")]
    Other(String),
}
