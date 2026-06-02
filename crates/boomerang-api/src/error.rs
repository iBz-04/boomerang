// Error handling definitions and IntoResponse mapping for the API server.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

/// Error type returned by the API endpoints.
#[derive(thiserror::Error, Debug)]
pub enum ApiError {
    #[error("Multipart error: {0}")]
    Multipart(#[from] axum::extract::multipart::MultipartError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Core error: {0}")]
    Core(#[from] boomerang_core::error::CoreError),

    #[error("Clip trim error: {0}")]
    Trim(#[from] clip_trim::TrimError),

    #[error("Chunking error: {0}")]
    Chunking(#[from] video_chunking::chunker::ChunkingError),

    #[error("Internal server error: {0}")]
    Internal(#[from] anyhow::Error),

    #[error("Bad request: {0}")]
    BadRequest(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ApiError::Multipart(ref e) => (StatusCode::BAD_REQUEST, e.to_string()),
            ApiError::BadRequest(ref msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            ApiError::Io(ref e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            ApiError::Core(ref e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            ApiError::Trim(ref e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            ApiError::Chunking(ref e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            ApiError::Internal(ref e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        };

        let body = Json(json!({
            "error": message,
        }));

        (status, body).into_response()
    }
}
