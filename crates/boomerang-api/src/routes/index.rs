// Endpoint for uploading and indexing a video file (POST /index).

use axum::{
    extract::{Multipart, State},
    response::IntoResponse,
    Json,
};
use serde::Serialize;
use std::path::Path;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tracing::info;

use crate::error::ApiError;
use crate::state::AppState;

const SEARCH_CHUNK_DURATION_SECONDS: u32 = 8;
const SEARCH_CHUNK_OVERLAP_SECONDS: u32 = 2;
const SEARCH_TARGET_FPS: u32 = 4;

/// API response for the video indexing endpoint.
#[derive(Serialize)]
pub struct IndexResponse {
    pub file: String,
    pub source_file: String,
    pub chunks: usize,
}

/// Handle video upload and semantic indexing.
pub async fn index_handler(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, ApiError> {
    let mut filename = None;
    let mut video_bytes = None;

    while let Some(f) = multipart.next_field().await? {
        if f.name() == Some("video") {
            let name = f.file_name().ok_or_else(|| {
                ApiError::BadRequest("video upload must include a file name".to_string())
            })?;
            filename = Some(name.to_string());
            video_bytes = Some(f.bytes().await?);
            break;
        }
    }

    let filename = match filename {
        Some(name) => name,
        None => {
            return Err(ApiError::BadRequest(
                "missing 'video' field in multipart form".to_string(),
            ))
        }
    };

    let video_bytes = match video_bytes {
        Some(bytes) => bytes,
        None => {
            return Err(ApiError::BadRequest(
                "missing video bytes in multipart field".to_string(),
            ))
        }
    };
    let video_path = state.upload_dir.join(&filename);

    if let Some(parent) = video_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let mut file = File::create(&video_path).await?;
    file.write_all(&video_bytes).await?;
    file.flush().await?;

    info!(path = %video_path.display(), "saved uploaded video file");

    let config = boomerang_core::chunk::ChunkingConfig {
        chunk_duration: SEARCH_CHUNK_DURATION_SECONDS,
        overlap: SEARCH_CHUNK_OVERLAP_SECONDS,
        target_resolution: 480,
        target_fps: SEARCH_TARGET_FPS,
        skip_preprocess: false,
        skip_still_detection: false,
    };

    let embedder = semantic_embed::create_embedder(&state.backend, state.model.as_deref())?;
    let embedding_space = embedder.embedding_space()?;
    let store = vector_store::create_store("qdrant", &embedding_space).await?;

    let video_file_str = video_path.to_string_lossy().to_string();
    if store.is_file_indexed(&video_file_str).await? {
        info!(file = %video_file_str, "file already indexed; clearing old entries to re-index");
        store.remove_file(&video_file_str).await?;
    }

    let tmp_dir = tempfile::tempdir()?;
    let chunks = video_chunking::chunker::chunk_video(&video_path, &config, tmp_dir.path())
        .await
        .map_err(|e| anyhow::anyhow!("failed to chunk video: {e}"))?;

    let mut indexed_count = 0;
    let total = chunks.len();

    for (i, chunk) in chunks.iter().enumerate() {
        if !config.skip_still_detection {
            let chunk_p = Path::new(&chunk.chunk_path);
            if video_chunking::still_frame::is_still_frame(chunk_p, 0.98).await? {
                info!(chunk = i + 1, total, "skipping still frame chunk");
                continue;
            }
        }

        let chunk_p = Path::new(&chunk.chunk_path);
        let processed = video_chunking::chunker::preprocess_chunk(chunk_p, &config)
            .await
            .map_err(|e| anyhow::anyhow!("preprocessing failed: {e}"))?;

        let embedding = embedder
            .embed_video(&processed.to_string_lossy())
            .await
            .map_err(|e| anyhow::anyhow!("failed to embed chunk: {e}"))?;

        let metadata = boomerang_core::chunk::ChunkMetadata {
            source_file: chunk.source_file.clone(),
            start_time: chunk.time_range.start,
            end_time: chunk.time_range.end,
            indexed_at: chrono::Utc::now(),
            backend: embedding_space.backend,
            model: embedding_space.model.clone(),
            dimensions: embedding_space.dimensions,
        };

        store.add(&chunk.id, &embedding, &metadata).await?;
        indexed_count += 1;
        info!(chunk = i + 1, total, "indexed chunk");
    }

    Ok(Json(IndexResponse {
        file: filename,
        source_file: video_file_str,
        chunks: indexed_count,
    }))
}
