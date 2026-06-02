//! Vector store backed by Qdrant.
//!
//! Provides persistent storage and similarity search for video chunk
//! embeddings using Qdrant's vector database.

mod collection_name;
pub mod qdrant_store;

use boomerang_core::error::CoreError;
use boomerang_core::store::VectorStore;
use boomerang_core::types::EmbeddingSpace;

/// Create a vector store for the given backend.
pub async fn create_store(
    backend: &str,
    embedding_space: &EmbeddingSpace,
) -> Result<Box<dyn VectorStore>, CoreError> {
    match backend {
        "qdrant" => {
            let url =
                std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".into());
            Ok(Box::new(
                qdrant_store::QdrantStore::connect(&url, embedding_space.clone()).await?,
            ))
        }
        _ => Err(CoreError::Config(format!(
            "unknown vector store backend: {backend}"
        ))),
    }
}

/// Detect all embedding spaces that currently contain indexed data.
pub async fn list_spaces(backend: &str) -> Result<Vec<EmbeddingSpace>, CoreError> {
    match backend {
        "qdrant" => {
            let url =
                std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".into());
            qdrant_store::QdrantStore::list_spaces(&url).await
        }
        _ => Err(CoreError::Config(format!(
            "unknown vector store backend: {backend}"
        ))),
    }
}

/// Detect the one indexed embedding space currently available.
pub async fn detect_space(backend: &str) -> Result<Option<EmbeddingSpace>, CoreError> {
    let spaces = list_spaces(backend).await?;
    match spaces.len() {
        0 => Ok(None),
        1 => Ok(spaces.into_iter().next()),
        _ => Err(CoreError::Config(
            "multiple indexed embedding spaces found; specify --backend and --model explicitly"
                .to_string(),
        )),
    }
}
