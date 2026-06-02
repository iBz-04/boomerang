//! Vector store backed by Qdrant.
//!
//! Provides persistent storage and similarity search for video chunk
//! embeddings using Qdrant's vector database.

pub mod qdrant_store;

use boomerang_core::error::CoreError;
use boomerang_core::store::VectorStore;

/// Create a vector store for the given backend.
pub async fn create_store(
    backend: &str,
    _db_path: Option<&str>,
) -> Result<Box<dyn VectorStore>, CoreError> {
    match backend {
        "qdrant" => {
            let url = std::env::var("QDRANT_URL")
                .unwrap_or_else(|_| "http://localhost:6333".into());
            Ok(Box::new(qdrant_store::QdrantStore::connect(&url).await?))
        }
        _ => Err(CoreError::Config(format!(
            "unknown vector store backend: {backend}"
        ))),
    }
}
