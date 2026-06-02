//! Vector store trait for persisting and querying embeddings.

use async_trait::async_trait;

use crate::chunk::ChunkMetadata;
use crate::embedding::Embedding;
use crate::error::CoreError;
use crate::search::SearchResult;
use crate::types::{ChunkId, EmbeddingSpace};

/// Statistics about the vector store.
#[derive(Debug, Clone)]
pub struct StoreStats {
    pub total_chunks: usize,
    pub unique_source_files: usize,
    pub source_files: Vec<String>,
    pub embedding_space: EmbeddingSpace,
}

/// Trait for vector storage backends.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Store a single embedding with metadata.
    async fn add(
        &self,
        id: &ChunkId,
        embedding: &Embedding,
        metadata: &ChunkMetadata,
    ) -> Result<(), CoreError>;

    /// Batch-store multiple embeddings.
    async fn add_batch(
        &self,
        entries: &[(ChunkId, Embedding, ChunkMetadata)],
    ) -> Result<(), CoreError>;

    /// Search for nearest neighbors by embedding.
    async fn search(&self, query: &Embedding, limit: usize)
        -> Result<Vec<SearchResult>, CoreError>;

    /// Check if a chunk ID already exists.
    async fn contains(&self, id: &ChunkId) -> Result<bool, CoreError>;

    /// Check if any chunks from a source file are already stored.
    async fn is_file_indexed(&self, source_file: &str) -> Result<bool, CoreError>;

    /// Remove all chunks for a given source file.
    async fn remove_file(&self, source_file: &str) -> Result<usize, CoreError>;

    /// Get all embeddings and metadata (for highlight scoring).
    async fn fetch_all(&self) -> Result<(Vec<Embedding>, Vec<ChunkMetadata>), CoreError>;

    /// Return store statistics.
    async fn stats(&self) -> Result<StoreStats, CoreError>;

    /// Total number of stored chunks.
    async fn count(&self) -> Result<usize, CoreError>;

    /// Delete all data.
    async fn clear(&self) -> Result<(), CoreError>;
}
