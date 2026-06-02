//! Embedding vector type and embedder trait.

use async_trait::async_trait;

use crate::error::CoreError;
use crate::types::EmbeddingSpace;

/// A normalized embedding vector.
#[derive(Debug, Clone)]
pub struct Embedding {
    pub data: Vec<f32>,
    pub dimensions: usize,
}

impl Embedding {
    pub fn new(data: Vec<f32>) -> Self {
        let dimensions = data.len();
        Self { data, dimensions }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// Trait for embedding backends that convert video/text/image to vectors.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed a video chunk file into a vector.
    async fn embed_video(&self, chunk_path: &str) -> Result<Embedding, CoreError>;

    /// Embed a natural language query into a vector.
    async fn embed_query(&self, query: &str) -> Result<Embedding, CoreError>;

    /// Embed an image file into a vector.
    async fn embed_image(&self, image_path: &str) -> Result<Embedding, CoreError>;

    /// Number of dimensions this embedder produces.
    fn dimensions(&self) -> usize;

    /// Backend identifier.
    fn backend_name(&self) -> &'static str;

    /// Optional model identifier for backends that expose multiple models.
    fn model_name(&self) -> Option<&str> {
        None
    }

    /// Fully qualified vector space emitted by this embedder.
    fn embedding_space(&self) -> Result<EmbeddingSpace, CoreError>;
}
