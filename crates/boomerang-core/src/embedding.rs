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
    pub fn new(mut data: Vec<f32>) -> Result<Self, CoreError> {
        if data.is_empty() {
            return Err(CoreError::InvalidEmbedding(
                "embedding vector must not be empty".into(),
            ));
        }

        let norm = data
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();
        if norm <= f64::EPSILON {
            return Err(CoreError::InvalidEmbedding(
                "embedding vector must have non-zero norm".into(),
            ));
        }

        let inverse_norm = (1.0 / norm) as f32;
        for value in &mut data {
            *value *= inverse_norm;
        }

        let dimensions = data.len();
        Ok(Self { data, dimensions })
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_normalizes_embedding_to_unit_length() {
        let embedding = Embedding::new(vec![3.0, 4.0]).expect("embedding should normalize");
        let norm = embedding
            .as_slice()
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>()
            .sqrt();

        assert!((norm - 1.0).abs() < 1e-6);
        assert_eq!(embedding.dimensions, 2);
    }

    #[test]
    fn test_new_rejects_zero_norm_embedding() {
        let error = Embedding::new(vec![0.0, 0.0]).expect_err("zero vector should fail");

        assert!(matches!(error, CoreError::InvalidEmbedding(_)));
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
