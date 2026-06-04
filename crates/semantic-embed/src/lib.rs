//! Embedding backends for video, text, and image content.
//!
//! Provides the `Embedder` trait implementation for:
//! - Gemini Embedding API (default)
//! - Qwen Cloud (DashScope)
//! - Local Qwen3-VL model (to do :))

pub mod gemini;
pub mod query_expand;
pub mod qwen_cloud;

use boomerang_core::embedding::Embedder;
use boomerang_core::error::CoreError;
use boomerang_core::types::EmbeddingBackend;

/// Create an embedder for the given backend.
pub fn create_embedder(backend: &str, model: Option<&str>) -> Result<Box<dyn Embedder>, CoreError> {
    match backend.parse::<EmbeddingBackend>() {
        Ok(EmbeddingBackend::Gemini) => {
            let api_key = std::env::var("GEMINI_API_KEY")
                .map_err(|_| CoreError::MissingApiKey("GEMINI_API_KEY not set".into()))?;
            Ok(Box::new(gemini::GeminiEmbedder::new(api_key)))
        }
        Ok(EmbeddingBackend::QwenCloud) => {
            let api_key = std::env::var("DASHSCOPE_API_KEY")
                .map_err(|_| CoreError::MissingApiKey("DASHSCOPE_API_KEY not set".into()))?;
            let model = model.unwrap_or("qwen3-vl-embedding");
            Ok(Box::new(qwen_cloud::QwenCloudEmbedder::new(
                api_key,
                model.to_string(),
            )))
        }
        Ok(EmbeddingBackend::Local) => Err(CoreError::Config(
            "local backend not yet implemented in Rust".into(),
        )),
        Err(message) => Err(CoreError::Config(message)),
    }
}
