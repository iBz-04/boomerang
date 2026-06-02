//! Gemini Embedding API backend.
//!
//! Uses Google's Gemini Embedding 2 model which natively embeds video
//! — raw pixels are projected into the same vector space as text queries.

use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use boomerang_core::embedding::{Embedder, Embedding};
use boomerang_core::error::CoreError;
use boomerang_core::types::{EmbeddingBackend, EmbeddingSpace};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

const GEMINI_EMBED_URL: &str =
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:embedContent";

const GEMINI_DIMENSIONS: usize = 3072;

#[derive(Debug, Serialize)]
struct EmbedRequest {
    model: String,
    content: Content,
}

#[derive(Debug, Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Debug, Serialize)]
struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inline_data: Option<InlineData>,
}

#[derive(Debug, Serialize)]
struct InlineData {
    mime_type: String,
    data: String,
}

#[derive(Debug, Deserialize)]
struct EmbedResponse {
    #[serde(default)]
    embedding: Option<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: Option<ErrorDetail>,
}

#[derive(Debug, Deserialize)]
struct ErrorDetail {
    message: Option<String>,
}

pub struct GeminiEmbedder {
    api_key: String,
    client: reqwest::Client,
}

impl GeminiEmbedder {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
        }
    }

    async fn send_request(&self, body: &EmbedRequest) -> Result<Vec<f32>, CoreError> {
        let url = format!("{GEMINI_EMBED_URL}?key={}", self.api_key);

        let response = self
            .client
            .post(&url)
            .json(body)
            .send()
            .await
            .map_err(|e| CoreError::EmbeddingApi(format!("HTTP request failed: {e}")))?;

        let status = response.status();

        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(CoreError::QuotaExceeded("Gemini API quota exceeded".into()));
        }

        let resp_body = response
            .text()
            .await
            .map_err(|e| CoreError::EmbeddingApi(format!("failed to read response: {e}")))?;

        if !status.is_success() {
            if let Ok(err) = serde_json::from_str::<ErrorResponse>(&resp_body) {
                let msg = err
                    .error
                    .and_then(|e| e.message)
                    .unwrap_or_else(|| "unknown error".into());
                return Err(CoreError::EmbeddingApi(msg));
            }
            return Err(CoreError::EmbeddingApi(format!(
                "API error ({}): {resp_body}",
                status.as_u16()
            )));
        }

        let embed_resp: EmbedResponse = serde_json::from_str(&resp_body)
            .map_err(|e| CoreError::EmbeddingApi(format!("failed to parse response: {e}")))?;

        let values = embed_resp
            .embedding
            .ok_or_else(|| CoreError::EmbeddingApi("no embedding in response".into()))?
            .values;

        Ok(values)
    }
}

#[async_trait]
impl Embedder for GeminiEmbedder {
    async fn embed_video(&self, chunk_path: &str) -> Result<Embedding, CoreError> {
        debug!(path = chunk_path, "embedding video via Gemini");

        let video_bytes = tokio::fs::read(chunk_path).await.map_err(CoreError::Io)?;

        let b64 = BASE64.encode(&video_bytes);

        let body = EmbedRequest {
            model: "models/gemini-embedding-2".into(),
            content: Content {
                parts: vec![Part {
                    text: None,
                    inline_data: Some(InlineData {
                        mime_type: "video/mp4".into(),
                        data: b64,
                    }),
                }],
            },
        };

        let values = self.send_request(&body).await?;

        info!(dimensions = values.len(), "video embedded");
        Ok(Embedding::new(values))
    }

    async fn embed_query(&self, query: &str) -> Result<Embedding, CoreError> {
        debug!(query, "embedding text query via Gemini");

        let body = EmbedRequest {
            model: "models/gemini-embedding-2".into(),
            content: Content {
                parts: vec![Part {
                    text: Some(query.to_string()),
                    inline_data: None,
                }],
            },
        };

        let values = self.send_request(&body).await?;
        Ok(Embedding::new(values))
    }

    async fn embed_image(&self, image_path: &str) -> Result<Embedding, CoreError> {
        debug!(path = image_path, "embedding image via Gemini");

        let image_bytes = tokio::fs::read(image_path).await.map_err(CoreError::Io)?;

        let mime = mime_type(image_path);
        let b64 = BASE64.encode(&image_bytes);

        let body = EmbedRequest {
            model: "models/gemini-embedding-2".into(),
            content: Content {
                parts: vec![Part {
                    text: None,
                    inline_data: Some(InlineData {
                        mime_type: mime,
                        data: b64,
                    }),
                }],
            },
        };

        let values = self.send_request(&body).await?;
        Ok(Embedding::new(values))
    }

    fn dimensions(&self) -> usize {
        GEMINI_DIMENSIONS
    }

    fn backend_name(&self) -> &'static str {
        "gemini"
    }

    fn embedding_space(&self) -> Result<EmbeddingSpace, CoreError> {
        Ok(EmbeddingSpace::new(
            EmbeddingBackend::Gemini,
            None,
            self.dimensions(),
        ))
    }
}

fn mime_type(path: &str) -> String {
    let lower = path.to_lowercase();
    if lower.ends_with(".png") {
        "image/png".into()
    } else if lower.ends_with(".webp") {
        "image/webp".into()
    } else if lower.ends_with(".gif") {
        "image/gif".into()
    } else if lower.ends_with(".heic") || lower.ends_with(".heif") {
        "image/heic".into()
    } else {
        "image/jpeg".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mime_type_detection() {
        assert_eq!(mime_type("photo.jpg"), "image/jpeg");
        assert_eq!(mime_type("photo.png"), "image/png");
        assert_eq!(mime_type("photo.webp"), "image/webp");
        assert_eq!(mime_type("photo.HEIC"), "image/heic");
    }
}
