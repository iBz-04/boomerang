//! Qwen Cloud (DashScope) embedding backend.
//!
//! Uses Alibaba DashScope's multimodal embedding API with Qwen3-VL-Embedding.

use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use boomerang_core::embedding::{Embedder, Embedding};
use boomerang_core::error::CoreError;
use boomerang_core::types::{EmbeddingBackend, EmbeddingSpace};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

const DASHSCOPE_URL: &str =
    "https://dashscope-intl.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding";

#[derive(Debug, Serialize)]
struct EmbedRequest {
    model: String,
    input: Input,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Parameters>,
}

#[derive(Debug, Serialize)]
struct Input {
    #[serde(skip_serializing_if = "Option::is_none")]
    video: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct Parameters {
    dimension: usize,
}

#[derive(Debug, Deserialize)]
struct EmbedResponse {
    output: Option<OutputData>,
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OutputData {
    embeddings: Option<Vec<EmbeddingItem>>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
}

pub struct QwenCloudEmbedder {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl QwenCloudEmbedder {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            client: reqwest::Client::new(),
        }
    }

    async fn send_request(&self, request: &EmbedRequest) -> Result<Vec<f32>, CoreError> {
        let response = self
            .client
            .post(DASHSCOPE_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(request)
            .send()
            .await
            .map_err(|e| CoreError::EmbeddingApi(format!("DashScope request failed: {e}")))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| CoreError::EmbeddingApi(format!("failed to read response: {e}")))?;

        if !status.is_success() {
            return Err(CoreError::EmbeddingApi(format!(
                "DashScope error ({}): {body}",
                status.as_u16()
            )));
        }

        let embed_resp: EmbedResponse = serde_json::from_str(&body)
            .map_err(|e| CoreError::EmbeddingApi(format!("failed to parse: {e}")))?;

        if let Some(msg) = embed_resp.message {
            if !msg.is_empty() && msg != "Success" {
                return Err(CoreError::EmbeddingApi(msg));
            }
        }

        let embeddings = embed_resp
            .output
            .and_then(|o| o.embeddings)
            .ok_or_else(|| CoreError::EmbeddingApi("no embeddings in response".into()))?;

        embeddings
            .into_iter()
            .next()
            .map(|item| item.embedding)
            .ok_or_else(|| CoreError::EmbeddingApi("empty embedding list".into()))
    }
}

#[async_trait]
impl Embedder for QwenCloudEmbedder {
    async fn embed_video(&self, chunk_path: &str) -> Result<Embedding, CoreError> {
        debug!(path = chunk_path, "embedding video via DashScope");

        let video_url = upload_to_oss(chunk_path).await?;

        let request = EmbedRequest {
            model: self.model.clone(),
            input: Input {
                video: Some(vec![video_url]),
                text: None,
                image: None,
            },
            parameters: Some(Parameters { dimension: 768 }),
        };

        let values = self.send_request(&request).await?;
        info!(dimensions = values.len(), "video embedded via DashScope");
        Embedding::new(values)
    }

    async fn embed_query(&self, query: &str) -> Result<Embedding, CoreError> {
        debug!(query, "embedding text via DashScope");

        let request = EmbedRequest {
            model: self.model.clone(),
            input: Input {
                video: None,
                text: Some(vec![query.to_string()]),
                image: None,
            },
            parameters: Some(Parameters { dimension: 768 }),
        };

        let values = self.send_request(&request).await?;
        Embedding::new(values)
    }

    async fn embed_image(&self, image_path: &str) -> Result<Embedding, CoreError> {
        debug!(path = image_path, "embedding image via DashScope");

        let image_url = upload_to_oss(image_path).await?;

        let request = EmbedRequest {
            model: self.model.clone(),
            input: Input {
                video: None,
                text: None,
                image: Some(vec![image_url]),
            },
            parameters: Some(Parameters { dimension: 768 }),
        };

        let values = self.send_request(&request).await?;
        Embedding::new(values)
    }

    fn dimensions(&self) -> usize {
        768
    }

    fn backend_name(&self) -> &'static str {
        "qwen-cloud"
    }

    fn model_name(&self) -> Option<&str> {
        Some(self.model.as_str())
    }

    fn embedding_space(&self) -> Result<EmbeddingSpace, CoreError> {
        Ok(EmbeddingSpace::new(
            EmbeddingBackend::QwenCloud,
            Some(self.model.clone()),
            self.dimensions(),
        ))
    }
}

/// Upload a local file to DashScope-managed temporary OSS.
///
/// The DashScope multimodal embedding API expects URLs, not raw bytes.
/// The Python SDK handles this transparently; here we read the file
/// and return a data URI as a fallback until OSS upload is implemented.
async fn upload_to_oss(file_path: &str) -> Result<String, CoreError> {
    let bytes = tokio::fs::read(file_path).await.map_err(CoreError::Io)?;

    let encoded = BASE64.encode(bytes);
    let mime = if file_path.to_lowercase().ends_with(".mp4") {
        "video/mp4"
    } else if file_path.to_lowercase().ends_with(".png") {
        "image/png"
    } else {
        "image/jpeg"
    };

    Ok(format!("data:{mime};base64,{encoded}"))
}
