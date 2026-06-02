//! Qdrant HTTP request and response helpers for the vector store.

use boomerang_core::chunk::ChunkMetadata;
use boomerang_core::embedding::Embedding;
use boomerang_core::error::CoreError;
use boomerang_core::types::{ChunkId, EmbeddingSpace};
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::collection_name::parse_collection_name;

#[derive(Debug, Serialize)]
struct CreateCollectionRequest {
    vectors: VectorConfig,
}

#[derive(Debug, Serialize)]
struct VectorConfig {
    size: u64,
    distance: &'static str,
}

#[derive(Debug, Serialize)]
struct UpsertRequest {
    points: Vec<Point>,
}

#[derive(Debug, Serialize)]
pub(crate) struct Point {
    id: String,
    vector: Vec<f32>,
    payload: Payload,
}

impl Point {
    pub(crate) fn from_entry(
        id: &ChunkId,
        embedding: &Embedding,
        metadata: &ChunkMetadata,
    ) -> Self {
        Self {
            id: id.to_string(),
            vector: embedding.data.clone(),
            payload: Payload {
                source_file: metadata.source_file.clone(),
                start_time: metadata.start_time,
                end_time: metadata.end_time,
                indexed_at: metadata.indexed_at.to_rfc3339(),
                backend: metadata.backend.to_string(),
                model: metadata.model.clone(),
                dimensions: metadata.dimensions,
            },
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Payload {
    pub(crate) source_file: String,
    pub(crate) start_time: f64,
    pub(crate) end_time: f64,
    pub(crate) indexed_at: String,
    pub(crate) backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) model: Option<String>,
    pub(crate) dimensions: usize,
}

#[derive(Debug, Serialize)]
struct SearchRequest {
    vector: Vec<f32>,
    limit: usize,
    with_payload: bool,
    with_vector: bool,
}

#[derive(Debug, Deserialize)]
struct CollectionListResponse {
    result: CollectionsResult,
}

#[derive(Debug, Deserialize)]
struct CollectionsResult {
    collections: Vec<CollectionName>,
}

#[derive(Debug, Deserialize)]
struct CollectionName {
    name: String,
}

#[derive(Debug, Deserialize)]
struct CollectionResponse {
    result: CollectionInfo,
}

#[derive(Debug, Deserialize)]
struct CollectionInfo {
    config: CollectionConfig,
}

#[derive(Debug, Deserialize)]
struct CollectionConfig {
    params: CollectionParams,
}

#[derive(Debug, Deserialize)]
struct CollectionParams {
    vectors: VectorParams,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum VectorParams {
    Single(VectorSize),
}

#[derive(Debug, Deserialize)]
struct VectorSize {
    size: usize,
}

#[derive(Debug, Deserialize)]
struct SearchResponse {
    result: Vec<ScoredPoint>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ScoredPoint {
    pub(crate) id: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) score: f64,
    pub(crate) payload: Option<Payload>,
    pub(crate) vector: Option<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ScrollResponse {
    pub(crate) result: ScrollResult,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ScrollResult {
    pub(crate) points: Vec<ScoredPoint>,
    pub(crate) next_page_offset: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CountResponse {
    result: CountResult,
}

#[derive(Debug, Deserialize)]
struct CountResult {
    count: usize,
}

pub(crate) async fn list_spaces(base_url: &str) -> Result<Vec<EmbeddingSpace>, CoreError> {
    let client = reqwest::Client::new();
    let base_url = base_url.trim_end_matches('/').to_string();
    let response = client
        .get(format!("{base_url}/collections"))
        .send()
        .await
        .map_err(|error| CoreError::Store(format!("failed to list collections: {error}")))?;
    let listing: CollectionListResponse = response
        .json()
        .await
        .map_err(|error| CoreError::Store(format!("failed to parse collections: {error}")))?;

    let mut spaces = Vec::new();
    for collection in listing.result.collections {
        let Some(space) = parse_collection_name(
            &collection.name,
            collection_dimensions(&client, &base_url, &collection.name).await?,
        )?
        else {
            continue;
        };
        if collection_count(&client, &base_url, &collection.name).await? > 0 {
            spaces.push(space);
        }
    }
    Ok(spaces)
}

pub(crate) async fn create_collection(
    client: &reqwest::Client,
    base_url: &str,
    collection_name: &str,
    dimensions: usize,
) -> Result<(), CoreError> {
    let response = client
        .put(format!("{base_url}/collections/{collection_name}"))
        .json(&CreateCollectionRequest {
            vectors: VectorConfig {
                size: dimensions as u64,
                distance: "Cosine",
            },
        })
        .send()
        .await
        .map_err(|error| CoreError::Store(format!("failed to create collection: {error}")))?;
    if !response.status().is_success() {
        let body = response.text().await.map_err(|error| {
            CoreError::Store(format!(
                "failed to read create collection response: {error}"
            ))
        })?;
        return Err(CoreError::Store(format!(
            "failed to create collection {collection_name}: {}",
            body,
        )));
    }
    Ok(())
}

pub(crate) async fn upsert_points(
    client: &reqwest::Client,
    base_url: &str,
    collection_name: &str,
    points: Vec<Point>,
) -> Result<(), CoreError> {
    let response = client
        .put(format!("{base_url}/collections/{collection_name}/points"))
        .json(&UpsertRequest { points })
        .send()
        .await
        .map_err(|error| CoreError::Store(format!("upsert failed: {error}")))?;
    if !response.status().is_success() {
        let body = response.text().await.map_err(|error| {
            CoreError::Store(format!("failed to read upsert response: {error}"))
        })?;
        return Err(CoreError::Store(format!(
            "upsert failed for {collection_name}: {}",
            body,
        )));
    }
    debug!(collection = %collection_name, "stored embeddings");
    Ok(())
}

pub(crate) async fn search_points(
    client: &reqwest::Client,
    base_url: &str,
    collection_name: &str,
    query: &Embedding,
    limit: usize,
) -> Result<Vec<ScoredPoint>, CoreError> {
    let response = client
        .post(format!(
            "{base_url}/collections/{collection_name}/points/search"
        ))
        .json(&SearchRequest {
            vector: query.data.clone(),
            limit,
            with_payload: true,
            with_vector: false,
        })
        .send()
        .await
        .map_err(|error| CoreError::Store(format!("search failed: {error}")))?;
    if !response.status().is_success() {
        let body = response.text().await.map_err(|error| {
            CoreError::Store(format!("failed to read search response: {error}"))
        })?;
        return Err(CoreError::Store(format!(
            "search failed for {collection_name}: {}",
            body,
        )));
    }
    let search_response: SearchResponse = response
        .json()
        .await
        .map_err(|error| CoreError::Store(format!("parse search response: {error}")))?;
    Ok(search_response.result)
}

pub(crate) async fn collection_dimensions(
    client: &reqwest::Client,
    base_url: &str,
    collection_name: &str,
) -> Result<usize, CoreError> {
    let response = client
        .get(format!("{base_url}/collections/{collection_name}"))
        .send()
        .await
        .map_err(|error| CoreError::Store(format!("collection info failed: {error}")))?;
    let details: CollectionResponse = response
        .json()
        .await
        .map_err(|error| CoreError::Store(format!("parse collection info: {error}")))?;
    let VectorParams::Single(config) = details.result.config.params.vectors;
    Ok(config.size)
}

pub(crate) async fn collection_count(
    client: &reqwest::Client,
    base_url: &str,
    collection_name: &str,
) -> Result<usize, CoreError> {
    let response = client
        .post(format!(
            "{base_url}/collections/{collection_name}/points/count"
        ))
        .json(&serde_json::json!({"exact": true}))
        .send()
        .await
        .map_err(|error| CoreError::Store(format!("count failed: {error}")))?;
    let count: CountResponse = response
        .json()
        .await
        .map_err(|error| CoreError::Store(format!("parse count: {error}")))?;
    Ok(count.result.count)
}
