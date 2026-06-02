//! Qdrant-backed storage for one isolated embedding space.

use async_trait::async_trait;
use boomerang_core::chunk::ChunkMetadata;
use boomerang_core::embedding::Embedding;
use boomerang_core::error::CoreError;
use boomerang_core::search::SearchResult;
use boomerang_core::store::{StoreStats, VectorStore};
use boomerang_core::types::{ChunkId, EmbeddingSpace};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::collection_name::{collection_name, parse_collection_name};

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
struct Point {
    id: String,
    vector: Vec<f32>,
    payload: Payload,
}

#[derive(Debug, Serialize, Deserialize)]
struct Payload {
    source_file: String,
    start_time: f64,
    end_time: f64,
    indexed_at: String,
    backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    dimensions: usize,
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
struct ScoredPoint {
    #[serde(default)]
    score: f64,
    payload: Option<Payload>,
    vector: Option<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct ScrollResponse {
    result: ScrollResult,
}

#[derive(Debug, Deserialize)]
struct ScrollResult {
    points: Vec<ScoredPoint>,
    next_page_offset: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CountResponse {
    result: CountResult,
}

#[derive(Debug, Deserialize)]
struct CountResult {
    count: usize,
}

pub struct QdrantStore {
    base_url: String,
    client: reqwest::Client,
    collection_name: String,
    embedding_space: EmbeddingSpace,
}

impl QdrantStore {
    pub async fn connect(
        base_url: &str,
        embedding_space: EmbeddingSpace,
    ) -> Result<Self, CoreError> {
        let client = reqwest::Client::new();
        let store = Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client,
            collection_name: collection_name(&embedding_space),
            embedding_space,
        };
        store.ensure_collection().await?;
        Ok(store)
    }

    pub async fn list_spaces(base_url: &str) -> Result<Vec<EmbeddingSpace>, CoreError> {
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
            let Some(space) =
                parse_collection_name(&collection.name, collection_dimensions(&client, &base_url, &collection.name).await?)?
            else {
                continue;
            };
            if collection_count(&client, &base_url, &collection.name).await? > 0 {
                spaces.push(space);
            }
        }
        Ok(spaces)
    }

    async fn ensure_collection(&self) -> Result<(), CoreError> {
        let url = format!("{}/collections/{}", self.base_url, self.collection_name);
        let response = self.client.get(&url).send().await;
        if let Ok(value) = response {
            if value.status().is_success() {
                let dimensions = collection_dimensions(
                    &self.client,
                    &self.base_url,
                    &self.collection_name,
                )
                .await?;
                if dimensions != self.embedding_space.dimensions {
                    return Err(CoreError::Store(format!(
                        "collection {} expects {} dimensions but embedding space uses {}",
                        self.collection_name,
                        dimensions,
                        self.embedding_space.dimensions,
                    )));
                }
                return Ok(());
            }
        }

        let body = CreateCollectionRequest {
            vectors: VectorConfig {
                size: self.embedding_space.dimensions as u64,
                distance: "Cosine",
            },
        };
        let response = self
            .client
            .put(&url)
            .json(&body)
            .send()
            .await
            .map_err(|error| CoreError::Store(format!("failed to create collection: {error}")))?;
        if !response.status().is_success() {
            return Err(CoreError::Store(format!(
                "failed to create collection {}: {}",
                self.collection_name,
                response
                    .text()
                    .await
                    .unwrap_or_else(|_| "failed to read response".to_string()),
            )));
        }
        info!(collection = %self.collection_name, "created Qdrant collection");
        Ok(())
    }

    fn validate_embedding(&self, embedding: &Embedding) -> Result<(), CoreError> {
        if embedding.dimensions != self.embedding_space.dimensions {
            return Err(CoreError::Store(format!(
                "embedding dimension mismatch for {}: expected {}, got {}",
                self.collection_name,
                self.embedding_space.dimensions,
                embedding.dimensions,
            )));
        }
        Ok(())
    }
}

#[async_trait]
impl VectorStore for QdrantStore {
    async fn add(
        &self,
        id: &ChunkId,
        embedding: &Embedding,
        metadata: &ChunkMetadata,
    ) -> Result<(), CoreError> {
        self.validate_embedding(embedding)?;
        let point = Point {
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
        };
        upsert_points(&self.client, &self.base_url, &self.collection_name, vec![point]).await
    }

    async fn add_batch(
        &self,
        entries: &[(ChunkId, Embedding, ChunkMetadata)],
    ) -> Result<(), CoreError> {
        let mut points = Vec::with_capacity(entries.len());
        for (id, embedding, metadata) in entries {
            self.validate_embedding(embedding)?;
            points.push(Point {
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
            });
        }
        upsert_points(&self.client, &self.base_url, &self.collection_name, points).await
    }

    async fn search(
        &self,
        query: &Embedding,
        limit: usize,
    ) -> Result<Vec<SearchResult>, CoreError> {
        self.validate_embedding(query)?;
        let response = self
            .client
            .post(format!(
                "{}/collections/{}/points/search",
                self.base_url, self.collection_name
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
            return Err(CoreError::Store(format!(
                "search failed for {}: {}",
                self.collection_name,
                response
                    .text()
                    .await
                    .unwrap_or_else(|_| "failed to read response".to_string()),
            )));
        }
        let search_response: SearchResponse = response
            .json()
            .await
            .map_err(|error| CoreError::Store(format!("parse search response: {error}")))?;
        let mut results = Vec::new();
        for point in search_response.result {
            let Some(payload) = point.payload else {
                continue;
            };
            results.push(SearchResult::new(
                payload.source_file,
                payload.start_time,
                payload.end_time,
                point.score,
            ));
        }
        Ok(results)
    }

    async fn contains(&self, id: &ChunkId) -> Result<bool, CoreError> {
        let response = self
            .client
            .get(format!(
                "{}/collections/{}/points/{}",
                self.base_url, self.collection_name, id
            ))
            .send()
            .await
            .map_err(|error| CoreError::Store(format!("point lookup failed: {error}")))?;
        Ok(response.status().is_success())
    }

    async fn is_file_indexed(&self, source_file: &str) -> Result<bool, CoreError> {
        Ok(scroll_by_source_file(
            &self.client,
            &self.base_url,
            &self.collection_name,
            source_file,
            1,
        )
        .await?
        .result
        .points
        .is_empty()
            .not())
    }

    async fn remove_file(&self, source_file: &str) -> Result<usize, CoreError> {
        let response = self
            .client
            .post(format!(
                "{}/collections/{}/points/delete",
                self.base_url, self.collection_name
            ))
            .json(&serde_json::json!({
                "filter": {
                    "must": [{
                        "key": "source_file",
                        "match": { "value": source_file }
                    }]
                }
            }))
            .send()
            .await
            .map_err(|error| CoreError::Store(format!("delete failed: {error}")))?;
        if !response.status().is_success() {
            return Err(CoreError::Store(format!(
                "delete failed for {}: {}",
                self.collection_name,
                response
                    .text()
                    .await
                    .unwrap_or_else(|_| "failed to read response".to_string()),
            )));
        }
        Ok(1)
    }

    async fn fetch_all(&self) -> Result<(Vec<Embedding>, Vec<ChunkMetadata>), CoreError> {
        let mut embeddings = Vec::new();
        let mut metadatas = Vec::new();
        let mut offset: Option<String> = None;
        loop {
            let mut request = serde_json::json!({
                "limit": 100,
                "with_payload": true,
                "with_vector": true,
            });
            if let Some(value) = &offset {
                request["offset"] = serde_json::Value::String(value.clone());
            }
            let response = self
                .client
                .post(format!(
                    "{}/collections/{}/points/scroll",
                    self.base_url, self.collection_name
                ))
                .json(&request)
                .send()
                .await
                .map_err(|error| CoreError::Store(format!("scroll all failed: {error}")))?;
            let scroll: ScrollResponse = response
                .json()
                .await
                .map_err(|error| CoreError::Store(format!("parse scroll all: {error}")))?;
            for point in &scroll.result.points {
                if let (Some(vector), Some(payload)) = (&point.vector, &point.payload) {
                    embeddings.push(Embedding::new(vector.clone()));
                    metadatas.push(ChunkMetadata {
                        source_file: payload.source_file.clone(),
                        start_time: payload.start_time,
                        end_time: payload.end_time,
                        indexed_at: payload
                            .indexed_at
                            .parse()
                            .unwrap_or_else(|_| Utc::now()),
                        backend: self.embedding_space.backend,
                        model: payload.model.clone(),
                        dimensions: payload.dimensions,
                    });
                }
            }
            match scroll.result.next_page_offset {
                Some(value) => offset = Some(value),
                None => break,
            }
        }
        Ok((embeddings, metadatas))
    }

    async fn stats(&self) -> Result<StoreStats, CoreError> {
        let count = self.count().await?;
        let source_files = unique_source_files(
            &self.client,
            &self.base_url,
            &self.collection_name,
        )
        .await?;
        Ok(StoreStats {
            total_chunks: count,
            unique_source_files: source_files.len(),
            source_files,
            embedding_space: self.embedding_space.clone(),
        })
    }

    async fn count(&self) -> Result<usize, CoreError> {
        collection_count(&self.client, &self.base_url, &self.collection_name).await
    }

    async fn clear(&self) -> Result<(), CoreError> {
        let response = self
            .client
            .delete(format!("{}/collections/{}", self.base_url, self.collection_name))
            .send()
            .await
            .map_err(|error| CoreError::Store(format!("clear failed: {error}")))?;
        if !response.status().is_success() {
            return Err(CoreError::Store(format!(
                "clear failed for {}: {}",
                self.collection_name,
                response
                    .text()
                    .await
                    .unwrap_or_else(|_| "failed to read response".to_string()),
            )));
        }
        Ok(())
    }
}

async fn upsert_points(
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
        return Err(CoreError::Store(format!(
            "upsert failed for {collection_name}: {}",
            response
                .text()
                .await
                .unwrap_or_else(|_| "failed to read response".to_string()),
        )));
    }
    debug!(collection = %collection_name, "stored embeddings");
    Ok(())
}

async fn collection_dimensions(
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

async fn collection_count(
    client: &reqwest::Client,
    base_url: &str,
    collection_name: &str,
) -> Result<usize, CoreError> {
    let response = client
        .post(format!("{base_url}/collections/{collection_name}/points/count"))
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

async fn scroll_by_source_file(
    client: &reqwest::Client,
    base_url: &str,
    collection_name: &str,
    source_file: &str,
    limit: usize,
) -> Result<ScrollResponse, CoreError> {
    let response = client
        .post(format!("{base_url}/collections/{collection_name}/points/scroll"))
        .json(&serde_json::json!({
            "filter": {
                "must": [{
                    "key": "source_file",
                    "match": { "value": source_file }
                }]
            },
            "limit": limit,
            "with_payload": true,
            "with_vector": false
        }))
        .send()
        .await
        .map_err(|error| CoreError::Store(format!("scroll failed: {error}")))?;
    response
        .json()
        .await
        .map_err(|error| CoreError::Store(format!("parse scroll: {error}")))
}

async fn unique_source_files(
    client: &reqwest::Client,
    base_url: &str,
    collection_name: &str,
) -> Result<Vec<String>, CoreError> {
    let mut offset: Option<String> = None;
    let mut source_files = std::collections::BTreeSet::new();
    loop {
        let mut request = serde_json::json!({
            "limit": 100,
            "with_payload": true,
            "with_vector": false
        });
        if let Some(value) = &offset {
            request["offset"] = serde_json::Value::String(value.clone());
        }
        let response = client
            .post(format!("{base_url}/collections/{collection_name}/points/scroll"))
            .json(&request)
            .send()
            .await
            .map_err(|error| CoreError::Store(format!("stats scroll failed: {error}")))?;
        let scroll: ScrollResponse = response
            .json()
            .await
            .map_err(|error| CoreError::Store(format!("parse stats scroll: {error}")))?;
        for point in &scroll.result.points {
            if let Some(payload) = &point.payload {
                source_files.insert(payload.source_file.clone());
            }
        }
        match scroll.result.next_page_offset {
            Some(value) => offset = Some(value),
            None => break,
        }
    }
    Ok(source_files.into_iter().collect())
}

trait BoolExt {
    fn not(self) -> bool;
}

impl BoolExt for bool {
    fn not(self) -> bool {
        !self
    }
}
