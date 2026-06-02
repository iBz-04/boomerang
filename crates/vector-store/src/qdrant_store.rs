//! Qdrant-backed storage for one isolated embedding space.

use async_trait::async_trait;
use boomerang_core::chunk::ChunkMetadata;
use boomerang_core::embedding::Embedding;
use boomerang_core::error::CoreError;
use boomerang_core::search::SearchResult;
use boomerang_core::store::{StoreStats, VectorStore};
use boomerang_core::types::{ChunkId, EmbeddingSpace};
use tracing::info;

use crate::collection_name::collection_name;
use crate::qdrant_http::{
    collection_count, collection_dimensions, create_collection, create_keyword_index, list_spaces,
    search_points, upsert_points, Point,
};
use crate::qdrant_scroll::{
    scroll_all_points, scroll_all_points_by_source_file, scroll_by_source_file, unique_source_files,
};

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
        list_spaces(base_url).await
    }

    async fn ensure_collection(&self) -> Result<(), CoreError> {
        let url = format!("{}/collections/{}", self.base_url, self.collection_name);
        let response = self.client.get(&url).send().await;
        if let Ok(value) = response {
            if value.status().is_success() {
                let dimensions =
                    collection_dimensions(&self.client, &self.base_url, &self.collection_name)
                        .await?;
                if dimensions != self.embedding_space.dimensions {
                    return Err(CoreError::Store(format!(
                        "collection {} expects {} dimensions but embedding space uses {}",
                        self.collection_name, dimensions, self.embedding_space.dimensions,
                    )));
                }
                info!(
                    collection = %self.collection_name,
                    dimensions,
                    backend = %self.embedding_space.backend,
                    model = ?self.embedding_space.model,
                    "using existing Qdrant collection"
                );
                create_keyword_index(
                    &self.client,
                    &self.base_url,
                    &self.collection_name,
                    "source_file",
                )
                .await?;
                return Ok(());
            }
        }

        create_collection(
            &self.client,
            &self.base_url,
            &self.collection_name,
            self.embedding_space.dimensions,
        )
        .await?;
        create_keyword_index(
            &self.client,
            &self.base_url,
            &self.collection_name,
            "source_file",
        )
        .await?;
        info!(collection = %self.collection_name, "created Qdrant collection");
        Ok(())
    }

    fn validate_embedding(&self, embedding: &Embedding) -> Result<(), CoreError> {
        if embedding.dimensions != self.embedding_space.dimensions {
            return Err(CoreError::Store(format!(
                "embedding dimension mismatch for {}: expected {}, got {}",
                self.collection_name, self.embedding_space.dimensions, embedding.dimensions,
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
        let point = Point::from_entry(id, embedding, metadata);
        upsert_points(
            &self.client,
            &self.base_url,
            &self.collection_name,
            vec![point],
        )
        .await
    }

    async fn add_batch(
        &self,
        entries: &[(ChunkId, Embedding, ChunkMetadata)],
    ) -> Result<(), CoreError> {
        let mut points = Vec::with_capacity(entries.len());
        for (id, embedding, metadata) in entries {
            self.validate_embedding(embedding)?;
            points.push(Point::from_entry(id, embedding, metadata));
        }
        upsert_points(&self.client, &self.base_url, &self.collection_name, points).await
    }

    async fn search(
        &self,
        query: &Embedding,
        limit: usize,
    ) -> Result<Vec<SearchResult>, CoreError> {
        self.search_points(query, limit, None).await
    }

    async fn search_by_source_file(
        &self,
        query: &Embedding,
        limit: usize,
        source_file: &str,
    ) -> Result<Vec<SearchResult>, CoreError> {
        self.search_points(query, limit, Some(source_file)).await
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
            let body = response.text().await.map_err(|error| {
                CoreError::Store(format!("failed to read delete response: {error}"))
            })?;
            return Err(CoreError::Store(format!(
                "delete failed for {}: {}",
                self.collection_name, body,
            )));
        }
        Ok(1)
    }

    async fn fetch_all(&self) -> Result<(Vec<Embedding>, Vec<ChunkMetadata>), CoreError> {
        let mut embeddings = Vec::new();
        let mut metadatas = Vec::new();
        let mut offset: Option<String> = None;
        loop {
            let scroll =
                scroll_all_points(&self.client, &self.base_url, &self.collection_name, offset)
                    .await?;
            for (rank, point) in scroll.result.points.iter().enumerate() {
                let Some(vector) = &point.vector else {
                    return Err(CoreError::Store(format!(
                        "Qdrant scroll point missing vector in collection {} at page rank {}",
                        self.collection_name,
                        rank + 1,
                    )));
                };
                let Some(payload) = &point.payload else {
                    return Err(CoreError::Store(format!(
                        "Qdrant scroll point missing payload in collection {} at page rank {}",
                        self.collection_name,
                        rank + 1,
                    )));
                };
                let indexed_at = payload.indexed_at.parse().map_err(|error| {
                    CoreError::Store(format!(
                        "invalid indexed_at '{}' in collection {}: {}",
                        payload.indexed_at, self.collection_name, error
                    ))
                })?;
                embeddings.push(Embedding::new(vector.clone()));
                metadatas.push(ChunkMetadata {
                    source_file: payload.source_file.clone(),
                    start_time: payload.start_time,
                    end_time: payload.end_time,
                    indexed_at,
                    backend: self.embedding_space.backend,
                    model: payload.model.clone(),
                    dimensions: payload.dimensions,
                });
            }
            match scroll.result.next_page_offset {
                Some(value) => offset = Some(value),
                None => break,
            }
        }
        Ok((embeddings, metadatas))
    }

    async fn fetch_by_source_file(
        &self,
        source_file: &str,
    ) -> Result<(Vec<Embedding>, Vec<ChunkMetadata>), CoreError> {
        let mut embeddings = Vec::new();
        let mut metadatas = Vec::new();
        let mut offset: Option<String> = None;
        loop {
            let scroll = scroll_all_points_by_source_file(
                &self.client,
                &self.base_url,
                &self.collection_name,
                source_file,
                offset,
            )
            .await?;
            for (rank, point) in scroll.result.points.iter().enumerate() {
                let Some(vector) = &point.vector else {
                    return Err(CoreError::Store(format!(
                        "Qdrant scroll point missing vector in collection {} at page rank {}",
                        self.collection_name,
                        rank + 1,
                    )));
                };
                let Some(payload) = &point.payload else {
                    return Err(CoreError::Store(format!(
                        "Qdrant scroll point missing payload in collection {} at page rank {}",
                        self.collection_name,
                        rank + 1,
                    )));
                };
                let indexed_at = payload.indexed_at.parse().map_err(|error| {
                    CoreError::Store(format!(
                        "invalid indexed_at '{}' in collection {}: {}",
                        payload.indexed_at, self.collection_name, error
                    ))
                })?;
                embeddings.push(Embedding::new(vector.clone()));
                metadatas.push(ChunkMetadata {
                    source_file: payload.source_file.clone(),
                    start_time: payload.start_time,
                    end_time: payload.end_time,
                    indexed_at,
                    backend: self.embedding_space.backend,
                    model: payload.model.clone(),
                    dimensions: payload.dimensions,
                });
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
        let source_files =
            unique_source_files(&self.client, &self.base_url, &self.collection_name).await?;
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
            .delete(format!(
                "{}/collections/{}",
                self.base_url, self.collection_name
            ))
            .send()
            .await
            .map_err(|error| CoreError::Store(format!("clear failed: {error}")))?;
        if !response.status().is_success() {
            let body = response.text().await.map_err(|error| {
                CoreError::Store(format!("failed to read clear response: {error}"))
            })?;
            return Err(CoreError::Store(format!(
                "clear failed for {}: {}",
                self.collection_name, body,
            )));
        }
        Ok(())
    }
}

impl QdrantStore {
    async fn search_points(
        &self,
        query: &Embedding,
        limit: usize,
        source_file: Option<&str>,
    ) -> Result<Vec<SearchResult>, CoreError> {
        self.validate_embedding(query)?;
        info!(
            collection = %self.collection_name,
            query_dimensions = query.dimensions,
            limit,
            with_payload = true,
            with_vector = false,
            source_file = source_file.unwrap_or("*"),
            "sending Qdrant search request"
        );
        let points = search_points(
            &self.client,
            &self.base_url,
            &self.collection_name,
            query,
            limit,
            source_file,
        )
        .await?;
        info!(
            collection = %self.collection_name,
            raw_points = points.len(),
            "Qdrant search response parsed"
        );

        let mut results = Vec::with_capacity(points.len());
        for (rank, point) in points.into_iter().enumerate() {
            let Some(payload) = point.payload else {
                return Err(CoreError::Store(format!(
                    "Qdrant search point missing payload in collection {} at rank {}",
                    self.collection_name,
                    rank + 1,
                )));
            };
            info!(
                collection = %self.collection_name,
                rank = rank + 1,
                point_id = ?point.id,
                score = point.score,
                source_file = %payload.source_file,
                start = payload.start_time,
                end = payload.end_time,
                indexed_backend = %payload.backend,
                indexed_model = ?payload.model,
                indexed_dimensions = payload.dimensions,
                "Qdrant search candidate returned"
            );
            results.push(SearchResult::new(
                payload.source_file,
                payload.start_time,
                payload.end_time,
                point.score,
            ));
        }
        Ok(results)
    }
}

trait BoolExt {
    fn not(self) -> bool;
}

impl BoolExt for bool {
    fn not(self) -> bool {
        !self
    }
}
