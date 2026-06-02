//! Qdrant scroll helpers for source-file lookup and full-index scans.

use std::collections::BTreeSet;

use boomerang_core::error::CoreError;

use crate::qdrant_http::ScrollResponse;

pub(crate) async fn scroll_by_source_file(
    client: &reqwest::Client,
    base_url: &str,
    collection_name: &str,
    source_file: &str,
    limit: usize,
) -> Result<ScrollResponse, CoreError> {
    let response = client
        .post(format!(
            "{base_url}/collections/{collection_name}/points/scroll"
        ))
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

pub(crate) async fn scroll_all_points(
    client: &reqwest::Client,
    base_url: &str,
    collection_name: &str,
    offset: Option<String>,
) -> Result<ScrollResponse, CoreError> {
    let mut request = serde_json::json!({
        "limit": 100,
        "with_payload": true,
        "with_vector": true,
    });
    if let Some(value) = offset {
        request["offset"] = serde_json::Value::String(value);
    }
    let response = client
        .post(format!(
            "{base_url}/collections/{collection_name}/points/scroll"
        ))
        .json(&request)
        .send()
        .await
        .map_err(|error| CoreError::Store(format!("scroll all failed: {error}")))?;
    response
        .json()
        .await
        .map_err(|error| CoreError::Store(format!("parse scroll all: {error}")))
}

pub(crate) async fn unique_source_files(
    client: &reqwest::Client,
    base_url: &str,
    collection_name: &str,
) -> Result<Vec<String>, CoreError> {
    let mut offset: Option<String> = None;
    let mut source_files = BTreeSet::new();
    loop {
        let scroll = scroll_source_file_page(client, base_url, collection_name, offset).await?;
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

async fn scroll_source_file_page(
    client: &reqwest::Client,
    base_url: &str,
    collection_name: &str,
    offset: Option<String>,
) -> Result<ScrollResponse, CoreError> {
    let mut request = serde_json::json!({
        "limit": 100,
        "with_payload": true,
        "with_vector": false
    });
    if let Some(value) = offset {
        request["offset"] = serde_json::Value::String(value);
    }
    let response = client
        .post(format!(
            "{base_url}/collections/{collection_name}/points/scroll"
        ))
        .json(&request)
        .send()
        .await
        .map_err(|error| CoreError::Store(format!("stats scroll failed: {error}")))?;
    response
        .json()
        .await
        .map_err(|error| CoreError::Store(format!("parse stats scroll: {error}")))
}
