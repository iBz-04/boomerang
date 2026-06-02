//! Reproducible local benchmark for vector retrieval latency.

use std::cmp::Ordering;
use std::time::Instant;

use boomerang_core::chunk::ChunkMetadata;
use boomerang_core::embedding::Embedding;
use boomerang_core::types::{ChunkId, EmbeddingBackend, EmbeddingSpace};
use chrono::Utc;
use uuid::Uuid;
use vector_store::create_store;

const UPSERT_BATCH: usize = 1_000;
const VECTOR_COUNT: usize = 20_000;
const QUERY_COUNT: usize = 100;
const DIMENSIONS: usize = 256;
const TOP_K: usize = 10;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let space = EmbeddingSpace::new(
        EmbeddingBackend::Gemini,
        Some("benchmark-search-256".to_string()),
        DIMENSIONS,
    );
    let store = create_store("qdrant", &space).await?;
    store.clear().await?;
    let store = create_store("qdrant", &space).await?;

    let corpus = build_corpus(VECTOR_COUNT, DIMENSIONS);
    let entries: Vec<(ChunkId, Embedding, ChunkMetadata)> = corpus
        .iter()
        .enumerate()
        .map(|(index, embedding)| {
            let id = ChunkId(
                Uuid::new_v5(&Uuid::NAMESPACE_URL, format!("bench-{index:06}").as_bytes())
                    .to_string(),
            );
            let metadata = ChunkMetadata {
                source_file: format!("/benchmark/video_{index:06}.mp4"),
                start_time: index as f64,
                end_time: index as f64 + 5.0,
                indexed_at: Utc::now(),
                backend: space.backend,
                model: space.model.clone(),
                dimensions: DIMENSIONS,
            };
            (id, Embedding::new(embedding.clone()), metadata)
        })
        .collect();
    for batch in entries.chunks(UPSERT_BATCH) {
        store.add_batch(batch).await?;
    }

    let queries = build_queries(QUERY_COUNT, DIMENSIONS);

    let qdrant_start = Instant::now();
    for query in &queries {
        let embedding = Embedding::new(query.clone());
        let results = store.search(&embedding, TOP_K).await?;
        assert_eq!(results.len(), TOP_K);
    }
    let qdrant_elapsed = qdrant_start.elapsed();

    let brute_start = Instant::now();
    for query in &queries {
        let results = brute_force_search(&corpus, query, TOP_K);
        assert_eq!(results.len(), TOP_K);
    }
    let brute_elapsed = brute_start.elapsed();

    store.clear().await?;

    let qdrant_ms = qdrant_elapsed.as_secs_f64() * 1_000.0;
    let brute_ms = brute_elapsed.as_secs_f64() * 1_000.0;
    let speedup = brute_ms / qdrant_ms;

    println!("dataset_vectors={VECTOR_COUNT}");
    println!("dimensions={DIMENSIONS}");
    println!("queries={QUERY_COUNT}");
    println!("top_k={TOP_K}");
    println!("qdrant_total_ms={qdrant_ms:.2}");
    println!(
        "qdrant_avg_ms_per_query={:.3}",
        qdrant_ms / QUERY_COUNT as f64
    );
    println!("bruteforce_total_ms={brute_ms:.2}");
    println!(
        "bruteforce_avg_ms_per_query={:.3}",
        brute_ms / QUERY_COUNT as f64
    );
    println!("speedup_vs_bruteforce={speedup:.2}");

    Ok(())
}

fn build_corpus(count: usize, dimensions: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|index| normalized_vector(index as u64 + 1, dimensions))
        .collect()
}

fn build_queries(count: usize, dimensions: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|index| normalized_vector(10_000_000 + index as u64, dimensions))
        .collect()
}

fn normalized_vector(seed: u64, dimensions: usize) -> Vec<f32> {
    let mut state = seed;
    let mut values = Vec::with_capacity(dimensions);
    for _ in 0..dimensions {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = ((state >> 11) as f64) / ((1u64 << 53) as f64);
        values.push((unit as f32) * 2.0 - 1.0);
    }

    let norm = values
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1e-12);
    for value in &mut values {
        *value /= norm;
    }
    values
}

fn brute_force_search(corpus: &[Vec<f32>], query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
    let mut scores: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(index, vector)| (index, dot(vector, query)))
        .collect();
    scores.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
    });
    scores.truncate(top_k);
    scores
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}
