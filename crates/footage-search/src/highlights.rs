//! Anomaly-based highlight ranking over indexed embeddings.
//!
//! Scores each indexed chunk by how unusual its embedding is and returns
//! the top-N as candidates for trimming. Supports centroid distance,
//! KNN distance, local temporal contrast, and Local Outlier Factor (LOF).

use boomerang_core::chunk::ChunkMetadata;
use boomerang_core::error::CoreError;
use boomerang_core::search::{HighlightConfig, SearchResult};
use boomerang_core::store::VectorStore;
use ndarray::{Array1, Array2, Axis};
use tracing::debug;

/// Rank the most anomalous clips in the index.
pub async fn rank_highlights(
    store: &dyn VectorStore,
    config: &HighlightConfig,
) -> Result<Vec<SearchResult>, CoreError> {
    let (embeddings, metadatas) = if let Some(source_file) = config.source_file.as_deref() {
        store.fetch_by_source_file(source_file).await?
    } else {
        store.fetch_all().await?
    };
    let n = embeddings.len();

    if n == 0 {
        return Ok(vec![]);
    }

    let dims = embeddings.first().map(|e| e.dimensions).unwrap_or(0);
    if dims == 0 {
        return Ok(vec![]);
    }

    // Build matrix
    let mut data = Vec::with_capacity(n * dims);
    for emb in &embeddings {
        data.extend_from_slice(&emb.data);
    }
    let x = Array2::from_shape_vec((n, dims), data)
        .map_err(|e| CoreError::Other(format!("shape error: {e}")))?;

    let xn = normalize(&x);

    // Apply baseline exclusion
    let candidate_mask = if config.exclude_baseline
        && config.method != boomerang_core::types::ScoringMethod::LocalContrast
    {
        exclude_baseline_mask(&xn)
    } else {
        Array1::ones(n)
    };

    let cand_indices: Vec<usize> = candidate_mask
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0.5)
        .map(|(i, _)| i)
        .collect();

    if cand_indices.is_empty() {
        return Ok(vec![]);
    }

    // Subset matrix to candidates
    let xn_sub = xn.select(Axis(0), &cand_indices);
    let candidate_metadatas = cand_indices
        .iter()
        .map(|&index| &metadatas[index])
        .collect::<Vec<_>>();

    // Score candidates
    let scores = match config.method {
        boomerang_core::types::ScoringMethod::Centroid => score_centroid(&xn_sub),
        boomerang_core::types::ScoringMethod::Knn => score_knn(&xn_sub, config.neighbors),
        boomerang_core::types::ScoringMethod::Lof => score_lof(&xn_sub, config.neighbors),
        boomerang_core::types::ScoringMethod::LocalContrast => {
            score_local_contrast(&xn_sub, &candidate_metadatas, config.neighbors)
        }
    };

    // Sort by score descending
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Deduplicate and collect results
    let mut results = Vec::new();
    let mut seen_files: Vec<(String, f64, f64)> = Vec::new();

    for &idx in &order {
        if results.len() >= config.count {
            break;
        }

        let global_idx = cand_indices[idx];
        let meta = &metadatas[global_idx];

        // Simple dedup: skip if too similar to already-kept result
        let is_dup = seen_files.iter().any(|(file, start, end)| {
            if file == &meta.source_file {
                let overlap_start = start.max(meta.start_time);
                let overlap_end = end.min(meta.end_time);
                let overlap = (overlap_end - overlap_start).max(0.0);
                let min_dur = (end - start).min(meta.end_time - meta.start_time);
                if min_dur > 0.0 {
                    return overlap / min_dur > config.dedupe_threshold;
                }
            }
            false
        });

        if !is_dup {
            seen_files.push((meta.source_file.clone(), meta.start_time, meta.end_time));
            results.push(SearchResult::new(
                meta.source_file.clone(),
                meta.start_time,
                meta.end_time,
                scores[idx] as f64,
            ));
        }
    }

    debug!(count = results.len(), "highlights ranked");
    Ok(results)
}

/// L2-normalize rows of a matrix.
fn normalize(x: &Array2<f32>) -> Array2<f32> {
    let norms = x.map_axis(Axis(1), |row| {
        let sq_sum: f32 = row.iter().map(|v| v * v).sum();
        sq_sum.sqrt().max(1e-12)
    });
    let norms_broadcast = norms.insert_axis(Axis(1));
    x / &norms_broadcast
}

/// Cosine distance matrix for normalized rows. Diagonal is +inf.
fn cosine_distance_matrix(xn: &Array2<f32>) -> Array2<f32> {
    let n = xn.nrows();
    let mut d = xn.dot(&xn.t());
    d.mapv_inplace(|v| 1.0 - v);
    for i in 0..n {
        d[[i, i]] = f32::INFINITY;
    }
    d
}

/// Score by distance from centroid.
fn score_centroid(xn: &Array2<f32>) -> Vec<f32> {
    let mean = xn.mean_axis(Axis(0)).unwrap();
    let mean_norm = mean.mapv(|v| v * v).sum().sqrt().max(1e-12);
    let mean_unit = mean / mean_norm;
    xn.dot(&mean_unit).mapv(|v| 1.0 - v).to_vec()
}

/// Score by mean cosine distance to k nearest neighbors.
fn score_knn(xn: &Array2<f32>, k: usize) -> Vec<f32> {
    let n = xn.nrows();
    let k = k.max(1).min(n.saturating_sub(1));
    if k == 0 {
        return vec![0.0; n];
    }

    let d = cosine_distance_matrix(xn);
    let mut scores = Vec::with_capacity(n);

    for i in 0..n {
        let mut row: Vec<f32> = d.row(i).to_vec();
        row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean_dist: f32 = row[..k].iter().sum::<f32>() / k as f32;
        scores.push(mean_dist);
    }

    scores
}

/// Score by Local Outlier Factor.
fn score_lof(xn: &Array2<f32>, k: usize) -> Vec<f32> {
    let n = xn.nrows();
    let k = k.max(2).min(n.saturating_sub(1));
    if k < 2 {
        return vec![1.0; n];
    }

    let d = cosine_distance_matrix(xn);

    // Find k-nearest neighbors for each point
    let mut knn_indices: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut k_dist: Vec<f32> = Vec::with_capacity(n);

    for i in 0..n {
        let mut pairs: Vec<(f32, usize)> = d
            .row(i)
            .iter()
            .enumerate()
            .map(|(j, &dist)| (dist, j))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let indices: Vec<usize> = pairs[..k].iter().map(|(_, j)| *j).collect();
        k_dist.push(pairs[k - 1].0);
        knn_indices.push(indices);
    }

    // Compute reachability distance and LRD
    let mut lrd = vec![0.0f32; n];
    for i in 0..n {
        let mut sum_reach = 0.0f32;
        for &neighbor in &knn_indices[i] {
            let reach = k_dist[neighbor].max(d[[i, neighbor]]);
            sum_reach += reach;
        }
        lrd[i] = 1.0 / (sum_reach / k as f32 + 1e-12);
    }

    // LOF score
    let mut scores = vec![1.0f32; n];
    for i in 0..n {
        let mut sum_ratio = 0.0f32;
        for &neighbor in &knn_indices[i] {
            sum_ratio += lrd[neighbor] / (lrd[i] + 1e-12);
        }
        scores[i] = sum_ratio / k as f32;
    }

    scores
}

/// Score by deviation from the local temporal neighborhood within each source file.
fn score_local_contrast(xn: &Array2<f32>, metadatas: &[&ChunkMetadata], radius: usize) -> Vec<f32> {
    let n = xn.nrows();
    if n == 0 {
        return Vec::new();
    }

    let radius = radius.max(1);
    let mut scores = vec![0.0f32; n];
    let mut groups = std::collections::HashMap::<&str, Vec<usize>>::new();

    for (index, metadata) in metadatas.iter().enumerate() {
        groups
            .entry(metadata.source_file.as_str())
            .or_default()
            .push(index);
    }

    for indices in groups.values_mut() {
        indices.sort_by(|left, right| {
            metadatas[*left]
                .start_time
                .total_cmp(&metadatas[*right].start_time)
        });

        for (position, &current_index) in indices.iter().enumerate() {
            let start = position.saturating_sub(radius);
            let end = (position + radius + 1).min(indices.len());
            let mut weighted_mean = vec![0.0f32; xn.ncols()];
            let mut weight_total = 0.0f32;

            for (neighbor_position, &neighbor_index) in
                indices.iter().enumerate().take(end).skip(start)
            {
                if neighbor_position == position {
                    continue;
                }

                let distance = position.abs_diff(neighbor_position).max(1) as f32;
                let weight = 1.0 / distance;

                for (dimension, value) in xn.row(neighbor_index).iter().enumerate() {
                    weighted_mean[dimension] += weight * *value;
                }
                weight_total += weight;
            }

            if weight_total <= f32::EPSILON {
                continue;
            }

            for value in &mut weighted_mean {
                *value /= weight_total;
            }

            let mean_norm = weighted_mean
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            if mean_norm <= 1e-12 {
                continue;
            }

            let similarity = xn
                .row(current_index)
                .iter()
                .zip(weighted_mean.iter())
                .map(|(current_value, mean_value)| current_value * (mean_value / mean_norm))
                .sum::<f32>();
            scores[current_index] = 1.0 - similarity;
        }
    }

    scores
}

/// Keep the half of points farthest from the index centroid.
fn exclude_baseline_mask(xn: &Array2<f32>) -> Array1<f32> {
    let n = xn.nrows();
    if n < 4 {
        return Array1::ones(n);
    }

    let mean = xn.mean_axis(Axis(0)).unwrap();
    let mean_norm = mean.mapv(|v| v * v).sum().sqrt().max(1e-12);
    let mean_unit = mean / mean_norm;
    let dist: Vec<f32> = xn.dot(&mean_unit).mapv(|v| 1.0 - v).to_vec();

    let mut sorted = dist.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[n / 2];

    Array1::from_vec(
        dist.into_iter()
            .map(|d| if d >= median { 1.0 } else { 0.0 })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_unit_length() {
        let x = Array2::from_shape_vec((2, 3), vec![3.0, 4.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let xn = normalize(&x);
        let row0_norm: f32 = xn.row(0).mapv(|v| v * v).sum().sqrt();
        assert!((row0_norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_score_centroid_returns_values() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0]).unwrap();
        let xn = normalize(&x);
        let scores = score_centroid(&xn);
        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn test_score_knn_returns_values() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.9, 0.1, 0.0, 1.0, -1.0, 0.0]).unwrap();
        let xn = normalize(&x);
        let scores = score_knn(&xn, 2);
        assert_eq!(scores.len(), 4);
    }

    #[test]
    fn test_score_lof_returns_values() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 0.0, 0.9, 0.1, 0.0, 1.0, -1.0, 0.0, 0.5, 0.5],
        )
        .unwrap();
        let xn = normalize(&x);
        let scores = score_lof(&xn, 2);
        assert_eq!(scores.len(), 5);
    }

    #[test]
    fn test_score_local_contrast_prioritizes_local_temporal_change() {
        let x = Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.98, 0.02, 0.0, 1.0, 0.97, 0.03])
            .unwrap();
        let xn = normalize(&x);
        let metadatas = vec![
            ChunkMetadata {
                source_file: "/tmp/a.mp4".into(),
                start_time: 0.0,
                end_time: 6.0,
                indexed_at: chrono::Utc::now(),
                backend: boomerang_core::types::EmbeddingBackend::Gemini,
                model: None,
                dimensions: 2,
            },
            ChunkMetadata {
                source_file: "/tmp/a.mp4".into(),
                start_time: 4.0,
                end_time: 10.0,
                indexed_at: chrono::Utc::now(),
                backend: boomerang_core::types::EmbeddingBackend::Gemini,
                model: None,
                dimensions: 2,
            },
            ChunkMetadata {
                source_file: "/tmp/a.mp4".into(),
                start_time: 8.0,
                end_time: 14.0,
                indexed_at: chrono::Utc::now(),
                backend: boomerang_core::types::EmbeddingBackend::Gemini,
                model: None,
                dimensions: 2,
            },
            ChunkMetadata {
                source_file: "/tmp/a.mp4".into(),
                start_time: 12.0,
                end_time: 18.0,
                indexed_at: chrono::Utc::now(),
                backend: boomerang_core::types::EmbeddingBackend::Gemini,
                model: None,
                dimensions: 2,
            },
        ];
        let metadata_refs = metadatas.iter().collect::<Vec<_>>();
        let scores = score_local_contrast(&xn, &metadata_refs, 1);

        assert!(scores[2] > scores[0]);
        assert!(scores[2] > scores[1]);
        assert!(scores[2] > scores[3]);
    }
}
