//! Temporal reranking expands top hits with neighboring indexed chunks.

use boomerang_core::chunk::ChunkMetadata;
use boomerang_core::embedding::Embedding;
use boomerang_core::search::SearchResult;
use boomerang_core::store::VectorStore;

const MAX_REFINED_DURATION_SECONDS: f64 = 24.0;
const REFINE_DEDUPE_THRESHOLD: f64 = 0.6;
const REFINE_TOP_RESULTS: usize = 3;
const TEMPORAL_BASELINE_MARGIN: f64 = 0.03;
const CONTIGUITY_EPSILON_SECONDS: f64 = 0.25;

pub async fn refine_search_results(
    results: Vec<SearchResult>,
    query_embeddings: &[Embedding],
    store: &dyn VectorStore,
    threshold: f64,
) -> Result<Vec<SearchResult>, boomerang_core::error::CoreError> {
    if results.is_empty() || query_embeddings.is_empty() {
        return Ok(results);
    }

    let mut cache = std::collections::HashMap::<String, SourceSequence>::new();
    let mut refined = Vec::with_capacity(results.len());

    for (index, result) in results.into_iter().enumerate() {
        if index >= REFINE_TOP_RESULTS {
            refined.push(result);
            continue;
        }

        let source_file = result.source_file.clone();
        let sequence = match cache.entry(source_file.clone()) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let (embeddings, metadatas) = store.fetch_by_source_file(&source_file).await?;
                entry.insert(SourceSequence::new(
                    embeddings,
                    metadatas,
                    query_embeddings,
                    threshold,
                ))
            }
        };
        refined.push(sequence.refine_result(result, threshold));
    }

    refined.sort_by(compare_results);
    Ok(deduplicate_results(refined, REFINE_DEDUPE_THRESHOLD))
}

fn compare_results(left: &SearchResult, right: &SearchResult) -> std::cmp::Ordering {
    right
        .ranking_score
        .total_cmp(&left.ranking_score)
        .then_with(|| right.similarity_score.total_cmp(&left.similarity_score))
        .then_with(|| left.start_time.total_cmp(&right.start_time))
}

fn deduplicate_results(results: Vec<SearchResult>, threshold: f64) -> Vec<SearchResult> {
    let mut kept = Vec::new();

    for result in results {
        let duplicate = kept.iter().any(|existing: &SearchResult| {
            if existing.source_file != result.source_file {
                return false;
            }

            let overlap_start = existing.start_time.max(result.start_time);
            let overlap_end = existing.end_time.min(result.end_time);
            let overlap = (overlap_end - overlap_start).max(0.0);
            let min_duration =
                (existing.end_time - existing.start_time).min(result.end_time - result.start_time);
            min_duration > 0.0 && overlap / min_duration > threshold
        });

        if !duplicate {
            kept.push(result);
        }
    }

    kept
}

struct SourceSequence {
    chunks: Vec<ScoredChunk>,
}

impl SourceSequence {
    fn new(
        embeddings: Vec<Embedding>,
        metadatas: Vec<ChunkMetadata>,
        query_embeddings: &[Embedding],
        threshold: f64,
    ) -> Self {
        let baseline = (threshold - TEMPORAL_BASELINE_MARGIN).max(0.0);
        let mut chunks = embeddings
            .into_iter()
            .zip(metadatas)
            .map(|(embedding, metadata)| ScoredChunk {
                start_time: metadata.start_time,
                end_time: metadata.end_time,
                similarity: fused_similarity(query_embeddings, &embedding),
                contribution: fused_similarity(query_embeddings, &embedding) - baseline,
            })
            .collect::<Vec<_>>();
        chunks.sort_by(|left, right| left.start_time.total_cmp(&right.start_time));
        Self { chunks }
    }

    fn refine_result(&self, mut result: SearchResult, threshold: f64) -> SearchResult {
        if self.chunks.is_empty() {
            return result;
        }

        let baseline = (threshold - TEMPORAL_BASELINE_MARGIN).max(0.0);
        let anchor_indices = self.anchor_indices(&result);
        let best_interval = anchor_indices
            .into_iter()
            .filter_map(|anchor| self.best_interval_containing(anchor, baseline))
            .max_by(compare_interval)
            .unwrap_or_else(|| ChunkInterval::single(self.best_anchor(&result), &self.chunks));

        result.start_time = best_interval.start_time;
        result.end_time = best_interval.end_time;
        result.similarity_score = best_interval.peak_similarity;
        result.ranking_score += best_interval.gain;
        result
    }

    fn anchor_indices(&self, result: &SearchResult) -> Vec<usize> {
        let overlapping = self
            .chunks
            .iter()
            .enumerate()
            .filter_map(|(index, chunk)| {
                let overlap_start = chunk.start_time.max(result.start_time);
                let overlap_end = chunk.end_time.min(result.end_time);
                if overlap_end - overlap_start > CONTIGUITY_EPSILON_SECONDS {
                    Some(index)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        if overlapping.is_empty() {
            vec![self.best_anchor(result)]
        } else {
            overlapping
        }
    }

    fn best_anchor(&self, result: &SearchResult) -> usize {
        let target_midpoint = (result.start_time + result.end_time) / 2.0;
        self.chunks
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                let left_midpoint = (left.start_time + left.end_time) / 2.0;
                let right_midpoint = (right.start_time + right.end_time) / 2.0;
                (left_midpoint - target_midpoint)
                    .abs()
                    .total_cmp(&(right_midpoint - target_midpoint).abs())
            })
            .map(|(index, _)| index)
            .unwrap_or(0)
    }

    fn best_interval_containing(&self, anchor: usize, baseline: f64) -> Option<ChunkInterval> {
        let left_bound = self.left_bound(anchor);
        let right_bound = self.right_bound(anchor);
        let mut best: Option<ChunkInterval> = None;

        for start in left_bound..=anchor {
            for end in anchor..=right_bound {
                let start_time = self.chunks[start].start_time;
                let end_time = self.chunks[end].end_time;
                if end_time - start_time > MAX_REFINED_DURATION_SECONDS {
                    continue;
                }

                let interval = ChunkInterval::from_range(&self.chunks[start..=end], baseline);
                if best
                    .as_ref()
                    .is_none_or(|existing| compare_interval(&interval, existing).is_gt())
                {
                    best = Some(interval);
                }
            }
        }

        best
    }

    fn left_bound(&self, anchor: usize) -> usize {
        let mut index = anchor;
        while index > 0 {
            let current = &self.chunks[index];
            let previous = &self.chunks[index - 1];
            if current.start_time - previous.end_time > CONTIGUITY_EPSILON_SECONDS {
                break;
            }
            if current.end_time - previous.start_time > MAX_REFINED_DURATION_SECONDS {
                break;
            }
            index -= 1;
        }
        index
    }

    fn right_bound(&self, anchor: usize) -> usize {
        let mut index = anchor;
        while index + 1 < self.chunks.len() {
            let current = &self.chunks[index];
            let next = &self.chunks[index + 1];
            if next.start_time - current.end_time > CONTIGUITY_EPSILON_SECONDS {
                break;
            }
            if next.end_time - self.chunks[anchor].start_time > MAX_REFINED_DURATION_SECONDS {
                break;
            }
            index += 1;
        }
        index
    }
}

fn compare_interval(left: &ChunkInterval, right: &ChunkInterval) -> std::cmp::Ordering {
    left.gain
        .total_cmp(&right.gain)
        .then_with(|| left.peak_similarity.total_cmp(&right.peak_similarity))
        .then_with(|| {
            (right.end_time - right.start_time).total_cmp(&(left.end_time - left.start_time))
        })
}

fn fused_similarity(query_embeddings: &[Embedding], candidate: &Embedding) -> f64 {
    let mut best = f64::NEG_INFINITY;
    let mut total = 0.0;

    for query_embedding in query_embeddings {
        let similarity = cosine_similarity(query_embedding.as_slice(), candidate.as_slice());
        best = best.max(similarity);
        total += similarity;
    }

    let mean = total / query_embeddings.len() as f64;
    0.75 * best + 0.25 * mean
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(left_value, right_value)| f64::from(*left_value) * f64::from(*right_value))
        .sum()
}

#[derive(Clone)]
struct ScoredChunk {
    start_time: f64,
    end_time: f64,
    similarity: f64,
    contribution: f64,
}

#[derive(Clone)]
struct ChunkInterval {
    start_time: f64,
    end_time: f64,
    peak_similarity: f64,
    gain: f64,
}

impl ChunkInterval {
    fn single(index: usize, chunks: &[ScoredChunk]) -> Self {
        Self::from_range(&chunks[index..=index], 0.0)
    }

    fn from_range(chunks: &[ScoredChunk], baseline: f64) -> Self {
        let start_time = chunks.first().map(|chunk| chunk.start_time).unwrap_or(0.0);
        let end_time = chunks
            .last()
            .map(|chunk| chunk.end_time)
            .unwrap_or(start_time);
        let peak_similarity = chunks
            .iter()
            .map(|chunk| chunk.similarity)
            .max_by(|left, right| left.total_cmp(right))
            .unwrap_or(baseline);
        let gain = chunks.iter().map(|chunk| chunk.contribution).sum();

        Self {
            start_time,
            end_time,
            peak_similarity,
            gain,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedding(values: Vec<f32>) -> Embedding {
        Embedding::new(values).expect("embedding should normalize")
    }

    fn metadata(start_time: f64, end_time: f64) -> ChunkMetadata {
        ChunkMetadata {
            source_file: "/tmp/video.mp4".into(),
            start_time,
            end_time,
            indexed_at: chrono::Utc::now(),
            backend: boomerang_core::types::EmbeddingBackend::Gemini,
            model: None,
            dimensions: 2,
        }
    }

    #[test]
    fn test_refine_result_expands_over_supported_neighbors() {
        let sequence = SourceSequence::new(
            vec![
                embedding(vec![1.0, 0.0]),
                embedding(vec![0.95, 0.05]),
                embedding(vec![0.9, 0.1]),
                embedding(vec![0.0, 1.0]),
            ],
            vec![
                metadata(0.0, 6.0),
                metadata(4.0, 10.0),
                metadata(8.0, 14.0),
                metadata(12.0, 18.0),
            ],
            &[embedding(vec![1.0, 0.0])],
            0.41,
        );
        let refined = sequence.refine_result(
            SearchResult::new("/tmp/video.mp4".into(), 4.0, 10.0, 0.9),
            0.41,
        );

        assert_eq!(refined.start_time, 0.0);
        assert_eq!(refined.end_time, 14.0);
        assert!(refined.similarity_score > 0.9);
    }

    #[test]
    fn test_refine_result_keeps_anchor_when_neighbors_are_off_topic() {
        let sequence = SourceSequence::new(
            vec![
                embedding(vec![0.0, 1.0]),
                embedding(vec![1.0, 0.0]),
                embedding(vec![0.0, 1.0]),
            ],
            vec![metadata(0.0, 6.0), metadata(4.0, 10.0), metadata(8.0, 14.0)],
            &[embedding(vec![1.0, 0.0])],
            0.41,
        );
        let refined = sequence.refine_result(
            SearchResult::new("/tmp/video.mp4".into(), 4.0, 10.0, 0.9),
            0.41,
        );

        assert_eq!(refined.start_time, 4.0);
        assert_eq!(refined.end_time, 10.0);
    }

    #[test]
    fn test_deduplicate_results_removes_expanded_overlap() {
        let deduped = deduplicate_results(
            vec![
                SearchResult::new("/tmp/video.mp4".into(), 0.0, 14.0, 0.92).with_ranking(2.0, 2, 0),
                SearchResult::new("/tmp/video.mp4".into(), 4.0, 10.0, 0.9).with_ranking(1.0, 1, 1),
            ],
            0.6,
        );

        assert_eq!(deduped.len(), 1);
        assert_eq!(deduped[0].start_time, 0.0);
        assert_eq!(deduped[0].end_time, 14.0);
    }
}
