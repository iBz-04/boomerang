//! CLI command implementations.
//!
//! Each public function corresponds to a subcommand in the CLI.

use std::path::Path;

use anyhow::{Context, Result};
use boomerang_core::chunk::ChunkingConfig;
use boomerang_core::search::{HighlightConfig, SearchConfig};
use boomerang_core::types::EmbeddingSpace;
use tracing::info;

use crate::{HighlightsArgs, ImgArgs, IndexArgs, RemoveArgs, SearchArgs};

/// Initialize configuration and validate API keys.
pub async fn init() -> Result<()> {
    info!("Initializing boomerang...");

    let gemini_key = std::env::var("GEMINI_API_KEY").ok();
    if gemini_key.is_none() {
        info!("GEMINI_API_KEY not set. Set it in .env or your environment.");
        info!("Get a key at https://aistudio.google.com/apikey");
    } else {
        info!("GEMINI_API_KEY found.");
    }

    info!("Setup complete. Run 'boomerang index <directory>' to get started.");
    Ok(())
}

fn render_space(space: &EmbeddingSpace) -> String {
    match &space.model {
        Some(model) => format!("{} ({model})", space.backend),
        None => space.backend.to_string(),
    }
}

async fn resolve_space(backend: Option<&str>, model: Option<&str>) -> Result<EmbeddingSpace> {
    if let Some(backend_name) = backend {
        let embedder = semantic_embed::create_embedder(backend_name, model)
            .context("failed to create embedder for requested backend")?;
        return embedder
            .embedding_space()
            .context("failed to resolve requested embedding space");
    }

    vector_store::detect_space("qdrant")
        .await
        .context("failed to detect indexed embedding space")?
        .context("no indexed embedding space found")
}

/// Index video footage into the vector store.
pub async fn index(args: IndexArgs) -> Result<()> {
    let path = Path::new(&args.path);

    let config = ChunkingConfig {
        chunk_duration: args.chunk_duration,
        overlap: args.overlap,
        target_resolution: args.target_resolution,
        target_fps: args.target_fps,
        skip_preprocess: args.no_preprocess,
        skip_still_detection: args.no_skip_still,
    };

    // Find video files
    let video_files = if path.is_dir() {
        video_chunking::scanner::scan_directory(path)?
    } else if video_chunking::is_supported_video(path) {
        vec![path.to_path_buf()]
    } else {
        anyhow::bail!("path is not a directory or supported video file");
    };

    info!(count = video_files.len(), "found video files");

    if video_files.is_empty() {
        info!("no video files found");
        return Ok(());
    }

    // Create embedder
    let embedder = semantic_embed::create_embedder(&args.backend, args.model.as_deref())
        .context("failed to create embedder")?;
    let embedding_space = embedder
        .embedding_space()
        .context("failed to resolve embedding space")?;

    // Create vector store
    let store = vector_store::create_store("qdrant", &embedding_space)
        .await
        .context("failed to create vector store")?;

    let tmp_dir = tempfile::tempdir()?;

    for video_file in &video_files {
        let filename = video_file.file_name().unwrap_or_default().to_string_lossy();

        // Skip already-indexed files
        if store.is_file_indexed(&video_file.to_string_lossy()).await? {
            info!(file = %filename, "already indexed, skipping");
            continue;
        }

        info!(file = %filename, "indexing");

        // Chunk the video
        let chunks = video_chunking::chunker::chunk_video(video_file, &config, tmp_dir.path())
            .await
            .context(format!("failed to chunk {}", video_file.display()))?;

        let total = chunks.len();
        for (i, chunk) in chunks.iter().enumerate() {
            // Still-frame detection
            if !config.skip_still_detection {
                let chunk_p = Path::new(&chunk.chunk_path);
                if video_chunking::still_frame::is_still_frame(chunk_p, 0.98).await? {
                    info!(chunk = i + 1, total, "skipping still frame chunk");
                    continue;
                }
            }

            // Preprocess
            let chunk_p = Path::new(&chunk.chunk_path);
            let processed = video_chunking::chunker::preprocess_chunk(chunk_p, &config)
                .await
                .context("preprocessing failed")?;

            // Embed
            let embedding = embedder
                .embed_video(&processed.to_string_lossy())
                .await
                .context(format!("failed to embed chunk {}/{}", i + 1, total))?;

            // Store
            let metadata = boomerang_core::chunk::ChunkMetadata {
                source_file: chunk.source_file.clone(),
                start_time: chunk.time_range.start,
                end_time: chunk.time_range.end,
                indexed_at: chrono::Utc::now(),
                backend: embedding_space.backend,
                model: embedding_space.model.clone(),
                dimensions: embedding_space.dimensions,
            };

            store.add(&chunk.id, &embedding, &metadata).await?;
            info!(chunk = i + 1, total, "indexed chunk");
        }
    }

    let stats = store.stats().await?;
    info!(
        chunks = stats.total_chunks,
        files = stats.unique_source_files,
        "indexing complete"
    );

    Ok(())
}

/// Search indexed footage with a text query.
pub async fn search(args: SearchArgs) -> Result<()> {
    let embedding_space = resolve_space(args.backend.as_deref(), args.model.as_deref()).await?;
    let embedder = semantic_embed::create_embedder(
        embedding_space.backend.as_str(),
        embedding_space.model.as_deref(),
    )?;
    let store = vector_store::create_store("qdrant", &embedding_space).await?;

    let query_embedding = embedder.embed_query(&args.query).await?;

    let search_config = SearchConfig {
        max_results: args.results,
        threshold: args.threshold,
        dedupe_threshold: args.dedupe,
    };

    let results =
        footage_search::search_by_text(store.as_ref(), query_embedding.as_slice(), &search_config)
            .await?;

    if results.is_empty() {
        info!(space = %render_space(&embedding_space), "no results found");
        return Ok(());
    }

    // Display results
    for (i, result) in results.iter().enumerate() {
        let filename = Path::new(&result.source_file)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();
        println!(
            "  #{:<2} [{:.2}] {} @ {}s-{}s",
            i + 1,
            result.similarity_score,
            filename,
            result.start_time,
            result.end_time
        );
    }

    // Trim clips
    if !args.no_trim {
        let count = args.save_top.unwrap_or(1);
        let output_dir = Path::new(&args.output_dir);

        let clips = clip_trim::trim_top_results(&results, output_dir, count).await?;
        for clip in &clips {
            println!("\nSaved clip: {}", clip.display());
        }
    }

    Ok(())
}

/// Search indexed footage with an image query.
pub async fn img(args: ImgArgs) -> Result<()> {
    let embedding_space = resolve_space(args.backend.as_deref(), args.model.as_deref()).await?;
    let embedder = semantic_embed::create_embedder(
        embedding_space.backend.as_str(),
        embedding_space.model.as_deref(),
    )?;
    let store = vector_store::create_store("qdrant", &embedding_space).await?;

    let image_embedding = embedder.embed_image(&args.image_path).await?;

    let search_config = SearchConfig {
        max_results: args.results,
        threshold: args.threshold,
        dedupe_threshold: args.dedupe,
    };

    let results =
        footage_search::search_by_image(store.as_ref(), image_embedding.as_slice(), &search_config)
            .await?;

    if results.is_empty() {
        info!(space = %render_space(&embedding_space), "no results found");
        return Ok(());
    }

    for (i, result) in results.iter().enumerate() {
        let filename = Path::new(&result.source_file)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();
        println!(
            "  #{:<2} [{:.2}] {} @ {}s-{}s",
            i + 1,
            result.similarity_score,
            filename,
            result.start_time,
            result.end_time
        );
    }

    if !args.no_trim {
        let count = args.save_top.unwrap_or(1);
        let output_dir = Path::new(&args.output_dir);
        let clips = clip_trim::trim_top_results(&results, output_dir, count).await?;
        for clip in &clips {
            println!("\nSaved clip: {}", clip.display());
        }
    }

    Ok(())
}

/// Rank the most anomalous clips in the index.
pub async fn highlights(args: HighlightsArgs) -> Result<()> {
    let embedding_space = resolve_space(args.backend.as_deref(), args.model.as_deref()).await?;
    let store = vector_store::create_store("qdrant", &embedding_space).await?;

    let method = match args.method.as_str() {
        "centroid" => boomerang_core::types::ScoringMethod::Centroid,
        "knn" => boomerang_core::types::ScoringMethod::Knn,
        "lof" => boomerang_core::types::ScoringMethod::Lof,
        _ => anyhow::bail!(
            "unknown scoring method: {}. Use centroid, knn, or lof.",
            args.method
        ),
    };

    let config = HighlightConfig {
        count: args.count,
        method,
        neighbors: args.neighbors,
        dedupe_threshold: args.dedupe,
        exclude_baseline: args.exclude_baseline,
    };

    let results = footage_search::rank_highlights(store.as_ref(), &config).await?;

    if results.is_empty() {
        info!("no highlights found (index may be empty)");
        return Ok(());
    }

    for (i, result) in results.iter().enumerate() {
        let filename = Path::new(&result.source_file)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();
        println!(
            "  #{:<2} [{:.3}] {} @ {}s-{}s",
            i + 1,
            result.similarity_score,
            filename,
            result.start_time,
            result.end_time
        );
    }

    if !args.no_trim {
        let output_dir = Path::new(&args.output_dir);
        let clips = clip_trim::trim_top_results(&results, output_dir, args.count).await?;
        for clip in &clips {
            println!("\nSaved clip: {}", clip.display());
        }
    }

    Ok(())
}

/// Show index statistics.
pub async fn stats() -> Result<()> {
    let spaces = vector_store::list_spaces("qdrant").await?;
    if spaces.is_empty() {
        println!("No indexed embedding spaces found.");
        return Ok(());
    }

    for (index, embedding_space) in spaces.iter().enumerate() {
        if index > 0 {
            println!();
        }

        let store = vector_store::create_store("qdrant", embedding_space).await?;
        let stats = store.stats().await?;

        println!("Backend: {}", stats.embedding_space.backend);
        if let Some(ref model) = stats.embedding_space.model {
            println!("Model: {model}");
        }
        println!("Dimensions: {}", stats.embedding_space.dimensions);
        println!("Total chunks: {}", stats.total_chunks);
        println!("Unique source files: {}", stats.unique_source_files);

        if !stats.source_files.is_empty() {
            println!("\nSource files:");
            for file in &stats.source_files {
                let exists = Path::new(file).exists();
                let marker = if exists { "" } else { " [missing]" };
                println!("  {file}{marker}");
            }
        }
    }

    Ok(())
}

/// Remove specific files from the index.
pub async fn remove(args: RemoveArgs) -> Result<()> {
    let spaces = vector_store::list_spaces("qdrant").await?;
    let mut total_removed = 0usize;
    for embedding_space in &spaces {
        let store = vector_store::create_store("qdrant", embedding_space).await?;
        total_removed += store.remove_file(&args.path).await?;
    }
    info!(removed = total_removed, "removed chunks matching path");
    Ok(())
}

/// Wipe the entire index.
pub async fn reset() -> Result<()> {
    let spaces = vector_store::list_spaces("qdrant").await?;
    for embedding_space in &spaces {
        let store = vector_store::create_store("qdrant", embedding_space).await?;
        store.clear().await?;
    }
    info!(spaces = spaces.len(), "index wiped");
    Ok(())
}
