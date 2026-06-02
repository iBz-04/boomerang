//! Boomerang CLI — semantic search over video footage.
//!
//! Entry point for the boomerang command-line tool. Handles argument
//! parsing, logging setup, and command dispatch.

mod commands;

use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

/// Semantic search over video footage. Type what you're looking for, get a trimmed clip back.
#[derive(Parser)]
#[command(name = "boomerang", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Enable verbose debug output.
    #[arg(long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Set up API keys and configuration.
    Init,

    /// Index video footage into the vector store.
    Index(IndexArgs),

    /// Search indexed footage with a natural language query.
    Search(SearchArgs),

    /// Search indexed footage using an image as the query.
    Img(ImgArgs),

    /// Surface the most anomalous clips in your footage.
    Highlights(HighlightsArgs),

    /// Show index statistics.
    Stats,

    /// Remove specific files from the index.
    Remove(RemoveArgs),

    /// Wipe the entire index.
    Reset,
}

#[derive(clap::Args)]
struct IndexArgs {
    /// Path to video file or directory.
    path: String,

    /// Embedding backend: gemini, qwen-cloud, local.
    #[arg(long, default_value = "gemini")]
    backend: String,

    /// Duration of each chunk in seconds.
    #[arg(long, default_value = "30")]
    chunk_duration: u32,

    /// Overlap between chunks in seconds.
    #[arg(long, default_value = "5")]
    overlap: u32,

    /// Skip preprocessing (downscale + fps reduction).
    #[arg(long)]
    no_preprocess: bool,

    /// Skip still-frame detection.
    #[arg(long)]
    no_skip_still: bool,

    /// Target resolution height for preprocessing.
    #[arg(long, default_value = "480")]
    target_resolution: u32,

    /// Target frames per second for preprocessing.
    #[arg(long, default_value = "5")]
    target_fps: u32,
}

#[derive(clap::Args)]
struct SearchArgs {
    /// Natural language search query.
    query: String,

    /// Number of results to return.
    #[arg(short = 'n', long, default_value = "5")]
    results: usize,

    /// Minimum similarity threshold (0.0–1.0).
    #[arg(long, default_value = "0.41")]
    threshold: f64,

    /// Skip auto-trimming of top result.
    #[arg(long)]
    no_trim: bool,

    /// Output directory for trimmed clips.
    #[arg(long, default_value = ".")]
    output_dir: String,

    /// Save top N clips instead of just the best match.
    #[arg(long)]
    save_top: Option<usize>,

    /// Deduplication similarity ceiling (0.0–1.0).
    #[arg(long)]
    dedupe: Option<f64>,

    /// Burn Tesla metadata overlay onto clips.
    #[arg(long)]
    overlay: bool,
}

#[derive(clap::Args)]
struct ImgArgs {
    /// Path to the reference image.
    image_path: String,

    /// Number of results to return.
    #[arg(short = 'n', long, default_value = "5")]
    results: usize,

    /// Minimum similarity threshold.
    #[arg(long, default_value = "0.41")]
    threshold: f64,

    /// Skip auto-trimming.
    #[arg(long)]
    no_trim: bool,

    /// Output directory for trimmed clips.
    #[arg(long, default_value = ".")]
    output_dir: String,

    /// Save top N clips.
    #[arg(long)]
    save_top: Option<usize>,

    /// Deduplication threshold.
    #[arg(long)]
    dedupe: Option<f64>,
}

#[derive(clap::Args)]
struct HighlightsArgs {
    /// Number of highlights to return.
    #[arg(short = 'n', long, default_value = "5")]
    count: usize,

    /// Scoring method: centroid, knn, lof.
    #[arg(long, default_value = "knn")]
    method: String,

    /// Number of neighbors for KNN/LOF.
    #[arg(short = 'k', long, default_value = "10")]
    neighbors: usize,

    /// Deduplication threshold.
    #[arg(long, default_value = "0.9")]
    dedupe: f64,

    /// Exclude baseline (half nearest centroid).
    #[arg(long)]
    exclude_baseline: bool,

    /// Skip auto-trimming.
    #[arg(long)]
    no_trim: bool,

    /// Output directory for trimmed clips.
    #[arg(long, default_value = ".")]
    output_dir: String,
}

#[derive(clap::Args)]
struct RemoveArgs {
    /// Path substring to match for removal.
    path: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    let filter = if cli.verbose {
        EnvFilter::new("boomerang=debug")
    } else {
        EnvFilter::new("boomerang=info")
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    match cli.command {
        Command::Init => commands::init().await?,
        Command::Index(args) => commands::index(args).await?,
        Command::Search(args) => commands::search(args).await?,
        Command::Img(args) => commands::img(args).await?,
        Command::Highlights(args) => commands::highlights(args).await?,
        Command::Stats => commands::stats().await?,
        Command::Remove(args) => commands::remove(args).await?,
        Command::Reset => commands::reset().await?,
    }

    Ok(())
}
