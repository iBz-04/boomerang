// Entry point for the Axum REST API server. Binds to port 8080 and handles routes.

mod error;
mod routes;
mod state;

use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

use crate::state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    // Initialize tracing/logging
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(
            "boomerang_api=info,footage_search=info,vector_store=info,semantic_embed=info",
        )
    });
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    tracing::info!("Starting Boomerang REST API server...");

    let upload_dir = PathBuf::from("./uploads");

    tokio::fs::create_dir_all(&upload_dir).await?;

    // Load backend config
    let backend = std::env::var("BOOMERANG_BACKEND").unwrap_or_else(|_| "gemini".to_string());
    let model = std::env::var("BOOMERANG_MODEL").ok();

    tracing::info!(
        backend = %backend,
        model = ?model,
        uploads = %upload_dir.display(),
        "initialized backend and directories"
    );

    let state = AppState::new(upload_dir, backend, model);
    let app = routes::create_router(state);

    let addr = "127.0.0.1:8080";
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Server listening on http://{}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}
