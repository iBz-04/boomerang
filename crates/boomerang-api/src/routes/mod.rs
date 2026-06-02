// Router aggregation, CORS setup, and static clips directory serving.

pub mod highlights;
pub mod index;
pub mod search;

use axum::{
    extract::DefaultBodyLimit,
    routing::post,
    Router,
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;

use crate::state::AppState;

/// Combine routes, configure CORS, and set up static file serving for clips.
pub fn create_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route(
            "/index",
            post(index::index_handler).layer(DefaultBodyLimit::disable()),
        )
        .route("/search", post(search::search_handler))
        .route("/highlights", post(highlights::highlights_handler))
        .nest_service("/clips", ServeDir::new(&state.clips_dir))
        .with_state(state)
        .layer(cors)
}
