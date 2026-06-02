// Router aggregation and CORS setup for the REST API.

pub mod highlights;
pub mod index;
pub mod match_result;
pub mod search;

use axum::{extract::DefaultBodyLimit, routing::post, Router};
use tower_http::cors::{Any, CorsLayer};

use crate::state::AppState;

/// Combine routes and configure CORS.
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
        .with_state(state)
        .layer(cors)
}
