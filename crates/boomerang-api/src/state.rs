// Shared application state defining backend configuration and request counters.

use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

/// Shared application state for the Boomerang API.
#[derive(Clone, Debug)]
pub struct AppState {
    pub upload_dir: PathBuf,
    pub backend: String,
    pub model: Option<String>,
    request_counter: Arc<AtomicU64>,
}

impl AppState {
    /// Create a new AppState with the given directories and configuration.
    pub fn new(upload_dir: PathBuf, backend: String, model: Option<String>) -> Self {
        Self {
            upload_dir,
            backend,
            model,
            request_counter: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Return the next API request identifier for correlated logs.
    pub fn next_request_id(&self) -> u64 {
        self.request_counter.fetch_add(1, Ordering::Relaxed)
    }
}
