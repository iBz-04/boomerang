// Shared application state defining the backend configuration and directories.

use std::path::PathBuf;

/// Shared application state for the Boomerang API.
#[derive(Clone, Debug)]
pub struct AppState {
    pub upload_dir: PathBuf,
    pub backend: String,
    pub model: Option<String>,
}

impl AppState {
    /// Create a new AppState with the given directories and configuration.
    pub fn new(upload_dir: PathBuf, backend: String, model: Option<String>) -> Self {
        Self {
            upload_dir,
            backend,
            model,
        }
    }
}
