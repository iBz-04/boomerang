//! Recursive video file scanner.

use std::path::{Path, PathBuf};

use crate::is_supported_video;

/// Recursively find supported video files in a directory.
pub fn scan_directory(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut videos = Vec::new();
    scan_recursive(root, &mut videos)?;
    videos.sort();
    Ok(videos)
}

fn scan_recursive(dir: &Path, videos: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            scan_recursive(&path, videos)?;
        } else if is_supported_video(&path) {
            videos.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_scan_finds_video_files() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("test.mp4"), b"fake").unwrap();
        fs::write(dir.path().join("test.mov"), b"fake").unwrap();
        fs::write(dir.path().join("readme.txt"), b"fake").unwrap();

        let videos = scan_directory(dir.path()).unwrap();
        assert_eq!(videos.len(), 2);
    }

    #[test]
    fn test_scan_empty_directory() {
        let dir = TempDir::new().unwrap();
        let videos = scan_directory(dir.path()).unwrap();
        assert!(videos.is_empty());
    }
}
