# AGENTS.md

## Project Overview

Boomerang — semantic search over video footage. Type what you're looking for, get a trimmed clip back.
Rust rewrite of SentrySearch, following the NVIDIA VSS agent architecture pattern.

**Tech stack:** Rust 2024 edition, Tokio async runtime, Qdrant vector store, Gemini Embedding API.
Package manager: Cargo. Linter/formatter: clippy + rustfmt. Type checker: rustc.

## Commands

```bash
# Build
cargo build --release

# Test
cargo test --workspace
cargo test -p boomerang-core

# Lint & format
cargo clippy --workspace -- -D warnings
cargo fmt --check

# Run
cargo run -p boomerang-cli -- index /path/to/footage
cargo run -p boomerang-cli -- search "red truck running a stop sign"
```

## Project Structure

```
crates/
├── boomerang-core/      # Domain types, shared traits (Chunk, Embedding, SearchResult)
├── video-chunking/      # Video splitting, still-frame detection, preprocessing via ffmpeg
├── semantic-embed/      # Embedding backends (Gemini API, local model, qwen-cloud)
├── vector-store/        # Vector storage abstraction (Qdrant backend)
├── footage-search/      # Semantic search + anomaly-based highlights ranking
├── clip-trim/           # ffmpeg clip extraction with padding
└── boomerang-cli/       # CLI application (clap)
deployments/             # Docker, compose, config profiles
```

## Architecture Patterns

- **Traits for polymorphism**: `Embedder`, `VectorStore`, `Chunker` are traits with multiple backends.
- **Newtypes for safety**: `ChunkId`, `CameraId`, `Resolution` wrap primitives.
- **Builder pattern** for complex construction (3+ optional fields).
- **Config is environment-driven**: `.env` files with `${ENV_VAR}` substitution.

## Testing

- Unit tests in `#[cfg(test)] mod tests` within each crate.
- Integration tests in `tests/` directory of each crate.
- Test naming: `test_<unit>_<scenario>_<expected>`.
- Use `proptest` for numeric logic, `insta` for snapshot tests.

## Boundaries

- **Always**: run `cargo clippy` + `cargo fmt --check` after changes, write tests for new code.
- **Ask first**: adding new dependencies to `Cargo.toml`, changing trait signatures in `boomerang-core`.
- **Never**: commit secrets or API keys, hardcode IPs/URLs, use `.unwrap()` in library code.
