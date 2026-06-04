<div align="center">
   <h1>Boomerang</h1>
</div>

<div align="center">
<p><b>Semantic search over video footage. Type what you're looking for, find the exact clip back</b></p>
</div>


## Project Structure

```
crates/
├── boomerang-core/      # Domain types, shared traits
├── video-chunking/      # Video processing
├── semantic-embed/      # Embedding backends
├── vector-store/        # Vector storage 
├── footage-search/      # Semantic search + ranking
├── clip-trim/           # Clip extraction
└── boomerang-cli/       # CLI application 
deployments/             # Docker, compose, config profiles
```

## Architecture Patterns

- **Traits for polymorphism**: `Embedder`, `VectorStore`, `Chunker` are traits with multiple backends.
- **Newtypes for safety**: `ChunkId`, `CameraId`, `Resolution` wrap primitives.
- **Builder pattern** for complex construction (3+ optional fields).
- **Config is environment-driven**: `.env` files with `${ENV_VAR}` substitution.

## Core Algorithms

- **Overlapping chunking**: each video is split into windows of length `L` with overlap `O`, so chunk `i` covers `[i(L-O), i(L-O)+L]`. The overlap reduces boundary loss when an event spans two chunks.
- **Cross-modal retrieval**: every chunk and every query are embedded into the same vector space. Search ranks chunks by cosine similarity

  `cos(x, q) = (x · q) / (||x|| ||q||)`

  where `x` is a stored chunk embedding and `q` is the query embedding.
- **Confidence filtering**: results below the configured threshold `tau` are dropped, so the returned set is

  `R = {x : cos(x, q) >= tau}`.

- **Highlight scoring**:
  - `centroid`: anomaly score is distance from the normalized corpus mean `mu`, so `s(x) = 1 - x · mu`
  - `knn`: anomaly score is the mean cosine distance to the `k` nearest neighbors
  - `lof`: anomaly score is Local Outlier Factor, comparing local density around a point to the density of its neighbors
- **Index isolation**: embeddings are stored per `(backend, model, dimensions)` space, so incompatible vectors never mix.

## Commands

### Build And Quality

| Command | Purpose |
| --- | --- |
| `cargo build --release` | Build the release binary. |
| `cargo test --workspace` | Run the full workspace test suite. |
| `cargo test -p boomerang-core` | Run tests for one crate. |
| `cargo clippy --workspace -- -D warnings` | Run lints and fail on warnings. |
| `cargo fmt --check` | Verify formatting. |

### CLI Reference

Global flag:

| Flag | Purpose |
| --- | --- |
| `--verbose` | Enable debug logging for any subcommand. |

Subcommands:

| Command | Purpose | Important Flags | Example |
| --- | --- | --- | --- |
| `boomerang init` | Check local configuration and report whether required API keys are present. | None | `cargo run -p boomerang-cli -- init` |
| `boomerang index <path>` | Index one video file or a directory of videos into the vector store. | `--backend <gemini\|qwen-cloud\|local>`, `--model <name>`, `--chunk-duration <seconds>`, `--overlap <seconds>`, `--no-preprocess`, `--no-skip-still`, `--target-resolution <height>`, `--target-fps <fps>` | `cargo run -p boomerang-cli -- index /path/to/footage --chunk-duration 6 --overlap 2` |
| `boomerang search "<query>"` | Search indexed footage with a natural-language query and optionally trim the best results. | `-n, --results <count>`, `--threshold <0.0-1.0>`, `--match-mode <exact\|span>`, `--dedupe <0.0-1.0>`, `--save-top <count>`, `--no-trim`, `--output-dir <path>`, `--backend <name>`, `--model <name>` | `cargo run -p boomerang-cli -- search "everyone got seated" --match-mode exact` |
| `boomerang img <image-path>` | Search indexed footage using an image query and optionally trim the best results. | `-n, --results <count>`, `--threshold <0.0-1.0>`, `--match-mode <exact\|span>`, `--dedupe <0.0-1.0>`, `--save-top <count>`, `--no-trim`, `--output-dir <path>`, `--backend <name>`, `--model <name>` | `cargo run -p boomerang-cli -- img ./frame.png --match-mode span` |
| `boomerang highlights` | Rank the most anomalous clips in the index and optionally trim them. | `-n, --count <count>`, `--method <centroid\|knn\|lof\|local-contrast>`, `-k, --neighbors <count>`, `--dedupe <0.0-1.0>`, `--exclude-baseline`, `--no-trim`, `--output-dir <path>`, `--backend <name>`, `--model <name>` | `cargo run -p boomerang-cli -- highlights --method local-contrast -n 10` |
| `boomerang stats` | Show index statistics for the configured vector spaces. | None | `cargo run -p boomerang-cli -- stats` |
| `boomerang remove <path-substring>` | Remove indexed entries whose source path matches the provided substring. | None | `cargo run -p boomerang-cli -- remove dashcam_trip_042` |
| `boomerang reset` | Wipe the entire index. Use carefully. | None | `cargo run -p boomerang-cli -- reset` |

### Match Modes

| Mode | When To Use It | Behavior |
| --- | --- | --- |
| `exact` | Queries about a specific moment or state change, such as `everyone got seated`, `door closed`, or `car stopped`. | Runs coarse retrieval first, then refines the top hits with short local windows to return the earliest stable matching moment. This is the default. |
| `span` | Queries where surrounding context is useful, such as `people entering the room` or `crowd cheering during the play`. | Uses broader temporal reranking that can expand the result to include nearby relevant context. |

### Quick Start

```bash
# Index footage
cargo run -p boomerang-cli -- index /path/to/footage

# Search for an exact moment
cargo run -p boomerang-cli -- search "everyone got seated" --match-mode exact

# Search for a broader contextual span
cargo run -p boomerang-cli -- search "people entering the theater" --match-mode span
```


## Testing

- Unit tests in `#[cfg(test)] mod tests` within each crate.
- Integration tests in `tests/` directory of each crate.
- Test naming: `test_<unit>_<scenario>_<expected>`.
- Use `proptest` for numeric logic, `insta` for snapshot tests.

## Citation

```bibtex
@software{
  author       = {Rayamah, Ibrahim},
  title        = {Boomerang, Exact-Moment search & retrieval for videos},
  year         = {2025},
  publisher    = {GitHub},
  url={https://github.com/iBz-04/boomerang}
}
