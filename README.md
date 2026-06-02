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


## Testing

- Unit tests in `#[cfg(test)] mod tests` within each crate.
- Integration tests in `tests/` directory of each crate.
- Test naming: `test_<unit>_<scenario>_<expected>`.
- Use `proptest` for numeric logic, `insta` for snapshot tests.

## Citation

```bibtex
@software{
  author       = {Rayamah, Ibrahim},
  title        = {Boomerang, Ultra fast semantic video search},
  year         = {2025},
  publisher    = {GitHub},
  url={https://github.com/iBz-04/boomerang}
}

