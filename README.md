<div align="center">
   <h1>Boomerang</h1>
</div>

<div align="center">
<p><b>Semantic search over video footage. Type what you're looking for, find the exact clip back</b></p>
</div>





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

## Testing

- Unit tests in `#[cfg(test)] mod tests` within each crate.
- Integration tests in `tests/` directory of each crate.
- Test naming: `test_<unit>_<scenario>_<expected>`.
- Use `proptest` for numeric logic, `insta` for snapshot tests.



