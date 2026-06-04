<div align="center">
   <h1>Boomerang</h1>
</div>

<div align="center">
<p><b>Semantic search over video footage. Type what you're looking for, find the exact clip back</b></p>
</div>


## Architecture

- **Traits for polymorphism**: `Embedder`, `VectorStore`, `Chunker` are traits with multiple backends.
- **Newtypes for safety**: `ChunkId`, `CameraId`, `Resolution` wrap primitives.
- **Builder pattern** for complex construction (3+ optional fields).
- **Config is environment-driven**: `.env` files with `${ENV_VAR}` substitution.

## Algorithms

- **Overlapping chunking**: each video is split into windows of length `L` with overlap `O`, so chunk `i` covers `[i(L-O), i(L-O)+L]`. In the current CLI defaults, `L = 6s` and `O = 2s`. The overlap reduces boundary loss when an event straddles two chunks.

- **Preprocessed retrieval inputs**: before embedding, chunks can be downscaled and reduced in frame rate. The current defaults are `480p` and `4 fps`, which keep the retrieval signal while lowering embedding cost.

- **Unit-normalized embeddings**: every stored embedding and every query embedding is normalized to unit length. For a vector `x`,

  $$
  \hat{x} = \frac{x}{\lVert x \rVert}
  $$

  and zero-norm vectors are rejected. Because of that normalization, cosine similarity reduces to a dot product:

  $$
  \mathrm{sim}(\hat{x}, \hat{q}) = \hat{x} \cdot \hat{q}
  $$

- **Multi-query semantic retrieval**: text search does not rely on a single query embedding. The query is first expanded into multiple semantically related search phrasings, each phrasing is embedded, and each embedding retrieves candidate chunks from the vector store.

- **Reciprocal-rank fusion over expanded queries**: if a chunk appears in several per-query result lists, Boomerang merges those hits and scores them by consensus, not just by a single best match. For rank `r` and fusion constant `k`,

  $$
  \mathrm{rrf}(r) = \frac{1}{k + r + 1}
  $$

  and the final retrieval ranking is the sum of those reciprocal-rank contributions across query variants. The highest raw similarity is also preserved per chunk.

- **Thresholded candidate set**: after fusion, chunks whose best similarity is below the configured threshold `tau` are removed:

  $$
  R = \{x \mid \max_j \mathrm{sim}(x, q_j) \ge \tau\}
  $$

  where `q_j` are the expanded query embeddings.

- **Two temporal refinement modes**:
  - `exact`: after coarse retrieval, Boomerang re-extracts short local windows around the top hits, embeds those windows, and returns the earliest stable onset that stays relevant across nearby micro-windows. This is for queries like `everyone got seated` or `door closed`.
  - `span`: after coarse retrieval, Boomerang fetches neighboring indexed chunks from the same source file and searches for the best contiguous interval around the anchor hit.

- **Span refinement scoring**: in `span` mode, each neighboring chunk is rescored against the query embedding set using a fused score

  $$
  \mathrm{fused}(c) = 0.75 \cdot \max_j \mathrm{sim}(c, q_j) + 0.25 \cdot \mathrm{mean}_j\, \mathrm{sim}(c, q_j)
  $$

  and Boomerang picks the contiguous interval with the best aggregate gain above a threshold-derived baseline, capped at `24s`.

- **Exact-moment refinement**: in `exact` mode, Boomerang rescans only the top coarse hits using `1.0s` windows with `0.5s` stride, then selects the earliest window where relevance becomes stable across a short lookahead. This avoids returning too much lead-in context before the event actually happens.

- **Result materialization**: search returns timestamps and can immediately trim clips from the source footage, so retrieval is operational rather than just analytical.

- **Highlight scoring**:
  - `centroid`: anomaly score is distance from the normalized corpus mean `\mu`, so

    $$
    s(x) = 1 - x \cdot \mu
    $$
  - `knn`: anomaly score is the mean cosine distance to the `k` nearest neighbors
  - `lof`: anomaly score is Local Outlier Factor, comparing local density around a point to the density of its neighbors
  - `local-contrast`: anomaly score is deviation from the temporally local neighborhood within the same source video, so unusual moments relative to nearby context can surface even if they are not globally rare

- **Index isolation**: embeddings are stored per `(backend, model, dimensions)` space, so incompatible vectors never mix and searches cannot silently compare vectors from different embedding spaces.

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

### Demo Query

Use this exact-moment query when the target scene is “all people are seated and no one is standing”:

```bash
cargo run -p boomerang-cli -- search "all people are seated no one is standing" --match-mode exact
```

Reference scene:

![Seated scene demo](/scene.png)


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
