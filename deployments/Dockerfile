FROM rust:1.85-slim-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates/boomerang-core/Cargo.toml crates/boomerang-core/
COPY crates/video-chunking/Cargo.toml crates/video-chunking/
COPY crates/semantic-embed/Cargo.toml crates/semantic-embed/
COPY crates/vector-store/Cargo.toml crates/vector-store/
COPY crates/footage-search/Cargo.toml crates/footage-search/
COPY crates/clip-trim/Cargo.toml crates/clip-trim/
COPY crates/boomerang-cli/Cargo.toml crates/boomerang-cli/

RUN mkdir -p crates/boomerang-core/src && echo 'fn main() {}' > crates/boomerang-core/src/lib.rs
RUN mkdir -p crates/video-chunking/src && echo 'fn main() {}' > crates/video-chunking/src/lib.rs
RUN mkdir -p crates/semantic-embed/src && echo 'fn main() {}' > crates/semantic-embed/src/lib.rs
RUN mkdir -p crates/vector-store/src && echo 'fn main() {}' > crates/vector-store/src/lib.rs
RUN mkdir -p crates/footage-search/src && echo 'fn main() {}' > crates/footage-search/src/lib.rs
RUN mkdir -p crates/clip-trim/src && echo 'fn main() {}' > crates/clip-trim/src/lib.rs
RUN mkdir -p crates/boomerang-cli/src && echo 'fn main() {}' > crates/boomerang-cli/src/main.rs

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
RUN cargo build --release 2>/dev/null || true

COPY crates/ crates/
RUN cargo build --release -p boomerang-cli

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ffmpeg ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/boomerang /usr/local/bin/boomerang

ENTRYPOINT ["boomerang"]
