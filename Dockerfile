FROM rust:1.93-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y pkg-config libssl-dev g++ && rm -rf /var/lib/apt/lists/*

# Copy manifests first for layer caching
COPY Cargo.toml Cargo.lock ./

# Create dummy source to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs && echo "" > src/lib.rs
RUN cargo build --release --features serve 2>/dev/null || true
RUN rm -rf src

# Copy actual source and touch to invalidate cargo fingerprints
COPY src/ src/
RUN touch src/main.rs src/lib.rs

# Build the real binary
RUN cargo build --release --features serve

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/inference-lab /usr/local/bin/inference-lab

EXPOSE 8080

ENTRYPOINT ["inference-lab", "serve"]
CMD ["--port", "8080", "--host", "0.0.0.0"]
