# Getting Started

This guide will help you get started with Inference Lab.

## Installation

Install from crates.io:

```bash
cargo install --locked inference-lab
```

Or build from source:

```bash
cargo build --release
```

## Running Your First Simulation

From a checkout of the repository:

```bash
inference-lab --config configs/llama-3-70b.toml --workload workloads/quick.toml
```

`configs/` holds one file per model (each with its hardware entries) and
`workloads/` the arrival patterns and request shapes; pass `--hardware
<name>` when a model config has more than one entry.

## Next Steps

- Learn about [configuration options](./configuration.md)
- Explore [running simulations](./running-simulations.md)
