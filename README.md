# Inference Lab

LLM inference simulator for analyzing serving systems.
Simulates GPU clusters serving LLM inference workloads with realistic
performance modeling.

## Features

- **Accurate Performance Modeling**: Models compute (FLOPS) and memory bandwidth constraints
- **Multiple Scheduling Policies**: FCFS, Priority, SJF, and more
- **Chunked Prefill**: Simulates realistic request interleaving
- **KV Cache Management**: Models GPU memory and KV cache utilization
- **Workload Generation**: Supports Poisson, Gamma, and closed-loop patterns
- **WebAssembly Support**: Run simulations in the browser via WASM
- **CLI Tool**: Standalone binary for command-line usage

## How does it work?

`inference-lab` uses discrete-event simulation to model the behavior of a
multi-GPU node serving LLM inference requests with the vLLM library. It
contains a facsimile of the vLLM queueing, scheduling, and execution logic,
with only the actual model inference replaced by a performance model based on
the supplied GPU specs and model architecture.

Within each simulation step, the simulator:

1. Processes any newly arrived requests, adding them to the scheduling queue.
2. Schedules requests to serve based on the selected scheduling policy.
3. Calculates the compute and memory bandwidth usage for the workload that the
   scheduled requests represent, and the theoretical time required to execute
the workload on the specified hardware.
4. Increments the simulation time by the calculated execution time, updating the
   state of all requests accordingly.

Caveats:

1. This assumes perfectly optimized GPU execution, ignoring kernel launch
   overheads, poorly optimized kernels, application overhead, thermals, etc.
2. We simulate tensor parallel execution, but don't model multi-GPU
   communication overheads.

## Installation

### As a Rust Library

```bash
cargo add inference-lab
```

### As an npm Package (WASM)

```bash
npm install @doublewordai/inference-lab
```

### CLI Tool

```bash
cargo install inference-lab
```

## Usage

### CLI

```bash
# Run with default configuration
inference-lab --config configs/config.toml

# Example output shows TTFT, E2E latency, throughput, and utilization metrics
```

### Rust Library

```rust
use inference_lab::simulation::Simulator;
use inference_lab::config::SimulationConfig;

let config = SimulationConfig::from_file("config.toml")?;
let mut simulator = Simulator::new(config);
let results = simulator.run();

println!("Mean TTFT: {:.2}ms", results.ttft_mean * 1000.0);
println!("P99 E2E: {:.2}ms", results.e2e_p99 * 1000.0);
println!("Throughput: {:.1} tok/s", results.throughput);
```

### WebAssembly

```javascript
import init, { run_simulation } from '@doubleword/inference-lab';

await init();

const config = `
[hardware]
name = "H100"
compute_flops = 2e15
memory_bandwidth = 3.35e12
# ... rest of config
`;

const results = run_simulation(config);
console.log('TTFT P50:', results.ttft_p50);
console.log('Throughput:', results.throughput);
```

## Configuration

Configuration files use TOML format and specify:

- **Hardware**: GPU specs (FLOPS, bandwidth, VRAM)
- **Model**: LLM architecture (parameters, layers, heads)
- **Scheduler**: Policies, max tokens, chunked prefill settings
- **Workload**: Request arrival patterns and distributions

Example configurations are in the `configs/` directory:

- `config.toml` - Default H100 + Llama-3-70B setup
- `test_blog.toml` - Closed-loop benchmark (64 users)
- `qwen3_30b_a3b.toml` - Qwen model configuration

## Building

### Native Binary

```bash
cargo build --release
./target/release/inference-lab --config configs/config.toml
```

### WASM Package

```bash
npm run build
# Outputs to pkg/ directory
```

### Publishing

```bash
# Publish to npm (requires authentication)
npm run build
npm publish --access public

# Publish Rust crate
cargo publish
```

## Project Structure

```
inference-lab/
├── src/
│   ├── simulation/     # Core simulator logic
│   ├── scheduler/      # Scheduling policies (FCFS, Priority, SJF)
│   ├── compute/        # Performance calculations
│   ├── kv_cache/       # KV cache management
│   ├── request/        # Request generation and tracking
│   ├── metrics/        # Performance metrics collection
│   ├── config/         # Configuration structures
│   ├── lib.rs          # Library root
│   ├── main.rs         # CLI entry point
│   └── wasm.rs         # WebAssembly bindings
├── configs/            # Example configurations
├── Cargo.toml          # Rust package manifest
└── package.json        # npm package manifest
```

## Metrics

The simulator tracks:

- **TTFT** (Time to First Token): Prefill latency
- **E2E** (End-to-End): Total request latency
- **TPOT** (Time Per Output Token): Decode latency per token
- **Throughput**: Tokens generated per second
- **Utilization**: Compute and memory bandwidth usage
- **KV Cache**: Memory utilization over time

Results include percentiles (p50, p90, p95, p99) and means.

## License

MIT

## Repository

<https://github.com/doublewordai/inference-lab>
