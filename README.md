# Inference Lab

**[Documentation](https://doublewordai.github.io/inference-lab/)**

LLM inference simulator for analyzing serving systems.
Simulates GPU clusters serving LLM inference workloads with realistic
performance modeling.

## Features

- **Roofline performance model**: per-precision compute streams and memory
  bandwidth, MoE expert loading, MLA / sliding-window / hybrid-linear KV,
  TP / EP collectives
- **vLLM-style scheduling**: chunked prefill, preemption with recompute,
  FCFS / priority / length-based policies, preemption-free admission
- **KV cache**: block allocation from each model's exact KV footprint,
  prefix caching with block sharing, spillover tiers, cascade attention
- **Disaggregated serving**: prefill and decode pools with a shared hand-off link
- **Speculative decoding**: analytic or trace-replayed acceptance, fixed and
  goodput-adaptive draft policies, measured step-cost tables
- **Workloads**: Poisson / uniform / burst / batched / closed-loop arrivals,
  synthetic length distributions or real datasets
- **Shipped catalog**: hardware presets (B200/B300/GH200/H100) and ~40
  model presets with their HF-derived numbers, referenced by name from configs
- **CLI, Rust library and WebAssembly package**

## How does it work?

`inference-lab` uses discrete-event simulation to model the behavior of multiple multi-GPU nodes
serving LLM inference requests with the vLLM library. It
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

**Note:** The CLI tool is only available if you install it using `cargo install inference-lab` (see above).

```bash
# Run a configuration
inference-lab --config examples/config.toml

# Example output shows TTFT, E2E latency, throughput, and utilization metrics
```

### Rust Library

```rust
use inference_lab::config::Config;
use inference_lab::simulation::Simulator;

// Presets: inference_lab::catalog::hardware("b200"), catalog::model("gemma-4-31b-it")
let config = Config::from_file("config.toml")?;
let mut simulator = Simulator::new(config, None)?;
simulator.run_with_callback(|_| {})?;
let summary = simulator.summary();

println!("Mean TTFT: {:.2}ms", summary.latency_metrics.ttft_ms.mean);
println!("P99 E2E: {:.2}ms", summary.latency_metrics.e2e_ms.p99);
println!("Throughput: {:.1} tok/s", summary.throughput_metrics.output_tokens_per_sec);
```

### WebAssembly

```javascript
import init, { run_simulation } from '@doubleword/inference-lab';

await init();

const config = {
  hardware: "h100",              // catalog preset (or an inline object)
  model: "llama-3-70b-fp8",      // catalog preset (or an inline object)
  scheduler: {
    max_num_batched_tokens: 8192,
    max_num_seqs: 256,
    policy: "fcfs",
    enable_chunked_prefill: true,
    block_size: 16
  },
  workload: {
    arrival_pattern: "poisson",
    arrival_rate: 5.0,
    num_requests: 400,
    seed: 42,
    input_len_dist: {
      type: "lognormal",
      mean: 6.9,
      std_dev: 0.7
    },
    output_len_dist: {
      type: "lognormal",
      mean: 5.3,
      std_dev: 0.8
    }
  }
};

const results = run_simulation(JSON.stringify(config));
console.log('TTFT P50:', results.metrics.ttft_p50);
console.log('Throughput:', results.metrics.output_tokens_per_sec);
```

## Configuration

Configuration files use TOML format and specify:

- **Hardware**: a catalog preset name or GPU specs (FLOP rates, bandwidth, VRAM)
- **Model**: a catalog preset name or the architecture as weight streams + layer classes
- **Scheduler**: engine args (batching, KV blocks, memory utilisation, policy)
- **Workload**: Request arrival patterns and distributions

The catalog lives in `catalog/{hardware,models}/*.toml` and is compiled into
the crate (`inference_lab::catalog`).

Example configurations are in `examples/*.toml` (small H100 / Llama and
Qwen setups) and `configs/` (production-shaped DeepSeek-V4, Qwen3.5/3.6,
GLM-5.2 and estate catalogs).

## Building

### Native Binary

```bash
cargo build --release
./target/release/inference-lab --config examples/config.toml
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
