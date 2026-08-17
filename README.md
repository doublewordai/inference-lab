# Inference Lab

**[Documentation](https://doublewordai.github.io/inference-lab/)**

LLM inference simulator for analyzing serving systems.
Simulates GPU clusters serving LLM inference workloads with a vLLM-style
scheduler and KV cache over a datasheet roofline: every step is priced at
peak FLOP rate and HBM bandwidth with collectives added serially, and
nothing else. Simulated latencies and throughputs are therefore upper
bounds — the numbers a perfect engine on the datasheet hardware would
reach — and the deltas between configurations are what the simulator is
for. `[hardware.<name>] time_correction = { alpha, beta }` calibrates the
step time (`alpha × roofline + beta`) against a measured engine when
absolute figures are needed.

## Features

- **Roofline performance model**: per-precision compute streams and memory
  bandwidth, MoE expert loading, MLA / sliding-window / hybrid-linear KV,
  TP / EP collectives
- **vLLM-style scheduling**: chunked prefill, preemption with recompute,
  FCFS / priority / length-based policies, preemption-free admission
- **KV cache**: block allocation from each model's KV footprint (content
  blocks per token block, sliding windows and recurrent state per request),
  prefix caching with block sharing, cascade attention
- **Memory graph**: KV tiers beyond HBM as a graph of stores and links per
  hardware preset (host DRAM / NVMe behind PCIe, Grace memory behind
  NVLink-C2C, NVLink, NICs); per-GPU or node-shared stores; write-back /
  write-through / selective writes, FIFO / LRU / TTL eviction; transfers
  at max-min fair share over every edge of their path
- **Replicas and routing**: N identical workers behind a pluggable router —
  round-robin, least-loaded, prefix-affinity, KV-aware
- **Disaggregated serving**: prefill and decode pools; hand-offs ride the
  memory graph's NICs and network core
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
# A model config (× one of its hardware entries) plus a workload
inference-lab --config configs/qwen3.6-35b-a3b-fp8.toml --hardware b200 \
              --workload workloads/chat-closed-256.toml

# Output shows TTFT, E2E latency, throughput, and utilization metrics
```

### Rust Library

```rust
use inference_lab::config::{ModelConfig, WorkloadConfig};
use inference_lab::simulation::Simulator;

// A model config resolves to a Deployment per hardware entry; add a workload
// to get a runnable Config.
let deployment = ModelConfig::from_file("configs/gemma-4-31b-it.toml")?
    .deployment(Some("b200"))?;
let workload = WorkloadConfig::from_file("workloads/chat-closed-256.toml")?;
let mut simulator = Simulator::new(deployment.with_workload(workload), None)?;
simulator.run_with_callback(|_| {})?;
let summary = simulator.summary();

println!("Mean TTFT: {:.2}ms", summary.latency_metrics.ttft_ms.mean);
println!("P99 E2E: {:.2}ms", summary.latency_metrics.e2e_ms.p99);
println!("Throughput: {:.1} tok/s", summary.throughput_metrics.output_tokens_per_sec);
```

### WebAssembly

```javascript
import init, { run_simulation } from '@doublewordai/inference-lab';

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

A simulation is a **model config** × one of its **hardware entries** × a
**workload**:

- `configs/<model>.toml` — one file per model deployment: the model (a
  catalog preset name or an inline architecture), its engine args
  (`[scheduler]`), optional `[speculative]` and `[router]`, and a
  `[hardware.<name>]` entry per hardware it runs on (`tp`/`ep`, `replicas`,
  per-entry scheduler overrides). Every
  model the production fleet serves has a file here with its B200 / B300 /
  GH200 entries.
- `workloads/<name>.toml` — arrival pattern, request-length distributions or
  a dataset, request count, seed.

The hardware and model presets live in `catalog/{hardware,models}/*.toml`
and are compiled into the crate (`inference_lab::catalog`); a config refers
to them by name (`model = "gemma-4-31b-it"`, `[hardware.b200]`).

## Building

### Native Binary

```bash
cargo build --release
./target/release/inference-lab --config configs/llama-3-70b.toml --workload workloads/quick.toml
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
│   ├── simulation/     # Core simulator: engine, spec decoding, disagg
│   ├── scheduler/      # Scheduling policies (FCFS, priority, SJF/SOF)
│   ├── compute/        # Roofline performance model, measured step tables
│   ├── kv_cache/       # KV block manager, prefix cache, tiers, links
│   ├── router/         # Request routing across replicas
│   ├── request/        # Request generation and tracking
│   ├── metrics/        # Metrics collection and summaries
│   ├── config/         # Configuration structures
│   ├── serve/          # OpenAI-compatible server (`--features serve`)
│   ├── catalog.rs      # Shipped hardware/model presets (embedded by build.rs)
│   ├── dataset.rs      # Dataset loading for trace-driven workloads
│   ├── lib.rs          # Library root
│   ├── main.rs         # CLI entry point
│   └── wasm.rs         # WebAssembly bindings
├── catalog/            # hardware/*.toml and models/*.toml presets
├── configs/            # One model config per deployment (model × hardware entries)
├── workloads/          # Workload files
├── examples/           # Rust examples and a sample dataset
├── build.rs            # Embeds catalog/ into the crate
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
