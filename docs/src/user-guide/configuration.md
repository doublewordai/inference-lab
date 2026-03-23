# Configuration

Inference Lab uses TOML configuration files to define your simulation parameters. A configuration file has five main sections: hardware, model, scheduler, workload, and simulation.

## Configuration Sections Overview

- **[hardware]** - GPU specifications (compute, memory, bandwidth)
- **[model]** - LLM architecture (layers, parameters, dimensions)
- **[scheduler]** - Scheduling policy and batching behavior
- **[workload]** - Request arrival patterns and distributions
- **[simulation]** - Logging and output options

## Quick Start Example

Here's a minimal configuration to get started:

```toml
[hardware]
name = "H100"
compute_flops = 1.513e15        # 1513 TFLOPS bf16
memory_bandwidth = 3.35e12      # 3.35 TB/s
memory_capacity = 85899345920   # 80 GB
bytes_per_param = 2             # bf16

[model]
name = "Llama-3-70B"
num_parameters = 70000000000
num_layers = 80
hidden_dim = 8192
num_heads = 64
num_kv_heads = 8                # GQA with 8 KV heads
max_seq_len = 8192

[scheduler]
max_num_batched_tokens = 8192
max_num_seqs = 256
policy = "fcfs"
enable_chunked_prefill = true
block_size = 16

[workload]
arrival_pattern = "poisson"
arrival_rate = 5.0              # 5 requests/sec
num_requests = 100
seed = 42

[workload.input_len_dist]
type = "lognormal"
mean = 6.9                      # ~1000 tokens median
std_dev = 0.7

[workload.output_len_dist]
type = "lognormal"
mean = 5.3                      # ~200 tokens median
std_dev = 0.8

[simulation]
log_interval = 5
```

## Hardware Configuration

The hardware section defines your GPU specifications:

```toml
[hardware]
name = "H100"
compute_flops = 1.513e15        # bf16 TFLOPS
memory_bandwidth = 3.35e12      # bytes/sec
memory_capacity = 85899345920   # 80 GB
bytes_per_param = 2             # 2 for bf16, 1 for fp8
```

Optional fields:
- `kv_cache_capacity` - Explicit KV cache size (otherwise computed automatically)
- `gpu_memory_utilization` - Fraction of memory to use (default: 0.9)

## Model Configuration

Define your LLM architecture:

```toml
[model]
name = "Llama-3-70B"
num_parameters = 70000000000
num_layers = 80
hidden_dim = 8192
num_heads = 64
num_kv_heads = 8                # For GQA (omit for MHA)
max_seq_len = 8192
```

### Grouped Query Attention (GQA)

For models using GQA, set `num_kv_heads` to the number of KV heads:

```toml
num_kv_heads = 8  # Llama 3 uses 8 KV heads
```

Omit `num_kv_heads` for standard multi-head attention (MHA) models.

### Mixture of Experts (MoE)

For MoE models, specify active parameters separately:

```toml
num_parameters = 140000000000      # Total params
num_active_parameters = 12000000000 # Active per forward pass
```

### Sliding Window Attention

For models like GPT-OSS with sliding window attention:

```toml
sliding_window = 4096
num_sliding_layers = 28  # Number of layers using sliding window
```

## Scheduler Configuration

Control request scheduling and batching:

```toml
[scheduler]
max_num_batched_tokens = 8192
max_num_seqs = 256
policy = "fcfs"
enable_chunked_prefill = true
block_size = 16
```

### Scheduling Policies

Available policies:
- `fcfs` - First-Come-First-Served (default)
- `sof` - Shortest Output First
- `sif` - Shortest Input First
- `stf` - Shortest Total First
- `lif` - Longest Input First
- `lof` - Longest Output First
- `ltf` - Longest Total First

### Chunked Prefill

Enable chunked prefill to allow interleaving prompt processing with generation:

```toml
enable_chunked_prefill = true
long_prefill_token_threshold = 512  # Optional: chunk size limit
max_num_partial_prefills = 1        # Max concurrent partial prefills
```

### Preemption-Free Mode

Enable conservative admission control to guarantee zero preemptions:

```toml
enable_preemption_free = true
```

## Workload Configuration

Define how requests arrive and their characteristics.

### Synthetic Workload

```toml
[workload]
arrival_pattern = "poisson"
arrival_rate = 5.0
num_requests = 100
seed = 42

[workload.input_len_dist]
type = "lognormal"
mean = 6.9
std_dev = 0.7

[workload.output_len_dist]
type = "lognormal"
mean = 5.3
std_dev = 0.8
```

### Arrival Patterns

- `poisson` - Poisson process with exponential inter-arrival times
- `uniform` - Uniform random inter-arrival times
- `burst` - Bursty traffic
- `fixed_rate` - Fixed interval between requests
- `closed_loop` - Fixed number of concurrent users
- `batched` - Requests arrive in batches

### Length Distributions

Four distribution types are supported:

**Fixed:**
```toml
[workload.input_len_dist]
type = "fixed"
value = 1000
```

**Uniform:**
```toml
[workload.input_len_dist]
type = "uniform"
min = 100
max = 2000
```

**Normal:**
```toml
[workload.input_len_dist]
type = "normal"
mean = 1000.0
std_dev = 200.0
```

**LogNormal:**
```toml
[workload.input_len_dist]
type = "lognormal"
mean = 6.9      # ln(1000)
std_dev = 0.7
```

### Dataset Mode

Use real request traces instead of synthetic workloads:

```toml
[workload]
dataset_path = "path/to/dataset.jsonl"
arrival_pattern = "poisson"
arrival_rate = 1.0

# These are used for sampling actual generation length
input_len_dist = { type = "fixed", value = 100 }  # Ignored
output_len_dist = { type = "fixed", value = 50 }  # Samples EOS
```

**Dataset Format:** JSONL file in OpenAI batch API format. Each line should be a JSON object with a `messages` field containing an array of message objects.

Example:
```json
{"custom_id": "req-1", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 100}
```

**Tokenizer:** Dataset mode requires a tokenizer file to convert text to tokens. You'll need to provide this via the `--tokenizer` flag:
```bash
inference-lab -c config.toml --tokenizer tokenizer.json
```

The tokenizer should be a HuggingFace tokenizers JSON file (typically `tokenizer.json` from the model repository).

**Chat Template:** You'll also need to specify how to format messages via `--chat-template`:
- Use `"None"` for simple concatenation of messages
- Use a Jinja2 template string for custom formatting (e.g., `"{{user}}\n{{assistant}}"`)
- Most models have their own chat template format

Example with no template:
```bash
inference-lab -c config.toml \
  --tokenizer tokenizer.json \
  --chat-template None
```

### Closed-Loop Workload

Simulate a fixed number of concurrent users:

```toml
[workload]
arrival_pattern = "closed_loop"
num_concurrent_users = 10
# ... length distributions ...
```

## Simulation Configuration

Control logging and output:

```toml
[simulation]
log_interval = 5  # Log every 5 iterations
```

## Common Configuration Patterns

### High Throughput Setup

Maximize batch size and token throughput:

```toml
[scheduler]
max_num_batched_tokens = 16384
max_num_seqs = 512
enable_chunked_prefill = true
```

### Low Latency Setup

Prioritize request completion speed:

```toml
[scheduler]
max_num_batched_tokens = 4096
max_num_seqs = 64
policy = "sof"  # Shortest Output First
```

### Memory-Constrained Setup

Limit KV cache usage:

```toml
[hardware]
kv_cache_capacity = 34359738368  # 32 GB explicit limit

[scheduler]
max_num_seqs = 128
```

## Next Steps

- See the [Configuration Reference](../reference/config.md) for exhaustive field documentation
- Learn about [Running Simulations](./running-simulations.md)
