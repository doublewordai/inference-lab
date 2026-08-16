# Configuration

Inference Lab uses TOML configuration files to define your simulation parameters. A configuration names (or inlines) a hardware spec and a model, then sets the engine arguments (`[scheduler]`) and the workload; `[parallel]` and `[speculative]` are optional. Unknown fields are rejected.

## Configuration Sections Overview

- **hardware** - a catalog preset name (`hardware = "b200"`) or an inline `[hardware]` table: per-GPU FLOP rates per precision, memory, bandwidth, optional KV tiers
- **model** - a catalog preset name (`model = "deepseek-v4-flash"`) or an inline `[model]` table: weight streams and token-mixing layer classes
- **[scheduler]** - engine arguments: batching, KV blocks, memory utilisation, scheduling policy
- **[workload]** - Request arrival patterns and distributions
- **[speculative]** - Speculative decoding (optional)

The crate ships a catalog of hardware (`b200`, `b300`, `gh200-120`, `h100`, ...)
and model presets (`inference_lab::catalog::model_names()`), each model
with the derivation of its numbers from the HF config in its file.

## Quick Start Example

Here's a minimal configuration to get started:

```toml
hardware = "h100"               # catalog preset
model = "llama-3-70b-fp8"       # catalog preset

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
```

## Hardware Configuration

Name a shipped preset:

```toml
hardware = "b200"     # b200, b200-datasheet, b300, gh200-120, gh200-96, h100
```

or give the per-GPU spec inline (a FLOP rate for every precision the model
uses, bandwidth, capacity, optional spillover `kv_tiers`):

```toml
[hardware]
name = "H100"
flops_fp8 = 1.979e15            # dense FLOP/s at fp8
flops_bf16 = 9.895e14           # dense FLOP/s at bf16
memory_bandwidth = 3.35e12      # bytes/sec
memory_capacity = 85899345920   # 80 GB
```

How much of that memory the engine may use, and how much goes to KV, are
deployment settings and live in `[scheduler]` (`gpu_memory_utilization`,
`kv_cache_capacity`).

## Model Configuration

Name a shipped preset:

```toml
model = "gemma-4-31b-it"
```

or describe the architecture inline as weight streams plus layer classes:

```toml
[model]
name = "Llama-3-70B"
hidden_dim = 8192
max_seq_len = 8192
attention_precision = "fp8"

[[model.weights]]               # one per-token GEMM stream per precision
precision = "fp8"
active_params = 70000000000
resident_params = 70000000000

[[model.layers]]                # token-mixing layer classes
kind = "attention"              # or "mla", "linear"
count = 80
heads = 64
head_dim = 128
kv_heads = 8                    # GQA
kv_precision = "fp8"
```

MoE adds `routing = { routed_experts, experts_per_tok, moe_layers }` to the
expert stream; sliding-window layers are an `attention` class with
`window`; MLA / DeepSeek sparse attention is the `mla` kind; GatedDeltaNet or
Mamba layers are `linear` with their per-sequence `state_bytes`. See the
[Configuration Reference](../reference/config.md) for every field.

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

**Dataset Format:** JSONL file in OpenAI batch API format. Each line may target either `/v1/chat/completions` with a `messages` array or `/v1/completions` with a string `prompt`.

Example:
```json
{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 100}}
{"custom_id": "req-2", "method": "POST", "url": "/v1/completions", "body": {"model": "gpt-3.5-turbo-instruct", "prompt": "Write a haiku about Rust.", "max_tokens": 80}}
```

**Tokenizer:** Dataset mode requires a tokenizer file to convert text to tokens. You'll need to provide this via the `--tokenizer` flag:
```bash
inference-lab -c config.toml --tokenizer tokenizer.json
```

The tokenizer should be a HuggingFace tokenizers JSON file (typically `tokenizer.json` from the model repository).

**Chat Template:** You'll also need to specify how to format chat-style requests via `--chat-template`:
- Use `"None"` for simple concatenation of messages
- Use a Jinja2 template string for custom formatting (e.g., `"{{user}}\n{{assistant}}"`)
- Most models have their own chat template format
- Plain `/v1/completions` prompts are tokenized directly and do not use the chat template

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
[scheduler]
kv_cache_capacity = 34359738368  # 32 GB explicit limit
max_num_seqs = 128
```

## Next Steps

- See the [Configuration Reference](../reference/config.md) for exhaustive field documentation
- Learn about [Running Simulations](./running-simulations.md)
