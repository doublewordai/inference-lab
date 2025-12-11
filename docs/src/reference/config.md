# Configuration File Reference

Complete field-by-field reference for Inference Lab configuration files.

## Top-Level Structure

```toml
[hardware]
# ... hardware configuration ...

[model]
# ... model configuration ...

[scheduler]
# ... scheduler configuration ...

[workload]
# ... workload configuration ...

[simulation]
# ... simulation configuration ...
```

---

## [hardware]

GPU and accelerator specifications.

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | String | Accelerator name (e.g., "H100", "A100") |
| `compute_flops` | Float | Compute capacity in FLOPS for the specified precision |
| `memory_bandwidth` | Float | Memory bandwidth in bytes/second |
| `memory_capacity` | U64 | Total GPU memory capacity in bytes |
| `bytes_per_param` | U32 | Bytes per parameter (1 for fp8, 2 for bf16/fp16) |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `kv_cache_capacity` | U64 | Computed | KV cache capacity in bytes. If not specified, calculated as `(memory_capacity * gpu_memory_utilization) - model_size` |
| `gpu_memory_utilization` | Float | 0.9 | Fraction of GPU memory to use. Used to compute `kv_cache_capacity` if not explicitly set |

### Example

```toml
[hardware]
name = "H100"
compute_flops = 1.513e15
memory_bandwidth = 3.35e12
memory_capacity = 85899345920
bytes_per_param = 2
```

---

## [model]

LLM architecture parameters.

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | String | Model name |
| `num_parameters` | U64 | Total number of parameters (for MoE: all experts) |
| `num_layers` | U32 | Number of transformer layers |
| `hidden_dim` | U32 | Hidden dimension size |
| `num_heads` | U32 | Number of attention heads |
| `max_seq_len` | U32 | Maximum sequence length supported by the model |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_active_parameters` | U64 | `num_parameters` | Active parameters per forward pass (for MoE models with sparse activation) |
| `num_kv_heads` | U32 | `num_heads` | Number of KV heads. Set for GQA/MQA, omit for MHA |
| `sliding_window` | U32 | None | Sliding window size for sliding window attention layers |
| `num_sliding_layers` | U32 | 0 | Number of layers using sliding window attention (rest use full attention) |

### Example

```toml
[model]
name = "Llama-3-70B"
num_parameters = 70000000000
num_layers = 80
hidden_dim = 8192
num_heads = 64
num_kv_heads = 8
max_seq_len = 8192
```

---

## [scheduler]

Request scheduling and batching configuration.

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `max_num_batched_tokens` | U32 | Maximum number of tokens processed in a single iteration |
| `max_num_seqs` | U32 | Maximum number of sequences that can run concurrently |
| `policy` | String | Scheduling policy: `"fcfs"`, `"sof"`, `"sif"`, `"stf"`, `"lif"`, `"lof"`, or `"ltf"` |
| `enable_chunked_prefill` | Bool | Enable chunked prefilling to interleave prompt processing with generation |
| `block_size` | U32 | Block size for KV cache management (in tokens) |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `long_prefill_token_threshold` | U32 | 0 or 4% of `max_seq_len` | Maximum tokens to prefill in a single iteration. Defaults to 0 (no chunking within request) unless `max_num_partial_prefills > 1`, then defaults to 4% of `max_seq_len` |
| `max_num_partial_prefills` | U32 | 1 | Maximum number of sequences that can be partially prefilled concurrently. Limits how many new waiting requests can start prefilling per iteration |
| `enable_preemption_free` | Bool | false | Enable preemption-free scheduling mode with conservative admission control |

### Scheduling Policy Values

- `fcfs` - First-Come-First-Served
- `sof` - Shortest Output First
- `sif` - Shortest Input First
- `stf` - Shortest Total First
- `lif` - Longest Input First
- `lof` - Longest Output First
- `ltf` - Longest Total First

### Example

```toml
[scheduler]
max_num_batched_tokens = 8192
max_num_seqs = 256
policy = "fcfs"
enable_chunked_prefill = true
block_size = 16
```

---

## [workload]

Request arrival patterns and length distributions.

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `arrival_pattern` | String | Arrival pattern: `"poisson"`, `"uniform"`, `"burst"`, `"fixed_rate"`, `"closed_loop"`, or `"batched"` |
| `arrival_rate` | Float | Mean arrival rate in requests per second |
| `input_len_dist` | Distribution | Input sequence length distribution (ignored in dataset mode) |
| `output_len_dist` | Distribution | Output sequence length distribution (in dataset mode: samples actual generation length) |
| `seed` | U64 | Random seed for reproducibility |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_path` | String | None | Path to dataset file in OpenAI batch API format (JSONL). If provided, uses dataset mode instead of synthetic workload |
| `num_requests` | Usize | None | Total number of requests to simulate. If None, runs until `duration_secs` |
| `duration_secs` | Float | None | Simulation duration in seconds. If None, runs until `num_requests` |
| `num_concurrent_users` | Usize | None | Number of concurrent users for `closed_loop` pattern. Each user immediately sends a new request when their previous one completes |

### Length Distribution Types

Distributions are specified using TOML tables with a `type` field:

**Fixed:**
```toml
input_len_dist = { type = "fixed", value = 1000 }
```

**Uniform:**
```toml
input_len_dist = { type = "uniform", min = 100, max = 2000 }
```

**Normal:**
```toml
input_len_dist = { type = "normal", mean = 1000.0, std_dev = 200.0 }
```

**LogNormal:**
```toml
input_len_dist = { type = "lognormal", mean = 6.9, std_dev = 0.7 }
```

Or using TOML section syntax:

```toml
[workload.input_len_dist]
type = "lognormal"
mean = 6.9
std_dev = 0.7
```

### Example

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

---

## [simulation]

Simulation control and logging.

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `log_interval` | U64 | 100 | Log progress every N iterations |

### Example

```toml
[simulation]
log_interval = 5
```

---

## Type Reference

- **String**: Text string
- **Float**: 64-bit floating point number
- **U32**: 32-bit unsigned integer
- **U64**: 64-bit unsigned integer
- **Usize**: Platform-dependent unsigned integer
- **Bool**: Boolean (`true` or `false`)
- **Distribution**: Length distribution object (see [Length Distribution Types](#length-distribution-types))
