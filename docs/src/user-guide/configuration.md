# Configuration

A simulation is a **model config** × one of its **hardware entries** × a
**workload**. Model configs live in `configs/`, one file per model
deployment; workloads in `workloads/`. Unknown fields in either are rejected.

```bash
inference-lab --config configs/qwen3.6-35b-a3b-fp8.toml --hardware b200 \
              --workload workloads/chat-closed-256.toml
```

## Model config

```toml
model = "qwen3.6-35b-a3b-fp8"   # catalog preset, or an inline [model] table

[scheduler]                     # engine args shared by every hardware entry
max_num_batched_tokens = 16384
max_num_seqs = 4096
policy = "priority"
enable_chunked_prefill = true
block_size = 64

[hardware.b200]                 # one entry per hardware this model runs on
tp = 1

[hardware.b300]
tp = 1

[hardware.gh200]
tp = 1
scheduler = { max_num_batched_tokens = 8192 }   # per-entry override
```

- **model** — a catalog preset name or an inline `[model]` table: weight
  streams and token-mixing layer classes.
- **[scheduler]** — engine arguments: batching, KV blocks, memory
  utilisation, scheduling policy.
- **[hardware.\<name\>]** — one per hardware the model is deployed on. The
  name is a hardware preset (`b200`, `b300`, `gh200`, `h100`) unless the entry sets `spec`. Each entry gives
  the parallel layout — `tp` (replica world size, weights sharded, per-layer
  all-reduces), `ep` (experts sharded over `ep` ranks), `dp_attention`
  (data-parallel attention: replicated attention weights,
  all-gather/reduce-scatter around the FFN, or dispatch/combine all-to-alls
  when `ep > 1`) — and may
  override `scheduler` keys or carry its own `speculative` block. The
  shipped configs carry the layouts production runs (`tp8 + dp_attention`
  for DeepSeek-V4-Pro / Kimi / GLM-5, `tp = ep` for the Qwen3.5/VL MoEs and
  Nemotron Ultra, plain `tp` elsewhere).
- **[speculative]** — speculative decoding, optional; a shared default that
  an entry's `speculative` replaces (acceptance traces and measured step
  costs are per hardware).
- **[memory]** — KV tiers beyond HBM, picked from the stores the hardware
  offers (host DRAM over PCIe, Grace memory over NVLink-C2C, NVMe): evicted
  blocks fall through them and are promoted back instead of recomputed. A
  `per = "node"` store is shared by the workers on a node. How KV moves —
  fetch or recompute, what HBM and each tier evict, when blocks are
  written, whether a re-entry's prefix is prefetched — is a set of policies
  with two presets, `reactive` (decides from the past, like shipped
  stacks) and `oracle` (knows every session's next re-entry); see the
  reference.
- **replicas / [router] / [decode_router]** — an entry's `replicas`
  (default 1) runs that many identical workers, each with its own scheduler
  and KV cache; the shared `[router]` (or an entry's `router`) picks which
  one each request enters: `round_robin`, `least_loaded`,
  `prefix_affinity`, or `kv_aware`. On a disaggregated topology
  `[decode_router]` (default: `[router]`) picks the decode worker each
  hand-off goes to; `kv_aware_decode` prices the transfer and the decode
  batch (see the reference).

`--hardware` picks the entry; it can be omitted when a file has one.
`inference-lab serve --config configs/ --hardware b200` serves every model
with a `b200` entry.

### Hardware

Name a shipped preset as the entry:

```toml
[hardware.b200]
tp = 2
```

or point an entry at another preset, or at an inline per-GPU spec (a FLOP
rate for every precision the model uses, bandwidth, capacity, optional
`[fabric]` and `[memory]`):

```toml
[hardware.isambard]
spec = "gh200"
tp = 4

[hardware.custom]
tp = 1
[hardware.custom.spec]
name = "H100"
flops_fp8 = 1.979e15            # dense FLOP/s at fp8
flops_bf16 = 9.895e14           # dense FLOP/s at bf16
memory_bandwidth = 3.35e12      # bytes/sec
memory_capacity = 85899345920   # 80 GB
```

How much of that memory the engine may use, and how much goes to KV, are
deployment settings and live in `[scheduler]` (`gpu_memory_utilization`,
`kv_cache_capacity`).

A preset also carries its node's collective fabric — `gpus_per_node`,
`scale_up` (NVLink: bandwidth, latency, in-network reduction) and
`scale_out` (per-GPU NIC across nodes) — which prices the TP all-reduces and
EP all-to-alls of any entry with `tp > 1` or `ep > 1`. An inline spec that
omits `[fabric]` can only be used with `tp = 1`, `ep = 1`.

### Model

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

### Scheduler

Control request scheduling and batching:

```toml
[scheduler]
max_num_batched_tokens = 8192
max_num_seqs = 256
policy = "fcfs"
enable_chunked_prefill = true
block_size = 16
```

#### Scheduling Policies

Available policies:
- `fcfs` - First-Come-First-Served (default)
- `sof` - Shortest Output First
- `sif` - Shortest Input First
- `stf` - Shortest Total First
- `lif` - Longest Input First
- `lof` - Longest Output First
- `ltf` - Longest Total First

#### Chunked Prefill

Enable chunked prefill to allow interleaving prompt processing with generation:

```toml
enable_chunked_prefill = true
long_prefill_token_threshold = 512  # Optional: chunk size limit
max_num_partial_prefills = 1        # Max concurrent partial prefills
```

#### Preemption-Free Mode

Enable conservative admission control to guarantee zero preemptions:

```toml
enable_preemption_free = true
```

## Workload

A workload file is the workload table at top level: how requests arrive and
their shapes.

### Synthetic Workload

```toml
# workloads/chat-poisson-5rps.toml
arrival_pattern = "poisson"
arrival_rate = 5.0
num_requests = 100
seed = 42

[input_len_dist]
type = "lognormal"
mean = 6.9
std_dev = 0.7

[output_len_dist]
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
[input_len_dist]
type = "fixed"
value = 1000
```

**Uniform:**
```toml
[input_len_dist]
type = "uniform"
min = 100
max = 2000
```

**Normal:**
```toml
[input_len_dist]
type = "normal"
mean = 1000.0
std_dev = 200.0
```

**LogNormal:**
```toml
[input_len_dist]
type = "lognormal"
mean = 6.9      # ln(1000)
std_dev = 0.7
```

### Dataset Mode

Use real request traces instead of synthetic workloads:

```toml
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
inference-lab -c configs/llama-3-70b.toml -w workloads/dataset-poisson.toml --tokenizer tokenizer.json
```

The tokenizer should be a HuggingFace tokenizers JSON file (typically `tokenizer.json` from the model repository).

**Chat Template:** You'll also need to specify how to format chat-style requests via `--chat-template`:
- Use `"None"` for simple concatenation of messages
- Use a Jinja2 template string for custom formatting (e.g., `"{{user}}\n{{assistant}}"`)
- Most models have their own chat template format
- Plain `/v1/completions` prompts are tokenized directly and do not use the chat template

Example with no template:
```bash
inference-lab -c configs/llama-3-70b.toml -w workloads/dataset-poisson.toml \
  --tokenizer tokenizer.json \
  --chat-template None
```

### Sessions

Agentic traffic: chains of requests where each step re-enters with its
parent's whole context as prefix (prompt *and* the parent's output) plus some
novel tokens, after the gap the harness spent between the parent's completion
and this arrival (a tool call running, a user typing).

```toml
sessions_path = "data/sessions/tracelab.jsonl"
arrival_pattern = "poisson"   # governs session starts
arrival_rate = 0.05           # sessions/s
num_sessions = 200            # stop starting sessions after this many
seed = 42

input_len_dist = { type = "fixed", value = 1 }   # ignored in session mode
output_len_dist = { type = "fixed", value = 1 }  # ignored in session mode
```

The arrival pattern decides when sessions **start**: `poisson` / `uniform` /
`burst` at `arrival_rate` sessions per second, `closed_loop` keeps
`num_concurrent_users` sessions in flight (a slot starts a fresh session when
its session's last step completes), `batched` starts every session at t=0.
Every later step of a session arrives at its parent's completion plus the
step's `gap`, so the simulated latency feeds back into the arrival process
and long gaps are preserved. Sessions are taken from the file in order and
the file is cycled. `num_sessions` bounds session starts; `num_requests`
separately bounds total emitted request steps across all sessions.

**Session file:** JSONL, one session per line:

```json
{"id": "claude:000adcd5", "steps": [
  {"input": 15524, "new": 15524, "output": 111, "gap": 0.0, "kind": "user"},
  {"input": 17079, "new": 1444, "output": 96, "gap": 0.104, "kind": "tool"}
]}
```

`input` is the step's prompt length, `new` the tokens of it that are not the
parent's context (`input − new` is the reusable prefix, capped at what the
parent actually had), `output` the tokens generated, `gap` the seconds from
the parent's completion to this arrival (ignored on the first step), `kind`
free-form. Prefix identity is built from block hashes: the parent's hashes
over the shared prefix, fresh ones for the novel tail and the step's own
output, so a re-entry hits the parent's generated tokens too. Whole blocks
only: a partial block continued with new tokens is new content.

`examples/sessions/tracelab_export.py` exports TraceLab's
`per_step_stats.parquet` into this format.

**Per-request CSV** (`--request-csv`) carries, for session steps, `session`,
`step`, `worker` (the memory-graph id of the worker that served it), `gap`,
`shared_toks` (the most the prefix cache could serve),
`cached_toks` (what it did), and two reuse distances: `reuse_distance_bytes`
(KV bytes written into the caches between the parent's completion and this
arrival) and `reuse_touched_bytes` (the same plus the free blocks that hits
pulled back into use in between). Fresh writes undercount the LRU stack
distance and the touched count overcounts it, so the pair brackets it.

### Closed-Loop Workload

Simulate a fixed number of concurrent users:

```toml
arrival_pattern = "closed_loop"
num_concurrent_users = 256
closed_loop_jitter_secs = 0.05  # stagger the initial arrivals
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
