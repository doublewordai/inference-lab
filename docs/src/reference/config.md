# Configuration File Reference

Field-by-field reference for the two TOML files a simulation takes: a
**model config** (`configs/<name>.toml`) and a **workload**
(`workloads/<name>.toml`). Unknown fields anywhere in either file are
rejected.

```toml
# configs/<name>.toml
model = "deepseek-v4-flash"    # catalog preset, or an inline [model] table

[scheduler]                    # engine args shared by every hardware entry
[speculative]                  # optional: speculative decoding, shared default
[fault]                        # optional: serve-mode static fault injection

[hardware.b200]                # one entry per hardware the model runs on
tp = 4
[hardware.gh200-120]
tp = 4
scheduler = { max_num_batched_tokens = 4096 }
```

```toml
# workloads/<name>.toml — the [workload] table at top level
arrival_pattern = "closed_loop"
num_concurrent_users = 256
num_requests = 2000
seed = 7
input_len_dist = { type = "lognormal", mean = 7.0, std_dev = 0.5 }
output_len_dist = { type = "lognormal", mean = 6.5, std_dev = 0.8 }
```

`inference-lab --config <model config> --hardware <entry> --workload <workload>`
runs one; `--hardware` may be omitted when the file has one entry. In Rust,
`ModelConfig::from_file(..).deployment(Some("b200"))` gives a `Deployment`
(model on hardware) and `.with_workload(WorkloadConfig::from_file(..))` a
`Config`. The WASM API takes that resolved `Config` as JSON: `hardware`,
`parallel`, `model`, `scheduler`, `workload`, optional `speculative`.

## Catalog presets

Hardware and model presets ship inside the crate (`catalog/hardware/*.toml`,
`catalog/models/*.toml`, embedded at build time). A config names one with
`model = "<name>"` and `[hardware.<name>]`; `inference_lab::catalog::
{hardware_names, model_names, hardware, model}` list and load them from
Rust. Presets carry only the physical hardware spec and the model
architecture; everything about a deployment (TP, memory utilisation, batch
limits) is in the config. To tweak a preset, copy its table inline.

## [hardware.\<name\>]

One entry per hardware the model is deployed on. The entry name is the
hardware preset unless `spec` says otherwise.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tp` | U32 | 1 | Tensor-parallel group size: FLOP rate, bandwidth and memory scale by `tp` |
| `ep` | U32 | 1 | Expert-parallel group size (MoE all-to-all volume) |
| `dp_attention` | Bool | false | DP-attention layout: no per-layer TP all-reduce |
| `spec` | String or Table | the entry name | Another catalog preset, or an inline hardware table (fields below) |
| `scheduler` | Table | `{}` | Keys merged over the shared `[scheduler]` for this entry |
| `speculative` | Table | shared `[speculative]` | Replaces the shared block for this entry |

TP all-reduces and EP all-to-alls are priced on the hardware's `[fabric]`
(below); an entry with `tp > 1` or `ep > 1` on hardware without one is
rejected.

### Hardware spec

Per-GPU physical spec: a catalog preset (`catalog/hardware/<name>.toml`) or
an inline `spec` table.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | String | — | Accelerator name |
| `flops_fp4` | Float | unset | Dense FLOP/s at FP4. Unset means the hardware has no FP4 rate; a model that declares an FP4 stream then fails at run time |
| `flops_fp8` | Float | unset | Dense FLOP/s at FP8 |
| `flops_bf16` | Float | unset | Dense FLOP/s at BF16 |
| `flops_fp16` | Float | unset | Dense FLOP/s at FP16 (FP32 is taken as half of it) |
| `memory_bandwidth` | Float | — | HBM bandwidth, bytes/s |
| `memory_capacity` | U64 | — | HBM capacity, bytes |
| `kv_tiers` | Array | `[]` | Spillover tiers below HBM, closest first: `{ name, capacity_bytes, bandwidth_to_hbm }`. Evicted KV blocks fall through the tiers and can be promoted back over the tier's bandwidth instead of being recomputed |
| `fabric` | Table | unset | Collective fabric, see below |

#### `[fabric]`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpus_per_node` | U32 | — | GPUs sharing one scale-up domain; parallel groups are packed node by node |
| `scale_up` | Table | — | Inside a node (NVLink / NVSwitch): `{ bandwidth, latency, in_network_reduction }` — per-GPU injection bytes/s per direction, seconds per collective call, and whether the switch reduces in-network (NVLink SHARP) |
| `scale_out` | Table | unset | Across nodes, rail-optimised (GPU *i* drives NIC *i*): same fields. Required for a group wider than `gpus_per_node` |

Cost per collective, added serially to the step (no overlap): a TP
all-reduce of `V` bytes over `g ≤ gpus_per_node` ranks is `latency + f·V /
bandwidth` with `f = 2(g−1)/g` (ring) or `1` (in-network reduction); over
more ranks it is reduce-scatter and all-gather inside each node around an
all-reduce of the `V/k` shard across the `n` nodes on the NIC. An EP
all-to-all moves each rank's `(g−1)/g` share at the scale-up rate, or, across
nodes, its in-node and cross-node shares concurrently on their own links.
`dp_attention` skips the per-layer all-reduce.

Shipped presets: `b200` (180 GiB / 7.7 TB/s as reported on the fleet),
`b200-datasheet` (192 GB / 8 TB/s), `b300`, `gh200-120`, `gh200-96`, `h100`;
each carries its node's fabric (8-GPU NVSwitch + CX-7/CX-8 for the HGX
boxes, 4-GPU NVLink + Slingshot for GH200).

---

## [model]

A model is described by its per-token weight streams and its token-mixing
layer classes; there are no named architectures. Any transformer the
simulator serves is a composition of the pieces below.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | String | — | |
| `hidden_dim` | U32 | — | Residual width (sizes collectives; prices MLA attention when its head shape is not given) |
| `max_seq_len` | U32 | — | Architecture context limit (`max_position_embeddings`) |
| `attention_precision` | Precision | `"bf16"` | Rate the attention score / AV matmuls run at; KV reads charge this stream |
| `activation_bytes` | U32 | 2 | Bytes per activation element on the wire |
| `weights` | Array | — | One or more weight streams (below) |
| `layers` | Array | — | One or more layer classes (below) |

### `[[weights]]` — a per-token GEMM stream at one precision

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `precision` | `"fp4"`,`"fp8"`,`"bf16"`,`"fp16"`,`"fp32"` | — | |
| `active_params` | U64 | — | Parameters touched per token (FLOPs = 2×) |
| `resident_params` | U64 | — | Parameters resident in HBM |
| `routing` | Table | unset | MoE routing: `{ routed_experts, experts_per_tok, moe_layers }`. The per-step read then follows coupon-collector growth with the step's tokens (per-expert and shared params are recovered from the active/resident split); EP all-to-alls = 2 × `moe_layers` |

A dense fp8 model is one stream; DeepSeek-V4 is an fp4 expert stream with
routing plus an fp8 non-expert stream; gpt-oss is fp4 experts + bf16 rest.

### `[[layers]]` — a class of identical token-mixing layers

`kind = "attention"` — GQA / MHA with a growing KV cache:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `count` | U32 | — | Layers in the class |
| `heads`, `head_dim` | U32 | — | Query heads and head width (attention FLOPs = 4 × heads × head_dim per query-key pair) |
| `kv_heads` | U32 | — | KV heads (KV per token = 2 × kv_heads × head_dim × bytes) |
| `kv_shared` | Bool | false | K and V share one tensor (Gemma-4): half the KV |
| `window` | U32 | 0 | Sliding window: attend to and store only the last `window` tokens; 0 = full context |
| `kv_precision` | Precision | — | |

`kind = "mla"` — multi-head latent attention, optionally sparse:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `count` | U32 | — | |
| `latent_dim`, `rope_dim` | U32 | — / 0 | KV per token = (latent + rope) × bytes |
| `kv_precision` | Precision | — | |
| `window` | U32 | 0 | Recent tokens attended directly. Without a `history` path, 0 means the whole context; with one, 0 means no local window |
| `history` | Table | unset | Long-range path `{ compress_ratio, index_topk, indexer }`: the history at stride `compress_ratio` (1 = every position), all of it or the `index_topk` entries an `indexer = { heads, head_dim, kv_precision }` selects (the indexer scores every entry and keeps its own KV) |
| `heads`, `qk_head_dim`, `v_head_dim` | U32 | unset | Head shape for the score/AV FLOP count (2 × heads × (qk + v) per pair). All three or none; absent = 4 × hidden_dim per pair |

Kimi-K2 is one `mla` class (full context); DeepSeek-V4 is three (window 128
only; window + top-k of the ÷4 history with an indexer; window + the whole
÷128 history); GLM-5 is top-2048 over the uncompressed history.

`kind = "linear"` — linear attention / SSM (GatedDeltaNet, Mamba, KDA):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `count` | U32 | — | |
| `state_bytes` | U64 | — | Fixed per-sequence state per layer (reserved for the sequence's lifetime and read once per step); no context-scaling work |

Shipped model presets: `catalog/models/` (`inference_lab::catalog::model_names()`),
each with the derivation of its numbers from the HF config in its header.

---

## [scheduler]

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_num_batched_tokens` | U32 | — | Token budget per iteration |
| `max_num_seqs` | U32 | — | Running-request cap |
| `policy` | String | — | `fcfs`, `priority`, `sif`, `lif`, `sof`, `lof`, `stf`, `ltf` (`sjf` = `sof`) |
| `enable_chunked_prefill` | Bool | — | Split long prefills across iterations |
| `long_prefill_token_threshold` | U32 | 0 | Prefill chunk cap; 0 = no cap. Defaults to 4% of `max_seq_len` when `max_num_partial_prefills > 1` |
| `max_num_partial_prefills` | U32 | 1 | vLLM's knob; only its effect on the threshold default is modelled |
| `block_size` | U32 | — | KV block size, tokens |
| `gpu_memory_utilization` | Float | 0.9 | Fraction of GPU memory the engine may use (vLLM's `--gpu-memory-utilization`); the KV cache gets what is left after the weights |
| `kv_cache_capacity` | U64 | 0 | Explicit KV cache bytes across the TP group; 0 derives it from `gpu_memory_utilization` |
| `max_model_len` | U32 | model's `max_seq_len` | Serving-time context limit (only the chunked-prefill threshold default depends on it) |
| `enable_preemption_free` | Bool | false | Admit only what can grow to `prompt + max_output` without preemption |
| `enable_cascade_attention` | Bool | false | Load a batch's shared prompt prefix once per iteration |

---

## Workload file

The `[workload]` table, at top level of `workloads/<name>.toml`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `arrival_pattern` | String | — | `poisson`, `uniform` (= `fixed_rate`), `burst`, `closed_loop`, `batched` |
| `arrival_rate` | Float | 1.0 | Requests/s for the open-loop patterns |
| `rate_schedule` | Table | unset | Time-varying rate: `{ type = "sine", min, max, period_secs }`, `{ type = "square", low, high, period_secs, duty }`, or `{ type = "trace", points = [[t, rate], ...] }` |
| `num_concurrent_users` | U32 | unset | Users for `closed_loop` |
| `closed_loop_jitter_secs` | Float | unset | Uniform stagger of the initial closed-loop arrivals |
| `input_len_dist`, `output_len_dist` | Table | — | `{ type = "fixed", value }`, `{ type = "uniform", min, max }`, `{ type = "normal", mean, std_dev }`, `{ type = "lognormal", mean, std_dev }` (ignored in dataset mode for input) |
| `num_requests` | U32 | unset | Stop after this many; unset = run the dataset out |
| `duration_secs` | Float | unset | Reserved |
| `dataset_path` | String | unset | JSONL in OpenAI batch format; prompts are tokenised with `--tokenizer` and hashed per KV block so shared prefixes hit the prefix cache |
| `seed` | U64 | — | |

---

## [speculative]

Optional. Decode steps then verify `1 + draft` positions and advance by
`1 + accepted`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gamma` | U32 | — | Draft length (`fixed`) or maximum candidate depth (budget policies) |
| `acceptance` | Table | — | `{ kind = "constant", alpha }`, `{ kind = "per_position", a = [...] }`, or `{ kind = "trace_rounds", path }` (CSV bank of real rounds: `commits,category,a0..aD-1`) |
| `policy` | String | `"fixed"` | `fixed`, `goodput_budget`, `gated_budget`, `gated_aggregate` |
| `measured_cost` | Table | unset | `{ path, ref_seq_len }`: measured `(batch_size, num_draft_tokens, step_seconds)` grid that prices decode steps and the policy's cost curve instead of the roofline |
| `switch` | Table | unconstrained | `{ cooldown_rounds, max_step, cost_ms }` for `gated_aggregate` |
| `drafter` | Table | free drafter | `{ kind = "fraction", frac }`, `{ kind = "autoregressive", dense_params, expert_params, num_experts, experts_per_tok, shared_experts }`, or `{ kind = "block_parallel", params, block }` |
