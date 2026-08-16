# Configuration File Reference

Field-by-field reference for the TOML (or, for WASM, JSON) configuration a
simulation takes. Unknown fields anywhere in the file are rejected.

```toml
[hardware]     # per-GPU spec
[parallel]     # optional: TP / EP layout
[model]        # architecture cost model, chosen by `type`
[scheduler]    # batching, KV blocks, policy
[workload]     # arrivals and request shapes
[speculative]  # optional: speculative decoding
```

---

## [hardware]

Per-GPU spec. Aggregate figures across a TP group are derived from
`[parallel]`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | String | — | Accelerator name |
| `flops_fp4` | Float | unset | Dense FLOP/s at FP4. Unset means the hardware has no FP4 rate; a model that declares an FP4 stream then fails at run time |
| `flops_fp8` | Float | unset | Dense FLOP/s at FP8 |
| `flops_bf16` | Float | unset | Dense FLOP/s at BF16 |
| `flops_fp16` | Float | unset | Dense FLOP/s at FP16 (FP32 is taken as half of it) |
| `memory_bandwidth` | Float | — | HBM bandwidth, bytes/s |
| `memory_capacity` | U64 | — | HBM capacity, bytes |
| `kv_cache_capacity` | U64 | 0 | KV cache bytes. 0 means `aggregate_capacity × gpu_memory_utilization − model weights` |
| `gpu_memory_utilization` | Float | 0.9 | Fraction of memory the engine may use (vLLM's `--gpu-memory-utilization`) |
| `kv_tiers` | Array | `[]` | Spillover tiers below HBM, closest first: `{ name, capacity_bytes, bandwidth_to_hbm }`. Evicted KV blocks fall through the tiers and can be promoted back over the tier's bandwidth instead of being recomputed |

```toml
[hardware]
name = "B200"
flops_fp4 = 9.0e15
flops_fp8 = 4.5e15
flops_bf16 = 2.25e15
memory_bandwidth = 8.0e12
memory_capacity = 206158430208
gpu_memory_utilization = 0.9
```

## [parallel]

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tp` | U32 | 1 | Tensor-parallel group size: FLOP rate, bandwidth and memory scale by `tp` |
| `ep` | U32 | 1 | Expert-parallel group size (MoE all-to-all volume) |
| `dp_attention` | Bool | false | DP-attention layout: no per-layer TP all-reduce |

Collective comms (`allreduce`/`alltoall` latency and link bandwidth) are only
modelled when a `ClusterSpec` carries a `comms` block, which the library API
exposes; the single-cluster TOML has none.

---

## [model]

`type` selects the architecture; each has its own fields.

### `type = "dense"` (also accepted as `"sliding"`)

Dense / GQA transformer, optionally with sliding-window layers.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | String | — | |
| `num_parameters` | U64 | — | Resident parameters |
| `num_active_parameters` | U64 | `num_parameters` | Parameters touched per token (sparse models) |
| `num_layers` | U32 | — | |
| `hidden_dim` | U32 | — | |
| `num_heads` | U32 | — | |
| `num_kv_heads` | U32 | `num_heads` | GQA / MQA |
| `head_dim` | U32 | `hidden_dim / num_heads` | |
| `max_seq_len` | U32 | — | |
| `sliding_window` | U32 | 0 | Window of the windowed layers; 0 = none |
| `num_sliding_layers` | U32 | 0 | Layers attending only the last `sliding_window` tokens |
| `precision` | `"fp4"`,`"fp8"`,`"bf16"`,`"fp16"`,`"fp32"` | `"bf16"` | Weights, attention compute and KV |

### `type = "deepseek_v4"`

MoE + MLA with per-layer compressed-history attention (DeepSeek-V4, GLM-5.2).
Expert GEMMs and everything else are separate precision streams.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name`, `num_layers`, `hidden_dim`, `num_heads`, `max_seq_len` | | — | |
| `kv_latent_dim` | U32 | — | MLA latent width stored per token per layer |
| `qk_rope_head_dim` | U32 | 0 | RoPE'd K width stored alongside the latent |
| `kv_precision` | Precision | `"fp8"` | KV cache precision |
| `num_active_expert_params` | U64 | — | Expert params per token (routed + shared) |
| `num_active_non_expert_params` | U64 | — | Non-expert params per token (attention, indexer, dense FFN, head) |
| `num_resident_expert_params` | U64 | — | All experts' resident params |
| `num_resident_non_expert_params` | U64 | — | |
| `expert_precision` | Precision | `"fp4"` | |
| `non_expert_precision` | Precision | `"fp8"` | |
| `window_size` | U32 | — | Recent-token window every layer attends |
| `num_dense_layers` | U32 | — | Layers with window only |
| `num_near_layers` | U32 | — | Layers with window + indexer-selected compressed history |
| `num_far_layers` | U32 | — | Layers with window + full stride-compressed history |
| `near_compress_ratio`, `far_compress_ratio` | U32 | — | Compression strides |
| `index_topk` | U32 | — | Indexer top-k cap on near layers |
| `index_n_heads`, `index_head_dim` | U32 | — | Indexer scoring shape |
| `indexer_retained_layers` | U32 | all near layers | Near layers running their own indexer (others reuse a neighbour's scores) |
| `index_kv_precision` | Precision | `kv_precision` | |
| `num_experts_per_tok` | U32 | — | Routed experts per token |
| `num_routed_experts` | U32 | 0 | Routed-expert pool; when set, per-step expert weight traffic follows coupon-collector growth with batch tokens |
| `num_moe_layers` | U32 | — | Layers doing EP routing |

### `type = "qwen35"`

Hybrid MoE: a minority of full-attention (GQA) layers with a growing KV
cache, the rest GatedDeltaNet linear layers with a fixed per-sequence state.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name`, `num_layers`, `hidden_dim`, `max_seq_len` | | — | |
| `num_attention_layers` | U32 | — | Full-attention layers |
| `num_attention_heads`, `num_kv_heads`, `attn_head_dim` | U32 | — | |
| `linear_num_value_heads`, `linear_num_key_heads`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_conv_kernel` | U32 | — | GatedDeltaNet state shape |
| `num_active_expert_params`, `num_active_non_expert_params`, `num_resident_expert_params`, `num_resident_non_expert_params` | U64 | — | As for `deepseek_v4` |
| `num_experts_per_tok`, `num_routed_experts`, `num_moe_layers` | U32 | — / 0 / — | As for `deepseek_v4` |
| `expert_precision`, `non_expert_precision`, `kv_precision` | Precision | `"bf16"` | |

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
| `enable_preemption_free` | Bool | false | Admit only what can grow to `prompt + max_output` without preemption |
| `enable_cascade_attention` | Bool | false | Load a batch's shared prompt prefix once per iteration |

---

## [workload]

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
