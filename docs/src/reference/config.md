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
[router]                       # optional: how requests spread over replicas
[decode_router]                # optional: disagg decode pool (defaults to [router])
[memory]                       # optional: KV tiers beyond HBM, from the hardware's stores
[fault]                        # optional: serve-mode static fault injection

[hardware.b200]                # one entry per hardware the model runs on
tp = 4
replicas = 4                   # identical workers behind the router
[hardware.gh200]
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
`parallel`, `model`, `scheduler`, `workload`, optional `replicas`, `router`,
`decode_router`, `memory`, `speculative`.

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
| `tp` | U32 | 1 | Replica world size: its GPUs pool FLOP rate, HBM bandwidth and memory; weights are sharded across them; each layer's output is all-reduced twice (after attention, after the FFN) |
| `ep` | U32 | 1 | Experts sharded across `ep` of the ranks (divides `tp`). With TP attention every rank holds every token, so the MoE output is still combined by the FFN all-reduce (vLLM `--enable-expert-parallel`); under `dp_attention` the MoE layers dispatch + combine with all-to-alls over the `ep` group instead |
| `dp_attention` | Bool | false | Attention runs data-parallel over the `tp` ranks (sglang `--enable-dp-attention`): the attention projections are replicated (`tp×` resident and read per step), a sequence's KV lives on one rank, no attention all-reduce; the TP-sharded FFN gathers the ranks' tokens with an all-gather and returns them with a reduce-scatter (with `ep > 1`, DeepEP-style dispatch + combine all-to-alls). **Each rank is its own worker**: its own scheduler and KV cache over its GPU's HBM (`1/tp` of the replica's), its own port into the node's `[memory]` stores; `[scheduler]` limits apply per rank. The `[router]` routes flat over ranks — the placement decision a DP-aware front end (Dynamo's `(worker, dp_rank)` endpoints) makes. The ranks step in lockstep as one group: the step is priced on the replica-wide roofline over the union batch plus the attention skew, the slowest rank's own-GPU attention time over the mean rank's (the ranks meet at every layer's FFN collective). Speculative decoding is not modelled with rank groups |
| `replicas` | U32 | 1 | Identical workers of this deployment (each a `tp`-GPU replica with its own scheduler and KV cache; `tp` rank workers each under `dp_attention`) behind the router |
| `spec` | String or Table | the entry name | Another catalog preset, or an inline hardware table (fields below) |
| `scheduler` | Table | `{}` | Keys merged over the shared `[scheduler]` for this entry |
| `speculative` | Table | shared `[speculative]` | Replaces the shared block for this entry |
| `router` | Table | shared `[router]` | Replaces the shared block for this entry |
| `decode_router` | Table | shared `[decode_router]` | Replaces the shared block for this entry |
| `memory` | Table | shared `[memory]` | Replaces the shared block for this entry |

Collectives are priced on the hardware's `[fabric]` (below) and added
serially to the step; an entry with `tp > 1` or `ep > 1` on hardware without
one is rejected. Per layer: attention → one all-reduce over `tp` (none under
`dp_attention`); dense FFN, or MoE without `dp_attention` (any `ep`) → one
all-reduce; under `dp_attention`, dense FFN or MoE with `ep = 1` → an
all-gather + reduce-scatter, MoE with `ep > 1` → dispatch and combine
all-to-alls over `ep`, each rank moving its `tokens / ep` share of
`experts_per_tok` hidden vectors. Expert reads and FLOPs are taken as
balanced across ranks; there is no overlap of collectives with compute.

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
| `fabric` | Table | unset | Collective fabric, see below |
| `memory` | Table | unset | KV memory beyond HBM this node class offers (stores and links), see below. A deployment picks tiers from it with its own `[memory]` |

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

#### `[memory]` on the hardware

The stores a node offers for KV beyond HBM and the links that reach them,
templated per GPU or per node: a graph. Presets ship one (host DRAM and a
node NVMe pool behind each GPU's PCIe port on the HGX parts; Grace LPDDR5X
behind NVLink-C2C on `gh200`; NVLink ports into the node switch and one NIC
per GPU to the network on all); an inline hardware table can declare its
own.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpus_per_node` | U32 | fabric's, else 1 | GPUs sharing a node's `per = "node"` stores |
| `stores` | Array | `[]` | `{ name, per = "gpu" \| "node", capacity, bandwidth }` — bytes per instance; a `gpu` store is private to its GPU, a `node` store is one pool for the node's GPUs. `bandwidth` (optional) is the store's own throughput per instance, shared by every transfer in or out of it |
| `junctions` | Array | `[]` | `{ name, per }` — a point with no capacity of its own, so that several links can share one (a GPU's PCIe port feeding host DRAM and NVMe) |
| `links` | Array | `[]` | `{ name, from, to, bandwidth, latency = 0 }` — `from` is `"gpu"` (one port per GPU), a store or a junction; `to` is a store, a junction, `"switch"` (the node's scale-up fabric) or `"network"` (the scale-out core). One instance per instance of `from`, full duplex at `bandwidth` bytes/s each way |

Instances: a `tp`-GPU worker pools its GPUs' per-GPU stores, junctions and
ports (capacity and bandwidth × `tp`); `gpus_per_node / tp` workers share a
node's per-node instances. A transfer takes the shortest hop path between
its ends and runs at its max-min fair share on every edge of it: the most
contended edge fixes its transfers' rate first, the residual is shared
among the rest. Tier promotions run store → GPU; hand-offs GPU → network →
GPU (see `kv_link_bw`).

Shipped presets, at datasheet figures: `b200` (192 GB / 8 TB/s), `b300`
(288 GB / 8 TB/s), `gh200` (96 GB / 4 TB/s), `h100` (80 GB / 3.35 TB/s);
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
| `routing` | Table | unset | MoE routing: `{ routed_experts, experts_per_tok, moe_layers }`. The per-step read then follows coupon-collector growth with the step's tokens (per-expert and shared params are recovered from the active/resident split); EP all-to-alls (under `dp_attention`) = 2 × `moe_layers` |

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
| `q_latent_dim`, `o_latent_dim` | U32 | unset | Low-rank query / output projections (`q_lora_rank`, `o_lora_rank`); size the attention projections replicated under `dp_attention`. Absent = full-rank |

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
| `sessions_path` | String | unset | Session file (JSONL, one session per line, see [Sessions](../user-guide/configuration.md#sessions)). The arrival pattern then governs session *starts* (`arrival_rate` in sessions/s; `closed_loop` holds `num_concurrent_users` sessions in flight); each later step arrives at its parent's completion plus the step's gap. Mutually exclusive with `dataset_path`; length distributions are ignored |
| `num_sessions` | U32 | unset | Session mode: stop starting sessions after this many (the file is cycled, so it may exceed the file's count) |
| `seed` | U64 | — | |

---

## [memory]

Optional; no tiering when absent. Which of the hardware's stores hold KV
evicted from HBM, closest first, and how much of each. A worker (a
`tp`-GPU replica) reaches each tier over its own link — its GPUs' ports
pooled, so a `tp = 2` worker on `gh200` promotes at 2 × 450 GB/s from 2 ×
120 GB of Grace memory. A `per = "node"` store is shared by the workers on
that node (`gpus_per_node / tp` of them): what one demotes, its neighbours
can promote. A worker wider than a node pools the node stores it spans.

```toml
[memory]
tiers = ["host_dram", "nvme"]
preset = "reactive"                            # reactive | oracle (optional bundle)
source = { policy = "promote" }                # promote | min_time
hbm_eviction = { policy = "lru" }              # lru | outlook
write = { policy = "write_back" }              # write_back | write_through | selective | live
eviction = { policy = "fifo" }                 # fifo | lru | ttl | outlook
prefetch = { policy = "none" }                 # none | outlook
hbm_evict_backed_first = false
[memory.capacity]
host_dram = 1.0e12          # bytes per instance given to KV
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tiers` | Array | `[]` | Store names from the hardware's `[memory]`, closest first. Each must be reachable from a GPU over the hardware's links |
| `capacity` | Table | full | Per-store cap on bytes per instance |
| `preset` | String | unset | A named bundle of the five policies below plus `hbm_evict_backed_first`; any field set explicitly overrides the preset's choice. `reactive`: `promote` / `lru` / `selective` (`min_hits` 1) / `lru` / `none` / backed-first — decides only from what has already happened, as shipped stacks do. `oracle`: `min_time` / `outlook` / `live` / `outlook` / `outlook` / backed-first — reads every session's announced re-entry |
| `source` | Table | `promote` | Where a re-entry's tier-held prefix comes from: `promote` — fetch it (a hit is a hit); `min_time` — fetch it only if the transfer, at the fetch path's current fair share, beats recomputing those tokens at the worker's roofline; otherwise recompute (the tier keeps its copy) |
| `hbm_eviction` | Table | `lru` | Which free HBM block is recycled first: `lru` — least recently freed; `outlook` — blocks with no announced re-entry first (LRU among them), then the farthest re-entry first, each sequence tail first |
| `write` | Table | `write_back` | When a block's KV is written to the first tier: `write_back` — when its HBM block is recycled, if no tier holds it; `write_through` — as soon as it is produced; `selective` (`min_hits`, default 1) — on its `min_hits`-th HBM hit, and dropped on eviction otherwise (SGLang HiCache's three positions); `live` — when recycled, only if its session has announced a re-entry (a finished trajectory is dropped) |
| `eviction` | Table | `fifo` | How every tier picks what to recycle: `fifo` (least recently inserted), `lru` (least recently inserted or promoted from), `ttl` (`seconds`: LRU, and any block untouched that long is dropped whether or not the store is full), `outlook` (no announced re-entry first, then farthest re-entry first) |
| `prefetch` | Table | `none` | Whether a demoted prefix is pulled back ahead of its re-entry: `none`; `outlook` (`lead`, default 0 s) — when a session step completes, plan a promotion of the prefix its next step re-enters with, starting so it lands `lead` seconds before that arrival at the fetch path's fair share at planning time; whatever is still in HBM when the plan fires needs nothing, and a re-entry that arrives mid-transfer joins it |
| `hbm_evict_backed_first` | Bool | false | When HBM must recycle a block, take one whose KV a tier already holds (a free drop) over the policy's first choice, looking 16 blocks up the free queue |

The `outlook`, `live` and `min_time`/`prefetch` policies act on a session
step's **outlook**: on a session workload, when a step completes the
simulator knows its successor's arrival (completion plus the recorded gap)
and how much of the context it re-enters with, and marks those blocks —
in HBM and in the tiers — with that time. Other workloads announce
nothing, so under `live` nothing is written and `outlook` eviction reduces
to LRU. The gap between the `reactive` and `oracle` presets on the same
replay is the value of knowing the re-entry.

Tiers are inclusive: a promoted block keeps its tier copy (KV is
immutable), so its next eviction from HBM is a free drop and only blocks no
tier holds ever cost a write. Writes are transfers GPU → store on the same
graph as promotions (full duplex: they share a port with fetches only in
the reverse direction, but do share an NVMe pool's drives); a block is
resident once its write lands and a promotion of a block still arriving
waits for it (the write-before-reuse race). A full store evicts its victim
into the next tier as a store → store transfer, or drops it; a block
dropped without ever having been promoted counts as dead bytes. A prompt
whose blocks sit in a tier is promoted along the path from that store to
the worker's GPU — sharing every edge on it with whatever else is in
flight — while the request waits with its landing blocks reserved.

The summary's `memory` section reports, per store name, blocks held, bytes
written / read / dead, evictions and expiries; per link name, bytes moved
and utilisation; and totals of bytes written, bytes promoted, and
promotions that waited on a write. The `prefix_cache` section counts
lookups recomputed instead of fetched (`min_time`) and prefetches started
(`prefetch = outlook`), with their tokens.

## [router] and [decode_router]

Optional; `round_robin` when absent. `[router]` picks the replica each
arriving request enters (`replicas` on the hardware entry; the prefill
pool on a disaggregated topology). `[decode_router]` picks the decode
worker each hand-off goes to on a disaggregated topology, and defaults to
`[router]`. The KV-reading policies look up each replica's state for the
prompt on every decision — an estimate from the replica's block index, as
a KV-aware front end sees it, not the scheduler's admission-time lookup.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `policy` | String | `"round_robin"` | `round_robin`, `least_loaded`, `prefix_affinity`, `kv_aware`, `kv_aware_decode` |
| `max_load_ratio` | F64 | unset | `prefix_affinity` only: pass over the prefix holder for the least-loaded replica when its requests in system exceed `max_load_ratio ×` the pool mean (bounded-load affinity) |
| `load_weight` | F64 | 1.0 / 64.0 | `kv_aware`: weight on the replica's queued prefill tokens. `kv_aware_decode`: tokens of transfer one running sequence is worth (default one 64-token block) |

- `round_robin` cycles through the replicas.
- `least_loaded` picks the fewest requests in system (running + waiting),
  ties by queued prefill tokens, then index.
- `prefix_affinity` picks the replica holding the longest cached prefix of
  the prompt (any tier); with none anywhere it falls back to
  `least_loaded`.
- `kv_aware` minimises `(prompt − cached prefix) + load_weight × queued
  prefill tokens`: the prefill work the request adds plus the prefill work
  already ahead of it, in tokens. A prefill-side policy: on a decode pool
  the load term is always zero.
- `kv_aware_decode` minimises `(context − prompt prefix resident in the
  decoder's HBM) + load_weight × running sequences`: the KV the hand-off
  must move plus the decode batch it joins, in tokens. Decoders whose free
  KV cannot hold the incoming context are passed over while any can.

The summary's `router` section (and `decode_router` on a disaggregated
topology) reports requests per replica and, for the KV-reading policies,
how many decisions had a cached prefix on some replica, how many went to
a holder, and how many went away from the longest holder. `handoff`
reports transfers, bytes moved, and bytes skipped because the chosen
decoder already held the prefix.

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
