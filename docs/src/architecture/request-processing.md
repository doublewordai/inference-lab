# Request Processing

## Token accounting

A request has a prompt of `P` tokens and will produce `T` output tokens
(sampled from the workload's output distribution, capped by the client's
`max_output_tokens`). The forward passes are:

- one prefill pass over the prompt (chunked if `enable_chunked_prefill`),
  which produces output token 1 — TTFT is the end of that pass;
- one decode pass per further token.

So the request computes `P + T − 1` positions. `num_computed_tokens` is the
resident context; `is_prefill()` is `num_computed_tokens < prefill_len()`.

## Prefix caching

Dataset prompts carry incremental content hashes, one per KV block of
`block_size` tokens. At admission the scheduler looks the prompt up against
the KV cache: blocks resident in HBM are shared by reference and their
compute is skipped (block-aligned, and never the whole prompt — the last
block is always computed, as in vLLM); blocks in a memory tier (a store of
the pool's memory graph the worker can reach) move according to
`promote_fill`. The default, `through`, models HiCache read-through: an
external-store hit first transfers into the next closer store and becomes
resident there, repeating one level at a time, then loads from the closest
store into HBM. These storage stages consume real tier capacity and bandwidth
but reserve no HBM. Only the final host-to-device leg reserves landing blocks,
and it starts only when the whole prompt would fit. With
`load_overlap = "layerwise"`, its elapsed time overlaps the first prefill pass
as an aggregate pipeline (`max(load, compute)`); `none` serialises them.
`storage_prefetch` decides whether the demand waits for staging, abandons it
when admission is available, or waits to a deadline. Abandonment keeps fully
landed closer stages and recomputes the still-external suffix. `buffer` follows
the same path but releases transient intermediate copies after the HBM load;
`direct` retains the legacy store-to-HBM transfer. Requests can still join an
HBM landing already in flight for another request. Synthetic workloads carry
no prompt content and never hit.

## KV blocks

A block is the model's *content* KV of `block_size` tokens
(`kv_content_bytes(block_size)`): the part of the footprint that grows with
position for the sequence's whole life — full-context layers, compressed
history, indexer entries — which is linear in position, so a sequence of
`t` tokens holds `ceil(t / block_size)` content blocks and content block
`i` is prompt block `i`, the prefix-hashed unit (vLLM's full-attention KV
group). On top a sequence holds *auxiliary* blocks, unshared and released
with it: length-independent per-sequence state (GatedDeltaNet) plus each
sliding window's last `min(t, window)` positions (`kv_window_bytes`),
which vLLM's sliding-window group frees as the sequence slides past them.
Blocks are held by reference; freed blocks keep their content and can be
re-hit until they are recycled (least recently freed first, sequences tail
first — or, with `hbm_evict_backed_first`, a nearby run a tier already
holds). The blocks themselves are not tracked one by one: the topology's
KV state is a radix tree over block hashes (`kv_cache::radix`), a node
being a run of consecutive blocks no request diverges inside, and every
worker's HBM residency, pinning, free runs (stamped for LRU / outlook
order) and in-flight landings, and every store's holdings, are ranges of
tree nodes with breakpoints where requests ending at different positions
touched them. Because a request frees its whole prefix at once, free-time
stamps are non-increasing along a chain and eviction never punches a hole
in HBM; a chain forked at every session step compacts back into one run
once the side branches are gone. Recycling a block whose KV no tier holds writes it to the
worker's first tier under `write_back`; under `write_through` every fresh
block was written when produced, under `selective` on its n-th hit. Tiers
are stores of the topology's `MemoryGraph`, instantiated from the
hardware's `[memory]` template per GPU (private) or per node (shared by the
node's workers), or once per cluster (shared by every worker through its NIC
and the network core), inclusive of HBM. Cluster stores may stripe one access
over several bandwidth units while a separate aggregate edge is the single
contention point for all accesses; store access latency is paid before bytes
flow. Writes, promotions, cascades between tiers and hand-offs are all
transfers on the graph's edges.

`peer_hbm` is a virtual node-local tier rather than another store. A lookup
can source the next prefix span from a sibling worker's resident HBM and
promote it GPU → switch → GPU over the hardware's NVLink edges. It has no
capacity and receives no writes. Tier order still selects the closest
available source first.

A tier entry may pin fetch sources. While pinned, a peer's free HBM run or a
normal store range cannot be recycled; an allocation with no other victim
waits and records a pin stall. Staged store reads always protect their source
until that leg drains, because their destination is a real cache fill. For
legacy direct reads, peer HBM pins by default and normal stores retain the
historical unpinned default. If an unpinned direct source loses a suffix,
completion publishes only the surviving prefix, releases the unused landing
reservation, and recomputes the rest. This records a partial landing.

## Preemption

When a running request cannot get the block its next position needs, the
scheduler preempts (from the cursor onward, never a request already in the
batch) until it can, or skips it. A preempted request loses its resident KV
and keeps its generated tokens: it goes to the head of the waiting queue and,
on resume, re-prefills prompt plus generated tokens before decoding again
(vLLM v1 recompute). Nothing new is admitted in a pass that preempted.
`enable_preemption_free` instead admits only requests that can grow to
`prompt + max_output` alongside everything running.

## Speculative decoding

With `[speculative]`, at the end of each step the planner draws a *round* per
decode sequence (per-depth acceptance signal and full-depth commits, from an
analytic model or a replayed trace bank) and chooses each sequence's draft
depth for the next step. The scheduler reserves `1 + draft` positions in the
token budget and KV; the verify pass then advances by `1 + min(commits,
draft)`. Steps are priced from the roofline at the verify width, or from a
measured step-cost table when one is configured.

## Disaggregated hand-off

On a prefill/decode topology, a request whose prefill has completed leaves
the prefill worker (its KV freed there, but still hittable in that worker's
prefix cache until recycled), is routed to a decode worker, transfers its
KV over the memory graph — the prefill GPU's NIC to the network core and
the decode GPU's NIC in, at its max-min share of each — and joins that
worker when the transfer drains. The decode worker is chosen when the
transfer starts, so the
router's KV-aware policies can pick a decoder that already holds part of
the context; the transfer carries the context minus the prompt prefix that
decoder has resident in HBM. On admission the decode worker treats the
transferred positions as resident — it allocates blocks for them (sharing
any it already held) and computes nothing but the decode step; the
prompt's hashes are published into its prefix cache. `RequestTiming`
records `prefill_done_time` and `handoff_done_time` alongside TTFT and
completion.
