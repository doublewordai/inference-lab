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
block is always computed, as in vLLM); blocks in a spillover tier or already
in flight for another request are promoted / joined asynchronously while the
request waits with its landing blocks reserved. Synthetic workloads carry no
prompt content and never hit.

## KV blocks

The `KVCacheManager` charges a sequence of `t` tokens
`ceil(kv_storage_bytes(t) / kv_storage_bytes(block_size))` content blocks
from the model's own KV curve — linear models get `ceil(t / block_size)`,
sliding-window and compressed-history models their real footprint — plus a
fixed reservation for length-independent per-sequence state (GatedDeltaNet).
Blocks are reference-counted; freed blocks keep their content hash and can
be re-hit until they are recycled, at which point the hash falls into the
first spillover tier.

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
the prefill worker (its KV freed there), transfers its KV over the shared
hand-off link, and joins the decode pool. `RequestTiming` records
`prefill_done_time` and `handoff_done_time` alongside TTFT and completion.
