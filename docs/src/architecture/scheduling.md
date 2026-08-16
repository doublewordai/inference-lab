# Scheduling

`Scheduler::schedule` runs once per worker iteration and mirrors vLLM v1.

## Phases

0. **Reap.** Finished requests are removed and their KV freed before any
   allocation, so a request later in the running order can hand its blocks
   to one earlier in the same pass.
1. **Running.** In admission order, each running request is given its
   positions for the step: the remaining prefill (capped by
   `long_prefill_token_threshold` under chunked prefill) or, in decode,
   `1 + pending_draft_len`, all within `max_num_batched_tokens`. If the
   request needs KV blocks that are not free, the policy picks a victim at
   or after the cursor to preempt (repeating until the request fits or no
   victim remains).
2. **Waiting.** Only if nothing was preempted: the policy picks the next
   waiting request; preemption-free admission and KV space are checked; the
   prefix cache is consulted (see request processing); the request enters
   the running set with its first positions scheduled. Stops at
   `max_num_seqs`, an exhausted token budget, or a request that does not fit.

## Policies

`policy` selects both the admission order and the preemption victim:

| Policy | Admits | Preempts |
|--------|--------|----------|
| `fcfs` | oldest waiting | most recently admitted |
| `priority` | oldest waiting | lowest priority (highest value), latest arrival breaks ties |
| `sif` / `lif` | shortest / longest prompt | longest prompt |
| `sof` / `lof` | shortest / longest max output | longest remaining output |
| `stf` / `ltf` | shortest / longest prompt + max output | longest remaining |

Length policies use the client-visible bound (`max_output_tokens`), never
the sampled target.

## Preemption-free admission

With `enable_preemption_free`, a waiting request is admitted only if every
running request and the candidate could all reach `prompt + max_output`
tokens of context at once within the KV capacity. Conservative, and it
guarantees zero preemptions.
