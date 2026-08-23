# Lane: gate staged HiCache reads on destination-store capacity

You are working in the git worktree `/home/fergus/scratch/kv-policy-eval/agents/staged-gating`
of the Rust crate `inference-lab` (a discrete-event LLM inference simulator), on branch
`fix/staged-read-gating` (off `disagg-toml`). Build with `cargo build --release`; test with
`cargo test --release`; keep `cargo fmt` and `cargo clippy --release --all-targets` clean.

Rules: commit on this branch only, with clear messages; never push, never open PRs, never
touch files outside this worktree except the repro directory named below. When done, write
`REPORT.md` in this worktree: what the bug was, what you changed, how you verified it, and
anything you could not resolve. Keep `docs/src/reference/config.md` accurate for any
behaviour you change.

## The bug

`promote_fill = "through"` (#132) stages an external-store hit (NVMe) into the next closer
store (host DRAM) before the final DRAM→HBM load; a request whose prefix needs such a stage
is parked in `Scheduler::pending_storage` (`src/scheduler/scheduler.rs`,
`park_on_storage_stage`, `promote_finished_transfers`) until the stage lands. Under
`storage_prefetch = { policy = "wait_complete" }` (the `reactive` preset's default) it waits
indefinitely.

Stages are not admission-controlled against the destination store's capacity, and blocks a
stage has landed in DRAM are not pinned while the request waits. So when the parked
requests' stages together exceed the DRAM slice, the stages evict each other before any one
completes. Observed state (8 prefill DPA ranks, 250 GB private DRAM per rank = 82,036
blocks, 3.84 TB private NVMe per rank, binary built at commit `590598d`):

```
dump at t=40000: 0 running, 53 waiting
  staging 53: [0] s132/214(cached 578176, ready_at None, abandoned false)
  pool 0 worker 0: running 0, waiting 53, parked 0, free 22349/22349 HBM blocks
  host_dram: 656288 blocks held of 656288 (100%), evictions 157,866,112, read 340 TB
```

Every prefill rank idle with all HBM free, 53 requests each staging a ~10k-block prefix into
an 82k-block DRAM that is already full, 158 M evictions and counting, nothing ever lands;
sim time creeps at each transfer chunk so the "no pending events" stall detector never
fires. Two unrelated stalls were fixed on this branch already (`590598d`, `cd02843`); after
`cd02843` (a drained hand-off wakes the prefill worker) this exact run happens to drain, but
the gating gap is untouched and the next workload shape will hit it.

## Reproduce

`/home/fergus/scratch/kv-policy-eval/sweep/repro-staging/nvme-p1-l0.08/` holds
`config.toml` (a `[prefill]` disaggregated topology; the prefill pool carries the
`host_dram` + `nvme` tiers with `wait_complete`), `workload.toml` (7 h of session arrivals,
`duration_secs` cap), and `repro.sh`. The sim binary is `target/release/inference-lab`.

```
# broken: a binary at 590598d creeps forever; the dump prints the stuck state and exits 3
git stash -u; git checkout 590598d; cargo build --release; git checkout fix/staged-read-gating; git stash pop
BIN=target/release/inference-lab DUMP_AT=40000 TIMEOUT=120 sweep/.../repro.sh   # see the state above
# expected when fixed (and already on HEAD by accident): "Simulation Complete" at ~25,260 s within ~40 s wall
```

`INFERENCE_LAB_DUMP_AT=<sim seconds>` (env var, `src/simulation/simulator.rs`) prints the
stuck-worker report plus memory metrics at that sim time and exits 3. The stuck-worker
report (`Engine::describe_stuck_workers`) now shows the staging park and the head request's
prefix lookup. Use them.

## What to build

1. **Capacity gating of stages.** A stage into store S starts only if S can hold its bytes:
   free blocks plus blocks it may evict (not pinned, not landing, not in flight) minus the
   reservations of stages already in flight into S. A stage that cannot start stays parked
   (or, if nothing is in flight into S and it can never fit — prefix larger than the store —
   falls back: promote direct from the source if `promote_fill` allows, else recompute the
   external suffix, as `best_effort` cancellation does). Count in-flight stage bytes per
   store instance. Find where stages are submitted (`KVCacheManager::start_staging`,
   `needs_staging`; `src/kv_cache/graph.rs` staged transfer submission) and where the store's
   free/evictable accounting lives.
2. **Pin landed stage blocks until consumed.** Blocks a stage has landed in S on behalf of a
   parked request must not be evicted by later stages or writes until that request's final
   load takes them (or its stage is cancelled / times out / the request is preempted), then
   they return to normal LRU. Check how `pin` on `StoreTemplate` and landing reservations work
   for HBM promotions (`reserve_blocks_for_transfer`, `partial_landings`, `pin_stalls`) and do
   the analogous thing for the intermediate store.
3. **Wake on stage completion.** Confirm that a completed stage wakes its worker (an event
   that causes a `schedule` pass) even when nothing else is happening on that worker; if only
   other events (arrivals, hand-off drains) trigger the pass today, make the stage's drain do
   it. This is probably why `cd02843` hid the symptom.
4. **No new deadlock.** With gating, a head-of-line request waiting for stage capacity while
   nothing runs and nothing is in flight must still make progress (cf. the rule in
   `scheduler.rs` around `release_waiting_kv` / "Nothing runs and nothing is in flight").
5. **Tests.** Unit tests in the style of `src/scheduler/scheduler.rs` tests
   (`a_promotion_starts_only_when_the_whole_prompt_would_fit`,
   `a_promotion_at_the_head_takes_back_kv_held_by_waiting_promotions`) and
   `src/simulation/mod.rs` (`a_drained_handoff_wakes_the_prefill_worker_...`): a store too
   small for two concurrent stages serialises them rather than thrashing; landed stage blocks
   survive a competing stage; a stage completion alone wakes the worker; an oversized prefix
   falls back rather than parking forever. Each test must fail without your change.
6. **Verify on the repro** at `590598d`-equivalent behaviour: with your change the run must
   complete (~25,260 s simulated) and `host_dram` evictions must be sane (tens of thousands,
   not millions). Then check the healthy-window numbers are unchanged: run
   `/home/fergus/scratch/kv-policy-eval/sweep/disagg/nvme-p1-l0.06/config.toml` +
   `workload.toml` (that cell uses `storage_prefetch = timeout 60 s`) and compare
   `summary.json` throughput/latency with the existing one there (coverage 0.996, TTFT p50
   ≈ 0.04 s, ~5.9k recomputed tok/s over the 5–7 h window; `python3
   /home/fergus/scratch/kv-policy-eval/ladder_disagg.py table` prints the table if you
   write your outputs into a copy of `sweep/disagg` and set `LADDER_OUT`). Do not overwrite
   the existing cells.

Background on what the sim models here: SGLang HiCache's storage prefetch (NVMe/3FS → host
memory, then host → device), which bounds prefetch by host-pool free space and keeps the
prefetched host nodes until the request uses them. `docs/src/reference/config.md`
(`[memory]`: `promote_fill`, `storage_prefetch`, `load_overlap`) describes the current
semantics. The experiment this serves is in
`/home/fergus/notes/docs/campaigns/kv-tier-post/disagg-ladder.md` (read-only context).
