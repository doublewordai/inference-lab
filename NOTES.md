# NOTES — staged-read gating lane

Running log of established facts, decisions, and open questions.

## Context

- Worktree: `/home/fergus/scratch/kv-policy-eval/agents/staged-gating`, branch
  `fix/staged-read-gating` (off `disagg-toml`), crate `inference-lab`.
- HEAD = `cd02843` (drained hand-off wakes prefill worker). Branch history:
  - `590598d` fix(scheduler): promotion at head takes back KV held by waiting promotions
  - `21783c6` debug: storage-stage park in stall report + `INFERENCE_LAB_DUMP_AT` hook
  - `cd02843` fix: drained hand-off wakes the prefill worker
- Bug: `promote_fill="through"` stages external-store hits into the next closer store
  (NVMe→DRAM), but the intermediate store (host_dram) is not admission-controlled:
  stages are not gated on destination capacity, and landed stage blocks are not pinned.
  Result: parked requests' stages thrash (evict each other), nothing ever lands, stalls.
- Repro: `/home/fergus/scratch/kv-policy-eval/sweep/repro-staging/nvme-p1-l0.08/`
  (prefill DPA, 8 ranks, 250 GB private DRAM/rank, 3.84 TB NVMe/rank, `wait_complete`).

## TODO (from LANE.md, ordered as instructed: wake+pinning first, gating second)

1. [ ] Wake on stage completion (confirm/produde worker wake when nothing else happens)
2. [ ] Pin landed stage blocks until consumed by the final load
3. [ ] Capacity gating of stages (admission control on destination store)
4. [ ] No new deadlock (head-of-line must still progress)
5. [ ] Tests (each must fail without the change)
6. [ ] Verify repro completes with sane evictions; healthy-window numbers unchanged
7. [ ] REPORT.md, config.md accuracy, NOTES.md kept current

## Log

<!-- appended as I go -->
## Updated findings

- LANE names `590598d` as broken; the dump hook only exists from `21783c6`.
  Supervisor confirmed: broken commit = `21783c6` (promotion fix + dump hook, no
  `cd02843` wake). Baseline reproduced there: exit 3, "0 running, 53 waiting, next
  event Some(40000.37)", staging 53 all stuck, host_dram 656288/656288 held (100%),
  157,866,112 evictions. (baseline `/tmp/run-21783c6.log`)


## Established (architecture)

- Stage flow: scheduler `park_on_storage_stage`→`start_staging`→`graph.submit_stage_promotion`
  inserts missing destination spans as ARRIVING (consuming held immediately) then evicts to
  capacity. On drain `advance()` → `store_landed` (resident) + `landed_stage_fills` +
  `wake_workers.insert(worker)`. Engine `pump_flows` sees `Owner::Worker` → `maybe_wake_worker`
  → schedule pass → `promote_finished_transfers` collects `completed.staged` wake → push to
  waiting.
- WAKE-ON-STAGE-COMPLETION ALREADY FUNCTIONS (engine.rs pump_flows, graph advance).
  Item 3 is likely a non-issue; confirm with a test.
- The failure is the STAGE THRASH: stage blocks are neither admission-gated on destination
  capacity nor pinned, so concurrent stages evict each other before any lands (0 running,
  53 waiting).
- Store capacity is in BLOCKS (radix StoreMeta.capacity/held). Arriving stage blocks count in
  held. Radix pins keyed by (node,start,end); store_trim_capacity skips pinned ranges.
- Gating design: allow a stage into S iff need <= (C - held) + (held - pinned) == C - pinned.
  Pin in-flight AND landed stage blocks until the owner's final load consumes them.

## Implemented so far (graph-level pinning + capacity gating)

- radix.rs: `StoreMeta.pinned_blocks` counter (sum of protected span lengths),
  maintained in `pin_source`/`unpin_source`; `store_pinned_blocks`/
  `store_available_for_stage(s) = capacity - pinned_blocks`.
- graph.rs `submit_stage_promotion`: capacity gate — a stage into the
  destination store starts only if `need <= store_available_for_stage(dest)`,
  i.e. it may only displace unpinned resident blocks. On start it PINs the
  destination copy (in-flight AND landed) for the stage's lifetime.
- gated → submit returns None (stage doesn't start).
- Dest pins released on `promoted_batch` (final HBM load consumes), on
  `cancel_staged_batch` (abandon), and `cancel_stage_promotion` (in-flight abort);
  a `release_stage_pins` helper + `trim_store_after_unpin` clean up.
- Updated `an_evicted_stage_destination_still_completes...` →
  `a_landed_stage_copy_survives_a_competing_write_until_it_is_consumed` (was
  asserting the OLD buggy behaviour). All 253+ tests pass.
- STILL TODO: scheduler-level "impossible to fit → fall back" handling; wake
  test; capacity-serialisation test; oversized-prefix test; verify repro.
