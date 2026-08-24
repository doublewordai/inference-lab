# Branch state: fix/staged-read-gating (2026-08-24, verified)

## Committed

- `1f357d6` (agent) gate staged reads on destination-store capacity + pin
  landed stage copies until consumed (releases on consume/abandon/abort).
  Also corrected a test that asserted the old buggy behaviour.
- `f441685` (agent) scheduler: Blocked stages stay parked; Impossible
  prefixes (can never fit) fall back instead of parking forever.
- `0509760` close-out docs (REPORT/NOTES/LANE) at the point the run stopped.
- (2026-08-24) fix: submission is the only staging classifier. The
  "regression" the close-out reported was not a slowdown but an admission
  livelock: `staging_outcome` (per-span, and `need == 0` → Fits) could say
  Fits where `submit_stage_promotion` (whole-group gate) refused, and the
  scheduler respun on the disagreement with sim time frozen inside one
  `schedule` pass. `submit_stage_promotion` now returns
  `Result<String, StageDenied>` (AlreadyStaged | Blocked | Impossible; a
  denial mutates nothing), `start_staging` folds groups into `StageStart`,
  and the scheduler either changes an admission input before retrying or
  holds the head without retrying this pass. Also range-restricted
  `store_range_pinned` to the candidate's node (was a full-store pin scan,
  ~45% of the unwedged run). Details in NOTES.md.

## Verified (LANE.md item 6)

- Repro (`sweep/repro-staging/nvme-p1-l0.08`): completes 25,264.8 s
  simulated in 33.8 s wall vs cd02843's 25,267.3 s in 32.5 s. host_dram
  evictions 47.5M (cd02843: 62.5M), dead bytes 11.4 TB (70.5 TB), NVMe
  read 127.9 TB (173.5 TB).
- Suite: 259 tests green (two new livelock-regression tests), fmt/clippy
  clean; `docs/src/reference/config.md` updated with the gating semantics.
- Healthy cell (`sweep/disagg/nvme-p1-l0.06`, timeout 60 s): NOT identical,
  and cannot be — the banked reference reproduces bit-identically from
  cd02843, and this cell holds host DRAM at 100%, so the gate engages.
  Deltas under the fixed binary (outputs in `sweep/disagg-verify/`,
  originals untouched): peak_transfers_in_flight 601→402, prefix misses
  −10.3%, TTFT mean −17%, per_token mean +24%, preemptions 145→186.
  Re-running the disagg ladder on the fixed binary is Fergus's call.

## Session notes

Agent: pi, doubleword / DeepSeek-V4-Flash-0731, ~3 h. Recurrent failure:
tool calls emitted as literal `<｜DSML｜...>` text (nothing executes, pi sits
idle); recovered by re-prompting, 4+ occurrences, frequency increasing with
context. Plus one 20-min stalled request and one 500 earlier (first session,
01a02f86). Serving-side follow-ups, evidence in
`~/.pi/agent/sessions/--home-fergus-scratch-kv-policy-eval-agents-staged-gating--/`.
