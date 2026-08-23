# Branch state: fix/staged-read-gating (2026-08-23, closed out by supervisor)

Written by the supervising session after stopping the pi agent (Fergus's
call: bank what's good, stop the run).

## Committed (both by the agent, suite green: 255 tests, clippy/fmt clean)

- `1f357d6` gate staged reads on destination-store capacity + pin landed
  stage copies until consumed (releases on consume/abandon/abort). Also
  corrected a test that asserted the old buggy behaviour.
- `f441685` scheduler: Blocked stages stay parked; Impossible prefixes
  (can never fit) fall back instead of parking forever.

## NOT verified — known regression, do not merge as-is

The capped wait_complete repro
(`~/scratch/kv-policy-eval/sweep/repro-staging/nvme-p1-l0.08`) completes in
32 s wall on the pre-gating binary (`sweep/inference-lab-disagg3`, commit
`cd02843`); on this branch it reaches only t≈800 s of 25,200 s in 45 s wall
(~30× slower) and times out at 300 s. Direction is right (the livelock the
gate targets is real — see LANE.md + LANE-CORRECTION.md), the implementation
churns: suspect the gate/park/retry cycle re-evaluates parked stages every
scheduler pass, or the pin/trim bookkeeping. Needs a performance pass and
then the LANE.md verification steps (repro at 21783c6 semantics, healthy-cell
numbers unchanged vs `sweep/disagg/nvme-p1-l0.06`).

## Session notes

Agent: pi, doubleword / DeepSeek-V4-Flash-0731, ~3 h. Recurrent failure:
tool calls emitted as literal `<｜DSML｜...>` text (nothing executes, pi sits
idle); recovered by re-prompting, 4+ occurrences, frequency increasing with
context. Plus one 20-min stalled request and one 500 earlier (first session,
01a02f86). Serving-side follow-ups, evidence in
`~/.pi/agent/sessions/--home-fergus-scratch-kv-policy-eval-agents-staged-gating--/`.
