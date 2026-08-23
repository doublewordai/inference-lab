# Correction to LANE.md (2026-08-23 18:45)

LANE.md names `590598d` as the broken-binary commit. Wrong: the
`INFERENCE_LAB_DUMP_AT` hook and the staging-park report only exist from
`21783c6`. Use **`21783c6`** as the broken commit — it has the promotion fix
and the dump hook but not the `cd02843` hand-off wake, so the staging creep
reproduces there with diagnostics available. Everything else in LANE.md
stands.
