#!/usr/bin/env python3
"""Export TraceLab per-step stats to the simulator's session JSONL.

    python3 examples/sessions/tracelab_export.py per_step_stats.parquet \
        --out data/sessions/tracelab.jsonl [--provider claude] [--min-steps 2]

One session per line: {"id", "steps": [{"input", "new", "output", "gap", "kind"}]}.

Column mapping (per_step_stats.parquet):
  input  = input_total            prompt tokens the model saw
  new    = round(ideal_new)       prompt tokens that are not the parent's
                                  context (the provider `prefix` counter is
                                  not used: it reads 0 after the cache TTL)
  output = output                 generated tokens, reasoning included
  gap    = max(gap_s, 0)          seconds from the parent's end to this start;
                                  negative gaps (overlapping steps: parallel
                                  tool calls) are clamped to 0, step 0 is 0
  kind   = user | tool | null     from first_input_event_type

Requires pyarrow.
"""

import argparse
import json
import math
import sys
from collections import defaultdict

import pyarrow.parquet as pq

KIND = {"user_message": "user", "tool_result": "tool"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", help="keep only this provider (claude, codex)")
    ap.add_argument("--min-steps", type=int, default=1, help="drop sessions with fewer steps")
    args = ap.parse_args()

    cols = [
        "session_id", "step_idx", "provider", "input_total", "ideal_new",
        "output", "gap_s", "first_input_event_type",
    ]
    table = pq.read_table(args.parquet, columns=cols)
    if args.provider:
        import pyarrow.compute as pc
        table = table.filter(pc.equal(table["provider"], args.provider))
    d = table.to_pydict()

    sessions = defaultdict(list)
    for sid, step, inp, new, out, gap, kind in zip(
        d["session_id"], d["step_idx"], d["input_total"], d["ideal_new"],
        d["output"], d["gap_s"], d["first_input_event_type"],
    ):
        sessions[sid].append((step, inp, new, out, gap, kind))

    n_neg = 0
    n_written = 0
    with open(args.out, "w") as f:
        for sid in sorted(sessions):
            rows = sorted(sessions[sid])
            if len(rows) < args.min_steps:
                continue
            steps = []
            for i, (_, inp, new, out, gap, kind) in enumerate(rows):
                if gap is None or (isinstance(gap, float) and math.isnan(gap)) or i == 0:
                    g = 0.0
                else:
                    if gap < 0:
                        n_neg += 1
                    g = max(float(gap), 0.0)
                new_i = int(round(new)) if new is not None else int(inp)
                steps.append({
                    "input": int(inp),
                    "new": max(0, min(new_i, int(inp))),
                    "output": int(out),
                    "gap": round(g, 3),
                    "kind": KIND.get(kind),
                })
            f.write(json.dumps({"id": sid, "steps": steps}, separators=(",", ":")) + "\n")
            n_written += 1

    print(f"wrote {n_written} sessions to {args.out} ({n_neg} negative gaps clamped to 0)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
