#!/usr/bin/env python3
"""Re-export the speculative-decoding acceptance banks with the *draft-time*
confidence signal in the a-columns, so the gating policies use an estimate the
drafter could actually compute before verification, not the realized outcome.

The shipped `*_speedbench_rounds.csv` banks put the binary accept pattern in the
a-columns (`commits=3 -> 1,1,1,0,0,...`). That is the oracle: gating on it has
perfect foresight. The raw parquet keeps the real signal, `conf0..confN`, the
drafter's per-depth confidence. This writes `*_conf_rounds.csv` with
`a_k = conf_k` and `commits = accept`, matching `TraceBank::load`'s
`commits,category,a0..a{D-1}` header. A null confidence at depth k means the
drafter emitted no token there, so `a_k = 0` (the gate won't draft that deep);
this matches the oracle bank's always-zero boundary column. Only rows with a null
`accept` are dropped.
"""
import csv
import pyarrow.parquet as pq

BANKS = [
    ("qwen36_nextn_speedbench_rounds_raw.parquet", "qwen36_nextn_conf_rounds.csv", 8),
    ("qwen36_dflash_speedbench_rounds_raw.parquet", "qwen36_dflash_conf_rounds.csv", 16),
]
DATA = "/home/fergus/inference-lab/data"


def main():
    for src, dst, depth in BANKS:
        t = pq.read_table(f"{DATA}/{src}")
        cols = {c: t.column(c).to_pylist() for c in t.column_names}
        n = len(cols["accept"])
        conf_cols = [f"conf{k}" for k in range(depth)]
        header = ["commits", "category"] + [f"a{k}" for k in range(depth)]
        kept, dropped = 0, 0
        with open(f"{DATA}/{dst}", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in range(n):
                if cols["accept"][r] is None:
                    dropped += 1
                    continue
                # null conf at depth k -> 0 (no draft token there); clamp to [0,1].
                confs = [
                    0.0 if cols[c][r] is None else min(1.0, max(0.0, float(cols[c][r])))
                    for c in conf_cols
                ]
                w.writerow([int(cols["accept"][r]), cols["category"][r]] + confs)
                kept += 1
        print(f"{dst}: wrote {kept} rounds (dropped {dropped} null-conf), depth {depth}")


if __name__ == "__main__":
    main()
