"""Regenerate the speculating-on-the-margin acceptance chart data, grouped by dataset.

Sources: the canonical published banks (mirrored on HF as
Doubleword/specdec-calibration):
  data/qwen3.6-35b-a3b/speedbench/{mtp, dflash@42d3b34d}/acceptance
  data/qwen3.6-35b-a3b/humaneval/{mtp, dflash@42d3b34d}/acceptance

Emits one JSON: { mtp: {width, datasets: {key: {nRounds, hist, heat}}}, dflash: {...} }
  hist[k]  = % of rounds committing exactly k draft tokens (k = 0..width)
  heat[a][p] = % of rounds with actual accept a and predicted accept p,
               predicted = round(sum_d prod_{k<=d} conf_k)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parents[2] / "data/qwen3.6-35b-a3b"
KEYS = ["category", "prompt_idx", "turn", "round_idx"]
N_CONF = 16  # conf0..conf15 in both banks

DRAFTERS = {"mtp": "mtp", "dflash": "dflash@42d3b34d"}
CATEGORIES = [
    "coding", "writing", "qa", "rag", "math", "reasoning",
    "stem", "humanities", "summarization", "multilingual", "roleplay",
]


def load(corpus: str, drafter_dir: str):
    base = DATA / corpus / drafter_dir / "acceptance"
    acc = pq.read_table(base / "acceptance.parquet").to_pandas()
    spec = pq.read_table(base / "speculator.parquet").to_pandas()
    conf_cols = [c for c in spec.columns if c.startswith("conf") and c[4:].isdigit()]
    conf_cols.sort(key=lambda c: int(c[4:]))
    df = acc.merge(spec[KEYS + conf_cols], on=KEYS, validate="1:1")
    df = df[df["accept"].notna()].copy()
    df["accept"] = df["accept"].astype(int)
    confs = np.nan_to_num(df[conf_cols].to_numpy(dtype=float), nan=0.0)
    df["pred"] = np.cumprod(confs, axis=1).sum(axis=1)
    return df


def entry(df, width: int) -> dict:
    n = len(df)
    counts = df["accept"].value_counts()
    hist = [round(100.0 * counts.get(k, 0) / n, 2) for k in range(width + 1)]
    pred = df["pred"].round().astype(int).clip(0, width).to_numpy()
    m = np.zeros((width + 1, width + 1))
    np.add.at(m, (df["accept"].to_numpy(), pred), 1)
    heat = (100.0 * m / n).round(3).tolist()
    corr = float(np.corrcoef(df["accept"], df["pred"])[0, 1])
    return {"nRounds": n, "hist": hist, "heat": heat, "corr": round(corr, 3)}


def main():
    out = {}
    for name, drafter_dir in DRAFTERS.items():
        sb = load("speedbench", drafter_dir)
        he = load("humaneval", drafter_dir)
        width = int(max(sb["accept"].max(), he["accept"].max()))
        datasets = {"speedbench/all": entry(sb, width), "humaneval": entry(he, width)}
        for cat in CATEGORIES:
            g = sb[sb["category"] == cat]
            datasets[f"speedbench/{cat}"] = entry(g, width)
        out[name] = {"width": width, "datasets": datasets}
        print(f"{name} ({drafter_dir}): width={width}, "
              f"speedbench={datasets['speedbench/all']['nRounds']} rounds "
              f"(corr {datasets['speedbench/all']['corr']}), "
              f"humaneval={datasets['humaneval']['nRounds']} rounds")
        print(f"  all hist: {[round(v,1) for v in datasets['speedbench/all']['hist']]}")

    dest = Path(sys.argv[1])
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out))
    print(f"wrote {dest} ({dest.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
