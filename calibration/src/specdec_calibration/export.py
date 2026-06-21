"""Export calibration parquet banks to the simulator's TraceRounds CSV format.

The Rust simulator intentionally consumes a small, stable CSV:

    commits,category,a0,a1,...,a{D-1}

where `commits` is the realised number of accepted draft tokens and `a*` is the
policy signal. This module adapts the richer calibration parquet outputs into
that format without making the simulator depend on Arrow/Parquet.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pyarrow.parquet as pq

from .schema import KEY_COLUMNS


class TraceSignal(str, Enum):
    CONFIDENCE = "confidence"
    ORACLE = "oracle"


@dataclass
class ExportManifest:
    output: str
    signal: str
    rows: int
    dropped_rows: int
    depth: int
    key_columns: list[str]
    acceptance_path: str | None = None
    speculator_path: str | None = None
    raw_path: str | None = None


def export_trace_bank(
    *,
    output_path: str | Path,
    signal: TraceSignal | str = TraceSignal.CONFIDENCE,
    acceptance_path: str | Path | None = None,
    speculator_path: str | Path | None = None,
    raw_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    drop_trailing_null_signal: bool = True,
    metadata_path: str | Path | None = None,
) -> ExportManifest:
    """Write a simulator TraceRounds CSV from calibration parquet outputs.

    Inputs can be either:
      * `run_dir` containing `acceptance.parquet` and optionally `speculator.parquet`;
      * explicit `acceptance_path` / `speculator_path`;
      * an older single `raw_path` containing `accept` plus `conf*` columns.
    """
    signal = TraceSignal(signal)
    output_path = Path(output_path)

    if run_dir is not None:
        run = Path(run_dir)
        acceptance_path = acceptance_path or run / "acceptance.parquet"
        speculator_path = speculator_path or run / "speculator.parquet"

    df, key_columns, sources = _load_inputs(
        acceptance_path=acceptance_path,
        speculator_path=speculator_path,
        raw_path=raw_path,
    )
    signal_cols = _signal_columns(df, signal, drop_trailing_null_signal)
    if not signal_cols:
        raise ValueError(f"no {signal.value} signal columns found")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["commits", "category"] + [f"a{k}" for k in range(len(signal_cols))])
        for _, row in df.iterrows():
            accept = row.get("accept")
            if _is_null(accept):
                dropped += 1
                continue
            category = row.get("category")
            if _is_null(category):
                category = "all"
            writer.writerow(
                [int(accept), str(category)]
                + [_signal_value(row, col, signal, depth=k) for k, col in enumerate(signal_cols)]
            )
            kept += 1

    manifest = ExportManifest(
        output=str(output_path),
        signal=signal.value,
        rows=kept,
        dropped_rows=dropped,
        depth=len(signal_cols),
        key_columns=key_columns,
        acceptance_path=sources.get("acceptance_path"),
        speculator_path=sources.get("speculator_path"),
        raw_path=sources.get("raw_path"),
    )
    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(manifest.__dict__, indent=2))
    return manifest


def _load_inputs(
    *,
    acceptance_path: str | Path | None,
    speculator_path: str | Path | None,
    raw_path: str | Path | None,
):
    if raw_path is not None:
        df = _read_parquet(raw_path)
        _require_columns(df, ["accept"])
        return df, _present_keys(df), {"raw_path": str(raw_path)}

    if acceptance_path is None:
        raise ValueError("provide --run-dir, --raw, or --acceptance")

    acc = _read_parquet(acceptance_path)
    _require_columns(acc, ["accept"])
    sources = {"acceptance_path": str(acceptance_path)}

    if speculator_path is None or not Path(speculator_path).exists():
        return acc, _present_keys(acc), sources

    spc = _read_parquet(speculator_path)
    sources["speculator_path"] = str(speculator_path)
    keys = [c for c in KEY_COLUMNS if c in acc.columns and c in spc.columns]
    if not keys:
        raise ValueError("acceptance/speculator banks have no shared key columns")
    df = acc.merge(spc, on=keys, how="left", validate="one_to_one")
    return df, keys, sources


def _read_parquet(path: str | Path):
    return pq.read_table(path).to_pandas()


def _require_columns(df, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _present_keys(df) -> list[str]:
    keys = [c for c in KEY_COLUMNS if c in df.columns]
    if "i" in df.columns and "round_idx" not in keys:
        keys.append("i")
    return keys


def _indexed_cols(df, prefix: str) -> list[str]:
    return sorted(
        [c for c in df.columns if c.startswith(prefix) and c[len(prefix) :].isdigit()],
        key=lambda c: int(c[len(prefix) :]),
    )


def _signal_columns(df, signal: TraceSignal, drop_trailing_null_signal: bool) -> list[str]:
    if signal is TraceSignal.CONFIDENCE:
        cols = _indexed_cols(df, "conf")
    else:
        cols = _indexed_cols(df, "acc")
        if not cols:
            # Older raw banks only have `accept` + `conf*`; use the confidence
            # depth to build an oracle accept pattern.
            cols = _indexed_cols(df, "conf")
    if drop_trailing_null_signal:
        while cols and df[cols[-1]].isna().all():
            cols.pop()
    return cols


def _signal_value(row, col: str, signal: TraceSignal, *, depth: int) -> float:
    if signal is TraceSignal.ORACLE and col.startswith("conf"):
        return 1.0 if int(row["accept"]) > depth else 0.0
    value = row.get(col)
    if _is_null(value):
        return 0.0
    return min(1.0, max(0.0, float(value)))


def _is_null(value) -> bool:
    import pandas as pd

    return bool(pd.isna(value))
