"""Arrow schemas + writers for the two parquet banks and the derived stats.json.

Both banks share the same key columns and one row per draft round. `width` is the
number of draft-token positions emitted by the hook family; shallower speculators
NaN/null-pad trailing columns when a wider union schema is requested.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

KEY_COLUMNS = ["model", "speculator", "config", "category", "prompt_idx", "turn", "round_idx"]


def _key_fields() -> list[pa.Field]:
    return [
        pa.field("model", pa.string()),
        pa.field("speculator", pa.string()),
        pa.field("config", pa.string()),
        pa.field("category", pa.string()),
        pa.field("prompt_idx", pa.int64()),
        pa.field("turn", pa.int16()),  # conversation turn (0 for single-turn)
        pa.field("round_idx", pa.int64()),
    ]


def acceptance_schema(width: int) -> pa.Schema:
    """Verify side: committed count + per-position accept mask."""
    fields = _key_fields()
    fields.append(pa.field("accept", pa.int64()))  # committed draft tokens (excl. bonus)
    fields += [pa.field(f"acc{k}", pa.int8()) for k in range(width)]  # 1/0/null
    return pa.schema(fields)


def speculator_schema(width: int) -> pa.Schema:
    """Draft side: the drafter's confidence (softmax prob) per proposed token."""
    fields = _key_fields()
    fields += [pa.field(f"conf{k}", pa.float64()) for k in range(width)]
    return pa.schema(fields)


def _write(rows: list[dict], schema: pa.Schema, path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {f.name: [r.get(f.name) for r in rows] for f in schema}
    table = pa.table(columns, schema=schema)
    pq.write_table(table, path, compression="zstd")
    return len(rows)


def write_acceptance(rows: list[dict], width: int, path: str | Path) -> int:
    return _write(rows, acceptance_schema(width), path)


def write_speculator(rows: list[dict], width: int, path: str | Path) -> int:
    return _write(rows, speculator_schema(width), path)


def routing_meta_schema() -> pa.Schema:
    return pa.schema(
        _key_fields()
        + [pa.field("position", pa.int16()), pa.field("accepted", pa.int8())]
    )


def write_routing_meta(rows: list[dict], path: str | Path) -> int:
    return _write(rows, routing_meta_schema(), path)


def write_stats(accept_rows: list[dict], speculator_label: str, width: int, path: str | Path) -> dict:
    """Derived aggregates, same shape as the old `*_stats.json`."""
    commits = [r["accept"] for r in accept_rows if r.get("accept") is not None]
    total = len(commits)
    accept_hist = dict(sorted(Counter(commits).items()))
    # accept_by_position[k] = P(at least k+1 tokens committed)
    accept_by_position = [
        (sum(1 for c in commits if c > k) / total) if total else 0.0 for k in range(width)
    ]

    def _grouped(field: str) -> dict:
        g: dict[str, list[int]] = defaultdict(list)
        for r in accept_rows:
            if r.get("accept") is not None:
                g[r[field]].append(r["accept"])
        return {k: {"rounds": len(v), "mean_commits": sum(v) / len(v)} for k, v in g.items()}

    stats = {
        "speculator": speculator_label,
        "width": width,
        "total_rounds": total,
        "accept_hist": {str(k): v for k, v in accept_hist.items()},
        "mean_commits_overall": (sum(commits) / total) if total else 0.0,
        "accept_by_position": accept_by_position,
        "by_config": _grouped("config"),
        "by_category": _grouped("category"),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(stats, indent=2))
    return stats
