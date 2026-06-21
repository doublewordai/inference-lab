"""CaptureBuffer: accumulate per-round captures from the (separate) verify and
draft hooks, keyed by (request id, round index), then emit acceptance/speculator
rows once a run finishes.

Acceptance data (committed count + per-position mask) comes from the *verify* hook;
confidence (per-position drafter softmax prob) comes from the *draft* hook. They
arrive independently and are merged here on the (rid, round_idx) key. Engine
agnostic and fully unit-testable without SGLang.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


class FileSink:
    """Cross-process capture sink. SGLang runs the spec worker in a spawned child
    process, so hooks installed there (via the plugin) cannot reach the parent's
    in-memory buffer. Instead they append JSONL rows to a shared file path; the
    parent reads it back after generation. Duck-types CaptureBuffer's add_* methods
    so the hook code is identical for both."""

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "a", buffering=1)
        self._lock = threading.Lock()

    def _write(self, obj: dict) -> None:
        import json

        with self._lock:
            self._f.write(json.dumps(obj) + "\n")
            self._f.flush()

    def add_accept(self, rid, round_idx, accept, accept_mask) -> None:
        self._write({
            "k": "a", "rid": str(rid), "r": int(round_idx),
            "accept": int(accept), "mask": [int(x) for x in accept_mask],
        })

    def add_conf(self, rid, round_idx, conf) -> None:
        self._write({
            "k": "c", "rid": str(rid), "r": int(round_idx),
            "conf": [None if x is None else float(x) for x in conf],
        })


@dataclass
class _Round:
    accept: int | None = None
    accept_mask: list[int] | None = None  # per-position 1/0
    conf: list[float] | None = None  # per-position drafter prob


@dataclass
class CaptureBuffer:
    width: int = 0
    _rounds: dict[tuple[str, int], _Round] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    # rid -> human-facing metadata (config, category, prompt_idx) set by the driver
    _meta: dict[str, dict] = field(default_factory=dict)

    def register(self, rid: str, *, config: str, category: str, prompt_idx: int) -> None:
        """Associate an engine request id with its prompt metadata before running."""
        with self._lock:
            self._meta[rid] = {"config": config, "category": category, "prompt_idx": prompt_idx}

    def bind_unregistered(self, *, config: str, category: str, prompt_idx: int) -> int:
        """Attribute every captured-but-unbound rid to this prompt. Engine request
        ids aren't known ahead of time; with batch=1 sequential generation, any rid
        seen during a prompt's generate() call belongs to that prompt. Returns the
        number of rids newly bound."""
        n = 0
        with self._lock:
            seen = {rid for (rid, _round) in self._rounds}
            for rid in seen:
                if rid not in self._meta:
                    self._meta[rid] = {
                        "config": config,
                        "category": category,
                        "prompt_idx": prompt_idx,
                    }
                    n += 1
        return n

    def add_accept(self, rid: str, round_idx: int, accept: int, accept_mask: list[int]) -> None:
        with self._lock:
            r = self._rounds.setdefault((rid, round_idx), _Round())
            r.accept = int(accept)
            r.accept_mask = [int(x) for x in accept_mask]

    def add_conf(self, rid: str, round_idx: int, conf: list[float]) -> None:
        with self._lock:
            r = self._rounds.setdefault((rid, round_idx), _Round())
            r.conf = [float(x) for x in conf]

    def _pad(self, seq: list | None, fill) -> list:
        seq = list(seq or [])
        if len(seq) < self.width:
            seq = seq + [fill] * (self.width - len(seq))
        return seq[: self.width]

    def _common(self, rid: str, round_idx: int, model: str, speculator: str) -> dict:
        meta = self._meta.get(rid, {"config": "?", "category": "?", "prompt_idx": -1})
        return {
            "model": model,
            "speculator": speculator,
            "config": meta["config"],
            "category": meta["category"],
            "prompt_idx": meta["prompt_idx"],
            "round_idx": round_idx,
        }

    def acceptance_rows(self, model: str, speculator: str) -> list[dict]:
        rows = []
        with self._lock:
            for (rid, ridx), r in sorted(self._rounds.items()):
                if r.accept is None:
                    continue
                row = self._common(rid, ridx, model, speculator)
                row["accept"] = r.accept
                mask = self._pad(r.accept_mask, None)
                for k in range(self.width):
                    row[f"acc{k}"] = mask[k]
                rows.append(row)
        return rows

    def speculator_rows(self, model: str, speculator: str) -> list[dict]:
        rows = []
        with self._lock:
            for (rid, ridx), r in sorted(self._rounds.items()):
                if r.conf is None:
                    continue
                row = self._common(rid, ridx, model, speculator)
                conf = self._pad(r.conf, None)
                for k in range(self.width):
                    row[f"conf{k}"] = conf[k]
                rows.append(row)
        return rows

    def clear(self) -> None:
        with self._lock:
            self._rounds.clear()
            self._meta.clear()

    def __len__(self) -> int:
        return len(self._rounds)
