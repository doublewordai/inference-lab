#!/usr/bin/env python3
"""Export local Claude Code transcripts to the simulator's session JSONL.

    python3 examples/sessions/claude_code_export.py ~/.claude/projects \
        --out data/sessions/claude-code.jsonl [--days 30] [--min-steps 2] \
        [--arrivals arrivals.json] [--characterize characterization.json]

One session per line, same schema as tracelab.jsonl (consumed by the
simulator's session replay and by the live replay client):

    {"id", "steps": [{"input", "new", "output", "gap", "kind"}]}

The exporter reads only `message.usage` numbers, timestamps, entry types, and
content-block *types* — never transcript text. Token content is synthesized at
replay time.

Field mapping (per API request, i.e. per unique assistant `message.id` —
streaming writes one transcript entry per content block, all carrying the same
id and usage, so groups are deduplicated by id):

  input  = cache_read + cache_creation + input_tokens   prompt tokens the model saw
  new    = cache_creation + input_tokens                uncached prompt tokens
  output = output_tokens                                generated, reasoning included
  gap    = trigger_ts - previous request's t_end, >= 0  seconds; first step 0.
           trigger_ts is the timestamp of the request's nearest ancestor user
           entry (the tool_result or typed message that made it ready to send),
           so the gap is client think time — tool latency plus human wait —
           and excludes the request's own server time. t_end is the last
           transcript timestamp of the id group (~ end of streaming). When no
           trigger entry is found the request's own first timestamp is used,
           which then includes its TTFT + first content block.
  kind   = user | tool | null       nearest ancestor user entry: tool_result
                                    content -> tool, text content -> user

Streams: the main chain of one transcript file is one session. Sidechain
(subagent) entries are split into their own sessions, keyed by the sidechain
root uuid, since each is an independent API request stream.

Arrivals sidecar (--arrivals): the session JSONL schema has no absolute time
(the simulator's SessionSpec is deny_unknown_fields), so per-session start
epochs go in a sidecar JSON: {"t0": {session_id: epoch_seconds}, "stats": ...}.

Requests with no usage (errors, synthetic messages) are dropped. Sessions are
filtered by --days on the *session's first request*.
"""

import argparse
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone


def parse_ts(s):
    # 2026-08-23T11:44:45.516Z
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def content_kind(message):
    content = message.get("content")
    if isinstance(content, str):
        return "user"
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return "tool"
        return "user"
    return None


def extract_file(path):
    """Return list of request dicts: {t_start, t_end, input, new, output, kind, stream}."""
    # uuid -> (type, parentUuid, kind_if_user, sidechain, ts). Every entry with
    # a uuid is recorded so parent-chain walks survive attachment/system links.
    nodes = {}
    groups = {}  # message.id -> dict
    order = []
    for line in open(path, errors="ignore"):
        if '"uuid"' not in line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = d.get("type")
        uuid = d.get("uuid")
        if not uuid:
            continue
        ts_raw = d.get("timestamp")
        ts = parse_ts(ts_raw) if ts_raw else None
        kind = content_kind(d.get("message") or {}) if t == "user" else None
        nodes[uuid] = (t, d.get("parentUuid"), kind, bool(d.get("isSidechain")), ts)
        if t == "assistant":
            msg = d.get("message") or {}
            usage = msg.get("usage")
            mid = msg.get("id")
            if not mid or not usage:
                continue
            if msg.get("model") == "<synthetic>":
                continue
            if ts is None:
                continue
            g = groups.get(mid)
            if g is None:
                groups[mid] = g = {
                    "t_start": ts, "t_end": ts,
                    "parent": d.get("parentUuid"), "uuid": uuid,
                    "sidechain": bool(d.get("isSidechain")),
                    "usage": usage,
                }
                order.append(mid)
            else:
                g["t_start"] = min(g["t_start"], ts)
                g["t_end"] = max(g["t_end"], ts)

    def walk_to_user(uuid):
        """Nearest ancestor user entry -> (kind, ts) or (None, None)."""
        seen = set()
        while uuid and uuid not in seen and uuid in nodes:
            seen.add(uuid)
            typ, parent, kind, _, ts = nodes[uuid]
            if typ == "user":
                return kind, ts
            uuid = parent
        return None, None

    def sidechain_root(uuid):
        seen = set()
        last = uuid
        while uuid and uuid not in seen and uuid in nodes:
            seen.add(uuid)
            last = uuid
            uuid = nodes[uuid][1]
        return last

    requests = []
    for mid in order:
        g = groups[mid]
        u = g["usage"]
        cache_read = u.get("cache_read_input_tokens") or 0
        cache_creation = u.get("cache_creation_input_tokens") or 0
        raw_in = u.get("input_tokens") or 0
        out = u.get("output_tokens") or 0
        stream = "main"
        if g["sidechain"]:
            stream = "sc:" + str(sidechain_root(g["parent"] or g["uuid"]))
        kind, trigger_ts = walk_to_user(g["parent"])
        requests.append({
            "t_start": g["t_start"], "t_end": g["t_end"],
            "trigger_ts": trigger_ts if trigger_ts is not None else g["t_start"],
            "input": cache_read + cache_creation + raw_in,
            "new": cache_creation + raw_in,
            "output": out,
            "kind": kind,
            "stream": stream,
        })
    return requests


def percentiles(xs, pcts=(50, 90, 99)):
    if not xs:
        return {f"p{p}": None for p in pcts} | {"mean": None, "n": 0}
    xs = sorted(xs)
    out = {}
    for p in pcts:
        idx = min(len(xs) - 1, max(0, math.ceil(p / 100 * len(xs)) - 1))
        out[f"p{p}"] = xs[idx]
    out["mean"] = sum(xs) / len(xs)
    out["n"] = len(xs)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=os.path.expanduser("~/.claude/projects"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=float, help="keep sessions starting in the last N days")
    ap.add_argument("--min-steps", type=int, default=2)
    ap.add_argument("--arrivals", help="write per-session start epochs + arrival stats")
    ap.add_argument("--characterize", help="write distribution summary JSON")
    args = ap.parse_args()

    cutoff = time.time() - args.days * 86400 if args.days else None
    files = sorted(glob.glob(os.path.join(args.root, "*", "*.jsonl")))
    if not files:
        print(f"no transcripts under {args.root}", file=sys.stderr)
        return 1

    sessions = []  # (id, t0, [request...])
    for path in files:
        reqs = extract_file(path)
        streams = defaultdict(list)
        for r in reqs:
            streams[r["stream"]].append(r)
        base = "cc:" + os.path.splitext(os.path.basename(path))[0]
        for stream, rs in streams.items():
            rs.sort(key=lambda r: r["t_start"])
            sid = base if stream == "main" else f"{base}:{stream}"
            sessions.append((sid, rs[0]["t_start"], rs))

    sessions.sort(key=lambda s: s[1])
    n_dropped_cutoff = n_dropped_short = 0
    kept = []
    for sid, t0, rs in sessions:
        if cutoff and t0 < cutoff:
            n_dropped_cutoff += 1
            continue
        if len(rs) < args.min_steps:
            n_dropped_short += 1
            continue
        kept.append((sid, t0, rs))

    with open(args.out, "w") as f:
        for sid, t0, rs in kept:
            steps = []
            prev_end = None
            for r in rs:
                gap = 0.0 if prev_end is None else max(r["trigger_ts"] - prev_end, 0.0)
                prev_end = r["t_end"]
                steps.append({
                    "input": r["input"],
                    "new": max(0, min(r["new"], r["input"])),
                    "output": r["output"],
                    "gap": round(gap, 3),
                    "kind": r["kind"],
                })
            f.write(json.dumps({"id": sid, "steps": steps},
                               separators=(",", ":")) + "\n")

    all_reqs = [r for _, _, rs in kept for r in rs]
    total_in = sum(r["input"] for r in all_reqs)
    total_new = sum(r["new"] for r in all_reqs)
    total_out = sum(r["output"] for r in all_reqs)
    print(f"wrote {len(kept)} sessions / {len(all_reqs)} requests to {args.out} "
          f"(dropped {n_dropped_cutoff} outside window, {n_dropped_short} short)",
          file=sys.stderr)
    if total_out:
        print(f"r (uncached in:out) = {total_new/total_out:.2f}  "
              f"h (cached share of input) = {1 - total_new/total_in:.4f}  "
              f"raw in:out = {total_in/total_out:.1f}", file=sys.stderr)

    if args.arrivals:
        t0s = {sid: t0 for sid, t0, _ in kept}
        starts = sorted(t0s.values())
        inter = [b - a for a, b in zip(starts, starts[1:])]
        span_days = (starts[-1] - starts[0]) / 86400 if len(starts) > 1 else 0
        with open(args.arrivals, "w") as f:
            json.dump({
                "t0": t0s,
                "stats": {
                    "sessions": len(starts),
                    "span_days": round(span_days, 2),
                    "sessions_per_day": round(len(starts) / span_days, 2) if span_days else None,
                    "interarrival_s": percentiles(inter),
                },
            }, f, indent=1)

    if args.characterize:
        gaps, ctx, news, outs, steps_per = [], [], [], [], []
        by_kind_new = defaultdict(list)
        by_kind_gap = defaultdict(list)
        for _, _, rs in kept:
            steps_per.append(len(rs))
            prev_end = None
            for r in rs:
                kind = r["kind"] or "unknown"
                if prev_end is not None:
                    g = max(r["trigger_ts"] - prev_end, 0.0)
                    gaps.append(g)
                    by_kind_gap[kind].append(g)
                prev_end = r["t_end"]
                ctx.append(r["input"])
                news.append(r["new"])
                outs.append(r["output"])
                by_kind_new[kind].append(r["new"])
        with open(args.characterize, "w") as f:
            json.dump({
                "source": args.root,
                "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "days_filter": args.days,
                "sessions": len(kept),
                "requests": len(all_reqs),
                "aggregate": {
                    "r_uncached_in_per_out": round(total_new / total_out, 3),
                    "cache_hit_share": round(1 - total_new / total_in, 4),
                    "raw_in_per_out": round(total_in / total_out, 1),
                },
                "per_request": {
                    "context_tokens": percentiles(ctx),
                    "new_tokens_context_growth": percentiles(news),
                    "output_tokens": percentiles(outs),
                    "new_tokens_by_kind": {k: percentiles(v) for k, v in sorted(by_kind_new.items())},
                    "think_gap_s": percentiles(gaps),
                    "think_gap_s_by_kind": {k: percentiles(v) for k, v in sorted(by_kind_gap.items())},
                },
                "per_session": {
                    "requests_per_session": percentiles(steps_per),
                },
            }, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
