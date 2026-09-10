#!/usr/bin/env python3
"""Closed-loop OpenAI-compatible load generator, for control-layer's flex
live-relay Redis stress test (COR-648).

Adapted from replay_client.py's worker-pool/percentile skeleton — same
closed-loop shape, same incremental progress reporting, same JSONL output —
but speaking real OpenAI `/v1/chat/completions` SSE against dwctl instead of
SGLang's native `/generate` protocol, and issuing flat synthetic requests
instead of replaying recorded session traces (this benchmark cares about raw
concurrency through the relay, not realistic session/cache-hit-rate shapes).

    python3 examples/sessions/replay_openai.py \
        --base-url http://localhost:3001/ai/v1 --model flex-bench-alias \
        --api-key sk-... --concurrency 100 --duration 60 --out results.jsonl

Per request: TTFT (time to the first content delta), the timestamp of every
subsequent delta (-> inter-token gaps), total duration to [DONE], and
delivered-frame-count vs. the final chunk's usage.completion_tokens (a direct
signal for the chunk-relay's publish channel silently dropping chunks under
load - see chunk_relay.rs's publish_channel_capacity).

Output: one JSON line per completed request to --out (if given), progress
every 25 completions to stdout, and a final percentile summary (TTFT, E2E,
inter-token gap) as JSON on stdout.
"""

import argparse
import asyncio
import json
import math
import sys
import time

import httpx


def percentiles(xs, pcts=(50, 90, 95, 99)):
    if not xs:
        return {f"p{p}": None for p in pcts} | {"mean": None, "n": 0}
    xs = sorted(xs)
    out = {}
    for p in pcts:
        idx = min(len(xs) - 1, max(0, math.ceil(p / 100 * len(xs)) - 1))
        out[f"p{p}"] = round(xs[idx], 4)
    out["mean"] = round(sum(xs) / len(xs), 4)
    out["n"] = len(xs)
    return out


async def one_request(client, args, req_id):
    t0 = time.monotonic()
    ttft = None
    token_times = []
    reported_completion_tokens = None

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": True,
        "service_tier": "flex",
        "stream_options": {"include_usage": True},
    }
    try:
        async with client.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread())[:300]
                return {"req_id": req_id, "error": f"HTTP {resp.status_code}: {body.decode(errors='replace')}"}
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue  # chunks can split across lines under load
                usage = chunk.get("usage")
                if usage:
                    reported_completion_tokens = usage.get("completion_tokens")
                choices = chunk.get("choices") or []
                if choices and (choices[0].get("delta") or {}).get("content"):
                    now = time.monotonic()
                    if ttft is None:
                        ttft = now - t0
                    token_times.append(now)
    except (httpx.HTTPError, asyncio.TimeoutError) as e:
        return {"req_id": req_id, "error": f"{type(e).__name__}: {e}"}

    e2e = time.monotonic() - t0
    gaps = [round(b - a, 4) for a, b in zip(token_times, token_times[1:])]
    return {
        "req_id": req_id,
        "t_issue": round(t0, 3),
        "ttft_s": round(ttft, 4) if ttft is not None else None,
        "e2e_s": round(e2e, 4),
        "delivered_tokens": len(token_times),
        "reported_completion_tokens": reported_completion_tokens,
        "inter_token_gaps_s": gaps,
    }


class IncrementalResults(list):
    def __init__(self, out_path):
        super().__init__()
        self._f = open(out_path, "w", buffering=1) if out_path else None
        self._t0 = time.monotonic()

    def append(self, r):
        super().append(r)
        if self._f:
            self._f.write(json.dumps(r) + "\n")
        n = len(self)
        if n % 25 != 0:
            return
        ok = [x for x in self if "error" not in x]
        errs = n - len(ok)
        tf = sorted(x["ttft_s"] for x in ok if x.get("ttft_s") is not None)
        gaps = sorted(g for x in ok for g in x.get("inter_token_gaps_s", []))
        el = time.monotonic() - self._t0
        msg = f"[progress] n={n} ok={len(ok)} err={errs} elapsed={el:.0f}s"
        if tf:
            msg += f" ttft_p50={tf[len(tf) // 2]:.3f}s"
        if gaps:
            msg += f" gap_p50={gaps[len(gaps) // 2] * 1000:.1f}ms"
        print(msg, flush=True)


async def run(args):
    results = IncrementalResults(args.out)
    deadline = time.monotonic() + args.duration if args.duration else None
    issued = 0

    headers = {"Authorization": f"Bearer {args.api_key}"}
    async with httpx.AsyncClient(
        base_url=args.base_url,
        headers=headers,
        timeout=httpx.Timeout(args.timeout, connect=30),
        limits=httpx.Limits(max_connections=args.concurrency + 4),
    ) as client:

        async def worker(start_delay):
            nonlocal issued
            if start_delay:
                await asyncio.sleep(start_delay)
            while True:
                if deadline and time.monotonic() > deadline:
                    return
                if args.num_requests and issued >= args.num_requests:
                    return
                issued += 1
                rid = issued
                r = await one_request(client, args, rid)
                results.append(r)

        t_bench0 = time.monotonic()
        delays = [(i / args.concurrency) * args.ramp_seconds for i in range(args.concurrency)] if args.ramp_seconds else [0] * args.concurrency
        await asyncio.gather(*(worker(d) for d in delays))
    wall = time.monotonic() - t_bench0

    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    all_gaps = [g for r in ok for g in r.get("inter_token_gaps_s", [])]
    total_delivered = sum(r["delivered_tokens"] for r in ok)
    mismatches = [
        {"req_id": r["req_id"], "delivered": r["delivered_tokens"], "reported": r["reported_completion_tokens"]}
        for r in ok
        if r.get("reported_completion_tokens") is not None and r["delivered_tokens"] != r["reported_completion_tokens"]
    ]
    summary = {
        "concurrency": args.concurrency,
        "requests_ok": len(ok),
        "errors": len(errs),
        "wall_s": round(wall, 1),
        "delivered_tok_per_s": round(total_delivered / wall, 1) if wall else None,
        "ttft_s": percentiles([r["ttft_s"] for r in ok if r["ttft_s"] is not None]),
        "e2e_s": percentiles([r["e2e_s"] for r in ok]),
        "inter_token_gap_s": percentiles(all_gaps),
        # Non-empty here is a direct sign of the chunk-relay's global publish
        # channel dropping chunks under load (see chunk_relay.rs).
        "delivered_vs_reported_mismatches": len(mismatches),
        "mismatch_sample": mismatches[:10],
    }
    print(json.dumps(summary, indent=1))
    if errs:
        print(f"first error: {errs[0]}", file=sys.stderr)
    return 0 if not errs else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True, help="e.g. http://localhost:3001/ai/v1")
    ap.add_argument("--model", required=True, help="the deployment alias, e.g. flex-bench-alias")
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--concurrency", type=int, default=10, help="closed-loop concurrent slots")
    ap.add_argument(
        "--ramp-seconds",
        type=float,
        default=0,
        help="spread worker start times evenly over N seconds instead of connecting all at once "
        "(macOS somaxconn defaults to 128, so a simultaneous burst above that gets refused/timed out "
        "at the kernel level regardless of server or Redis behavior)",
    )
    ap.add_argument("--duration", type=float, help="stop issuing after N seconds")
    ap.add_argument("--num-requests", type=int, help="stop after roughly N total requests issued")
    ap.add_argument("--prompt", default="Tell me something interesting.", help="fixed user message content")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", help="per-request JSONL path")
    args = ap.parse_args()
    if not args.duration and not args.num_requests:
        ap.error("one of --duration or --num-requests is required")
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
