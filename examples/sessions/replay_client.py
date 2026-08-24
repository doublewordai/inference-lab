#!/usr/bin/env python3
"""Live OpenAI-API session replay: Claude Code workload against a real server.

    python3 examples/sessions/replay_client.py sessions.jsonl \
        --base-url http://HOST:PORT/v1 --model MODEL \
        --concurrency 8 [--duration 600] [--gap-scale 1.0] \
        [--window 131072] [--clip compact|cap] [--post-compact 15000] \
        [--tokenizer /path/to/tokenizer.json [--corpus text.txt]] \
        [--out requests.jsonl]

Replays sessions from the exporter's JSONL (claude_code_export.py /
tracelab_export.py schema) as *shapes*: per request the client reproduces the
recorded prompt length, cached/uncached split, output length, and preceding
think-time gap. Content is synthesized — no transcript text exists anywhere
in this pipeline.

Mechanics
---------
Prompts are token-id arrays on /v1/completions (exact lengths, exact prefix
control; no tokenizer needed). Output length is forced with max_tokens +
ignore_eos. Per step i the prompt is: the previous session prompt+output
truncated/extended to (input_i - new_i) ids — the recorded cached prefix —
plus new_i fresh ids. This reproduces r and h per request by construction,
and the shared prefix makes the server's prefix cache behave as recorded.

Output-span cache continuity: the server caches the ids it actually
generated. Without a tokenizer the client appends fresh synthetic ids for the
output span, so the next request re-prefills the previous output (mean ~870
tokens/request extra uncached work — reported at the end as achieved r vs
target). With --tokenizer the client retokenizes the returned completion text
and appends the server's own ids, keeping the radix/prefix cache warm across
steps (a few tokens of BPE-boundary drift are reconciled at the next step's
construction).

Content: fresh ids are uniform random in [1000, vocab-1000) by default.
--corpus FILE (requires --tokenizer) slices pre-tokenized windows of real
public text instead, for measurements that care about token statistics
(speculative-decode acceptance — lane P).

Window rule (GLM-5.2 architectural window is 1,048,576; the *served* window
at the campaign operating point is smaller). Effective cap per request is
C = window - max_tokens - 8.
  --clip compact (default): when the recorded input exceeds C, emulate Claude
      Code compaction: the session context resets to --post-compact fresh ids
      (a full cache miss, like the real re-prefill after /compact) and
      subsequent recorded inputs are shifted down by the drop; per-step new,
      output, and gap replay unchanged. --post-compact defaults to
      0.28 * window: the recorded corpus compacts at the 1M window to a
      post-compaction context of p50 293k = 0.280 of the window.
  --clip cap: lane-B parity (cap_tracelab.py): input = min(input, C) with the
      prefix preserved (tail-truncate). Optimistic about caching — a real
      sliding window would miss; use for comparisons against the simulator.

Timing: each step sleeps gap * --gap-scale after the previous step's response
completes (gap is recorded client think time). --gap-scale 0 = saturation.
Sessions are drawn in file order (--shuffle N for seeded shuffle) by
--concurrency closed-loop slots; a slot finishing a session takes the next.

Output: one JSON line per completed request (session, step, kind, timings,
target vs achieved usage) to --out, and a summary table (TTFT/TPOT/E2E
percentiles split by trigger kind, aggregate output tok/s, achieved r/h) to
stdout at the end.
"""

import argparse
import asyncio
import json
import math
import random
import sys
import time

import httpx


def percentiles(xs, pcts=(50, 90, 99)):
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


class ContentSource:
    """Fresh token ids: uniform random, or sliced pre-tokenized corpus text."""

    def __init__(self, vocab, seed, corpus_ids=None):
        self.rng = random.Random(seed)
        self.vocab = vocab
        self.corpus = corpus_ids

    def fresh(self, n):
        if n <= 0:
            return []
        if self.corpus:
            start = self.rng.randrange(len(self.corpus))
            out = []
            while len(out) < n:
                take = self.corpus[start:start + (n - len(out))]
                out.extend(take)
                start = 0 if start + len(take) >= len(self.corpus) else start + len(take)
            return out
        return [self.rng.randrange(1000, self.vocab - 1000) for _ in range(n)]


class SessionPlayer:
    def __init__(self, args, content, tokenizer, results, client):
        self.args = args
        self.content = content
        self.tok = tokenizer
        self.results = results
        self.client = client

    async def play(self, session, deadline):
        args = self.args
        ids = []  # replayed context: prompt of last step + its output span
        offset = 0  # cumulative context dropped by emulated compactions
        for step_no, step in enumerate(session["steps"]):
            if deadline and time.monotonic() > deadline:
                return
            if args.max_steps and step_no >= args.max_steps:
                return
            rec_in, rec_new = step["input"], step["new"]
            out_len = max(1, min(step["output"], args.max_output))
            gap = step["gap"] * args.gap_scale
            if gap > 0 and step_no > 0:
                await asyncio.sleep(gap)

            cap = args.window - out_len - 8
            eff_in = rec_in - offset
            if args.clip == "compact" and eff_in > cap:
                offset = rec_in - args.post_compact
                eff_in = args.post_compact
                ids = []  # full miss: context replaced by the compact summary
            elif args.clip == "cap":
                eff_in = min(eff_in, cap)
            eff_in = max(eff_in, 1)
            new = min(rec_new, eff_in)
            cached_target = eff_in - new

            # prefix-preserving construction of the recorded cached span
            if len(ids) > cached_target:
                ids = ids[:cached_target]
            elif len(ids) < cached_target:
                ids = ids + self.content.fresh(cached_target - len(ids))
            prompt = ids + self.content.fresh(new)

            t0 = time.monotonic()
            ttft = None
            text_parts = []
            usage = None
            try:
                async with self.client.stream(
                    "POST", "/completions",
                    json={
                        "model": args.model,
                        "prompt": prompt,
                        "max_tokens": out_len,
                        "temperature": 1.0,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "ignore_eos": True,
                    },
                ) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread())[:300]
                        self.results.append({
                            "session": session["id"], "step": step_no,
                            "error": f"HTTP {resp.status_code}: {body.decode(errors='replace')}",
                        })
                        return
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload.strip() == "[DONE]":
                            break
                        chunk = json.loads(payload)
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        for choice in chunk.get("choices", []):
                            piece = choice.get("text")
                            if piece:
                                if ttft is None:
                                    ttft = time.monotonic() - t0
                                text_parts.append(piece)
            except (httpx.HTTPError, asyncio.TimeoutError) as e:
                self.results.append({
                    "session": session["id"], "step": step_no,
                    "error": f"{type(e).__name__}: {e}",
                })
                return
            e2e = time.monotonic() - t0

            # output span for the next step's context
            out_ids = None
            if self.tok:
                enc = self.tok.encode("".join(text_parts), add_special_tokens=False)
                out_ids = list(enc.ids)[:out_len]
            if not out_ids:
                out_ids = self.content.fresh(out_len)
            ids = prompt + out_ids

            details = (usage or {}).get("prompt_tokens_details") or {}
            self.results.append({
                "session": session["id"], "step": step_no,
                "kind": step.get("kind"),
                "t_issue": round(t0, 3),
                "ttft_s": round(ttft, 4) if ttft is not None else None,
                "e2e_s": round(e2e, 4),
                "tpot_s": round((e2e - ttft) / max(out_len - 1, 1), 5)
                if ttft is not None else None,
                "prompt_len": len(prompt), "target_cached": cached_target,
                "target_new": new, "out_len": out_len,
                "usage_prompt": (usage or {}).get("prompt_tokens"),
                "usage_cached": details.get("cached_tokens"),
                "usage_completion": (usage or {}).get("completion_tokens"),
                "compacted": bool(offset) and eff_in == args.post_compact and not cached_target,
            })


async def run(args):
    sessions = [json.loads(line) for line in open(args.sessions)]
    if args.shuffle is not None:
        random.Random(args.shuffle).shuffle(sessions)
    corpus_ids = None
    tokenizer = None
    if args.tokenizer:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(args.tokenizer)
        if args.corpus:
            text = open(args.corpus, errors="ignore").read()
            corpus_ids = list(tokenizer.encode(text, add_special_tokens=False).ids)
            if len(corpus_ids) < 4096:
                print("corpus too small (<4096 tokens)", file=sys.stderr)
                return 1
    elif args.corpus:
        print("--corpus requires --tokenizer", file=sys.stderr)
        return 1

    content = ContentSource(args.vocab, args.seed, corpus_ids)
    results = []
    queue = asyncio.Queue()
    for s in sessions:
        queue.put_nowait(s)

    deadline = time.monotonic() + args.duration if args.duration else None
    t_bench0 = time.monotonic()
    async with httpx.AsyncClient(
        base_url=args.base_url, timeout=httpx.Timeout(args.timeout, connect=30),
        limits=httpx.Limits(max_connections=args.concurrency + 4),
    ) as client:
        player = SessionPlayer(args, content, tokenizer, results, client)

        async def worker():
            while not (deadline and time.monotonic() > deadline):
                try:
                    s = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                await player.play(s, deadline)

        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    wall = time.monotonic() - t_bench0

    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]
    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")

    total_out = sum(r["out_len"] for r in ok)
    total_new = sum(r["target_new"] for r in ok)
    total_prompt = sum(r["prompt_len"] for r in ok)
    cached = [r for r in ok if r["usage_cached"] is not None]
    summary = {
        "requests_ok": len(ok), "errors": len(errs), "wall_s": round(wall, 1),
        "output_tok_per_s": round(total_out / wall, 1) if wall else None,
        "offered_r": round(total_new / total_out, 2) if total_out else None,
        "offered_h": round(1 - total_new / total_prompt, 4) if total_prompt else None,
        "achieved_h_server": round(
            sum(r["usage_cached"] for r in cached)
            / sum(r["prompt_len"] for r in cached), 4) if cached else None,
        "ttft_s": percentiles([r["ttft_s"] for r in ok if r["ttft_s"] is not None]),
        "tpot_s": percentiles([r["tpot_s"] for r in ok if r["tpot_s"] is not None]),
        "e2e_s": percentiles([r["e2e_s"] for r in ok]),
    }
    for kind in ("tool", "user"):
        rs = [r for r in ok if r.get("kind") == kind]
        summary[f"ttft_s_{kind}"] = percentiles(
            [r["ttft_s"] for r in rs if r["ttft_s"] is not None])
    print(json.dumps(summary, indent=1))
    if errs:
        print(f"first error: {errs[0]}", file=sys.stderr)
    return 0 if not errs else 2


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions")
    ap.add_argument("--base-url", required=True, help="e.g. http://host:30000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=8,
                    help="closed-loop concurrent session slots")
    ap.add_argument("--duration", type=float, help="stop issuing after N seconds")
    ap.add_argument("--gap-scale", type=float, default=1.0,
                    help="scale recorded think gaps (0 = saturation)")
    ap.add_argument("--max-steps", type=int, help="cap steps per session")
    ap.add_argument("--max-output", type=int, default=32768,
                    help="cap per-request max_tokens")
    ap.add_argument("--window", type=int, default=1048576,
                    help="served context window (tokens)")
    ap.add_argument("--clip", choices=["compact", "cap"], default="compact")
    ap.add_argument("--post-compact", type=int,
                    help="context size after an emulated compaction "
                         "(default 0.28 * window, the recorded corpus ratio)")
    ap.add_argument("--tokenizer", help="tokenizer.json: keep output spans cache-warm")
    ap.add_argument("--corpus", help="text file for naturalistic content (needs --tokenizer)")
    ap.add_argument("--vocab", type=int, default=154880, help="vocab size for random ids")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", type=int, help="seeded session order shuffle")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", help="per-request JSONL")
    args = ap.parse_args()
    if args.post_compact is None:
        args.post_compact = int(0.28 * args.window)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
