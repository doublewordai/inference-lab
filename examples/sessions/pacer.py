#!/usr/bin/env python3
"""Realistic-pacing SSE proxy in front of inference-lab, for the flex
live-relay Redis stress test (COR-648).

inference-lab's own `serve` mode paces sim-time to wall-time internally, but
that pacing is only reliably realistic once a request's batch has been
running a while (a ramp-up window during which tokens arrive far faster than
any real model) - see the investigation notes in the accompanying plan. This
proxy sidesteps that entirely: it asks inference-lab for the FULL response
non-streaming (fast and content-realistic, since that part works fine), then
re-streams it to the real client itself, one real token at a time, at an
explicit, deterministic, configurable rate - decoupled from inference-lab's
own scheduler/engine timing altogether.

Token boundaries come from a real production tokenizer via gigatoken
(https://github.com/marcelroed/gigatoken), not word-splitting, so pacing is
against genuine sub-word units the way a real model actually streams.

    python3 examples/sessions/pacer.py \
        --upstream http://localhost:8080/v1 \
        --tokenizer deepseek-ai/DeepSeek-V3 \
        --tokens-per-second 60 \
        --port 9000

Then point dwctl's registered endpoint at this proxy's port instead of
inference-lab's directly.
"""

import argparse
import asyncio
import json
import time
import uuid

import gigatoken as gt
import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

_tokenizer = None


def get_tokenizer(model_name):
    # Not every model's tokenizer is loaded from a standard HF tokenizer.json
    # (Kimi-K2's isn't), so `.as_hf()` isn't always available - the raw
    # gigatoken.Tokenizer's own encode()/decode() work directly regardless,
    # just returning bytes instead of str (decoded at the call site below).
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = gt.Tokenizer(model_name)
    return _tokenizer


def sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


def make_app(upstream_base, tokens_per_second, tokenizer_model):
    interval = 1.0 / tokens_per_second
    # httpx defaults to 100 max connections - at real concurrency that
    # queues most requests for a pool slot until they time out upstream as
    # 502s, well before this proxy's own pacing is ever the bottleneck.
    client = httpx.AsyncClient(
        base_url=upstream_base,
        timeout=httpx.Timeout(60.0, connect=10.0),
        limits=httpx.Limits(max_connections=100000, max_keepalive_connections=1000),
    )

    async def chat_completions(request):
        body = await request.json()
        model = body.get("model")
        stream = bool(body.get("stream", False))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        upstream_body = dict(body)
        upstream_body["stream"] = False
        try:
            resp = await client.post("/chat/completions", json=upstream_body)
        except httpx.HTTPError as e:
            return JSONResponse({"error": {"message": f"upstream error: {e}", "type": "server_error"}}, status_code=502)

        if resp.status_code != 200:
            return Response(resp.content, status_code=resp.status_code, media_type="application/json")

        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}

        if not stream:
            return JSONResponse(data)

        tokenizer = get_tokenizer(tokenizer_model)
        ids = tokenizer.encode(text)
        pieces = [tokenizer.decode([i]).decode("utf-8", errors="replace") for i in ids]
        req_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())

        async def gen():
            yield sse(
                {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
            )
            for piece in pieces:
                await asyncio.sleep(interval)
                yield sse(
                    {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                    }
                )
            yield sse(
                {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            if include_usage:
                yield sse(
                    {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": len(pieces),
                            "total_tokens": usage.get("prompt_tokens", 0) + len(pieces),
                        },
                    }
                )
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def list_models(request):
        resp = await client.get("/models")
        return Response(resp.content, status_code=resp.status_code, media_type="application/json")

    async def health(request):
        return JSONResponse({"status": "ok"})

    return Starlette(
        routes=[
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
            Route("/v1/models", list_models, methods=["GET"]),
            Route("/health", health, methods=["GET"]),
        ]
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", required=True, help="inference-lab base URL, e.g. http://localhost:8080/v1")
    ap.add_argument("--tokenizer", default="deepseek-ai/DeepSeek-V3", help="HF model name for gigatoken")
    ap.add_argument("--tokens-per-second", type=float, default=60.0, help="per-stream pacing rate")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    args = ap.parse_args()

    app = make_app(args.upstream, args.tokens_per_second, args.tokenizer)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
