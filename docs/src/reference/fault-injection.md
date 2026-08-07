# Fault Injection

The serve mode can kill a streaming chat completion mid-generation in every way a real
downstream does — on demand, deterministically, per request. It exists as the test double
for gateway resilience work (mid-stream continuation/resume middleware): if the sim can
produce every death signature, that middleware can be e2e-tested without waiting for real
incidents.

Every faulting stream first emits real partial output — the initial `role` frame plus
`after_chunks` content-bearing delta frames of deterministic placeholder text — so there
is always something to resume. The fault path bypasses the simulation engine (like
echo-directives) and contains no randomness: the same trigger produces the same frames
and the same death at the same byte, every time.

**Scope**: streaming `POST /v1/chat/completions` only. A fault header on a non-streaming
request, or on `/v1/completions`, is rejected with a 400 (`invalid_fault_directive`) —
never silently ignored. With no trigger present, behavior is completely unchanged.

## Trigger: the `x-inference-lab-fault` header

```text
x-inference-lab-fault: <mode>[;after_chunks=<u32>][;delay_ms=<u64>][;utf8=<bool>]
```

| Parameter | Default | Meaning |
|---|---|---|
| `<mode>` | required | one of the eleven mode names below |
| `after_chunks` | `3` | content-bearing delta frames emitted before the fault fires (the initial role-only frame is always sent and not counted) |
| `delay_ms` | `10` | fixed pacing between frames, milliseconds |
| `utf8` | `false` | `cut_mid_frame` only: cut inside a multi-byte UTF-8 character |

Parts are `;`-separated; whitespace around parts and `=` values is ignored. Unknown modes,
unknown keys, or malformed values are a 400 listing the valid modes.

A header was chosen over a body extension because the platform's proxy layer runs strict
request sanitization that strips unknown **body** fields; headers pass through. Verify
this empirically through the full stack (client → dwctl → onwards → sim) when the branch
reaches a preview environment — full-platform verification is out of scope here and
happens after deploy.

## Fallback trigger: static per-model config

For clients that cannot set a header, a model's TOML config can apply one fault to every
**streaming** chat completion on that model (non-streaming requests are served normally;
an explicit header on the request still wins):

```toml
[fault]
mode = "cut_mid_frame"   # same names as the header
after_chunks = 5         # optional, default 3
delay_ms = 10            # optional, default 10
utf8 = true              # optional, cut_mid_frame only
```

An invalid `[fault]` block fails server startup, not individual requests.

Precedence per request: header > model `[fault]` config > (echo-directives >) normal path.

## Modes

| Mode | After the N content frames… | Client observes (curl) |
|---|---|---|
| `cut_between_frames` | connection closes on a frame boundary, without the chunked-encoding terminator (FIN) | exit 18, transfer closed with outstanding read data |
| `cut_mid_frame` | half of the next frame's bytes, then close — torn JSON. `utf8=true` cuts one byte into a 2-byte UTF-8 character (`é`) in the delta text | exit 18, partial `data:` line |
| `reset` | abortive close: `SO_LINGER=0` then drop → TCP RST, not FIN | exit 56, connection reset by peer |
| `stall` | nothing, forever; connection stays open until the client gives up | exit 28 (client timeout) |
| `error_envelope_200` | OpenRouter-style error envelope as an SSE data frame, then `[DONE]`; HTTP status stays 200 | exit 0, `{"error":{"message":…,"code":502,"metadata":{…}}}` |
| `error_400_in_sse` | vLLM-style 400 object in-stream (the nemotron-incident signature), then clean close, no `[DONE]` | exit 0, `{"object":"error",…,"code":400}` |
| `no_done` | finish_reason frame (with usage if requested), then clean close — `[DONE]` never comes | exit 0, stream just ends |
| `no_usage` | finish_reason frame **without** usage (even when `stream_options.include_usage` was set), then `[DONE]` | exit 0, usage missing |
| `cancelled_499` | the exact dynamo frontend-cancellation body, then clean close | exit 0, `{"error":{"code":499,"message":"CancelledError: ","type":"request_cancelled"}}` |
| `mid_reasoning` | frames carry `reasoning_content` deltas instead of `content`; dies cut_between_frames-style | exit 18, last deltas are reasoning |
| `mid_tool_call` | frame 1 announces a tool call (id + name), later frames stream `arguments` fragments that never terminate; dies cut_between_frames-style | exit 18, partial tool call |

Notes:

- The error body shapes for `error_envelope_200` and `error_400_in_sse` are
  representative; exact shapes sync with the death-taxonomy workstream as it lands. The
  `cancelled_499` body is exact.
- `delay_ms=0` is fine for the graceful modes, but the abrupt modes always wait a short
  flush grace (~25 ms) before killing the connection so the partial output reliably
  reaches the wire first.
- `reset` needs the raw socket, which the server threads through per-connection; on
  non-unix platforms (or when handlers are driven outside the real server, as in unit
  tests) it degrades to a FIN with a warning log.

## Examples

All against a local sim (`inference-lab serve --config configs/ --port 8080`); `$BODY` is
any streaming chat request:

```bash
BODY='{"model":"DeepSeek-V4-Flash","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hello"}],"max_tokens":16}'
URL=http://localhost:8080/v1/chat/completions
```

```bash
# 1. cut_between_frames — 5 frames then FIN (curl exit 18)
curl -N $URL -H 'content-type: application/json' -H 'x-inference-lab-fault: cut_between_frames;after_chunks=5' -d "$BODY"

# 2a. cut_mid_frame — torn JSON frame
curl -N $URL -H 'content-type: application/json' -H 'x-inference-lab-fault: cut_mid_frame' -d "$BODY"

# 2b. cut_mid_frame, cut inside a multi-byte UTF-8 character (pipe through xxd to see it)
curl -sN $URL -H 'content-type: application/json' -H 'x-inference-lab-fault: cut_mid_frame;utf8=true' -d "$BODY" | xxd | tail

# 3. reset — TCP RST (curl exit 56, "connection reset by peer")
curl -N $URL -H 'content-type: application/json' -H 'x-inference-lab-fault: reset' -d "$BODY"

# 4. stall — 3 frames then silence; bound the wait client-side
curl -N --max-time 10 $URL -H 'content-type: application/json' -H 'x-inference-lab-fault: stall' -d "$BODY"

# 5. error_envelope_200 — OpenRouter-style envelope then [DONE]
curl -N $URL -H 'content-type: application/json' -H 'x-inference-lab-fault: error_envelope_200' -d "$BODY"

# 6. error_400_in_sse — 400-shaped object inside the 200 stream
curl -N $URL -H 'content-type: application/json' -H 'x-inference-lab-fault: error_400_in_sse' -d "$BODY"

# 7. no_done — finish_reason + usage, then the stream ends without [DONE]
curl -N $URL -H 'content-type: application/json' -H 'x-inference-lab-fault: no_done' -d "$BODY"

# 8. no_usage — [DONE] arrives but the requested usage never does
curl -N $URL -H 'content-type: application/json' -H 'x-inference-lab-fault: no_usage' -d "$BODY"

# 9. cancelled_499 — exact dynamo cancellation body
curl -N $URL -H 'content-type: application/json' -H 'x-inference-lab-fault: cancelled_499' -d "$BODY"

# 10. mid_reasoning — dies while streaming reasoning_content deltas
curl -N $URL -H 'content-type: application/json' -H 'x-inference-lab-fault: mid_reasoning' -d "$BODY"

# 11. mid_tool_call — dies with a tool call's arguments unterminated
curl -N $URL -H 'content-type: application/json' -H 'x-inference-lab-fault: mid_tool_call;after_chunks=4' -d "$BODY"
```
