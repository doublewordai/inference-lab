# Saturation and Capacity

`serve` mode queues arrivals without bound by default: there is no request
volume at which it refuses work. That is fine for latency experiments, but it
makes the server useless for testing a client that is supposed to *react* to
overload — a controller that discovers a model's sustainable concurrency from
downstream rejections has nothing to react to, because no rejection ever
arrives.

Two knobs change that, and they do different things:

| Knob | What it changes | Typical use |
|------|-----------------|-------------|
| `max_waiting` | How long the queue may get before the server refuses | Switch 529s on and off |
| `max_num_seqs` | How fast the engine drains work | Model a scale up/down |

`max_waiting` does **not** change capacity. Lowering it makes rejections start
sooner, but the engine serves at exactly the same rate, so the concurrency a
client settles at is set by queue policy rather than by anything physical.
`max_num_seqs` is the dial that changes the drain rate: halve it and the queue
backs up at roughly half the offered load.

## The waiting bound

Set it in a model's `[scheduler]` table:

```toml
[scheduler]
max_num_seqs = 256
max_waiting = 64   # 0 (the default) = unbounded, never refuse
```

or override every model's value at the command line:

```bash
inference-lab serve --config configs/ --hardware b200 --max-waiting 64
```

Once `max_waiting` requests are queued, further arrivals get:

```
HTTP/1.1 529
content-type: application/json

{"error": {
  "message": "Server is at capacity: 64 requests waiting (max_waiting = 64). Retry with reduced concurrency.",
  "type": "overloaded_error",
  "code": "queue_saturated"
}}
```

### Why 529, and not 503 or 429

529 is the convention for "the engine has nowhere to put this request".
Clients that adapt their concurrency generally key on 529 alone, because the
other two mean something a client should *not* answer by shedding load: 503 is
"this service is unavailable" and 429 is "you exceeded a quota or a proxy's own
limit". `serve` already returns 503 when the engine channel is closed, which is
a liveness failure rather than saturation, and stays a distinct code.

### Two properties the rejection is built to have

**It is returned before the response starts.** A real engine that admits a
request, sends `200` plus streaming headers, and only then discovers it cannot
schedule it has spent its status code: the failure can then only be an error
object inside the stream, or a stream that stops with no content. Neither is
classifiable as overload, so the client learns nothing. The check therefore
runs in `submit_engine_request`, the single funnel into the engine, before any
handler has built a response — a refused request returns a bare status and
envelope, never a `text/event-stream`.

**It is immediate, never a stall.** A bounded queue that parks requests until
some later timeout produces a client that waits and eventually gives up, which
consumes a client slot for the whole timeout and still carries no overload
signal. The bound is a synchronous check against a published queue depth.

### What "waiting" counts

The depth compared against `max_waiting` is every worker's `num_waiting()`
(queued requests plus those parked on a KV transfer or a staged read), *plus*
arrivals the HTTP layer has admitted that the engine has not stepped into a
scheduler yet. The second term matters: `Engine::submit` only queues an
`Arrival` event, so without it a burst would read as depth 0 and be admitted
wholesale.

## Runtime capacity control

`GET /control/capacity` reports every model's knobs and live depth:

```bash
curl localhost:8080/control/capacity
[{"model":"gpt-oss-20b","max_waiting":64,"max_num_seqs":256,"waiting":12,"running":256}]
```

`POST /control/capacity` retunes them without a restart — which matters
because a restart drops every in-flight request, destroying the before-and-after
that a capacity-change experiment depends on. Both fields are optional, and
`model` defaults to every loaded model:

```bash
# Scale down: the engine now drains at a sixteenth of the rate.
curl -X POST localhost:8080/control/capacity \
  -H 'content-type: application/json' -d '{"max_num_seqs": 16}'

# Turn 529s on (or off again with 0) against a running server.
curl -X POST localhost:8080/control/capacity \
  -H 'content-type: application/json' -d '{"max_waiting": 64}'
```

Both changes act on live state:

- **Lowering `max_num_seqs` drains, it does not evict.** The cap gates
  admission only, so requests already running above the new cap run to
  completion and the batch shrinks to the new size. Nothing in flight is lost.
- **Lowering `max_waiting` only affects requests that have not arrived yet.**
  Anything already queued keeps its place; its status code is long since spent.

`max_num_seqs` must be at least 1 — a cap of 0 would accept requests and then
never schedule them, which is exactly the stall the bound exists to replace.
`max_waiting: 0` is meaningful (unbounded) and allowed.

## Cold start: the first burst after idle is unpaced

`serve` paces simulated time to wall-clock from an epoch fixed when the engine
starts, and simulated time only advances while there is work to do. An idle
server therefore accumulates a deficit, and the first requests after a quiet
period are served as fast as the CPU allows rather than at the rate the model
predicts — a single 200-token stream into a freshly booted server returns in
single-digit milliseconds.

The deficit burns off once there is continuous work: under sustained load the
sim clock catches up within about a second, after which pacing is correct and
steady. Measured on `gpt-oss-20b` / `b200` at 24 concurrent requests, a
200-token response settles at a flat ~293 ms; drop `max_num_seqs` to 8 and the
same load settles at ~646 ms with throughput down by roughly the same factor.

The practical consequence is only for short experiments: a benchmark that fires
one burst at a just-started server measures the CPU, not the modelled hardware.
Give it a second of warm-up load first.
