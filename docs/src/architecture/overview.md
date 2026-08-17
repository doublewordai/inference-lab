# Architecture Overview

Inference Lab is a discrete-event simulator of an LLM serving engine. The
engine's queueing, scheduling and KV-cache logic are simulated faithfully
(vLLM v1 shapes); the forward pass is replaced by a roofline cost model
built from the hardware spec and the model's architecture.

## Layers

```
catalog       shipped hardware / model presets (embedded TOML)
config        TOML/JSON -> Config, ClusterSpec, DisaggTopology
  |
  |  ModelSpec: weight streams (per-precision FLOPs and bytes, MoE routing)
  |  + layer classes (attention FLOPs, KV read / storage, per-sequence state)
  v
compute       ComputeEngine: roofline step cost for a batch on a cluster
kv_cache      KVCacheManager (blocks, prefix cache, spillover tiers), Link
scheduler     Scheduler: waiting/running sets, one schedule() per iteration
  |
  v
simulation    Engine: worker pools + hand-off links + event heap
              SpecPlanner: speculative-decoding draft depth and outcomes
              drivers: Simulator (batch), RealtimeEngine (HTTP), simulate_closed_loop
metrics       MetricsCollector over RequestTiming -> MetricsSummary
```

Every consumer of a model goes through `ModelSpec`'s cost methods. There
are no named architectures: a model is a composition of weight streams and
layer classes (`attention` with optional window and shared KV, `mla` with
optional window / history / indexer, `linear` state), so nothing else in the
simulator knows an architecture's knobs.

## One worker

A worker is one `Scheduler` plus one `ComputeEngine`; the scheduler owns the
worker's `KVCacheManager`. Each iteration:

1. `Scheduler::schedule(now)` reaps finished requests (freeing their KV),
   grows the running set inside the token budget (preempting under KV
   pressure), and admits waiting requests into the leftover budget. It
   returns the batch as `(index, positions)` pairs.
2. `ComputeEngine::step_cost` prices the batch: per-precision streams,
   `max(compute, memory)` per stream summed, plus collectives. Speculative
   decoding may instead price the decode portion from a measured table.
3. The engine commits progress to each request (`record_progress`), hands
   completed prefills to the decode pool on a disaggregated topology, and,
   when speculating, asks the `SpecPlanner` for next step's drafts.

## Topologies

`Topology::aggregated` is one pool of identical workers; a `Router`
(round-robin by default; least-loaded, prefix-affinity or KV-aware by
config) picks the worker each arrival enters. `Topology::from_disagg` is a
prefill pool and a decode pool joined by a KV hand-off `Link` whose
bandwidth is shared by every hand-off in flight (processor sharing,
event-driven); `[router]` fronts the prefill pool and `[decode_router]`
the decode pool.

## Time

The engine is a pure event loop over arrivals, worker-ready events and link
drains; the drivers decide what to submit and when. The batch `Simulator`
pumps a `RequestGenerator` through the engine and feeds `RequestTiming`s
into a `MetricsCollector`; the serve driver paces engine time to the wall
clock and streams tokens to HTTP clients.
