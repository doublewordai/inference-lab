# Architecture Overview

Inference Lab is a discrete-event simulator of an LLM serving engine. The
engine's queueing, scheduling and KV-cache logic are simulated faithfully
(vLLM v1 shapes); the forward pass is replaced by a roofline cost model
built from the hardware spec and the model's architecture.

## Layers

```
config        TOML/JSON -> Config, ClusterSpec, DisaggTopology
  |
  |  ModelCosts trait: per-precision FLOPs and weight bytes, attention
  |  FLOPs, KV read / storage bytes, per-sequence state, collective volumes
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

Every consumer of a model goes through `ModelCosts`; architectures
(`dense`, `deepseek_v4`, `qwen35`) implement it and nothing else in the
simulator knows their knobs.

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

`Topology::aggregated` is one pool of identical workers; arrivals are
round-robined across them. `Topology::from_disagg` is a prefill pool and a
decode pool joined by a KV hand-off `Link` whose bandwidth is shared by every
hand-off in flight (processor sharing, event-driven).

## Time

The engine is a pure event loop over arrivals, worker-ready events and link
drains; the drivers decide what to submit and when. The batch `Simulator`
pumps a `RequestGenerator` through the engine and feeds `RequestTiming`s
into a `MetricsCollector`; the serve driver paces engine time to the wall
clock and streams tokens to HTTP clients.
