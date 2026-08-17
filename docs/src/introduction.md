# Introduction

Inference Lab is a simulation framework designed to evaluate and analyze LLM
workloads.

It uses discrete-event simulation to model the behavior of a
multi-GPU node serving LLM inference requests with the vLLM library. It
contains a facsimile of the vLLM queueing, scheduling, and execution logic,
with only the actual model inference replaced by a performance model based on
the supplied GPU specs and model architecture.

Within each simulation step, the simulator:

- Processes any newly arrived requests, adding them to the scheduling queue.
- Schedules requests to serve based on the selected scheduling policy.
- Calculates the compute and memory bandwidth usage for the workload that the scheduled requests represent, and the theoretical time required to execute the workload on the specified hardware.
- Increments the simulation time by the calculated execution time, updating the state of all requests accordingly.

Caveats:

- Step times are a datasheet roofline: peak FLOP rate and HBM bandwidth
  per precision stream, collectives on the fabric preset added serially,
  and no kernel-efficiency or fixed per-step overhead term. Every
  latency and throughput is an upper bound; the optional
  `time_correction = { alpha, beta }` on a hardware entry calibrates
  the step (`alpha × roofline + beta`) against a measured engine.

## Features

- Roofline Performance Modeling: compute (FLOPS) and memory bandwidth constraints per precision stream
- Multiple Scheduling Policies: FCFS, Priority, SJF, and more
- Chunked Prefill: Simulates realistic request interleaving
- KV Cache Management: Models GPU memory and KV cache utilization
- Workload Generation: Supports Poisson, Gamma, and closed-loop patterns
- WebAssembly Support: Run simulations in the browser via WASM

## Quick Start

See the [Getting Started](./user-guide/getting-started.md) guide to begin using Inference Lab.
