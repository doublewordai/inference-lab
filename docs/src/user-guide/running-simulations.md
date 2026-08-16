# Running Simulations

This guide covers how to run simulations and interpret results.

## Basic Usage

Run a model config on one of its hardware entries against a workload:

```bash
inference-lab -c configs/gpt-oss-120b.toml --hardware b200 -w workloads/chat-closed-256.toml
```

For dataset mode, add tokenizer and chat template:

```bash
inference-lab -c configs/llama-3-70b.toml -w workloads/dataset-poisson.toml \
  --tokenizer tokenizer.json \
  --chat-template None
```

See [Configuration](./configuration.md) for details on configuring workloads, policies, and hardware.

## Output Modes

### Console Output (Default)

By default, the simulator displays:
- Real-time progress bar
- Current simulation time
- Queue status (running/waiting requests)
- KV cache utilization

Final output includes:
- Latency metrics (TTFT, E2E, per-token)
- Throughput metrics (tokens/sec, requests/sec)
- Utilization statistics (KV cache, FLOPS, bandwidth)
- Preemption statistics

### JSON Output

Save results to a file:

```bash
inference-lab -c configs/gpt-oss-120b.toml --hardware b200 -w workloads/chat-closed-256.toml -o results.json
```

Combine with `-q` for batch processing:

```bash
inference-lab -c configs/gpt-oss-120b.toml --hardware b200 -w workloads/chat-closed-256.toml -q -o results.json
```

## Running Multiple Experiments

### Comparing Hardware

```bash
for hw in b200 b300 gh200-120; do
  inference-lab -c configs/gpt-oss-120b.toml --hardware $hw \
    -w workloads/chat-closed-256.toml -q -o results_$hw.json
done
```

### Sweeping Engine Args

```bash
for batch_size in 4096 8192 16384; do
  sed "s/max_num_batched_tokens = .*/max_num_batched_tokens = $batch_size/" \
    configs/gpt-oss-120b.toml > /tmp/gpt-oss-120b_$batch_size.toml
  inference-lab -c /tmp/gpt-oss-120b_$batch_size.toml --hardware b200 \
    -w workloads/chat-closed-256.toml -o results_$batch_size.json
done
```

### Multiple Seeds

Override the workload's seed:

```bash
for seed in {1..10}; do
  inference-lab -c configs/gpt-oss-120b.toml --hardware b200 \
    -w workloads/chat-closed-256.toml --seed $seed -q -o results_$seed.json
done
```

## Understanding Results

### Latency Metrics

**Time to First Token (TTFT)**
- Time from request arrival to first token generation
- Lower is better for interactive applications
- Affected by: queue wait time, prefill computation

**End-to-End (E2E) Latency**
- Total time from request arrival to completion
- Includes prefill and all decode steps
- Key metric for overall user experience

**Per-Token Latency**
- Average time between consecutive output tokens
- Lower is better for streaming applications
- Primarily affected by batch size and model size

### Throughput Metrics

**Input Tokens/sec**
- Rate of processing prompt tokens
- Indicates prefill throughput

**Output Tokens/sec**
- Rate of generating output tokens
- Indicates decode throughput

**Requests/sec**
- Overall request completion rate
- Key metric for capacity planning

### Utilization Metrics

**KV Cache**
- Percentage of KV cache memory in use
- High utilization may lead to preemptions

**FLOPS**
- Percentage of compute capacity utilized
- Low FLOPS may indicate memory bottleneck

**Bandwidth**
- Percentage of memory bandwidth utilized
- High bandwidth utilization indicates memory-bound workload

### Preemption Statistics

Preemptions occur when new requests need memory but the KV cache is full:
- Total number of preemptions
- Average preemptions per request
- Can significantly impact TTFT for preempted requests

## Troubleshooting

**Simulation running slowly?**
- Reduce `num_requests` or use `-q` flag
- Build with `--release`

**Too many preemptions?**
- Raise `gpu_memory_utilization` or set `kv_cache_capacity` in `[scheduler]`
- Reduce `max_num_seqs` or `max_num_batched_tokens` in scheduler config

**Dataset loading errors?**
- Verify `--tokenizer` and `--chat-template` flags are provided
- Check JSONL format matches OpenAI batch API format

For more details, see [CLI Reference](../reference/cli.md) and [Configuration](./configuration.md).
