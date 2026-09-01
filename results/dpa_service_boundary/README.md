# DPA sustainable-service TTFT sweep

Generated on 2026-08-30 from commit `30cad0a` plus the current uncommitted
experiment changes:

```sh
target/release/examples/dpa_service_boundary --sweep --json \
  > results/dpa_service_boundary/ttft-sweep.json \
  2> results/dpa_service_boundary/ttft-sweep.stderr
```

Each run uses seed 42 and starts its 7,200-second measurement only after every
prefill and decode rank has cumulatively evicted one local HBM capacity.
Arrivals remain enabled throughout the measurement, and the simulator then
drains so every admitted request's TTFT is observed.

The pass criterion is p99 TTFT at or below 30 seconds in each of twelve
600-second arrival windows. The 120-second cohorts are diagnostic only; they do
not change the pass criterion.

- `ttft-sweep.json`: full structured output, including queue samples.
- `ttft-cohorts.tsv`: 600 diagnostic 120-second cohorts (60 per rate).
- `ttft-slo-windows.tsv`: 120 SLO windows (12 per rate).
- `ttft-sweep-summary.tsv`: one-row-per-rate summary and first SLO breach.
- `ttft-sweep.stderr`: concise run log.

For this single seeded sweep, 0.020 sessions/s/rank passes and 0.02125
sessions/s/rank first breaches in the final SLO window. This brackets rather
than precisely estimates the service boundary; replication across seeds is a
separate experiment.

This two-hour sweep predates the just-in-time arrival-admission fix used by the
long run below. A fixed-seed short rerun changed output throughput by 0.03%
without changing its SLO classification, but these artifacts should be
refreshed before pooling their statistics with the long-run results.

## Long empty-start run

`dpa_long_run` keeps arrivals enabled until the configured endpoint and stops
without a drain. It logs one-hour progress epochs and stops early if either:

- aggregate waiting reaches 1,000 requests; or
- the endpoint queue grows for six consecutive one-hour epochs without
  returning to zero.

The bounded `0.020` sessions/s/rank run stopped after 26.1 wall seconds at
26,390.8 simulated seconds (7.33 hours), when waiting first reached 1,001.
The final partial epoch's completed-request p99 TTFT was 496.7 seconds.

- `ten-day-lambda-0.02-limited.json`: structured run result and stop reason.
- `ten-day-lambda-0.02-limited-progress.tsv`: hourly and terminal progress.
- `ten-day-lambda-0.02-limited-slo-windows.tsv`: 600-second TTFT windows with
  explicit endpoint censoring.
- `ten-day-lambda-0.02-uncapped-cancelled.stderr`: progress from the manually
  cancelled uncapped run, which had reached 2.42 simulated days.

## Excursion-cause intervention

The `0.0035` sessions/s/rank run was repeated with one causal intervention:
double only the prefiller GPUs' physical HBM. Model weights, prefiller compute,
decode hardware, workload, and seed remained unchanged. Because weights are a
fixed allocation, this increased the prefiller pool's usable KV capacity from
544.9 GB to 1,927.3 GB (3.54x), while decode KV capacity stayed at 544.9 GB.

- `excursion-cause-lambda-0.0035-prefill-hbm-1x.json` and `.stderr`: baseline.
- `excursion-cause-lambda-0.0035-prefill-hbm-2x.json` and `.stderr`: HBM
  intervention.

The JSON records one-minute prefill/decode queue samples, hourly TTFT and work
breakdowns, reusable-prefix miss cohorts, the largest individual completed
miss in each hour, and pool-separated cache eviction and occupancy.

The baseline has recoverable prefill-queue excursions before entering a
non-recovering one around day 6.7. It reached the 1,001-waiting-request safety
limit at day 9.16. Its full observed-window prefill queue p99 was 946, versus 0
for decode; completed-request p95 TTFT was 694.7 seconds over the truncated run.
During the transition:

| phase (days) | mean prefill queue | p95 TTFT | parent-prefill recompute/h | parent-decode recompute/h | reusable-token miss |
|---|---:|---:|---:|---:|---:|
| 5.5-6.67 | 1.0 | 5.5 s | 100.5 M | 3.22 M | 9.8% |
| 6.67-6.84 | 19.7 | 80.8 s | 253.8 M | 3.23 M | 21.1% |
| 6.84-7.5 | 81.1 | 245.7 s | 364.2 M | 2.66 M | 41.0% |
| 7.5-stop | 668.3 | 923.3 s | 504.1 M | 1.85 M | 99.3% |

This is not initiated by one exceptional request. The largest individual
reusable-prefix miss was already 999k tokens before the non-recovering onset,
was 1.000M during the runaway, and the healthy 2x-HBM run also processed 999k
token individual misses. What changes is the distribution: hourly reusable
miss volume grows from 109M tokens before onset to 322M at onset and 377-507M
afterwards, while its per-request p99 grows from 511k to roughly 0.8M tokens.

The 2x-HBM run completed all ten days. Prefill queue p99/max were 2/7, decode
queue p99/max were 2/12, and completed-request p95 TTFT was 330 ms. In the same
late windows its parent-prefill recomputation stayed around 7-11M tokens/hour
and its reusable-token miss fraction around 1%.

The measured cause of the excursions is therefore broad loss of prefiller
prefix reuse, not decode congestion or a single large eviction. The likely
feedback is that the prefiller working set exceeds usable KV capacity; misses
then consume finite prefill service, the resulting delay leaves more active
session state competing for that cache, and the miss/recompute load either
subsides (a recoverable excursion) or crosses into sustained queue growth. The
large intervention establishes this mechanism for seed 42 but does not locate
the minimum sufficient cache capacity or establish a seed-independent rate.
