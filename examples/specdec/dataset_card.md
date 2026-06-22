---
license: apache-2.0
task_categories:
  - text-generation
tags:
  - speculative-decoding
  - calibration
  - mixture-of-experts
pretty_name: Qwen3.6-35B-A3B speculative-decoding calibration
---

# Qwen3.6-35B-A3B speculative-decoding calibration

Per-round speculative-decoding **acceptance** and **speculator** banks for
Qwen3.6-35B-A3B, collected by driving SGLang and logging every draft round.
Used to drive the discrete-event simulator in
[inference-lab](https://github.com/doublewordai/inference-lab) (see
`examples/specdec/README.md` for figure reproduction).

## Layout

```
<dataset>/<drafter>/<experiment>/
```

- **dataset** — prompt corpus:
  - `speedbench` — SPEED-Bench qualitative split (coding, math, qa, rag,
    reasoning, stem, writing, humanities, multilingual, summarization, roleplay,
    plus high/low/mixed entropy buckets).
  - `humaneval` — HumanEval, 164 coding prompts.
- **drafter** — the speculator head:
  - `mtp` — native one-layer MTP/NextN head.
  - `dflash@42d3b34d` — DFlash block-diffusion head (8 dense SwiGLU layers + a
    5-layer hidden-state fusion), checkpoint `42d3b34d`.
- **experiment**:
  - `acceptance` — the standard acceptance + speculator collection.
  - `routing` — same prompts at batch 10, additionally dumping the MoE
    expert-selection capture (`routing.npy` + `routing_meta.parquet`). HumanEval
    only.

```
qwen3.6-35b-a3b/
  speedbench/{mtp, dflash@42d3b34d}/acceptance/
  humaneval/{mtp, dflash@42d3b34d}/{acceptance, routing}/
```

Each leaf is a calibration run directory: the parquet banks, durable checkpoint
shards under `parts/`, and JSON sidecars (`run_manifest.json`, `stats.json`,
`metainfo.json`).

## Schemas

Banks share the key `(model, speculator, config, category, prompt_idx, turn,
round_idx)` — one row per draft round, so they JOIN.

**acceptance.parquet** — verify side: `… , accept, acc0..acc15`.
`accept` is the committed draft-token count (excludes the bonus); `acc_k` is the
per-position accept mask (1/0/null). Acceptance is a contiguous prefix, so
`acc_k = 1 iff k < accept`. Shallower drafters null-pad trailing columns.

**speculator.parquet** — draft side: `… , conf0..conf15`. `conf_k` is the
drafter's softmax probability of the token proposed at depth `k` (null where no
token proposed).

**routing_meta.parquet + routing.npy** (routing experiments) — paired and
aligned row-for-row by `routing_idx`. `routing.npy` is shape `(N, L, k)` `uint8`
(N routing positions × L=40 layers × k=8 routed experts per token) holding the
expert IDs; `routing_meta.parquet` indexes it
(`… , routing_idx, routing_block_idx, request_idx, position, accepted`).

## Usage

Download everything, or just the slice you need:

```bash
# the lightweight acceptance banks (a few MB):
hf download Doubleword/qwen3.6-specdec-calibration --repo-type dataset \
  --include "qwen3.6-35b-a3b/speedbench/*/acceptance/acceptance.parquet" --local-dir data/

# everything:
hf download Doubleword/qwen3.6-specdec-calibration --repo-type dataset \
  --local-dir data/
```

To turn a run directory into the simulator's trace-bank CSV, use the
`export-trace` command shipped with inference-lab's `calibration/` package
(`specdec-calibrate export-trace --run-dir <leaf> --signal oracle -o <out>.csv`).
The simulator's homogeneous policy uses only the committed count; the per-depth
mask and confidence feed the gated policies.
