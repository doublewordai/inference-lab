---
license: apache-2.0
task_categories:
  - text-generation
tags:
  - speculative-decoding
  - calibration
  - mixture-of-experts
pretty_name: Speculative-decoding calibration banks
---

# Speculative-decoding calibration banks

Per-round speculative-decoding **acceptance** and **speculator** banks, plus
MoE expert-**routing** captures, collected by driving SGLang and logging every
draft round. Used to drive the discrete-event simulator in
[inference-lab](https://github.com/doublewordai/inference-lab) (see
`examples/specdec/README.md` for figure reproduction).

Supersedes
[`Doubleword/qwen3.6-specdec-calibration`](https://huggingface.co/datasets/Doubleword/qwen3.6-specdec-calibration):
this dataset adds a model level to the path, the per-category SPEED-Bench
routing captures, and DeepSeek-V4-Flash.

## Layout

```
<model>/<dataset>/<drafter>/<experiment>/
```

- **model** — the verifier:
  - `qwen3.6-35b-a3b` — `Qwen/Qwen3.6-35B-A3B`.
  - `deepseek-v4-flash` — `deepseek-ai/DeepSeek-V4-Flash`.
- **dataset** — prompt corpus:
  - `speedbench` — SPEED-Bench qualitative split (coding, math, qa, rag,
    reasoning, stem, writing, humanities, multilingual, summarization,
    roleplay).
  - `humaneval` — HumanEval, 164 coding prompts (80 for DeepSeek-V4-Flash).
- **drafter** — the speculator head:
  - `mtp` — the model's native MTP/NextN head. (For DeepSeek-V4-Flash the
    `run_manifest.json` records `speculator: eagle` — SGLang serves DeepSeek
    MTP through its EAGLE worker; it is the native MTP module.)
  - `dflash@42d3b34d` — DFlash block-diffusion head (8 dense SwiGLU layers + a
    5-layer hidden-state fusion), checkpoint `42d3b34d`. Qwen only.
- **experiment**:
  - `acceptance` — the standard acceptance + speculator collection.
  - `routing` — same prompts at batch 10, additionally dumping the MoE
    expert-selection capture (`routing.npy` + `routing_meta.parquet`). For the
    SPEED-Bench routing captures (both models) the capture is one run directory
    per category (`routing/<category>/`), each with its own manifests. The
    DeepSeek-V4-Flash HumanEval routing run was captured with CUDA graphs
    disabled (`eager`); its SPEED-Bench runs with EP2 + attention DP
    (`ep2dpa`).

```
qwen3.6-35b-a3b/
  speedbench/{mtp, dflash@42d3b34d}/acceptance/
  speedbench/mtp/routing/<category>/
  humaneval/{mtp, dflash@42d3b34d}/{acceptance, routing}/
deepseek-v4-flash/
  humaneval/mtp/routing/
  speedbench/mtp/routing/<category>/
```

Each leaf is a calibration run directory: the materialized parquet banks and
JSON sidecars (`run_manifest.json`, `stats.json`, `metainfo.json`). Unlike the
superseded dataset, internal `parts/` checkpoint shards are not included — the
top-level parquets are the full union.

## Schemas

Banks share the key `(model, speculator, config, category, prompt_idx, turn,
round_idx)` — one row per draft round, so they JOIN.

**acceptance.parquet** — verify side: `… , accept, acc0..acc{D-1}`.
`accept` is the committed draft-token count (excludes the bonus); `acc_k` is the
per-position accept mask (1/0/null). Acceptance is a contiguous prefix, so
`acc_k = 1 iff k < accept`. Shallower drafters null-pad trailing columns.

**speculator.parquet** — draft side: `… , conf0..conf{D-1}`. `conf_k` is the
drafter's softmax probability of the token proposed at depth `k` (null where no
token proposed).

**routing_meta.parquet + routing.npy** (routing experiments) — paired and
aligned row-for-row by `routing_idx`. `routing.npy` is shape `(N, L, k)` `uint8`
(N routing positions × L layers × k routed experts per token; L=40, k=8 for
Qwen3.6-35B-A3B) holding the expert IDs; `routing_meta.parquet` indexes it
(`… , routing_idx, routing_block_idx, request_idx, position, accepted`).

Sampling is `temperature: 0.6`. See `run_manifest.json` in each leaf for the
exact configuration hashes.

## Usage

Download everything, or just the slice you need:

```bash
# the lightweight acceptance banks (a few MB):
hf download Doubleword/specdec-calibration --repo-type dataset \
  --include "qwen3.6-35b-a3b/speedbench/*/acceptance/acceptance.parquet" --local-dir data/

# everything:
hf download Doubleword/specdec-calibration --repo-type dataset \
  --local-dir data/
```

To turn a run directory into the simulator's trace-bank CSV, use the
`export-trace` command shipped with inference-lab's `calibration/` package
(`specdec-calibrate export-trace --run-dir <leaf> --signal oracle -o <out>.csv`).
The simulator's homogeneous policy uses only the committed count; the per-depth
mask and confidence feed the gated policies.
