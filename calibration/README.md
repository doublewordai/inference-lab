# specdec-calibration

Drive **SGLang** to emit speculative-decoding **acceptance** and **speculator**
parquet banks for any supported (target model, speculator) combination, on Modal.

## Outputs

Two parquet files per run, both keyed by
`(model, speculator, config, category, prompt_idx, turn, round_idx)` — one row
per draft round, so they JOIN:

- **`acceptance.parquet`** — verify side: `accept` (committed draft tokens, excl. the
  bonus) + per-position accept mask `acc0..acc{D-1}` (1/0/null).
- **`speculator.parquet`** — draft side: `conf0..conf{D-1}`, the drafter's softmax
  probability of each proposed token (recomputed; SGLang discards it internally).
- **`stats.json`** — accept histogram, mean commits, accept-by-position, by config/category.
- **`metainfo.json`** — SGLang's own public `meta_info` per prompt (used for the
  reconciliation gate below).

`D` is the measured draft-token width (EAGLE: `num_steps`; DFLASH:
`dflash_block_size - 1`, because SGLang's DFLASH block also includes the
bonus/root verify position). Shallower speculators null-pad trailing columns so
banks union cleanly when written to a wider schema.

## Usage

```bash
cd inference-lab/calibration
uv sync                                                              # CPU-only base install
uv run specdec-calibrate run --config configs/qwen36-mtp.yaml --dry-run   # no GPU
uv run specdec-calibrate run --config configs/qwen36-mtp.yaml --modal     # real run on Modal
uv run specdec-calibrate export-trace --run-dir ../data/qwen36_mtp_bench \
  --signal confidence --output ../data/qwen36_mtp_conf_rounds.csv
```

Flags: `--modal/--local`, `--hooks/--no-hooks` (meta_info only), `--out-dir`, `--dry-run`.

Prompt sources:

- `dataset: smoke` uses the built-in smoke prompts.
- `dataset: speedbench` loads `nvidia/SPEED-Bench`; `dataset_configs` selects
  SPEED-Bench configs and `dataset_split` selects the Hugging Face split
  (`test` by default).

Sampling defaults are `temperature: 0.6` and `seed: 0`. `max_tokens` is unset by
default; set it in a run YAML when you want a bounded calibration run.

### Simulator export

The Rust simulator does not read parquet directly. Its stable empirical acceptance
input is a `TraceRounds` CSV:

```csv
commits,category,a0,a1,...,a{D-1}
```

`commits` is the realised number of accepted draft tokens. The `a*` columns are
the policy signal:

- `--signal confidence` writes `a_k = conf{k}` for realizable draft-time gating.
- `--signal oracle` writes `a_k = acc{k}` (or derives `accept > k`) for
  perfect-foresight comparisons.

Use `export-trace` on either a run directory containing `acceptance.parquet` and
`speculator.parquet`, explicit `--acceptance/--speculator` paths, or an older
combined `--raw` parquet.

### Example run matrix (Qwen3.6-35B-A3B)

| config | speculator |
|---|---|
| `qwen36-mtp.yaml` | shipped MTP (NEXTN path, no draft repo) |
| `qwen36-dflash-before.yaml` | DFLASH draft, pre Modal-retrain (`42d3b34d`, 2026-04-26) |
| `qwen36-dflash-after.yaml` | DFLASH draft, post Modal-retrain (`0ce151d0`, 2026-06-19) |

The `speculator` column encodes the draft + revision (e.g. `dflash:…-DFlash@0ce151d0`),
so before/after banks stay distinct.

## How it works (and the pin)

SGLang's public `meta_info` only exposes aggregate accept-length / a per-step
accept-count histogram — not per-position accept masks or per-token draft confidence.
To reproduce the `accept + conf0..confN` schema we **monkeypatch SGLang internals**,
pinned to an exact commit (`SGLANG_COMMIT` in `src/specdec_calibration/__init__.py`,
= tag `v0.5.13.post1`) because the speculative `*_v2` internals churn across releases.

Wrapped symbols (all behavior-preserving — capture is best-effort, never raises into
the engine):

| family | accept side | confidence side |
|---|---|---|
| EAGLE / EAGLE3 / MTP / STANDALONE | `EagleVerifyInputV2Mixin.sample` | `EagleDraftWorker.draft_forward` (tees per-step draft logits) |
| DFLASH | `DFlashVerifyInput.verify` | `DFlashWorker._greedy_sample_from_vocab_parallel_head` |

Runs are driven in turn-synchronized batches. For turn `t`, every conversation
that has a user turn `t` is generated with its accumulated chat history, and the
request id is set to `"<prompt_idx>:<turn>"`, so each captured round attributes
cleanly to a prompt and conversation turn. `eagle_topk=1` is still assumed by the
accept-mask interpretation.

**Reconciliation gate.** After each run, the hook-derived mean accept length is checked
against SGLang's own `meta_info.spec_accept_length` (engine-version-independent). A
`[MISMATCH]` line means the hook drifted from the pinned source and needs attention.

### Known v1 limitations

- EAGLE confidence capture assumes `eagle_topk=1`. Draft position 0 is captured
  from the draft-extend logits; positions 1..D-1 are captured from `draft_forward`.
- DFLASH confidence requires the **TP=1 / no-added-vocab** fast path (single-GPU
  offline Engine); otherwise confidence is skipped and only acceptance is captured.

`sgl_src/` (gitignored) holds the pinned-commit SGLang source used to author the hooks.
