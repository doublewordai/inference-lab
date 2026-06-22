# Reproducing the adaptive-speculation figures

How to regenerate the four figures in the
[*Adaptive speculative decoding*](https://fergusfinn.com/blog/adaptive-speculation/)
blog post from the calibration base data. Nothing data-shaped is committed here:
pull the base data from Hugging Face, generate the simulator banks, run the
example, copy the results into the blog.

## What reproduces what

| Figure (blog component)        | Generator / source                                  |
|--------------------------------|-----------------------------------------------------|
| `DrafterCrossover.tsx`         | analytic roofline in the component — **no data step**|
| `AcceptanceCurve.tsx`          | `accept_by_position` from each run's `stats.json`    |
| `ParetoFrontier.tsx`           | `drafterGrid.json` ← `examples/spec_drafter_grid.rs` |
| `PricingEnvelope.tsx`          | the same `drafterGrid.json` (arrays read off it)     |

The two charted drafters are the **MTP** head and the **DFlash** head
(checkpoint `@42d3b34d`).

## Prerequisites

- Rust toolchain (`cargo`) — runs the simulator example.
- Calibration Python env — `cd calibration && uv sync` — provides `export-trace`.
- `hf` (`pip install -U huggingface_hub`) — pulls the base data.

## Step 0 — pull the base data

The calibration corpus lives as an HF dataset mirroring the local `data/` layout
(schema documented in [`data/CLAUDE.md`](../../data/CLAUDE.md)):

```bash
hf download Doubleword/qwen3.6-specdec-calibration \
  --repo-type dataset --local-dir data/
```

The figures only need the SpeedBench **acceptance** banks for the two charted
drafters:

- MTP:     `data/qwen3.6-35b-a3b/speedbench/mtp/acceptance`
- DFlash:  `data/qwen3.6-35b-a3b/speedbench/dflash@42d3b34d/acceptance`

## Step 1 — export the simulator banks

The examples read a flat CSV bank (`commits,category,a0..a{D-1}`) from
`data/banks/`. Generate them from the run dirs with:

```bash
./examples/specdec/export_banks.sh
```

This writes `data/banks/{mtp,dflash}_speedbench_rounds.csv` (oracle accept mask,
used by `spec_drafter_grid`) and `..._conf_rounds.csv` (drafter confidence, used
by `spec_gating_ladder`). Under the hood it runs
`specdec-calibrate export-trace --run-dir <leaf> --signal oracle|confidence`.

## Step 2 — run the simulator → `drafterGrid.json`

```bash
cargo run --release --no-default-features --example spec_drafter_grid > drafterGrid.json
```

This sweeps the whole (γ × batch) goodput/TPOT surface for both drafters plus
the adaptive (priced-budget) envelope. It feeds two charts:

- **ParetoFrontier** imports `drafterGrid.json` directly.
- **PricingEnvelope** arrays come from the same JSON, per concurrency:
  `NOSPEC` = `nospec.goodput`, `*_BEST` = best `fixed[γ].goodput`,
  `*_ADP` = `adaptive.goodput`, `*_G` = `adaptive.gamma`.

## Step 3 — acceptance curve

`MTP_SURV` / `DF_SURV` in `AcceptanceCurve.tsx` are `accept_by_position`:

```bash
jq .accept_by_position data/qwen3.6-35b-a3b/speedbench/mtp/acceptance/stats.json
jq .accept_by_position "data/qwen3.6-35b-a3b/speedbench/dflash@42d3b34d/acceptance/stats.json"
```

## Step 4 — copy into the blog

```bash
cp drafterGrid.json ../fergusfinn-blog/src/components/specdec/drafterGrid.json
```

Then update `MTP_SURV` / `DF_SURV` (Step 3) and the `PricingEnvelope` arrays
(Step 2) in the blog components. `DrafterCrossover` needs nothing.

## Publishing the base data (maintainers)

The HF dataset is produced from the local `data/` tree by
[`./publish_dataset.sh`](./publish_dataset.sh), which uploads an explicit
allowlist of run directories plus [`dataset_card.md`](./dataset_card.md) as the
repo README. Requires `hf auth login` and write access to the
`Doubleword` org. This is an outward-facing publish — do it deliberately, not
as part of a repro run.
