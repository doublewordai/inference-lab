#!/usr/bin/env bash
# Generate the simulator trace banks for the adaptive-speculation figures from
# the calibration base data (pull it first; see README.md Step 0).
#
# Writes to data/banks/ (gitignored, and outside the published qwen3.6-... tree):
#   <drafter>_speedbench_rounds.csv   oracle accept-mask bank (spec_drafter_grid)
#   <drafter>_conf_rounds.csv         drafter-confidence bank (spec_gating_ladder)
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
src="$repo_root/data/qwen3.6-35b-a3b/speedbench"
out="$repo_root/data/banks"

if [[ ! -d "$src/mtp/acceptance" ]]; then
  echo "base data not found under $src" >&2
  echo "pull it first:  hf download Doubleword/specdec-calibration --repo-type dataset --local-dir data/" >&2
  exit 1
fi

mkdir -p "$out"
cd "$repo_root/calibration"

# drafter dir name -> output bank stem
gen() {
  local run_dir="$1" stem="$2"
  uv run specdec-calibrate export-trace --run-dir "$src/$run_dir/acceptance" \
    --signal oracle     -o "$out/${stem}_speedbench_rounds.csv" --no-metadata
  uv run specdec-calibrate export-trace --run-dir "$src/$run_dir/acceptance" \
    --signal confidence -o "$out/${stem}_conf_rounds.csv"       --no-metadata
}

gen "mtp"            "mtp"
gen "dflash@42d3b34d" "dflash"

echo "banks written to data/banks/"
echo "next:  cargo run --release --no-default-features --example spec_drafter_grid > drafterGrid.json"
