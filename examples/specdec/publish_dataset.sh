#!/usr/bin/env bash
# Publish the calibration base data to the Hugging Face dataset
# Doubleword/specdec-calibration (layout: <model>/<dataset>/<drafter>/<experiment>).
# Run after `hf auth login` (needs write access to the Doubleword org).
#
# Uploads an explicit allowlist of run directories plus the dataset card. The
# allowlist is the source of truth for what goes public — only paths named here
# are uploaded. Internal `parts/` checkpoint shards are excluded everywhere:
# the top-level parquets are the materialized union.
set -euo pipefail

REPO="Doubleword/specdec-calibration"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"

if ! hf auth whoami >/dev/null 2>&1; then
  echo "not logged in -- run: hf auth login" >&2
  exit 1
fi

cd "$repo_root"

up() { # up <local-dir> <path-in-repo> <message>
  hf upload "$REPO" "$1" "$2" --repo-type dataset \
    --exclude "parts/*" --exclude "*/parts/*" \
    --commit-message "$3"
}

# Qwen acceptance (+ HumanEval routing) banks: the two published drafters.
for ds in speedbench humaneval; do
  for drafter in mtp "dflash@42d3b34d"; do
    up "data/qwen3.6-35b-a3b/$ds/$drafter" "qwen3.6-35b-a3b/$ds/$drafter" \
      "Add qwen3.6-35b-a3b/$ds/$drafter banks"
  done
done

# Qwen SPEED-Bench per-category routing captures (one run dir per category).
# coding_clean supersedes the original coding run.
for d in data/qwen36_mtp15_speedbench_qualitative_*_routing; do
  name="$(basename "$d")"
  [[ "$name" == "qwen36_mtp15_speedbench_qualitative_coding_routing" ]] && continue
  cat="${name#qwen36_mtp15_speedbench_qualitative_}"
  cat="${cat%_routing}"
  cat="${cat%_clean}"
  up "$d" "qwen3.6-35b-a3b/speedbench/mtp/routing/$cat" \
    "Add qwen speedbench routing capture: $cat"
done

# DeepSeek-V4-Flash HumanEval routing (eager capture; native MTP head, which
# the SGLang manifest records as speculator "eagle").
up data/deepseek_v4_flash_humaneval_routing_eager deepseek-v4-flash/humaneval/mtp/routing \
  "Add deepseek-v4-flash humaneval routing capture"

# DeepSeek-V4-Flash SPEED-Bench per-category routing captures (ep2dpa runs).
for d in data/deepseek_v4_flash_speedbench_qualitative_*_ep2dpa; do
  cat="$(basename "$d")"
  cat="${cat#deepseek_v4_flash_speedbench_qualitative_}"
  cat="${cat%_ep2dpa}"
  up "$d" "deepseek-v4-flash/speedbench/mtp/routing/$cat" \
    "Add deepseek-v4-flash speedbench routing capture: $cat"
done

# Dataset card -> repo README.md.
hf upload "$REPO" examples/specdec/dataset_card.md README.md --repo-type dataset \
  --commit-message "Add dataset card"

echo "published $REPO"
