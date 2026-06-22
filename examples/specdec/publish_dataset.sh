#!/usr/bin/env bash
# Publish the calibration base data to the Hugging Face dataset.
# Run after `hf auth login` (needs write access to the Doubleword org).
#
# Uploads an explicit allowlist of run directories plus the dataset card. The
# allowlist is the source of truth for what goes public — only paths matching
# these include patterns are uploaded.
set -euo pipefail

REPO="Doubleword/qwen3.6-specdec-calibration"
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"

if ! hf auth whoami >/dev/null 2>&1; then
  echo "not logged in -- run: hf auth login" >&2
  exit 1
fi

cd "$repo_root"

# Run-directory banks. Patterns are relative to data/qwen3.6-35b-a3b/ and select
# only the two published drafters; nothing else can match.
hf upload "$REPO" data/qwen3.6-35b-a3b qwen3.6-35b-a3b --repo-type dataset --no-private \
  --include "*/mtp/*" \
  --include "*dflash@42d3b34d*" \
  --commit-message "Add MTP + DFlash acceptance/speculator/routing banks"

# Dataset card -> repo README.md.
hf upload "$REPO" examples/specdec/dataset_card.md README.md --repo-type dataset \
  --commit-message "Add dataset card"

echo "published $REPO"
