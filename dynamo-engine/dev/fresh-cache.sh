#!/usr/bin/env bash
# Start one extra copy of a worker service with an empty Hugging Face cache and
# metadata fetching on; print its fetch and registration lines.
# Usage: fresh-cache.sh <service> [seconds]
service=$1
wait=${2:-60}
here=$(dirname "$0")
token=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)
docker compose -f "$here/compose.yaml" run -d --name "ildyn-fresh-$service" \
  -e HF_HOME=/tmp/empty-hf -e INFERENCE_LAB_FETCH_METADATA=1 -e HF_TOKEN="$token" \
  "$service" >/dev/null
sleep "$wait"
docker logs "ildyn-fresh-$service" 2>&1 | grep -E 'metadata_fetched|engine_start|Registered base model|Error|Traceback' | grep -v 'omni\|error_response' | cut -c1-240
docker rm -f "ildyn-fresh-$service" >/dev/null
