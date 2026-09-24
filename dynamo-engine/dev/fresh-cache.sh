#!/usr/bin/env bash
# Start one extra copy of a worker service with an empty Hugging Face cache,
# online, with metadata fetching on — as a production entry that does not set
# HF_HUB_OFFLINE — and show what it fetched and whether it registered.
# Usage: fresh-cache.sh <env file> <service> [seconds]
env_file=$1
service=$2
wait=${3:-60}
here=$(dirname "$0")
token=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)
docker compose --env-file "$env_file" -f "$here/compose.yaml" run -d --name "ildyn-fresh-$service" \
  -e HF_HOME=/tmp/empty-hf -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
  -e INFERENCE_LAB_FETCH_METADATA=1 -e HF_TOKEN="$token" "$service" >/dev/null
sleep "$wait"
docker logs "ildyn-fresh-$service" 2>&1 \
  | grep -E 'metadata_fetched|engine_start|Registered base model|Error|Traceback' \
  | grep -v 'omni\|error_response' | cut -c1-200
docker exec "ildyn-fresh-$service" du -sh /tmp/empty-hf
docker rm -f "ildyn-fresh-$service" >/dev/null
