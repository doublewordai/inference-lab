# Simulated engine for Dynamo workers

Runs production [Dynamo](https://github.com/ai-dynamo/dynamo) worker images
without GPUs. The worker process is unchanged — `python3 -m dynamo.sglang` or
`python3 -m dynamo.vllm` parses its production arguments, registers its model
card, handles requests and reports load exactly as it does on a GPU — and only
the engine object that would load weights is replaced. The replacement returns
junk tokens (or scripted text) and logs every request it receives.

Everything upstream of the engine is therefore real: the frontend's chat
templating, tokenisation, routing and parsers, and the worker's translation of
each request into engine arguments.

## Switching it on

The `inference-lab-dynamo` chart (`charts/inference-lab-dynamo`) installs a
namespace-scoped `MutatingAdmissionPolicy` and the adapter as a ConfigMap.
Every Dynamo worker pod created in a bound namespace keeps its image, command
and placement, and changes only in:

- resources: GPU requests removed, small CPU/memory requests;
- two environment variables that load the adapter (`PYTHONPATH` points at the
  adapter zip; `INFERENCE_LAB_SIMULATE=1`), plus the GPU the pod was placed
  for, so the engine resolves the same defaults it would on that GPU;
- hostPath model caches become `emptyDir`, and `hostNetwork` is turned off.

```sh
helm install sim oci://ghcr.io/doublewordai/charts/inference-lab-dynamo \
  --namespace <worker namespace>
```

The adapter loads through `sitecustomize`, so it needs no change to the worker
command; any process other than a Dynamo worker is untouched.

## What it records

One JSON object per line on stdout, each with `"record":"inference_lab"`:

| `event` | Fields |
| --- | --- |
| `engine_start` | engine, served model name, data-parallel size, block/page size |
| `engine_request` | the full engine call as the worker made it (SGLang `async_generate` arguments, or vLLM prompt and `SamplingParams`), and the decoded prompt |
| `engine_response` | prompt and completion tokens, time to first token, end-to-end time, finish reason |

With Loki: `{namespace="<ns>"} |= "\"record\":\"inference_lab\"" | json`.

## Scripted output

A prompt containing a directive makes the engine emit scripted output and stop,
following the same contract as `inference-lab serve`:

```text
<<respond:{"reasoning": "...", "text": "...", "tool_calls": [{"name": "Read", "arguments": {"file_path": "/x"}}]}>>
```

The engine writes the output in the model's own format, so the frontend's
parsers do the real work: reasoning goes in a thinking block (closing one the
chat template has opened), and tool calls are rendered for the worker's
`--dyn-tool-call-parser` (`hermes`, `qwen25`, `qwen3_coder`, `glm47`). The last
well-formed directive in the prompt wins, so a chained agent loop advances on
the directive carried by its newest tool result.

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `SIM_ITL_S` | `0.02` | seconds per generated token |
| `SIM_DEFAULT_MAX_TOKENS` | `16` | tokens generated when a request sets no limit |
| `SIM_MAX_TOTAL_NUM_TOKENS` | `2055872` | SGLang KV capacity per data-parallel rank |
| `SIM_NUM_GPU_BLOCKS` | `74403` | vLLM KV blocks |
| `INFERENCE_LAB_FETCH_METADATA` | `1` in the chart | fetch config/tokenizer files (never weights) at start-up |

## Local check

`dev/compose.yaml` runs etcd, NATS, a Dynamo frontend and SGLang and vLLM
workers on CPU. Put `FRONTEND_IMAGE`, `SGLANG_WORKER_IMAGE` and
`VLLM_WORKER_IMAGE` in an env file (`HARNESS_ENV_FILE` for the scripts), the
models' metadata in `dev/hf`, run `python3 dynamo-engine/build-zip.py`, then
`docker compose --env-file <file> -f dynamo-engine/dev/compose.yaml up`. The
`dev/*_check.py` scripts and `dev/probe.py` exercise request shapes, choices
and tool-call parsing through the frontend; `dev/card_diff.py` compares a
registered model card with a production one.

Tests: `python3 -m unittest discover -s dynamo-engine/tests`.
