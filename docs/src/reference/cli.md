# CLI Reference

Command-line interface reference for Inference Lab.

## Usage

```bash
inference-lab [OPTIONS]
```

A binary built with `--features serve` (the Docker image) has subcommands
instead: `inference-lab sim [OPTIONS]` takes the options below and
`inference-lab serve` starts the OpenAI-compatible server. `serve` takes
`--config` (a model config or a directory of them), `--hardware` (models
without that entry are skipped) and an optional `--workload`, whose
`output_len_dist` samples each response's length; without one responses run
to their `max_tokens`.

## Options

### Configuration

**`-c, --config <PATH>`**

Model config file (`configs/<name>.toml`).

- Default: `config.toml`

**`--hardware <NAME>`**

Which `[hardware.<name>]` entry of the model config to run. Optional when
the file has exactly one entry.

**`-w, --workload <PATH>`**

Workload file (`workloads/<name>.toml`). Required for `sim`.

```bash
inference-lab -c configs/gpt-oss-120b.toml --hardware gh200-120 -w workloads/quick.toml
```

### Dataset Mode

**`-t, --tokenizer <PATH>`**

Path to tokenizer file (required for dataset mode).

- Required when using `dataset_path` in configuration
- Example: `inference-lab -c configs/llama-3-70b.toml -w workloads/quick.toml --tokenizer tokenizer.json`

**`--chat-template <TEMPLATE>`**

Chat template for formatting messages in dataset mode.

- Required when using datasets
- Use `"None"` for simple message concatenation (no template)
- Example: `inference-lab -c configs/llama-3-70b.toml -w workloads/dataset-poisson.toml --tokenizer tokenizer.json --chat-template None`
- Example with template: `inference-lab ... --tokenizer tokenizer.json --chat-template "{{system}}\n{{user}}\n{{assistant}}"`

### Output Options

**`-o, --output <PATH>`**

Path to output JSON file for results.

- If not specified, results are only displayed to console
- Example: `inference-lab -c configs/llama-3-70b.toml -w workloads/quick.toml -o results.json`

**`-q, --quiet`**

Suppress progress output (only show final results).

- Example: `inference-lab -c configs/llama-3-70b.toml -w workloads/quick.toml -q`

**`-v, --verbose`**

Enable verbose output.

- Example: `inference-lab -c configs/llama-3-70b.toml -w workloads/quick.toml -v`

**`--debug`**

Enable debug logging.

- Example: `inference-lab -c configs/llama-3-70b.toml -w workloads/quick.toml --debug`

**`--no-color`**

Disable colored output.

- Useful for logging to files or CI environments
- Example: `inference-lab -c configs/llama-3-70b.toml -w workloads/quick.toml --no-color`

### Simulation Options

**`--seed <NUMBER>`**

Override the random seed from configuration.

- Useful for reproducible runs with different seeds
- Example: `inference-lab -c configs/llama-3-70b.toml -w workloads/quick.toml --seed 12345`

## Examples

### Basic Simulation

```bash
inference-lab -c configs/llama-3-70b.toml -w workloads/quick.toml
```

### Dataset Mode

```bash
inference-lab -c configs/llama-3-70b.toml -w workloads/quick.toml \
  --tokenizer tokenizer.json \
  --chat-template None
```

### Save Results to File

```bash
inference-lab -c configs/llama-3-70b.toml -w workloads/quick.toml -o results.json
```

### Quiet Mode with Output

```bash
inference-lab -c configs/llama-3-70b.toml -w workloads/quick.toml -q -o results.json
```

### Multiple Runs with Different Seeds

```bash
for seed in 42 43 44; do
  inference-lab -c configs/llama-3-70b.toml -w workloads/quick.toml --seed $seed -o results_$seed.json
done
```

## Exit Codes

- `0` - Simulation completed successfully
- `1` - Error occurred (configuration error, file not found, etc.)
