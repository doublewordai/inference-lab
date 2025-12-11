# CLI Reference

Command-line interface reference for Inference Lab.

## Usage

```bash
inference-lab [OPTIONS]
```

## Options

### Required Options

None - all options have defaults or are optional.

### Configuration

**`-c, --config <PATH>`**

Path to the TOML configuration file.

- Default: `config.toml`
- Example: `inference-lab -c my-config.toml`

### Dataset Mode

**`-t, --tokenizer <PATH>`**

Path to tokenizer file (required for dataset mode).

- Required when using `dataset_path` in configuration
- Example: `inference-lab -c config.toml --tokenizer tokenizer.json`

**`--chat-template <TEMPLATE>`**

Chat template for formatting messages in dataset mode.

- Required when using datasets
- Use `"None"` for simple message concatenation (no template)
- Example: `inference-lab --tokenizer tokenizer.json --chat-template None`
- Example with template: `inference-lab --tokenizer tokenizer.json --chat-template "{{system}}\n{{user}}\n{{assistant}}"`

### Output Options

**`-o, --output <PATH>`**

Path to output JSON file for results.

- If not specified, results are only displayed to console
- Example: `inference-lab -c config.toml -o results.json`

**`-q, --quiet`**

Suppress progress output (only show final results).

- Example: `inference-lab -c config.toml -q`

**`-v, --verbose`**

Enable verbose output.

- Example: `inference-lab -c config.toml -v`

**`--debug`**

Enable debug logging.

- Example: `inference-lab -c config.toml --debug`

**`--no-color`**

Disable colored output.

- Useful for logging to files or CI environments
- Example: `inference-lab -c config.toml --no-color`

### Simulation Options

**`--seed <NUMBER>`**

Override the random seed from configuration.

- Useful for reproducible runs with different seeds
- Example: `inference-lab -c config.toml --seed 12345`

## Examples

### Basic Simulation

```bash
inference-lab -c config.toml
```

### Dataset Mode

```bash
inference-lab -c config.toml \
  --tokenizer tokenizer.json \
  --chat-template None
```

### Save Results to File

```bash
inference-lab -c config.toml -o results.json
```

### Quiet Mode with Output

```bash
inference-lab -c config.toml -q -o results.json
```

### Multiple Runs with Different Seeds

```bash
for seed in 42 43 44; do
  inference-lab -c config.toml --seed $seed -o results_$seed.json
done
```

## Exit Codes

- `0` - Simulation completed successfully
- `1` - Error occurred (configuration error, file not found, etc.)
