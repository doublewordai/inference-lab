"""specdec-calibration: drive SGLang to emit acceptance + speculator parquet banks.

Two parquet outputs per (target model, speculator) run, both keyed by
(model, speculator, config, category, prompt_idx, turn, round_idx):

  acceptance.parquet  -- verify side: per-round `accept` + per-position `acc*` mask
  speculator.parquet  -- draft side:  per-position drafter confidence `conf*`

See README.md for the SGLang pin and the wrapped internal symbols.
"""

from .config import RunConfig, Speculator

__all__ = ["RunConfig", "Speculator"]
__version__ = "0.1.0"

# SGLang is pinned to an exact commit because the speculative `*_v2` internals we
# hook churn across patch releases (v0.5.13.post1 already rewrote the EAGLE worker
# within the minor version). Bumping this is a deliberate, contained edit.
SGLANG_COMMIT = "85fd90072d1a9f2432842b03588f63b745e524e4"  # tag v0.5.13.post1
SGLANG_VERSION = "0.5.13.post1"
