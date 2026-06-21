"""Modal runner. Builds a GPU image with SGLang pinned to SGLANG_COMMIT, runs the
collection remotely, and writes the parquet/JSON artifacts back on the local side.

Imported lazily (only on `--modal`) so the base package needs neither modal nor a
GPU. The SGLang install is the first thing to validate on a real run -- it is
version-coupled (flashinfer wheel index, torch backend) and cannot be checked
without launching Modal.
"""

from __future__ import annotations

from pathlib import Path

import modal

from . import SGLANG_COMMIT, SGLANG_VERSION
from .config import RunConfig

# .../calibration (package root, holds pyproject.toml)
_CALIB_DIR = Path(__file__).resolve().parents[2]

# --- image -----------------------------------------------------------------
# Base off SGLang's official prebuilt image for the pinned version (tag
# v0.5.13.post1 == SGLANG_COMMIT). The cu130 build matches SGLang's
# cuda-python>=13 requirement; reconstructing the install from nvidia/cuda + uv
# hits an unsatisfiable torch/cuda-bindings conflict. We only add our small CPU
# deps on top and never touch torch/transformers/sglang.
image = (
    modal.Image.from_registry(f"lmsysorg/sglang:v{SGLANG_VERSION}-cu130")
    .entrypoint([])  # clear the sglang server ENTRYPOINT so Modal runs our function
    # `datasets` so prompts are built IN-CONTAINER (the long-context SPEED-Bench
    # configs are huge -- building locally + shipping prompts over pickle is infeasible).
    .pip_install("pyarrow>=15", "pyyaml>=6", "hf_transfer", "datasets>=2.18")
    # Mount the HF cache volume at a fresh path (the sglang image already populates
    # /root/.cache/huggingface, which Modal refuses to mount over) and point HF there.
    # SGLANG_ENABLE_SPEC_V2=1 selects the v2 speculative workers (eagle_worker_v2,
    # eagle_info_v2, ...) -- the exact symbols our hooks patch -- so it is required
    # for capture to fire, independent of the model.
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache/huggingface",
            "SGLANG_ENABLE_SPEC_V2": "1",
        }
    )
    # pip-install (not just mount) so the `sglang.srt.plugins` entry point is
    # registered and discoverable in the spawned scheduler child. --no-deps: torch/
    # transformers/sglang already in the base; pyarrow/pyyaml added above.
    .add_local_dir(
        str(_CALIB_DIR),
        "/pkg",
        copy=True,
        ignore=["**/.venv/**", "**/__pycache__/**", "sgl_src/**", "data/**",
                "**/*.pyc", "**/.git/**", "**/.pytest_cache/**", "**/*.egg-info/**"],
    )
    .run_commands("pip install --no-deps /pkg")
)

app = modal.App("specdec-calibration")
hf_cache = modal.Volume.from_name("specdec-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("specdec-outputs", create_if_missing=True)


@app.function(
    image=image,
    gpu="H200",
    volumes={"/cache/huggingface": hf_cache, "/out": out_vol},
    timeout=60 * 60 * 12,
)
def _collect_remote(cfg_dict: dict, with_hooks: bool) -> dict:
    """Build prompts, run the collection, and write ALL outputs to the /out volume
    in-container (parquets can be GB once routing is on), then return only a manifest.
    The caller streams the files down -- nothing large crosses the pickle channel."""
    import os

    from specdec_calibration import collect
    from specdec_calibration.config import RunConfig as RC
    from specdec_calibration.datasets import build_prompts

    cfg = RC.from_dict(cfg_dict)
    prompts = build_prompts(
        cfg.dataset,
        cfg.dataset_configs,
        cfg.n_prompts,
        cfg.multiturn_only,
        cfg.dataset_split,
    )
    mt = sum(1 for p in prompts if p.is_multiturn)
    print(f"built {len(prompts)} prompts in-container ({mt} multi-turn)", flush=True)
    result = (
        collect.collect_banks(cfg, prompts)
        if with_hooks
        else {"metainfo": collect.run_metainfo(cfg, prompts)}
    )
    run = os.path.basename(cfg.out_dir.rstrip("/")) or "run"
    outdir = f"/out/{run}"
    collect.write_result(result, cfg, outdir)
    out_vol.commit()
    files = [(f, os.path.getsize(os.path.join(outdir, f))) for f in sorted(os.listdir(outdir))]
    return {"run": run, "files": files}


def launch(cfg: RunConfig, with_hooks: bool = True) -> dict:
    cfg_dict = cfg.to_dict()

    fn = _collect_remote
    if cfg.gpu and cfg.gpu != "H200":
        fn = _collect_remote.with_options(gpu=cfg.gpu)

    with modal.enable_output(), app.run():
        manifest = fn.remote(cfg_dict, with_hooks)

    # stream each output file down from the volume (routing.npy may be GB)
    from pathlib import Path

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run = manifest["run"]
    for fname, size in manifest["files"]:
        with open(out / fname, "wb") as f:
            for chunk in out_vol.read_file(f"{run}/{fname}"):
                f.write(chunk)
        print(f"downloaded {fname} ({size/1e6:.1f} MB)")
    return manifest
