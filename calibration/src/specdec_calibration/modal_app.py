"""Modal runner. Builds a GPU image with SGLang pinned to SGLANG_COMMIT, runs the
collection remotely, and writes the parquet/JSON artifacts back on the local side.

Imported lazily (only on `--modal`) so the base package needs neither modal nor a
GPU. The SGLang install is the first thing to validate on a real run -- it is
version-coupled (flashinfer wheel index, torch backend) and cannot be checked
without launching Modal.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

from . import SGLANG_COMMIT, SGLANG_VERSION
from .config import RunConfig

# .../calibration (package root, holds pyproject.toml)
_CALIB_DIR = Path(__file__).resolve().parents[2]
_SGLANG_IMAGE_TAG = os.environ.get("SPECDEC_SGLANG_IMAGE_TAG", f"v{SGLANG_VERSION}-cu130")
_IMAGE_ENV = {
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_HOME": "/cache/huggingface",
    "SGLANG_ENABLE_SPEC_V2": "1",
}
if os.environ.get("SPECDEC_ENABLE_OVERLAP_PLAN_STREAM"):
    _IMAGE_ENV["SGLANG_ENABLE_OVERLAP_PLAN_STREAM"] = "1"

# --- image -----------------------------------------------------------------
# Base off SGLang's official prebuilt image for the pinned version (tag
# v0.5.13.post1 == SGLANG_COMMIT). The cu130 build matches SGLang's
# cuda-python>=13 requirement; reconstructing the install from nvidia/cuda + uv
# hits an unsatisfiable torch/cuda-bindings conflict. We only add our small CPU
# deps on top and never touch torch/transformers/sglang.
image = (
    modal.Image.from_registry(f"lmsysorg/sglang:{_SGLANG_IMAGE_TAG}")
    .entrypoint([])  # clear the sglang server ENTRYPOINT so Modal runs our function
    # `datasets` so prompts are built IN-CONTAINER (the long-context SPEED-Bench
    # configs are huge -- building locally + shipping prompts over pickle is infeasible).
    .pip_install("pyarrow>=15", "pyyaml>=6", "hf_transfer", "datasets>=2.18")
    # Mount the HF cache volume at a fresh path (the sglang image already populates
    # /root/.cache/huggingface, which Modal refuses to mount over) and point HF there.
    # SGLANG_ENABLE_SPEC_V2=1 selects the v2 speculative workers (eagle_worker_v2,
    # eagle_info_v2, ...) -- the exact symbols our hooks patch -- so it is required
    # for capture to fire, independent of the model.
    .env(_IMAGE_ENV)
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


def _remote_context(cfg_dict: dict):
    import os

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
    run = os.path.basename(cfg.out_dir.rstrip("/")) or "run"
    cfg.out_dir = f"/out/{run}"
    return cfg, prompts, run


@app.function(
    image=image,
    volumes={"/cache/huggingface": hf_cache, "/out": out_vol},
    timeout=60 * 60,
)
def _prepare_remote_run(cfg_dict: dict, with_hooks: bool, resume: bool) -> dict:
    from pathlib import Path

    from specdec_calibration import collect

    cfg, prompts, run = _remote_context(cfg_dict)
    mt = sum(1 for p in prompts if p.is_multiturn)
    print(f"built {len(prompts)} prompts in-container ({mt} multi-turn)", flush=True)
    out = Path(cfg.out_dir)
    collect._prepare_checkpoint_run(out, cfg, prompts, with_hooks, resume)
    completed = collect._completed_parts(out)
    parts = collect.checkpoint_parts(cfg, prompts)
    pending = [
        {"part_idx": p.part_idx, "start": p.start, "end": p.end}
        for p in parts
        if p.part_idx not in completed
    ]
    out_vol.commit()
    print(
        f"prepared {len(parts)} part(s), {len(completed)} complete, "
        f"{len(pending)} pending",
        flush=True,
    )
    return {
        "run": run,
        "n_parts": len(parts),
        "pending": pending,
        "files": _remote_selected_files(
            cfg.out_dir,
            ["run_manifest.json", "completed_parts.jsonl", "run_complete.json"],
        ),
    }


@app.function(
    image=image,
    gpu="H200",
    volumes={"/cache/huggingface": hf_cache, "/out": out_vol},
    timeout=60 * 60 * 12,
)
def _collect_part_remote(
    cfg_dict: dict,
    with_hooks: bool,
    part_idx: int,
    start: int,
    end: int,
) -> dict:
    from pathlib import Path

    from specdec_calibration import collect

    cfg, prompts, run = _remote_context(cfg_dict)
    out = Path(cfg.out_dir)
    collect._prepare_checkpoint_run(out, cfg, prompts, with_hooks, resume=True)
    part = collect.CheckpointPart(part_idx=part_idx, start=start, end=end)
    manifest = collect.run_checkpoint_part(
        cfg,
        prompts,
        part,
        with_hooks=with_hooks,
        append_completed=False,
    )
    out_vol.commit()
    part_name = f"part-{part_idx:06d}"
    return {
        **manifest,
        "run": run,
        "files": _remote_files(str(out / "parts" / part_name), f"parts/{part_name}"),
    }


@app.function(
    image=image,
    volumes={"/cache/huggingface": hf_cache, "/out": out_vol},
    timeout=60 * 60 * 3,
)
def _finalize_remote_run(cfg_dict: dict, with_hooks: bool, resume: bool) -> dict:
    from specdec_calibration import collect

    cfg, prompts, run = _remote_context(cfg_dict)
    collect.run_checkpointed(cfg, prompts, with_hooks=with_hooks, resume=resume)
    out_vol.commit()
    return _remote_file_manifest(run, cfg.out_dir)


def _remote_file_manifest(run: str, outdir: str) -> dict:
    return {"run": run, "files": _remote_files(outdir)}


def _remote_files(root_dir: str, prefix: str = "") -> list[tuple[str, int]]:
    import os

    files = []
    for root, _dirs, names in os.walk(root_dir):
        for name in sorted(names):
            path = os.path.join(root, name)
            rel = os.path.relpath(path, root_dir)
            if prefix:
                rel = os.path.join(prefix, rel)
            files.append((rel, os.path.getsize(path)))
    files.sort()
    return files


def _remote_selected_files(root_dir: str, names: list[str]) -> list[tuple[str, int]]:
    import os

    files = []
    for name in names:
        path = os.path.join(root_dir, name)
        if os.path.isfile(path):
            files.append((name, os.path.getsize(path)))
    return files


def launch(cfg: RunConfig, with_hooks: bool = True, resume: bool = True) -> dict:
    cfg_dict = cfg.to_dict()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = None
    with modal.enable_output(), app.run():
        plan = _prepare_remote_run.remote(cfg_dict, with_hooks, resume)
        _download_files(plan["run"], plan.get("files") or [], out)
        pending = plan["pending"]
        print(
            f"modal checkpoint parallelism={cfg.checkpoint_parallelism}; "
            f"{len(pending)}/{plan['n_parts']} part(s) pending"
        )
        part_fn = _collect_part_remote.with_options(
            gpu=cfg.gpu or "H200",
            max_containers=cfg.checkpoint_parallelism,
        )
        args = [
            (cfg_dict, with_hooks, p["part_idx"], p["start"], p["end"])
            for p in pending
        ]
        for part_manifest in part_fn.starmap(args, order_outputs=False):
            print(
                "completed remote part "
                f"{part_manifest['part_idx']:06d} "
                f"prompts {part_manifest['start']}:{part_manifest['end']}"
            )
            _download_files(part_manifest["run"], part_manifest["files"], out)
        manifest = _finalize_remote_run.remote(cfg_dict, with_hooks, resume)
    if manifest is None:
        raise RuntimeError("Modal run did not return a manifest")

    _download_files(manifest["run"], manifest["files"], out)
    return manifest


def _download_files(run: str, files: list[tuple[str, int]], out: Path) -> None:
    for fname, size in files:
        target = out / fname
        if target.exists() and target.stat().st_size == size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            for chunk in out_vol.read_file(f"{run}/{fname}"):
                f.write(chunk)
        print(f"downloaded {fname} ({size/1e6:.1f} MB)")
