"""CLI: `specdec-calibrate run --config cfg.yaml [--dry-run|--modal]`.

--dry-run validates the config and builds prompts with no GPU and no SGLang.
--modal launches the run on Modal (default). Without --modal it runs locally,
which requires SGLang installed on a CUDA box.
"""

from __future__ import annotations

import typer

from .config import RunConfig
from .datasets import build_prompts
from .export import TraceSignal, export_trace_bank

app = typer.Typer(add_completion=False, help="SGLang acceptance/speculator parquet collator.")


@app.callback()
def _main() -> None:
    """SGLang acceptance/speculator parquet collator."""


@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c", help="Path to a run YAML."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate + build prompts, no GPU."),
    modal: bool = typer.Option(True, "--modal/--local", help="Run on Modal (default) or locally."),
    hooks: bool = typer.Option(True, "--hooks/--no-hooks", help="Capture banks (on) vs meta_info only."),
    out_dir: str = typer.Option(None, "--out-dir", help="Override config out_dir."),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume completed checkpoint parts."),
    checkpoint_batches: int | None = typer.Option(
        None,
        "--checkpoint-batches",
        help="Durable output shard size in checkpoint batches.",
    ),
    checkpoint_batch_size: int | None = typer.Option(
        None,
        "--checkpoint-batch-size",
        help="Calibration-only prompt batch multiple for output shards.",
    ),
    checkpoint_parallelism: int | None = typer.Option(
        None,
        "--checkpoint-parallelism",
        help="Bounded Modal shard workers for checkpointed runs.",
    ),
) -> None:
    cfg = RunConfig.from_yaml(config)
    if out_dir:
        cfg.out_dir = out_dir
    if checkpoint_batches is not None:
        cfg.checkpoint_batches = checkpoint_batches
    if checkpoint_batch_size is not None:
        cfg.checkpoint_batch_size = checkpoint_batch_size
    if checkpoint_parallelism is not None:
        cfg.checkpoint_parallelism = checkpoint_parallelism
    cfg.validate()

    typer.echo(
        f"model={cfg.target_model}  speculator={cfg.speculator_label} "
        f"({cfg.speculator.sglang_algorithm}, width={cfg.speculator.column_width})  "
        f"dataset={cfg.dataset}  routing={cfg.capture_routing}  "
        f"checkpoint={cfg.checkpoint_batch_size}x{cfg.checkpoint_batches}  "
        f"parallelism={cfg.checkpoint_parallelism}"
    )

    if modal and not dry_run:
        # prompts are built in-container (avoids downloading long-context configs locally)
        from .modal_app import launch

        launch(cfg, with_hooks=hooks, resume=resume)
        return

    # local path (and --dry-run) build prompts here
    prompts = build_prompts(
        cfg.dataset,
        cfg.dataset_configs,
        cfg.n_prompts,
        cfg.multiturn_only,
        cfg.dataset_split,
    )
    typer.echo(f"  built {len(prompts)} prompts")
    if dry_run:
        for p in prompts[:8]:
            typer.echo(f"  [{p.config}/{p.category}] {p.content[:64]}")
        if len(prompts) > 8:
            typer.echo(f"  ... (+{len(prompts) - 8} more)")
        raise typer.Exit()

    from .collect import run_local

    run_local(cfg, prompts, with_hooks=hooks, resume=resume)


@app.command("export-trace")
def export_trace(
    output: str = typer.Option(..., "--output", "-o", help="Simulator trace CSV to write."),
    signal: TraceSignal = typer.Option(
        TraceSignal.CONFIDENCE,
        "--signal",
        help="'confidence' uses conf* as the policy signal; 'oracle' uses accept masks.",
    ),
    run_dir: str | None = typer.Option(
        None,
        "--run-dir",
        help="Calibration run directory containing acceptance.parquet/speculator.parquet.",
    ),
    acceptance: str | None = typer.Option(None, "--acceptance", help="acceptance.parquet path."),
    speculator: str | None = typer.Option(None, "--speculator", help="speculator.parquet path."),
    raw: str | None = typer.Option(
        None,
        "--raw",
        help="Older combined parquet containing accept plus conf* columns.",
    ),
    metadata: bool = typer.Option(True, "--metadata/--no-metadata", help="Write a .meta.json sidecar."),
    drop_trailing_null_signal: bool = typer.Option(
        True,
        "--drop-trailing-null-signal/--keep-trailing-null-signal",
        help="Drop all-null trailing signal columns, useful for older DFLASH raw banks.",
    ),
) -> None:
    """Export calibration parquet to the simulator's TraceRounds CSV format."""
    from pathlib import Path

    out = Path(output)
    meta = out.with_suffix(".meta.json") if metadata else None
    manifest = export_trace_bank(
        output_path=out,
        signal=signal,
        acceptance_path=acceptance,
        speculator_path=speculator,
        raw_path=raw,
        run_dir=run_dir,
        drop_trailing_null_signal=drop_trailing_null_signal,
        metadata_path=meta,
    )
    typer.echo(
        f"wrote {manifest.rows} rounds, depth={manifest.depth}, "
        f"signal={manifest.signal} -> {manifest.output}"
    )
    if metadata:
        typer.echo(f"wrote metadata -> {meta}")


if __name__ == "__main__":
    app()
