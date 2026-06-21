"""Engine-driving core. Runs inside the GPU context (Modal remote or a local CUDA
box). Imports SGLang lazily so the rest of the package stays importable on a laptop.

This module owns both the meta-info-only path and the hook-fed bank path. Both use
the same turn-synchronized generation loop so dry/debug runs exercise the same
prompt histories as real captures.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import RunConfig
from .datasets import PromptSpec

# meta_info keys SGLang populates for any speculative run (read defensively: absent
# when spec_verify_ct == 0). See tokenizer_manager._calculate_spec_decoding_metrics.
SPEC_META_KEYS = [
    "spec_verify_ct",
    "spec_accept_length",
    "spec_accept_rate",
    "spec_num_correct_drafts",
    "spec_num_proposed_drafts",
]

_PIN_WARNED = False


@dataclass
class _Engine:
    engine: object
    tokenizer: object

    def shutdown(self) -> None:
        try:
            self.engine.shutdown()
        except Exception:
            pass


def build_engine(cfg: RunConfig) -> _Engine:
    import sglang as sgl
    from transformers import AutoTokenizer

    _warn_if_sglang_pin_mismatch(sgl)
    engine = sgl.Engine(**cfg.engine_kwargs())
    tok = AutoTokenizer.from_pretrained(cfg.target_model, trust_remote_code=True)
    return _Engine(engine=engine, tokenizer=tok)


def _warn_if_sglang_pin_mismatch(sgl) -> None:
    """Warn when the local SGLang package is not the version the hooks target."""
    global _PIN_WARNED
    if _PIN_WARNED:
        return
    _PIN_WARNED = True

    import importlib.metadata as metadata
    import sys

    from . import SGLANG_COMMIT, SGLANG_VERSION

    version = getattr(sgl, "__version__", None)
    if not version:
        try:
            version = metadata.version("sglang")
        except metadata.PackageNotFoundError:
            version = None
    if version != SGLANG_VERSION:
        print(
            "[specdec-calibration] warning: hooks target "
            f"SGLang {SGLANG_VERSION} ({SGLANG_COMMIT[:8]}), "
            f"but local package reports {version or 'unknown'}",
            file=sys.stderr,
            flush=True,
        )


def _templated_chat(tok, history: list[dict]) -> str:
    return tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)


def _sampling_params(cfg: RunConfig) -> dict:
    params = {
        "temperature": cfg.temperature,
        "seed": cfg.seed,
    }
    if cfg.max_tokens is not None:
        params["max_new_tokens"] = cfg.max_tokens
    return params


def run_metainfo(
    cfg: RunConfig,
    prompts: list[PromptSpec],
    prompt_offset: int = 0,
) -> list[dict]:
    """Per-request public meta_info rows. No hooks, same turn loop as captures."""
    eng = build_engine(cfg)
    sp = _sampling_params(cfg)
    rows: list[dict] = []
    try:
        histories: list[list[dict]] = [[] for _ in prompts]
        max_turns = max((len(p.turns) for p in prompts), default=0)
        for t in range(max_turns):
            idxs = [i for i, p in enumerate(prompts) if t < len(p.turns)]
            if not idxs:
                break
            texts = []
            for i in idxs:
                histories[i].append({"role": "user", "content": prompts[i].turns[t]})
                texts.append(_templated_chat(eng.tokenizer, histories[i]))
            outs = eng.engine.generate(prompt=texts, sampling_params=sp)
            if isinstance(outs, dict):
                outs = [outs]
            for i, out in zip(idxs, outs):
                prompt_idx = prompt_offset + i
                mi = out.get("meta_info", {}) or {}
                histories[i].append({"role": "assistant", "content": out.get("text", "")})
                row = {
                    "model": cfg.target_model,
                    "speculator": cfg.speculator_label,
                    "config": prompts[i].config,
                    "category": prompts[i].category,
                    "prompt_idx": prompt_idx,
                    "turn": t,
                    "question_id": prompts[i].question_id,
                    "completion_tokens": mi.get("completion_tokens"),
                }
                for k in SPEC_META_KEYS:
                    row[k] = mi.get(k)
                rows.append(row)
    finally:
        eng.shutdown()
    return rows


def collect_banks(
    cfg: RunConfig,
    prompts: list[PromptSpec],
    prompt_offset: int = 0,
) -> dict:
    """Hook-fed acceptance + speculator banks for the EAGLE/DFLASH families.

    SGLang runs the spec worker in a spawned child process, so the hooks are
    installed there via the `sglang.srt.plugins` entry point (plugin.py), gated on
    env vars we set here. The child appends JSONL capture rows to a shared file; we
    pass rid="<prompt_idx>:<turn>" per active conversation turn so each captured row
    attributes to its prompt and turn, then read the file back. Returns
    {"acceptance": [...], "speculator": [...], "metainfo": [...]}.
    """
    import os
    import tempfile
    from pathlib import Path

    env_keys = [
        "SPECDEC_CAPTURE_PATH",
        "SPECDEC_HOOK_FAMILY",
        "SPECDEC_ROUTING_BIN",
        "SPECDEC_ROUTING_META",
        "SPECDEC_NO_CONF",
    ]
    old_env = {k: os.environ.get(k) for k in env_keys}

    def _restore_env() -> None:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    with tempfile.TemporaryDirectory(prefix="specdec_capture_") as tmp:
        tmp_path = Path(tmp)
        path = tmp_path / "capture.jsonl"
        path.write_text("")
        os.environ["SPECDEC_CAPTURE_PATH"] = str(path)
        os.environ["SPECDEC_HOOK_FAMILY"] = cfg.speculator.family

        rbin = tmp_path / "routing.bin"
        rmeta = tmp_path / "routing_meta.jsonl"
        if cfg.capture_routing:
            rbin.write_bytes(b"")
            rmeta.write_text("")
            os.environ["SPECDEC_ROUTING_BIN"] = str(rbin)
            os.environ["SPECDEC_ROUTING_META"] = str(rmeta)
        else:
            os.environ.pop("SPECDEC_ROUTING_BIN", None)
            os.environ.pop("SPECDEC_ROUTING_META", None)
        # Confidence capture is CUDA-graph safe (pinned-buffer copy recorded into the
        # graph), so it works under graphs alongside correct acceptance -- no eager needed.
        if cfg.capture_conf:
            os.environ.pop("SPECDEC_NO_CONF", None)
        else:
            os.environ["SPECDEC_NO_CONF"] = "1"

        eng = None
        metainfo: list[dict] = []
        try:
            eng = build_engine(cfg)  # child spawns here, inherits env, plugin installs hooks
            sp = _sampling_params(cfg)
            try:
                # Turn-synchronized batched generation. Each conversation builds up its
                # history; at turn t we batch-generate for every conversation that has a turn
                # t, using its accumulated history. Single-turn prompts = one iteration (same
                # as a flat batch). Captures attribute by rid = "<prompt_idx>:<turn>".
                histories: list[list[dict]] = [[] for _ in prompts]
                max_turns = max((len(p.turns) for p in prompts), default=0)
                for t in range(max_turns):
                    idxs = [i for i, p in enumerate(prompts) if t < len(p.turns)]
                    if not idxs:
                        break
                    texts, rids = [], []
                    for i in idxs:
                        prompt_idx = prompt_offset + i
                        histories[i].append({"role": "user", "content": prompts[i].turns[t]})
                        texts.append(_templated_chat(eng.tokenizer, histories[i]))
                        rids.append(f"{prompt_idx}:{t}")
                    outs = eng.engine.generate(prompt=texts, sampling_params=sp, rid=rids)
                    if isinstance(outs, dict):
                        outs = [outs]
                    for i, out in zip(idxs, outs):
                        prompt_idx = prompt_offset + i
                        mi = out.get("meta_info", {}) or {}
                        histories[i].append({"role": "assistant", "content": out.get("text", "")})
                        row = {
                            "model": cfg.target_model,
                            "speculator": cfg.speculator_label,
                            "config": prompts[i].config,
                            "category": prompts[i].category,
                            "prompt_idx": prompt_idx,
                            "turn": t,
                            "question_id": prompts[i].question_id,
                            "completion_tokens": mi.get("completion_tokens"),
                        }
                        for k in SPEC_META_KEYS:
                            row[k] = mi.get(k)
                        metainfo.append(row)
                    lens = [o.get("meta_info", {}).get("spec_accept_length") for o in outs]
                    lens = [x for x in lens if x]
                    print(
                        f"  turn {t}: {len(idxs)} convs, "
                        f"mean accept_len={sum(lens)/len(lens):.3f}"
                        if lens
                        else f"  turn {t}: {len(idxs)} convs"
                    )
            finally:
                eng.shutdown()

            acceptance, speculator = _assemble_captures(str(path), cfg, prompts, prompt_offset)
            print(
                f"  captured {len(acceptance)} acceptance rows, "
                f"{len(speculator)} speculator rows"
            )
            _reconcile(metainfo, acceptance)
            result = {"acceptance": acceptance, "speculator": speculator, "metainfo": metainfo}
            if cfg.capture_routing:
                arr, meta = _assemble_routing(str(rbin), str(rmeta), cfg, prompts, prompt_offset)
                result["routing_npy"] = arr
                result["routing_meta"] = meta
                print(
                    f"  captured routing: {arr.shape if arr is not None else None} "
                    f"({len(meta)} position rows)"
                )
            return result
        finally:
            _restore_env()


def _assemble_routing(
    bin_path: str,
    meta_path: str,
    cfg: RunConfig,
    prompts: list[PromptSpec],
    prompt_offset: int = 0,
):
    """Expand the per-round routing blocks into a uint8 array [N_positions, L, S] and
    per-position metadata rows."""
    import json

    import numpy as np

    blocks, meta_rows = [], []
    with open(meta_path) as f:
        meta_lines = [json.loads(x) for x in f if x.strip()]
    with open(bin_path, "rb") as bf:
        for m in meta_lines:
            nt, L, S = m["nt"], m["L"], m["S"]
            raw = bf.read(nt * L * S)
            if len(raw) < nt * L * S:
                break
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(nt, L, S)
            blocks.append(arr)
            npos = m["npos"]
            for r, (rid, rnd, acc) in enumerate(zip(m["rids"], m["rounds"], m["accepts"])):
                pidx, turn = _parse_rid(rid)
                local_idx = pidx - prompt_offset
                p = prompts[local_idx] if 0 <= local_idx < len(prompts) else None
                for pos in range(npos):
                    meta_rows.append({
                        "model": cfg.target_model,
                        "speculator": cfg.speculator_label,
                        "config": p.config if p else "?",
                        "category": p.category if p else "?",
                        "prompt_idx": pidx,
                        "turn": turn,
                        "round_idx": rnd,
                        "position": pos,
                        "accepted": 1 if pos < acc else 0,
                    })
    full = np.concatenate(blocks, axis=0) if blocks else None
    return full, meta_rows


def _pad(seq, width: int, fill):
    seq = list(seq or [])
    if len(seq) < width:
        seq = seq + [fill] * (width - len(seq))
    return seq[:width]


def _parse_rid(rid: str) -> tuple[int, int]:
    """rid is '<prompt_idx>:<turn>' (turn 0 for single-turn)."""
    s = str(rid)
    if ":" in s:
        a, b = s.split(":", 1)
        return (int(a) if a.isdigit() else -1, int(b) if b.isdigit() else 0)
    return (int(s) if s.isdigit() else -1, 0)


def _assemble_captures(
    path: str,
    cfg: RunConfig,
    prompts: list[PromptSpec],
    prompt_offset: int = 0,
) -> tuple[list, list]:
    """Merge the child's JSONL accept/conf rows (keyed by rid=str(prompt_idx), round)
    into acceptance + speculator rows."""
    import json

    width = cfg.speculator.column_width
    merged: dict[tuple[str, int], dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            d = merged.setdefault((o["rid"], o["r"]), {})
            if o["k"] == "a":
                d["accept"], d["mask"] = o["accept"], o["mask"]
            else:
                d["conf"] = o["conf"]

    def _sort_key(item):
        (rid, rnd), _ = item
        pidx, turn = _parse_rid(rid)
        return (pidx if pidx >= 0 else 1 << 30, turn, rnd)

    acceptance, speculator = [], []
    for (rid, rnd), d in sorted(merged.items(), key=_sort_key):
        pidx, turn = _parse_rid(rid)
        local_idx = pidx - prompt_offset
        p = prompts[local_idx] if 0 <= local_idx < len(prompts) else None
        common = {
            "model": cfg.target_model,
            "speculator": cfg.speculator_label,
            "config": p.config if p else "?",
            "category": p.category if p else "?",
            "prompt_idx": pidx,
            "turn": turn,
            "round_idx": rnd,
        }
        if "accept" in d:
            row = dict(common, accept=d["accept"])
            mask = _pad(d.get("mask"), width, None)
            row.update({f"acc{k}": mask[k] for k in range(width)})
            acceptance.append(row)
        if "conf" in d:
            row = dict(common)
            conf = _pad(d.get("conf"), width, None)
            row.update({f"conf{k}": conf[k] for k in range(width)})
            speculator.append(row)
    return acceptance, speculator


def _reconcile(metainfo: list[dict], acceptance: list[dict]) -> None:
    """Correctness gate comparing each side's OWN accept-length (tokens committed per
    verify round), which is averaging- and round-count-invariant:

      hook_len = (sum(accept) + n_hook_rounds) / n_hook_rounds   # +1 bonus per round
      meta_len = sum(completion_tokens) / sum(spec_verify_ct)

    The hook may log ~1 extra (truncated final) verify round per sequence -- a real
    verification event SGLang's emitted-token count excludes -- so the round COUNTS
    differ slightly, but the per-round accept-length must agree. Logs; does not raise."""
    commits = [r["accept"] for r in acceptance if r.get("accept") is not None]
    n_rounds = len(commits)
    if not n_rounds:
        print("[reconcile] no hook rounds captured -- accept hook may have missed")
        return
    hook_len = (sum(commits) + n_rounds) / n_rounds
    meta_completion = sum(m.get("completion_tokens") or 0 for m in metainfo)
    meta_verify = sum(m.get("spec_verify_ct") or 0 for m in metainfo)
    meta_len = (meta_completion / meta_verify) if meta_verify else float("nan")
    delta = abs(hook_len - meta_len)
    status = "OK" if delta < 0.1 else "MISMATCH"
    print(
        f"[reconcile] hook accept_len={hook_len:.3f} vs meta spec_accept_length={meta_len:.3f}  "
        f"delta={delta:.3f}  [{status}]  ({n_rounds} hook rounds, {meta_verify} meta verifies)"
    )


def write_result(result: dict, cfg: RunConfig, out_dir: str) -> None:
    """Write a collection result (meta_info + optional banks) to out_dir. Shared by
    the local and Modal paths."""
    import json
    from pathlib import Path

    from . import schema

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    width = cfg.speculator.column_width

    if result.get("metainfo"):
        (out / "metainfo.json").write_text(json.dumps(result["metainfo"], indent=2))
        print(f"wrote {len(result['metainfo'])} meta_info rows -> {out/'metainfo.json'}")
    if "acceptance" in result:
        n = schema.write_acceptance(result["acceptance"], width, out / "acceptance.parquet")
        print(f"wrote {n} rows -> {out/'acceptance.parquet'}")
        schema.write_stats(result["acceptance"], cfg.speculator_label, width, out / "stats.json")
        print(f"wrote stats -> {out/'stats.json'}")
    if "speculator" in result:
        n = schema.write_speculator(result["speculator"], width, out / "speculator.parquet")
        print(f"wrote {n} rows -> {out/'speculator.parquet'}")
    if result.get("routing_npy") is not None:
        import numpy as np

        np.save(out / "routing.npy", result["routing_npy"])
        schema.write_routing_meta(result["routing_meta"], out / "routing_meta.parquet")
        print(f"wrote routing.npy {result['routing_npy'].shape} + "
              f"{len(result['routing_meta'])} meta rows -> {out}")


def run_checkpointed(
    cfg: RunConfig,
    prompts: list[PromptSpec],
    with_hooks: bool = True,
    resume: bool = True,
    after_part=None,
) -> dict:
    """Run collection in durable prompt shards.

    Full shards are sized as `effective_batch_size * checkpoint_batches`, so the
    checkpoint boundary does not force partial scheduler batches. The final shard
    may be smaller when the prompt count is not a multiple of the target size.
    """
    import math
    from pathlib import Path

    out = Path(cfg.out_dir)
    manifest = _prepare_checkpoint_run(out, cfg, prompts, with_hooks, resume)
    completed = _completed_parts(out)
    shard_size = cfg.checkpoint_shard_size
    n_parts = math.ceil(len(prompts) / shard_size) if prompts else 0
    print(
        "checkpointing "
        f"{len(prompts)} prompts -> {n_parts} part(s) "
        f"({cfg.effective_batch_size} prompt batch x {cfg.checkpoint_batches})"
    )

    written = 0
    skipped = 0
    for part_idx, start in enumerate(range(0, len(prompts), shard_size)):
        end = min(start + shard_size, len(prompts))
        existing = completed.get(part_idx)
        if existing is not None:
            if existing.get("start") != start or existing.get("end") != end:
                raise ValueError(
                    f"{out}: completed part {part_idx:06d} has range "
                    f"{existing.get('start')}:{existing.get('end')}, expected {start}:{end}"
                )
            print(f"skipping complete part {part_idx:06d} prompts {start}:{end}")
            skipped += 1
            continue

        chunk = prompts[start:end]
        print(f"running part {part_idx:06d} prompts {start}:{end}")
        result = (
            collect_banks(cfg, chunk, prompt_offset=start)
            if with_hooks
            else {"metainfo": run_metainfo(cfg, chunk, prompt_offset=start)}
        )
        part_manifest = _write_part_result(result, cfg, out, part_idx, start, end)
        written += 1
        if after_part is not None:
            after_part(part_manifest)

    final_files = _existing_final_files(out) if written == 0 else []
    if not final_files:
        final_files = _materialize_final_outputs(out, cfg)
    complete = {
        **manifest,
        "complete": True,
        "parts_written": written,
        "parts_skipped": skipped,
        "final_files": final_files,
    }
    _write_json_atomic(out / "run_complete.json", complete)
    return complete


def _prepare_checkpoint_run(
    out,
    cfg: RunConfig,
    prompts: list[PromptSpec],
    with_hooks: bool,
    resume: bool,
) -> dict:
    import json

    out.mkdir(parents=True, exist_ok=True)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    manifest = _run_manifest(cfg, prompts, with_hooks)
    path = out / "run_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        for key in ("schema_version", "config_hash", "prompt_hash", "with_hooks"):
            if existing.get(key) != manifest.get(key):
                raise ValueError(
                    f"{out}: existing checkpoint manifest does not match this run "
                    f"({key}: {existing.get(key)!r} != {manifest.get(key)!r})"
                )
    elif _completed_parts(out):
        raise ValueError(f"{out}: found checkpoint parts but no run_manifest.json")
    elif _legacy_outputs(out):
        raise ValueError(f"{out}: existing outputs found but no run_manifest.json; use a clean out_dir")
    else:
        _write_json_atomic(path, manifest)

    if not resume and _completed_parts(out):
        raise ValueError(f"{out}: checkpoint parts already exist; rerun with --resume or clean out_dir")
    return manifest


def _legacy_outputs(out) -> list:
    names = [
        "acceptance.parquet",
        "speculator.parquet",
        "metainfo.json",
        "stats.json",
        "run_complete.json",
        "completed_parts.jsonl",
    ]
    return [out / name for name in names if (out / name).exists()]


def _existing_final_files(out) -> list[str]:
    import json

    complete = out / "run_complete.json"
    if not complete.exists():
        return []
    final_files = json.loads(complete.read_text()).get("final_files") or []
    return final_files if all((out / name).exists() for name in final_files) else []


def _run_manifest(cfg: RunConfig, prompts: list[PromptSpec], with_hooks: bool) -> dict:
    return {
        "schema_version": 1,
        "target_model": cfg.target_model,
        "speculator": cfg.speculator_label,
        "with_hooks": with_hooks,
        "n_prompts": len(prompts),
        "effective_batch_size": cfg.effective_batch_size,
        "checkpoint_batches": cfg.checkpoint_batches,
        "checkpoint_shard_size": cfg.checkpoint_shard_size,
        "config_hash": _stable_hash(_config_payload(cfg)),
        "prompt_hash": _stable_hash(_prompt_payload(prompts)),
    }


def _config_payload(cfg: RunConfig) -> dict:
    payload = cfg.to_dict()
    payload.pop("out_dir", None)
    return payload


def _prompt_payload(prompts: list[PromptSpec]) -> list[dict]:
    return [
        {
            "config": p.config,
            "category": p.category,
            "turns": p.turns,
            "question_id": p.question_id,
        }
        for p in prompts
    ]


def _stable_hash(obj) -> str:
    import hashlib
    import json

    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _write_json_atomic(path, obj) -> None:
    import json

    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


def _completed_parts(out) -> dict[int, dict]:
    import json

    completed: dict[int, dict] = {}
    for part in sorted((out / "parts").glob("part-*")):
        if not (part / "_SUCCESS").exists():
            continue
        meta = part / "part_manifest.json"
        if meta.exists():
            row = json.loads(meta.read_text())
            completed[int(row["part_idx"])] = row
        else:
            completed[int(part.name.rsplit("-", 1)[-1])] = {"part_dir": str(part)}
    return completed


def _write_part_result(
    result: dict,
    cfg: RunConfig,
    out,
    part_idx: int,
    start: int,
    end: int,
) -> dict:
    import shutil

    parts = out / "parts"
    final = parts / f"part-{part_idx:06d}"
    tmp = parts / f".part-{part_idx:06d}.tmp"
    if final.exists():
        if (final / "_SUCCESS").exists():
            raise ValueError(f"{final}: part is already complete")
        raise ValueError(f"{final}: incomplete non-temporary part directory exists")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    write_result(result, cfg, str(tmp))
    manifest = {
        "part_idx": part_idx,
        "start": start,
        "end": end,
        "prompt_count": end - start,
        "metainfo_rows": len(result.get("metainfo") or []),
        "acceptance_rows": len(result.get("acceptance") or []),
        "speculator_rows": len(result.get("speculator") or []),
        "routing_rows": len(result.get("routing_meta") or []),
    }
    _write_json_atomic(tmp / "part_manifest.json", manifest)
    (tmp / "_SUCCESS").write_text("ok\n")
    tmp.rename(final)

    with (out / "completed_parts.jsonl").open("a") as f:
        import json

        f.write(json.dumps({**manifest, "part_dir": str(final)}) + "\n")
    return {**manifest, "part_dir": str(final)}


def _materialize_final_outputs(out, cfg: RunConfig) -> list[str]:
    from . import schema

    final_files: list[str] = []
    part_dirs = [p for p in sorted((out / "parts").glob("part-*")) if (p / "_SUCCESS").exists()]

    metainfo_paths = [p / "metainfo.json" for p in part_dirs if (p / "metainfo.json").exists()]
    if metainfo_paths:
        n = _merge_json_arrays(metainfo_paths, out / "metainfo.json")
        print(f"materialized {n} meta_info rows -> {out/'metainfo.json'}")
        final_files.append("metainfo.json")

    acceptance_paths = [p / "acceptance.parquet" for p in part_dirs if (p / "acceptance.parquet").exists()]
    if acceptance_paths:
        n = _merge_parquets(acceptance_paths, out / "acceptance.parquet")
        print(f"materialized {n} rows -> {out/'acceptance.parquet'}")
        schema.write_stats_from_acceptance_parquet(
            out / "acceptance.parquet",
            cfg.speculator_label,
            cfg.speculator.column_width,
            out / "stats.json",
        )
        print(f"materialized stats -> {out/'stats.json'}")
        final_files += ["acceptance.parquet", "stats.json"]

    speculator_paths = [p / "speculator.parquet" for p in part_dirs if (p / "speculator.parquet").exists()]
    if speculator_paths:
        n = _merge_parquets(speculator_paths, out / "speculator.parquet")
        print(f"materialized {n} rows -> {out/'speculator.parquet'}")
        final_files.append("speculator.parquet")

    return final_files


def _merge_parquets(paths, output_path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    tmp = output_path.with_name(f".{output_path.name}.tmp")
    if tmp.exists():
        tmp.unlink()
    writer = None
    rows = 0
    try:
        for path in paths:
            pf = pq.ParquetFile(path)
            if writer is None:
                writer = pq.ParquetWriter(tmp, pf.schema_arrow, compression="zstd")
            rows += pf.metadata.num_rows
            for batch in pf.iter_batches(batch_size=65536):
                writer.write_table(pa.Table.from_batches([batch], schema=pf.schema_arrow))
        if writer is None:
            return 0
        writer.close()
        writer = None
        tmp.replace(output_path)
        return rows
    finally:
        if writer is not None:
            writer.close()


def _merge_json_arrays(paths, output_path) -> int:
    import json

    tmp = output_path.with_name(f".{output_path.name}.tmp")
    count = 0
    first = True
    with tmp.open("w") as f:
        f.write("[\n")
        for path in paths:
            rows = json.loads(path.read_text())
            for row in rows:
                if not first:
                    f.write(",\n")
                f.write(json.dumps(row, sort_keys=True))
                first = False
                count += 1
        f.write("\n]\n")
    tmp.replace(output_path)
    return count


def run_local(
    cfg: RunConfig,
    prompts: list[PromptSpec],
    with_hooks: bool = True,
    resume: bool = True,
) -> None:
    """Local (non-Modal) entry: run on this box's GPU and write outputs."""
    run_checkpointed(cfg, prompts, with_hooks=with_hooks, resume=resume)
