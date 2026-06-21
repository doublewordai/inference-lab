import json
import os

import pyarrow.parquet as pq

from specdec_calibration import collect
from specdec_calibration.collect import _assemble_captures, _sampling_params
from specdec_calibration.config import RunConfig, Speculator
from specdec_calibration.datasets import PromptSpec


def test_assemble_captures_preserves_turn_and_dflash_width(tmp_path):
    cfg = RunConfig(
        target_model="M",
        speculator=Speculator(
            algorithm="dflash",
            draft_model_path="draft",
            dflash_block_size=16,
        ),
    )
    prompts = [
        PromptSpec("qualitative", "coding", ["a"]),
        PromptSpec("qualitative", "math", ["b", "c", "d"]),
    ]
    path = tmp_path / "capture.jsonl"
    rows = [
        {"k": "a", "rid": "1:2", "r": 7, "accept": 3, "mask": [1, 1, 1] + [0] * 12},
        {"k": "c", "rid": "1:2", "r": 7, "conf": [0.9] * 15},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    acceptance, speculator = _assemble_captures(str(path), cfg, prompts)

    assert cfg.speculator.column_width == 15
    assert acceptance[0]["prompt_idx"] == 1
    assert acceptance[0]["turn"] == 2
    assert acceptance[0]["round_idx"] == 7
    assert acceptance[0]["acc14"] == 0
    assert "acc15" not in acceptance[0]
    assert speculator[0]["conf14"] == 0.9
    assert "conf15" not in speculator[0]


def test_sampling_params_include_seed():
    cfg = RunConfig(target_model="M", speculator=Speculator(algorithm="mtp"), seed=123)
    assert _sampling_params(cfg)["seed"] == 123


def test_sampling_params_omit_max_tokens_by_default():
    cfg = RunConfig(target_model="M", speculator=Speculator(algorithm="mtp"))
    assert "max_new_tokens" not in _sampling_params(cfg)


def test_sampling_params_include_explicit_max_tokens():
    cfg = RunConfig(target_model="M", speculator=Speculator(algorithm="mtp"), max_tokens=64)
    assert _sampling_params(cfg)["max_new_tokens"] == 64


def test_collect_banks_uses_scoped_capture_env(monkeypatch):
    old_env = {
        "SPECDEC_CAPTURE_PATH": "old-capture",
        "SPECDEC_HOOK_FAMILY": "old-family",
        "SPECDEC_ROUTING_BIN": "old-routing-bin",
        "SPECDEC_ROUTING_META": "old-routing-meta",
        "SPECDEC_NO_CONF": "old-no-conf",
    }
    for key, value in old_env.items():
        monkeypatch.setenv(key, value)

    seen = {}

    class FakeTokenizer:
        def apply_chat_template(self, history, tokenize=False, add_generation_prompt=True):
            return history[-1]["content"]

    class FakeBackend:
        def generate(self, *, prompt, sampling_params, rid):
            seen["capture_path"] = os.environ["SPECDEC_CAPTURE_PATH"]
            assert os.environ["SPECDEC_HOOK_FAMILY"] == "eagle"
            assert "SPECDEC_ROUTING_BIN" not in os.environ
            assert "SPECDEC_ROUTING_META" not in os.environ
            assert "SPECDEC_NO_CONF" not in os.environ
            assert rid == ["0:0"]
            return [
                {
                    "text": "ok",
                    "meta_info": {
                        "completion_tokens": 1,
                        "spec_verify_ct": 0,
                    },
                }
            ]

    class FakeEngine:
        tokenizer = FakeTokenizer()
        engine = FakeBackend()

        def shutdown(self):
            seen["shutdown"] = True

    monkeypatch.setattr(collect, "build_engine", lambda cfg: FakeEngine())
    cfg = RunConfig(target_model="M", speculator=Speculator(algorithm="mtp"))
    prompts = [PromptSpec("qualitative", "coding", ["hello"])]

    result = collect.collect_banks(cfg, prompts)

    assert seen["shutdown"] is True
    assert seen["capture_path"] != old_env["SPECDEC_CAPTURE_PATH"]
    assert result["metainfo"][0]["prompt_idx"] == 0
    assert result["acceptance"] == []
    assert result["speculator"] == []
    for key, value in old_env.items():
        assert os.environ[key] == value


def test_checkpointed_run_writes_batch_aligned_parts_and_resumes(tmp_path, monkeypatch):
    cfg = RunConfig(
        target_model="M",
        speculator=Speculator(algorithm="mtp"),
        n_prompts=5,
        max_running_requests=2,
        checkpoint_batches=2,
        out_dir=str(tmp_path / "run"),
    )
    cfg.validate()
    prompts = [PromptSpec("qualitative", f"cat{i}", [f"prompt {i}"]) for i in range(5)]
    calls = []

    def fake_collect(cfg_arg, chunk, prompt_offset=0):
        calls.append((prompt_offset, len(chunk)))
        acceptance = []
        speculator = []
        metainfo = []
        for local_idx, prompt in enumerate(chunk):
            prompt_idx = prompt_offset + local_idx
            common = {
                "model": cfg_arg.target_model,
                "speculator": cfg_arg.speculator_label,
                "config": prompt.config,
                "category": prompt.category,
                "prompt_idx": prompt_idx,
                "turn": 0,
                "round_idx": 0,
            }
            acceptance.append(
                dict(common, accept=1, acc0=1, acc1=0, acc2=0, acc3=0)
            )
            speculator.append(
                dict(common, conf0=0.5, conf1=0.4, conf2=0.3, conf3=0.2)
            )
            metainfo.append(
                {
                    "model": cfg_arg.target_model,
                    "speculator": cfg_arg.speculator_label,
                    "config": prompt.config,
                    "category": prompt.category,
                    "prompt_idx": prompt_idx,
                    "turn": 0,
                    "question_id": prompt.question_id,
                    "completion_tokens": 1,
                }
            )
        return {"acceptance": acceptance, "speculator": speculator, "metainfo": metainfo}

    monkeypatch.setattr(collect, "collect_banks", fake_collect)

    collect.run_checkpointed(cfg, prompts)

    assert calls == [(0, 4), (4, 1)]
    run = tmp_path / "run"
    assert (run / "parts" / "part-000000" / "_SUCCESS").exists()
    assert (run / "parts" / "part-000001" / "_SUCCESS").exists()
    assert pq.read_table(run / "acceptance.parquet").num_rows == 5
    assert json.loads((run / "run_manifest.json").read_text())["checkpoint_shard_size"] == 4

    collect.run_checkpointed(cfg, prompts)

    assert calls == [(0, 4), (4, 1)]
