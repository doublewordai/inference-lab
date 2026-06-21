import json
import os

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
