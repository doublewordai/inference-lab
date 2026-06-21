import textwrap

import pytest

from specdec_calibration.config import RunConfig, Speculator


def _write(tmp_path, body):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_yaml_roundtrip(tmp_path):
    p = _write(
        tmp_path,
        """
        target_model: meta-llama/Llama-3.1-8B-Instruct
        speculator:
          algorithm: eagle3
          draft_model_path: yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
          num_steps: 5
          eagle_topk: 1
          num_draft_tokens: 6
        dataset: smoke
        n_prompts: 4
        temperature: 0.6
        """,
    )
    cfg = RunConfig.from_yaml(p)
    cfg.validate()
    assert cfg.target_model.endswith("Llama-3.1-8B-Instruct")
    assert cfg.speculator.sglang_algorithm == "EAGLE3"
    assert cfg.speculator.column_width == 5
    assert cfg.speculator.family == "eagle"
    assert cfg.speculator_label == "eagle3:EAGLE3-LLaMA3.1-Instruct-8B"


def test_mtp_needs_no_draft(tmp_path):
    p = _write(
        tmp_path,
        """
        target_model: Qwen/Qwen3-Next-80B
        speculator:
          algorithm: mtp
          num_steps: 3
        """,
    )
    cfg = RunConfig.from_yaml(p)
    cfg.validate()
    assert cfg.speculator.sglang_algorithm == "NEXTN"
    assert cfg.speculator.family == "eagle"


def test_mtp_rejects_draft_path():
    spec = Speculator(algorithm="mtp", draft_model_path="some/draft")
    with pytest.raises(ValueError, match="own MTP weights"):
        spec.validate()


def test_eagle_requires_draft_path():
    spec = Speculator(algorithm="eagle")
    with pytest.raises(ValueError, match="requires `draft_model_path`"):
        spec.validate()


def test_dflash_width_from_block_size():
    spec = Speculator(algorithm="dflash", draft_model_path="d", dflash_block_size=16)
    assert spec.family == "dflash"
    assert spec.column_width == 15


def test_dflash_block_size_must_leave_draft_position():
    spec = Speculator(algorithm="dflash", draft_model_path="d", dflash_block_size=1)
    with pytest.raises(ValueError, match="dflash_block_size must be >= 2"):
        spec.validate()


def test_dflash_requires_explicit_block_size():
    spec = Speculator(algorithm="dflash", draft_model_path="d")
    with pytest.raises(ValueError, match="dflash requires `dflash_block_size`"):
        spec.validate()


def test_eagle_capture_requires_topk_one():
    spec = Speculator(algorithm="eagle3", draft_model_path="d", eagle_topk=2)
    with pytest.raises(ValueError, match="eagle_topk=1"):
        spec.validate()


def test_engine_kwargs_caps_eagle_conf_capture_batch():
    cfg = RunConfig(target_model="M", speculator=Speculator(algorithm="mtp"))
    cfg.validate()
    assert cfg.capture_max_running_requests == 512
    assert cfg.engine_kwargs()["max_running_requests"] == 512
    assert cfg.effective_batch_size == 512
    assert cfg.checkpoint_shard_size == 2048


def test_checkpoint_shards_are_batch_aligned_when_batch_is_explicit():
    cfg = RunConfig(
        target_model="M",
        speculator=Speculator(algorithm="mtp"),
        max_running_requests=128,
        checkpoint_batches=3,
    )
    cfg.validate()
    assert cfg.effective_batch_size == 128
    assert cfg.checkpoint_shard_size == 384


def test_checkpoint_batches_must_be_positive():
    cfg = RunConfig(
        target_model="M",
        speculator=Speculator(algorithm="mtp"),
        checkpoint_batches=0,
    )
    with pytest.raises(ValueError, match="checkpoint_batches must be >= 1"):
        cfg.validate()


def test_engine_kwargs_caps_routing_verify_positions():
    cfg = RunConfig(
        target_model="M",
        speculator=Speculator(algorithm="dflash", draft_model_path="d", dflash_block_size=16),
        capture_routing=True,
    )
    cfg.validate()
    assert cfg.capture_max_running_requests == 256
    assert cfg.engine_kwargs()["max_running_requests"] == 256


def test_explicit_max_running_requests_must_fit_capture_shape():
    cfg = RunConfig(
        target_model="M",
        speculator=Speculator(algorithm="dflash", draft_model_path="d", dflash_block_size=16),
        capture_routing=True,
        max_running_requests=300,
    )
    with pytest.raises(ValueError, match="max_running_requests must be <="):
        cfg.validate()


def test_unknown_algorithm_rejected():
    with pytest.raises(ValueError, match="unknown speculator algorithm"):
        Speculator(algorithm="ngram")


def test_unknown_yaml_key_rejected(tmp_path):
    p = _write(
        tmp_path,
        """
        target_model: m
        speculator:
          algorithm: mtp
        bogus_key: 1
        """,
    )
    with pytest.raises(ValueError, match="unknown config keys"):
        RunConfig.from_yaml(p)
