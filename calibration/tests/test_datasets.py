import sys
import types

from specdec_calibration.datasets import build_prompts


def test_speedbench_split_is_forwarded(monkeypatch):
    calls = []

    def load_dataset(name, cfg, split):
        calls.append((name, cfg, split))
        return [
            {
                "category": "coding",
                "turns": ["turn one"],
                "question_id": "q0",
            }
        ]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))

    prompts = build_prompts("speedbench", ["qualitative"], 1, split="validation")

    assert calls == [("nvidia/SPEED-Bench", "qualitative", "validation")]
    assert prompts[0].question_id == "q0"
