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


def test_humaneval_uses_task_id_and_prompt(monkeypatch):
    calls = []

    def load_dataset(name, split):
        calls.append((name, split))
        return [
            {
                "task_id": "HumanEval/0",
                "prompt": "def f():\n",
                "canonical_solution": "    pass\n",
            }
        ]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))

    prompts = build_prompts("humaneval", ["humaneval"], 1, split="test")

    assert calls == [("openai/openai_humaneval", "test")]
    assert prompts[0].config == "humaneval"
    assert prompts[0].category == "coding"
    assert prompts[0].content == "def f():\n"
    assert prompts[0].question_id == "HumanEval/0"
