"""Prompt sources. A prompt is (config, category, raw user turn).

The chat template is applied later, on the GPU, where the target tokenizer is
loaded (collect.py) -- so prompt building here is dependency-free and works in
--dry-run with no model. Default source is a tiny built-in "smoke" set; the real
runs use external datasets such as SPEED-Bench and HumanEval, which need the
`data` extra.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptSpec:
    config: str
    category: str
    turns: list[str]  # full conversation of user turns (>1 = multi-turn), pre template
    question_id: str = ""  # stable SPEED-Bench id, for matching across runs

    @property
    def content(self) -> str:
        return self.turns[0] if self.turns else ""

    @property
    def is_multiturn(self) -> bool:
        return len(self.turns) > 1


_SMOKE: list[tuple[str, str, str]] = [
    ("qualitative", "coding", "Write a Python function that reverses a singly linked list."),
    ("qualitative", "math", "Compute the 12th Fibonacci number and show the steps."),
    ("qualitative", "reasoning", "A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?"),
    ("qualitative", "writing", "Write a short haiku about entropy in a data stream."),
]


def build_prompts(
    dataset: str,
    configs: list[str],
    n_prompts: int,
    multiturn_only: bool = False,
    split: str = "test",
) -> list[PromptSpec]:
    if dataset == "smoke":
        out = [PromptSpec(c, cat, [txt]) for (c, cat, txt) in _SMOKE][: max(1, n_prompts)]
    elif dataset == "speedbench":
        out = _speedbench(configs, n_prompts, split)
    elif dataset == "humaneval":
        out = _humaneval(configs, n_prompts, split)
    else:
        raise ValueError(f"unknown dataset {dataset!r}; use 'smoke', 'speedbench', or 'humaneval'")
    if multiturn_only:
        out = [p for p in out if p.is_multiturn]
    return out


def _speedbench(configs: list[str], n_prompts: int, split: str) -> list[PromptSpec]:
    from collections import defaultdict

    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - only on real runs
        raise ImportError(
            "the 'speedbench' dataset needs the `data` extra: "
            "uv sync --extra data"
        ) from e

    out: list[PromptSpec] = []
    for cfg in configs:
        ds = load_dataset("nvidia/SPEED-Bench", cfg, split=split)
        by_cat: dict[str, list] = defaultdict(list)
        for ex in ds:
            by_cat[ex["category"]].append(ex)
        for cat, exs in by_cat.items():
            for ex in exs[:n_prompts]:
                out.append(
                    PromptSpec(cfg, cat, list(ex["turns"]), str(ex.get("question_id", "")))
                )
    return out


def _humaneval(configs: list[str], n_prompts: int, split: str) -> list[PromptSpec]:
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - only on real runs
        raise ImportError(
            "the 'humaneval' dataset needs the `data` extra: "
            "uv sync --extra data"
        ) from e

    ds = load_dataset("openai/openai_humaneval", split=split)
    config = configs[0] if configs else "humaneval"
    out: list[PromptSpec] = []
    for ex in ds:
        out.append(
            PromptSpec(
                config,
                "coding",
                [str(ex["prompt"])],
                str(ex.get("task_id", "")),
            )
        )
        if len(out) >= n_prompts:
            break
    return out
