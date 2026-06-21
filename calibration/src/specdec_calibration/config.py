"""Run configuration: a target model + a speculator spec + a prompt source.

Loaded from YAML (see ../configs/). `validate()` rejects (model, speculator)
combinations that SGLang cannot serve, with an explicit message, before any GPU
time is spent.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# EAGLE family share one SGLang verify/draft path (one hook family); DFLASH is
# separate. NGRAM is intentionally out of scope for v1.
EAGLE_FAMILY = {"eagle", "eagle3", "mtp", "nextn", "standalone"}
VALID_ALGORITHMS = EAGLE_FAMILY | {"dflash"}

# Methods that need a separate draft checkpoint vs. those that ride on the target.
NEEDS_DRAFT_PATH = {"eagle", "eagle3", "standalone", "dflash"}
TARGET_INTRINSIC = {"mtp", "nextn"}  # speculator weights ship inside the target

# Hook-side fixed buffers. Keep these in config too so launch validation can bound
# SGLang before the capture hooks see unsupported active batch shapes.
EAGLE_CONF_CAPTURE_MAX_BATCH = 512
ROUTING_CAPTURE_MAX_TOKENS = 4096


@dataclass
class Speculator:
    """One speculator. `algorithm` selects the SGLang `--speculative-algorithm`.

    `width` is the number of accepted-draft-token positions per round = the count
    of `conf*` / `acc*` columns. Defaults to `num_steps`; DFLASH exposes
    `dflash_block_size - 1` draft positions because SGLang's block includes the
    bonus/root verify position.
    """

    algorithm: str
    draft_model_path: str | None = None
    draft_model_revision: str | None = None  # HF git revision/commit to pin the draft
    num_steps: int = 4
    eagle_topk: int = 1
    num_draft_tokens: int | None = None
    dflash_block_size: int | None = None
    width: int | None = None  # explicit override for the conf/acc column count

    def __post_init__(self) -> None:
        self.algorithm = self.algorithm.lower()
        if self.algorithm not in VALID_ALGORITHMS:
            raise ValueError(
                f"unknown speculator algorithm {self.algorithm!r}; "
                f"v1 supports {sorted(VALID_ALGORITHMS)} (no ngram)"
            )

    @property
    def family(self) -> str:
        """Which hook family handles this method."""
        return "dflash" if self.algorithm == "dflash" else "eagle"

    @property
    def column_width(self) -> int:
        """Number of conf*/acc* columns to emit."""
        if self.width is not None:
            return self.width
        if self.algorithm == "dflash" and self.dflash_block_size:
            return self.dflash_block_size - 1
        return self.num_steps

    @property
    def sglang_algorithm(self) -> str:
        """The value SGLang's --speculative-algorithm expects.

        MTP/NEXTN are served through the EAGLE code path (NEXTN is an SGLang alias
        of EAGLE; MTP weights ride the target). EAGLE3/STANDALONE/DFLASH map 1:1.
        """
        if self.algorithm in {"mtp", "nextn"}:
            return "NEXTN"
        return self.algorithm.upper()

    @property
    def label(self) -> str:
        """Stable identifier written into the `speculator` parquet column.

        Includes a short draft revision when pinned, so before/after runs of the
        same draft checkpoint stay distinct in the bank (e.g. two DFLASH revisions).
        """
        if self.draft_model_path:
            short = self.draft_model_path.rstrip("/").split("/")[-1]
            label = f"{self.algorithm}:{short}"
            if self.draft_model_revision:
                label += f"@{self.draft_model_revision[:8]}"
            return label
        return self.algorithm

    def validate(self) -> None:
        if self.algorithm in NEEDS_DRAFT_PATH and not self.draft_model_path:
            raise ValueError(
                f"speculator {self.algorithm!r} requires `draft_model_path` "
                f"(a draft trained for the target model)"
            )
        if self.algorithm in TARGET_INTRINSIC and self.draft_model_path:
            raise ValueError(
                f"speculator {self.algorithm!r} uses the target's own MTP weights; "
                f"do not set `draft_model_path`"
            )
        if self.algorithm == "dflash" and self.dflash_block_size is not None:
            if self.dflash_block_size < 2:
                raise ValueError("dflash_block_size must be >= 2")
        if self.algorithm == "dflash" and self.dflash_block_size is None:
            raise ValueError("dflash requires `dflash_block_size` so capture width is explicit")
        if self.family == "eagle" and self.eagle_topk != 1:
            raise ValueError("eagle-family capture currently requires eagle_topk=1")
        if self.column_width < 1:
            raise ValueError("speculator column_width must be >= 1")


@dataclass
class RunConfig:
    target_model: str
    speculator: Speculator
    dataset: str = "smoke"  # "smoke" (built-in, no deps) | "speedbench"
    dataset_configs: list[str] = field(default_factory=lambda: ["qualitative"])
    dataset_split: str = "test"
    n_prompts: int = 8
    max_tokens: int | None = None
    temperature: float = 0.6
    seed: int = 0
    out_dir: str = "data"
    gpu: str = "H200"
    # Default: graphs on. The current hooks capture confidence through graph-safe
    # pinned-buffer copies. Eager is retained only for source-level debugging or for
    # comparing graph/eager behavior; on hybrid-Mamba targets it can badly degrade
    # acceptance.
    eager: bool = False
    capture_conf: bool = True  # set False to install only the accept hook
    capture_routing: bool = False  # also capture target MoE expert routing per verify position
    multiturn_only: bool = False  # restrict to multi-turn examples (proper turn-by-turn run)
    # For hybrid-Mamba targets, SGLang's mamba state buffering during speculation.
    # "extra_buffer" is required for radix+spec and may be needed for correct eager
    # spec acceptance; None leaves SGLang's default ("no_buffer").
    mamba_scheduler_strategy: str | None = None
    disable_radix_cache: bool = True  # extra_buffer requires this False
    mem_fraction_static: float | None = None  # lower to leave activation headroom (OOM)
    max_running_requests: int | None = None  # cap concurrency to bound KV+activation memory
    disable_overlap_schedule: bool = False  # test whether non-overlap lets eager draft

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        raw = yaml.safe_load(Path(path).read_text())
        spec_raw = raw.pop("speculator", None)
        if spec_raw is None:
            raise ValueError(f"{path}: config must define a `speculator` block")
        known = {f.name for f in dataclasses.fields(Speculator)}
        unknown = set(spec_raw) - known
        if unknown:
            raise ValueError(f"{path}: unknown speculator keys {sorted(unknown)}")
        speculator = Speculator(**spec_raw)
        known_run = {f.name for f in dataclasses.fields(cls)} - {"speculator"}
        unknown_run = set(raw) - known_run
        if unknown_run:
            raise ValueError(f"{path}: unknown config keys {sorted(unknown_run)}")
        return cls(speculator=speculator, **raw)

    def validate(self) -> None:
        self.speculator.validate()
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.n_prompts < 1:
            raise ValueError("n_prompts must be >= 1")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1 when set")
        capture_limit = self.capture_max_running_requests
        if (
            capture_limit is not None
            and self.max_running_requests is not None
            and self.max_running_requests > capture_limit
        ):
            raise ValueError(
                "max_running_requests must be <= "
                f"{capture_limit} for this capture shape"
            )

    @property
    def speculator_label(self) -> str:
        return self.speculator.label

    def to_dict(self) -> dict:
        """Plain-dict form for crossing the Modal boundary (picklable, no classes)."""
        d = dataclasses.asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        d = dict(d)
        spec = Speculator(**d.pop("speculator"))
        return cls(speculator=spec, **d)

    @property
    def capture_max_running_requests(self) -> int | None:
        """Largest active request count that the selected capture hooks can store."""
        limits: list[int] = []
        if self.capture_conf and self.speculator.family == "eagle":
            limits.append(EAGLE_CONF_CAPTURE_MAX_BATCH)
        if self.capture_routing:
            # Routing stores verify positions, including the root/bonus position.
            verify_positions = self.speculator.column_width + 1
            limits.append(max(1, ROUTING_CAPTURE_MAX_TOKENS // verify_positions))
        return min(limits) if limits else None

    def engine_kwargs(self) -> dict:
        """Map the run config to `sgl.Engine(**kwargs)` / SGLang ServerArgs.

        The offline Engine forwards these straight into ServerArgs, so the same
        keys work for the HTTP server. Speculative knobs are only included when
        set, letting SGLang auto-tune the rest.
        """
        s = self.speculator
        kw: dict = {
            "model_path": self.target_model,
            "trust_remote_code": True,
            "speculative_algorithm": s.sglang_algorithm,
            "speculative_num_steps": s.num_steps,
            "speculative_eagle_topk": s.eagle_topk,
        }
        if self.disable_radix_cache:
            # Calibration runs use unique short prompts (no shared prefixes), so radix
            # cache buys nothing -- and disabling it sidesteps the radix+spec+mamba
            # incompatibility on hybrid-Mamba targets (e.g. Qwen3.6-35B-A3B).
            kw["disable_radix_cache"] = True
        if self.mem_fraction_static is not None:
            kw["mem_fraction_static"] = self.mem_fraction_static
        if self.max_running_requests is not None:
            kw["max_running_requests"] = self.max_running_requests
        elif self.capture_max_running_requests is not None:
            kw["max_running_requests"] = self.capture_max_running_requests
        if self.eager:
            # Eager disables CUDA graphs for debugging/comparison. It can degrade
            # acceptance on some targets -- see RunConfig.eager.
            kw["disable_cuda_graph"] = True
        if self.mamba_scheduler_strategy:
            kw["mamba_scheduler_strategy"] = self.mamba_scheduler_strategy
        if self.disable_overlap_schedule:
            kw["disable_overlap_schedule"] = True
        if s.draft_model_path:
            kw["speculative_draft_model_path"] = s.draft_model_path
        if s.draft_model_revision:
            kw["speculative_draft_model_revision"] = s.draft_model_revision
        if s.num_draft_tokens is not None:
            kw["speculative_num_draft_tokens"] = s.num_draft_tokens
        if s.algorithm == "dflash" and s.dflash_block_size is not None:
            kw["speculative_dflash_block_size"] = s.dflash_block_size
        return kw
