"""A stand-in for vLLM's ``AsyncLLM`` inside the unmodified production worker
image. dynamo.vllm's own code (argument parsing, the real ``VllmConfig``,
registration, request handlers, stat-logger load reports) runs unchanged;
only the engine that would load weights onto a GPU is replaced.

``install()`` must run before ``dynamo.vllm`` is imported: vLLM resolves its
platform when its argument module is first imported."""

import asyncio
import inspect
import os
import time
from types import SimpleNamespace

from inference_lab_dynamo import simulation
from inference_lab_dynamo.records import emit


def _builtins(value):
    """Plain-data view of vLLM request objects for the record."""
    if hasattr(value, "__struct_fields__"):
        return {f: _builtins(getattr(value, f)) for f in value.__struct_fields__}
    if hasattr(value, "__dataclass_fields__"):
        return {f: _builtins(getattr(value, f)) for f in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {k: _builtins(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_builtins(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def install():
    from inference_lab_dynamo.accelerator import pin_vllm_platform

    pin_vllm_platform()
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.metrics.stats import SchedulerStats

    real_signature = inspect.signature(AsyncLLM.generate)

    class SimAsyncLLM(AsyncLLM):
        def __init__(self, vllm_config, stat_loggers=None):
            from transformers import AutoTokenizer

            model = vllm_config.model_config
            self.vllm_config = vllm_config
            self.model_config = model
            self.log_stats = True
            self._tokenizer = AutoTokenizer.from_pretrained(
                model.tokenizer, revision=model.tokenizer_revision or model.revision,
                trust_remote_code=model.trust_remote_code,
            )
            cache = vllm_config.cache_config
            cache.num_gpu_blocks = int(os.environ.get("SIM_NUM_GPU_BLOCKS", "74403"))
            self.dp_size = vllm_config.parallel_config.data_parallel_size
            self.loggers = [
                factory(vllm_config, rank)
                for factory in (stat_loggers or [])
                for rank in range(self.dp_size)
            ]
            self.running = {rank: 0 for rank in range(self.dp_size)}
            self.aborted = set()
            self._stopped = False
            emit(
                "engine_start",
                engine="vllm",
                served_model_name=model.served_model_name,
                dp_size=self.dp_size,
                block_size=cache.block_size,
                max_model_len=model.max_model_len,
            )

        @property
        def tokenizer(self):
            return self._tokenizer

        @property
        def errored(self):
            return False

        @property
        def is_running(self):
            return not self._stopped

        @property
        def is_stopped(self):
            return self._stopped

        async def generate(self, *args, **kwargs):
            bound = real_signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            call = {k: v for k, v in bound.arguments.items() if k != "self"}
            prompt = call["prompt"]
            token_ids = list(prompt.get("prompt_token_ids") or []) if isinstance(prompt, dict) else []
            request_id = call["request_id"]
            prompt_text = self._tokenizer.decode(token_ids, skip_special_tokens=False)
            emit(
                "engine_request",
                engine="vllm",
                rid=request_id,
                call={k: _builtins(v) for k, v in call.items() if k != "prompt"},
                prompt={k: _builtins(v) for k, v in prompt.items() if k != "prompt_token_ids"}
                if isinstance(prompt, dict) else repr(prompt),
                prompt_token_count=len(token_ids),
                prompt_text=prompt_text,
            )
            output = simulation.plan(self._tokenizer, prompt_text, call["sampling_params"].max_tokens)
            rank = call.get("data_parallel_rank") or 0
            started, produced, first, finish = time.time(), 0, None, None
            self.running[rank] += 1
            try:
                last = len(output.token_ids) - 1
                for index, token in enumerate(output.token_ids):
                    await asyncio.sleep(simulation.INTER_TOKEN_SECONDS)
                    if request_id in self.aborted:
                        finish = "abort"
                        yield self._output(request_id, token_ids, [], finish)
                        return
                    produced += 1
                    first = first or time.time()
                    if index == last:
                        finish = "stop" if output.scripted else "length"
                    yield self._output(request_id, token_ids, [token], finish)
            finally:
                self.running[rank] -= 1
                emit(
                    "engine_response",
                    engine="vllm",
                    rid=request_id,
                    prompt_tokens=len(token_ids),
                    completion_tokens=produced,
                    ttft_s=(first - started) if first else None,
                    e2e_s=time.time() - started,
                    finish_reason=finish,
                )

        @staticmethod
        def _output(request_id, prompt_ids, new_ids, finish):
            return SimpleNamespace(
                request_id=request_id,
                prompt_token_ids=prompt_ids,
                num_cached_tokens=0,
                prompt_logprobs=None,
                finished=finish is not None,
                outputs=[
                    SimpleNamespace(
                        index=0, token_ids=new_ids, finish_reason=finish,
                        stop_reason=None, logprobs=None, routed_experts=None,
                    )
                ],
            )

        async def abort(self, request_id, *args, **kwargs):
            ids = request_id if isinstance(request_id, (list, tuple)) else [request_id]
            self.aborted.update(ids)

        async def check_health(self):
            return None

        async def do_log_stats(self, *args, **kwargs):
            for logger in self.loggers:
                rank = getattr(logger, "engine_idx", 0) or 0
                logger.record(
                    scheduler_stats=SchedulerStats(
                        num_running_reqs=self.running.get(rank, 0),
                        num_waiting_reqs=0,
                        kv_cache_usage=0.0,
                    ),
                    iteration_stats=None,
                    engine_idx=rank,
                )

        async def collective_rpc(self, method, *args, **kwargs):
            if method == "dynamo_get_kv_cache_group_metadata":
                block = self.vllm_config.cache_config.block_size
                return [[{"group_idx": 0, "kind": "full_attention", "block_size": block}]]
            return [None]

        async def reset_prefix_cache(self, *args, **kwargs):
            return True

        async def pause_generation(self, *args, **kwargs):
            return None

        async def resume_generation(self, *args, **kwargs):
            return None

        async def sleep(self, *args, **kwargs):
            return None

        async def wake_up(self, *args, **kwargs):
            return None

        def shutdown(self, *args, **kwargs):
            self._stopped = True

    SimAsyncLLM.generate.__signature__ = real_signature

    def from_vllm_config(cls, vllm_config, *args, stat_loggers=None, **kwargs):
        return SimAsyncLLM(vllm_config, stat_loggers)

    AsyncLLM.from_vllm_config = classmethod(from_vllm_config)

    import dynamo.vllm.main as worker_main

    async def fetch_metadata_only(model, *args, **kwargs):
        return model

    worker_main.fetch_model = fetch_metadata_only
