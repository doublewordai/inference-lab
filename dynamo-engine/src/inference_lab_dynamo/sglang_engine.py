"""A stand-in for ``sglang.Engine`` inside the unmodified production worker
image. dynamo.sglang's own code (argument parsing, model registration,
request handlers, metrics relay) runs unchanged; only the engine object that
would load weights onto a GPU is replaced."""

import asyncio
import inspect
import os
import random
import threading
import time
from types import SimpleNamespace

import sglang
import zmq
from sglang.srt.server_args import PortArgs

from inference_lab_dynamo import simulation
from inference_lab_dynamo.records import emit

_REAL_ENGINE = sglang.Engine
_REAL_SIGNATURE = inspect.signature(_REAL_ENGINE.async_generate)


class SimTokenizerManager:
    def __init__(self, server_args):
        from sglang.srt.configs.model_config import ModelConfig
        from transformers import AutoTokenizer

        self.server_args = server_args
        self.model_config = ModelConfig.from_server_args(server_args)
        self.tokenizer = AutoTokenizer.from_pretrained(
            server_args.tokenizer_path or server_args.model_path,
            revision=server_args.revision,
            trust_remote_code=server_args.trust_remote_code,
        )
        self.initial_weights_loaded = True
        self.aborted = set()

    def abort_request(self, rid="", abort_all=False, **_):
        self.aborted.add(rid)

    async def pause_generation(self, *args, **kwargs):
        return None

    async def continue_generation(self, *args, **kwargs):
        return None


class SimEngine:
    def __init__(self, server_args=None, **kwargs):
        from sglang.srt.utils.common import set_prometheus_multiproc_dir

        self.server_args = server_args
        if server_args.enable_metrics:
            set_prometheus_multiproc_dir()
        self.port_args = PortArgs.init_new(server_args)
        self.tokenizer_manager = SimTokenizerManager(server_args)
        self.dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
        max_total = int(os.environ.get("SIM_MAX_TOTAL_NUM_TOKENS", "2055872"))
        self._scheduler_init_result = SimpleNamespace(
            scheduler_infos=[
                {"max_total_num_tokens": max_total, "max_req_input_len": max_total}
                for _ in range(self.dp_size)
            ]
        )
        self.running = {rank: 0 for rank in range(self.dp_size)}
        self.waiting = {rank: 0 for rank in range(self.dp_size)}
        self.max_running = max(
            1, (server_args.max_running_requests or 256) // self.dp_size
        )
        self._stopped = False
        threading.Thread(target=self._push_load_reports, daemon=True).start()
        emit(
            "engine_start",
            engine="sglang",
            served_model_name=server_args.served_model_name,
            dp_size=self.dp_size,
            page_size=server_args.page_size,
        )

    def _push_load_reports(self):
        from sglang.srt.managers.scheduler_components.kv_events_publisher import (
            KvMetrics,
        )

        sock = zmq.Context().socket(zmq.PUSH)
        sock.connect(self.port_args.metrics_ipc_name)
        total = self._scheduler_init_result.scheduler_infos[0]["max_total_num_tokens"]
        while not self._stopped:
            for rank in range(self.dp_size):
                sock.send_pyobj(
                    KvMetrics(
                        request_active_slots=self.running[rank],
                        request_total_slots=self.max_running,
                        kv_active_blocks=0,
                        kv_total_blocks=total // max(1, self.server_args.page_size),
                        num_requests_waiting=self.waiting[rank],
                        data_parallel_rank=rank,
                    )
                )
            time.sleep(1.0)

    def get_all_child_pids(self):
        return []

    def shutdown(self):
        self._stopped = True

    async def async_generate(self, *args, **kwargs):
        bound = _REAL_SIGNATURE.bind(self, *args, **kwargs)
        bound.apply_defaults()
        call = {k: v for k, v in bound.arguments.items() if k != "self"}
        rid = call.get("rid") or f"sim-{random.getrandbits(48):x}"
        input_ids = call.get("input_ids") or []
        tokenizer = self.tokenizer_manager.tokenizer
        prompt = (
            tokenizer.decode(input_ids, skip_special_tokens=False)
            if input_ids
            else call.get("prompt")
        )
        emit("engine_request", engine="sglang", rid=rid, call=call, prompt_text=prompt)
        sampling = call.get("sampling_params") or {}
        output = simulation.plan(tokenizer, prompt, sampling.get("max_new_tokens"))
        rank = call.get("data_parallel_rank") or 0
        return self._stream(rid, len(input_ids), output, rank)

    async def _stream(self, rid, prompt_len, output, rank):
        started = time.time()
        self.running[rank] = self.running.get(rank, 0) + 1
        produced, first_token_at, finish = 0, None, None

        def chunk(ids, finish_reason):
            return {
                "output_ids": ids,
                "meta_info": {
                    "id": rid,
                    "finish_reason": finish_reason,
                    "prompt_tokens": prompt_len,
                    "completion_tokens": produced,
                    "cached_tokens": 0,
                },
            }

        try:
            last = len(output.token_ids) - 1
            for index, token in enumerate(output.token_ids):
                await asyncio.sleep(simulation.INTER_TOKEN_SECONDS)
                if rid in self.tokenizer_manager.aborted:
                    finish = {"type": "abort", "message": "aborted"}
                    yield chunk([], finish)
                    return
                produced += 1
                first_token_at = first_token_at or time.time()
                if index == last:
                    finish = (
                        {"type": "stop", "matched": None}
                        if output.scripted
                        else {"type": "length", "length": produced}
                    )
                yield chunk([token], finish)
        finally:
            self.running[rank] -= 1
            emit(
                "engine_response",
                engine="sglang",
                rid=rid,
                prompt_tokens=prompt_len,
                completion_tokens=produced,
                ttft_s=(first_token_at - started) if first_token_at else None,
                e2e_s=time.time() - started,
                finish_reason=finish,
            )


SimEngine.async_generate.__signature__ = _REAL_SIGNATURE


def install():
    sglang.Engine = SimEngine
