"""Does SGLang's own default resolution run on CPU against a simulated GPU?
Parses the given worker arguments with Dynamo's parser, presents the GPU named
by SIM_GPU_* through the simulated accelerator, runs ServerArgs.resolve_once()
and prints the fields a model card derives from."""

import asyncio
import sys

sys.path.insert(0, "/opt/inference-lab-dynamo/inference-lab-dynamo.zip")
from inference_lab_dynamo import accelerator  # noqa: E402

accelerator.pin_sglang_platform()

from dynamo.sglang.args import parse_args  # noqa: E402

config = asyncio.run(parse_args(sys.argv[1:]))
server_args = config.server_args
with accelerator.simulated_sglang_device():
    server_args.resolve_once()
from sglang.srt.arg_groups.overrides import resolving_view  # noqa: E402

with accelerator.simulated_sglang_device():
    server_args.check_server_args()
    view = resolving_view(server_args)
for field in ("page_size", "context_length", "max_running_requests", "chunked_prefill_size",
              "max_prefill_tokens", "cuda_graph_max_bs", "attention_backend", "mem_fraction_static",
              "dp_size", "tp_size", "device", "skip_tokenizer_init"):
    print(f"{field:24s} {getattr(view, field, '<absent>')}")
