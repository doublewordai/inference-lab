"""Install the simulated engine for a Dynamo worker module before it runs."""

import os

from inference_lab_dynamo import accelerator


def install(module, argv):
    if module == "dynamo.vllm":
        accelerator.ensure_driver_stub(argv)
    if os.environ.get("INFERENCE_LAB_FETCH_METADATA") == "1":
        from inference_lab_dynamo import metadata

        metadata.fetch(argv)
    # From here the worker must only read what is already in the cache: its
    # own start-up paths (SGLang's argument parser, Dynamo's model
    # registration) would otherwise download full weights for any model the
    # production entry does not already mark offline.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from huggingface_hub import constants

        constants.HF_HUB_OFFLINE = True
    except ImportError:
        pass
    if module == "dynamo.sglang":
        from inference_lab_dynamo import sglang_engine

        sglang_engine.install()
    elif module == "dynamo.vllm":
        from inference_lab_dynamo import vllm_engine

        vllm_engine.install()
    else:
        raise ValueError(f"no simulated engine for {module}")
