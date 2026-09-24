"""Python loads this module at start-up when its directory is on PYTHONPATH.

With INFERENCE_LAB_SIMULATE=1, a process running ``-m dynamo.sglang`` or
``-m dynamo.vllm`` gets the simulated engine installed before the worker
module runs; the worker's command line stays exactly as production renders
it. Every other process is untouched. The image's own sitecustomize (which
this module shadows) runs afterwards."""

import importlib.machinery
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _command_line():
    try:
        with open("/proc/self/cmdline", "rb") as handle:
            return [part.decode() for part in handle.read().split(b"\0") if part]
    except OSError:
        return list(sys.argv)


def _worker_module(argv):
    if "-m" in argv[:-1]:
        return argv[argv.index("-m") + 1]
    return None


def _run_shadowed_sitecustomize():
    path = [entry for entry in sys.path if os.path.abspath(entry or ".") != _HERE]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", path)
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


if os.environ.get("INFERENCE_LAB_SIMULATE") == "1":
    _argv = _command_line()
    _module = _worker_module(_argv)
    if _module in ("dynamo.sglang", "dynamo.vllm"):
        from inference_lab_dynamo import bootstrap

        bootstrap.install(_module, _argv)

_run_shadowed_sitecustomize()
