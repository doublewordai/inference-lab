"""Runtime hooks that capture per-round acceptance + draft confidence from SGLang.

Two families (selected by `Speculator.family`):
  - "eagle"  -> hooks/eagle.py   (EAGLE, EAGLE3, MTP/NEXTN, STANDALONE)
  - "dflash" -> hooks/dflash.py  (DFLASH)

Each family monkeypatches SGLang `*_v2` internals pinned to SGLANG_COMMIT, writing
into a shared CaptureBuffer. The buffer (buffer.py) is engine-agnostic and unit
tested without SGLang; the family modules import SGLang lazily so this package
imports fine on a laptop.
"""

from __future__ import annotations

from .buffer import CaptureBuffer

__all__ = ["CaptureBuffer", "install", "uninstall"]


def install(family: str, buffer: CaptureBuffer):
    """Install the hook family and return an uninstall handle."""
    if family == "eagle":
        from . import eagle

        return eagle.install(buffer)
    if family == "dflash":
        from . import dflash

        return dflash.install(buffer)
    raise ValueError(f"no hook family {family!r}")


def uninstall(handle) -> None:
    if handle is not None:
        handle()
