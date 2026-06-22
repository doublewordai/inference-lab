"""SGLang plugin entry point. Registered under the `sglang.srt.plugins` group, so
`load_plugins()` calls register() at the very start of every scheduler child process
-- which is where the speculative workers actually run. This is the only place a
monkeypatch can reach them (the child is spawned, not forked, so parent-process
patches are not inherited).

Activated only when SPECDEC_CAPTURE_PATH + SPECDEC_HOOK_FAMILY are set (the parent
sets them before building the Engine; spawn inherits os.environ), so the plugin is
inert in any other SGLang process.
"""

from __future__ import annotations

import os


def register() -> None:
    path = os.environ.get("SPECDEC_CAPTURE_PATH")
    family = os.environ.get("SPECDEC_HOOK_FAMILY")
    if not path or not family:
        return
    try:
        from .hooks import install
        from .hooks.buffer import FileSink

        install(family, FileSink(path))
        print(f"[specdec-calibration] plugin armed {family} hooks -> {path}", flush=True)
    except Exception as e:  # never break the scheduler
        import sys

        print(f"[specdec-calibration] plugin install failed: {e!r}", file=sys.stderr, flush=True)

    rbin = os.environ.get("SPECDEC_ROUTING_BIN")
    rmeta = os.environ.get("SPECDEC_ROUTING_META")
    if rbin and rmeta:
        try:
            from .hooks import routing

            routing.install(rbin, rmeta)
            print(f"[specdec-calibration] plugin armed routing -> {rbin}", flush=True)
        except Exception as e:
            import sys

            print(f"[specdec-calibration] routing install failed: {e!r}", file=sys.stderr, flush=True)
