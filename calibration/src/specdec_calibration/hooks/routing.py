"""Expert-routing capture for the MoE target during spec-decode verify.

Hooks Qwen2MoeSparseMoeBlock (40 MoE layers in Qwen3.6-35B-A3B): tees the per-layer
TopK to record the selected expert ids (topk_ids, int32 [num_tokens, top_k]) into a
PINNED per-layer buffer via a device->pinned copy. As with confidence, the copy is
recorded into the verify CUDA graph so it replays without our Python.

The family verify hook (eagle.sample / dflash.verify) calls emit_round() once per
round -- it has the rid/round/accept context. We segment the flat verify tokens
(draft_token_num per request, in batch.reqs order) and write each round's expert-id
block to a binary sink (routing.bin) + a compact metadata JSONL the parent expands
into routing.npy + routing_meta.parquet.

Only target-verify forwards are captured (the draft model's own MoE is skipped).
"""

from __future__ import annotations

import json
import sys

from ..config import ROUTING_CAPTURE_MAX_TOKENS

_MAX_LAYERS = 64
_MAX_TOKENS = ROUTING_CAPTURE_MAX_TOKENS


def _warn(msg: str) -> None:
    print(f"[specdec-calibration] routing: {msg}", file=sys.stderr, flush=True)


class RoutingSink:
    """Binary block sink: routing.bin holds uint8 [nt, num_layers, top_k] per round;
    routing_meta.jsonl holds one line per round with the per-request segmentation."""

    def __init__(self, bin_path: str, meta_path: str):
        self.bin = open(bin_path, "ab")
        self.meta = open(meta_path, "a", buffering=1)

    def write_block(self, arr_bytes: bytes, meta: dict) -> None:
        self.bin.write(arr_bytes)
        self.bin.flush()
        self.meta.write(json.dumps(meta) + "\n")
        self.meta.flush()


class RoutingCapture:
    def __init__(self, sink: RoutingSink):
        import torch

        self.torch = torch
        self.sink = sink
        # [layer, token, slot] uint8; top_k discovered on first fill (<=_MAX? use 8)
        self.buf = torch.zeros(_MAX_LAYERS, _MAX_TOKENS, 8, dtype=torch.uint8).pin_memory()
        self.top_k = 8
        self.num_layers = 0  # max layer_id+1 actually seen
        self._host = None
        self._overflow_warned = False

    # --- fill side: patch the MoE block ------------------------------------
    def install(self):
        from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

        assert hasattr(Qwen2MoeSparseMoeBlock, "forward"), "Qwen2MoeSparseMoeBlock.forward missing"
        orig_forward = Qwen2MoeSparseMoeBlock.forward
        cap = self

        def patched(self, hidden_states, *args, **kwargs):
            fb = args[0] if args else kwargs.get("forward_batch")
            # only the target verify forward; skip draft-model MoE
            try:
                if fb is not None and hasattr(fb, "forward_mode"):
                    itv = getattr(fb.forward_mode, "is_target_verify", None)
                    if itv is not None and not itv():
                        return orig_forward(self, hidden_states, *args, **kwargs)
            except Exception:
                pass
            layer_id = int(getattr(self, "layer_id", 0))
            topk = self.topk
            orig_topk = topk.forward

            def teed(hs, router_logits, *a, **k):
                out = orig_topk(hs, router_logits, *a, **k)
                try:
                    ids = out.topk_ids  # [num_tokens, top_k] int32
                    nt = min(ids.shape[0], _MAX_TOKENS)
                    slots = min(ids.shape[1], cap.buf.shape[2])
                    cap.top_k = slots
                    if layer_id < _MAX_LAYERS:
                        cap.buf[layer_id, :nt, :slots].copy_(
                            ids[:nt, :slots].to(cap.torch.uint8), non_blocking=True
                        )
                        if layer_id + 1 > cap.num_layers:
                            cap.num_layers = layer_id + 1
                except Exception:
                    pass
                return out

            topk.forward = teed
            try:
                return orig_forward(self, hidden_states, *args, **kwargs)
            finally:
                topk.forward = orig_topk

        Qwen2MoeSparseMoeBlock.forward = patched

        def uninstall():
            Qwen2MoeSparseMoeBlock.forward = orig_forward

        return uninstall

    # --- drain side: called by the family verify hook per round ------------
    def emit_round(self, rids, rounds, npos, accepts):
        """rids/rounds/accepts: per request (batch order). npos: draft tokens per
        request (uniform). Writes the round's [nt, num_layers, top_k] block."""
        try:
            n = len(rids)
            if npos <= 0:
                return
            max_reqs = _MAX_TOKENS // npos
            if max_reqs <= 0:
                if not self._overflow_warned:
                    self._overflow_warned = True
                    _warn(f"round has {npos} verify positions, exceeding capture buffer")
                return
            if n > max_reqs:
                if not self._overflow_warned:
                    self._overflow_warned = True
                    _warn(
                        f"routing batch has {n * npos} verify positions; "
                        f"capturing first {max_reqs * npos}"
                    )
                n = max_reqs
                rids = rids[:n]
                rounds = rounds[:n]
                accepts = accepts[:n]
            nt = n * npos
            if nt == 0 or self.num_layers == 0:
                return
            self.torch.cuda.synchronize()  # ensure graphed copies landed
            L, S = self.num_layers, self.top_k
            # [L, nt, S] -> [nt, L, S]
            block = self.buf[:L, :nt, :S].numpy().transpose(1, 0, 2).copy()
            self.sink.write_block(
                block.tobytes(),
                {
                    "rids": [str(r) for r in rids],
                    "rounds": [int(r) for r in rounds],
                    "npos": int(npos),
                    "accepts": [int(a) for a in accepts],
                    "nt": int(nt),
                    "L": int(L),
                    "S": int(S),
                },
            )
        except Exception as e:
            _warn(f"emit_round failed: {e!r}")


# module singleton (installed by the plugin in the worker process)
_ROUTING: RoutingCapture | None = None


def install(bin_path: str, meta_path: str):
    global _ROUTING
    _ROUTING = RoutingCapture(RoutingSink(bin_path, meta_path))
    return _ROUTING.install()


def emit_round(rids, rounds, npos, accepts) -> None:
    if _ROUTING is not None:
        _ROUTING.emit_round(rids, rounds, npos, accepts)
