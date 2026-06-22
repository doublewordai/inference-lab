"""EAGLE-family capture hooks (EAGLE, EAGLE3, MTP/NEXTN, STANDALONE).

Pinned to SGLANG_COMMIT. Three behavior-preserving monkeypatches:

  * EagleVerifyInputV2Mixin.sample  -> accept side. Runs eagerly each round even
    under CUDA graphs. Reads accept_lens/accept_index (topk==1: accept mask is the
    front chain) AND drains the pinned confidence buffer for the same round.

  * EagleDraftWorker._draft_extend_for_{prefill,decode} -> confidence for draft
    position 0. This is the first proposed token for the next verify round; SGLang's
    topk=1 fast path overwrites its public `topk_p` with 1.0, so we capture from
    the draft-extend logits before that signal is discarded.

  * EagleDraftWorker.draft_forward -> confidence for draft positions 1..D-1,
    CUDA-GRAPH SAFE. The draft logits live inside the decode graph; SGLang hardcodes
    the draft prob to 1.0 on the topk==1 fast path. We tee the per-step draft
    forward and write the real confidence (max softmax prob of the argmax-chosen
    token) into a PINNED host buffer via a device->pinned copy. That copy is legal
    during graph capture and is recorded into the graph, so it re-executes on every
    replay -- delivering real per-step confidence during generation even though our
    Python never runs on replay.

This is why EAGLE now yields both banks in a single graphs-on run (correct
acceptance), the same as DFLASH. Eager mode is no longer required for confidence.

Pairing: draft-extend after round N fills slot 0 for round N+1, draft_forward fills
slots 1..D-1 just before verify, and sample() drains the buffer for that round.
"""

from __future__ import annotations

import sys

from ..config import EAGLE_CONF_CAPTURE_MAX_BATCH
from .buffer import CaptureBuffer

_MAX_BS = EAGLE_CONF_CAPTURE_MAX_BATCH  # pinned buffer width


def _warn(what: str, err: Exception) -> None:
    print(f"[specdec-calibration] {what} capture failed: {err!r}", file=sys.stderr)


def _warn_once(state: dict, key: str, msg: str) -> None:
    warned = state.setdefault("warned", set())
    if key not in warned:
        warned.add(key)
        print(f"[specdec-calibration] {msg}", file=sys.stderr, flush=True)


def install(buffer: CaptureBuffer):
    import os

    import torch
    from sglang.srt.speculative.eagle_info_v2 import EagleVerifyInputV2Mixin
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

    assert hasattr(EagleVerifyInputV2Mixin, "sample"), "EagleVerifyInputV2Mixin.sample missing"
    assert hasattr(EagleDraftWorker, "draft_forward"), "EagleDraftWorker.draft_forward missing"
    assert hasattr(EagleDraftWorker, "_draft_extend_for_prefill"), (
        "EagleDraftWorker._draft_extend_for_prefill missing"
    )
    assert hasattr(EagleDraftWorker, "_draft_extend_for_decode"), (
        "EagleDraftWorker._draft_extend_for_decode missing"
    )

    orig_sample = EagleVerifyInputV2Mixin.sample
    orig_draft = EagleDraftWorker.draft_forward
    orig_extend_prefill = EagleDraftWorker._draft_extend_for_prefill
    orig_extend_decode = EagleDraftWorker._draft_extend_for_decode
    capture_conf = not os.environ.get("SPECDEC_NO_CONF")

    # Pinned host buffer, allocated once, never reallocated (so the graph-recorded
    # copies keep targeting a valid host address). [num_steps, _MAX_BS], filled per
    # round by draft-extend (slot 0) and draft_forward (slots 1..D-1).
    state: dict = {"pinned": None, "num_steps": None}
    counters: dict = {}  # rid -> next round_idx (EAGLE v2 does not bump spec_verify_ct)

    def _pinned(num_steps: int):
        if state["pinned"] is None:
            state["pinned"] = torch.zeros(num_steps, _MAX_BS, dtype=torch.float32).pin_memory()
            state["num_steps"] = num_steps
        return state["pinned"]

    def _capture_logits_slot(num_steps: int, slot: int, logits) -> None:
        if slot < 0 or slot >= num_steps:
            return
        bs = min(logits.shape[0], _MAX_BS)
        # max softmax prob == prob of the argmax-chosen draft token (topk==1).
        p = torch.softmax(logits.float(), dim=-1).amax(dim=-1)
        _pinned(num_steps)[slot, :bs].copy_(p[:bs], non_blocking=True)

    def patched_extend_prefill(self, batch, target_hidden_states, next_token_ids, mm_input_embeds=None):
        num_steps = self.speculative_num_steps
        runner = self.draft_runner
        orig_fwd = runner.forward

        def teed(fb, *a, **k):
            r = orig_fwd(fb, *a, **k)
            try:
                _capture_logits_slot(num_steps, 0, r.logits_output.next_token_logits)
            except Exception:
                pass
            return r

        runner.forward = teed
        try:
            return orig_extend_prefill(
                self, batch, target_hidden_states, next_token_ids, mm_input_embeds
            )
        finally:
            runner.forward = orig_fwd

    def patched_extend_decode(self, batch, batch_result):
        num_steps = self.speculative_num_steps
        # Mirror SGLang's row selection inside _draft_extend_for_decode: the full
        # draft-extend logits have `speculative_num_draft_tokens` rows per request,
        # and the next round's root token is the accepted path's last row.
        try:
            select_index = (
                torch.arange(len(batch.seq_lens), device=self.device)
                * self.speculative_num_draft_tokens
                + batch_result.accept_lens
                - 1
            )
        except Exception:
            select_index = None

        runner = self.draft_runner
        orig_fwd = runner.forward
        graph_runner = self.cuda_graph_runner_for_draft_extend
        orig_replay = getattr(graph_runner, "replay", None) if graph_runner is not None else None
        patched_replay = False

        def _capture_from_output(logits_output):
            if select_index is None:
                return
            logits = logits_output.next_token_logits[select_index]
            _capture_logits_slot(num_steps, 0, logits)

        def teed_forward(fb, *a, **k):
            r = orig_fwd(fb, *a, **k)
            try:
                _capture_from_output(r.logits_output)
            except Exception:
                pass
            return r

        def teed_replay(fb, *a, **k):
            out = orig_replay(fb, *a, **k)
            try:
                _capture_from_output(out)
            except Exception:
                pass
            return out

        runner.forward = teed_forward
        if graph_runner is not None and orig_replay is not None:
            try:
                graph_runner.replay = teed_replay
                patched_replay = True
            except Exception:
                patched_replay = False
        try:
            return orig_extend_decode(self, batch, batch_result)
        finally:
            runner.forward = orig_fwd
            if patched_replay:
                graph_runner.replay = orig_replay

    def patched_draft_forward(self, forward_batch):
        num_steps = self.speculative_num_steps
        runner = self.draft_runner
        orig_fwd = runner.forward
        step_box = {"i": 0}

        def teed(fb, *a, **k):
            r = orig_fwd(fb, *a, **k)
            try:
                logits = r.logits_output.next_token_logits  # [bs, vocab]
                i = step_box["i"]
                _capture_logits_slot(num_steps, i + 1, logits)  # slot 0 belongs to draft-extend
                step_box["i"] = i + 1
            except Exception:
                pass
            return r

        runner.forward = teed
        try:
            return orig_draft(self, forward_batch)
        finally:
            runner.forward = orig_fwd

    def patched_sample(self, batch, logits_output, vocab_mask=None):
        out = orig_sample(self, batch, logits_output, vocab_mask)
        try:
            _capture(batch, out)
        except Exception as e:
            _warn("eagle accept", e)
        return out

    def _capture(batch, out):
        _predict, accept_lens, accept_index = out
        if accept_index is None or accept_index.ndim != 2:
            return
        bs, ncols = accept_index.shape
        width = ncols - 1  # num_steps draft positions (col 0 = root/bonus)
        lens = accept_lens.detach().cpu().tolist()

        pinned = state["pinned"]
        conf_host = None
        if capture_conf and pinned is not None:
            torch.cuda.synchronize()  # ensure the graphed device->pinned copies landed
            conf_host = pinned.cpu().tolist()  # [num_steps][_MAX_BS]

        reqs = getattr(batch, "reqs", None) or []
        r_rids, r_rounds, r_accepts = [], [], []
        for i in range(min(bs, len(reqs))):
            rid = reqs[i].rid
            rnd = counters.get(rid, 0)
            counters[rid] = rnd + 1
            accepted = max(0, int(lens[i]) - 1)  # drop the bonus token
            mask = [1 if d < accepted else 0 for d in range(width)]
            buffer.add_accept(rid, rnd, accepted, mask)
            if conf_host is not None:
                if i >= _MAX_BS:
                    _warn_once(
                        state,
                        "conf_batch_overflow",
                        "eagle confidence batch exceeds pinned capture width; "
                        "overflow request confidence is null-padded",
                    )
                    vec = [None] * width
                else:
                    vec = [conf_host[k][i] for k in range(min(width, len(conf_host)))]
                buffer.add_conf(rid, rnd, vec[:width])
            r_rids.append(rid)
            r_rounds.append(rnd)
            r_accepts.append(int(lens[i]))  # accepted chain length incl. bonus
        # expert routing for this round's verify forward (npos = num_draft_tokens = ncols)
        from . import routing

        routing.emit_round(r_rids, r_rounds, ncols, r_accepts)

    EagleVerifyInputV2Mixin.sample = patched_sample
    if capture_conf:
        EagleDraftWorker._draft_extend_for_prefill = patched_extend_prefill
        EagleDraftWorker._draft_extend_for_decode = patched_extend_decode
        EagleDraftWorker.draft_forward = patched_draft_forward

    def uninstall():
        EagleVerifyInputV2Mixin.sample = orig_sample
        if capture_conf:
            EagleDraftWorker._draft_extend_for_prefill = orig_extend_prefill
            EagleDraftWorker._draft_extend_for_decode = orig_extend_decode
            EagleDraftWorker.draft_forward = orig_draft

    return uninstall
