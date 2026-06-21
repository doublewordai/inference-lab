"""DFLASH capture hooks. Pinned to SGLANG_COMMIT.

DFLASH drafts a whole non-causal block in one forward (no per-depth loop) and is
spec-v1 (non-overlap). Two behavior-preserving monkeypatches:

  * DFlashVerifyInput.verify -> accept side. Returns (new_bonus_tokens, commit_lens,
    next_target_hidden, num_correct_drafts_per_req_cpu). `num_correct` is the accepted
    draft prefix length per request; DFLASH increments req.spec_verify_ct here, giving
    a free per-request round counter. Per-position accept mask = arange(W) < accepted.

  * DFlashWorker._greedy_sample_from_vocab_parallel_head -> confidence side. DFLASH
    argmaxes the target LM head over draft hidden states and keeps only token ids, so
    we recompute softmax(hidden @ lm_head.T)[chosen] for the block. Supported on the
    TP=1 / no-added-vocab fast path (single-GPU offline Engine); otherwise confidence
    is skipped (accept still captured).

Pairing: _greedy_sample runs in draft preparation, before verify, within one round;
the block confidence is held in a pending slot keyed by batch row and consumed by the
next verify().
"""

from __future__ import annotations

import sys

from .buffer import CaptureBuffer

_WARNED: set = set()


def _warn(what: str, err: Exception) -> None:
    print(f"[specdec-calibration] {what} capture failed: {err!r}", file=sys.stderr)


def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"[specdec-calibration] {msg}", file=sys.stderr)


def install(buffer: CaptureBuffer):
    import torch
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput
    from sglang.srt.speculative.dflash_worker import DFlashWorker

    assert hasattr(DFlashVerifyInput, "verify"), "DFlashVerifyInput.verify missing"
    assert hasattr(DFlashWorker, "_greedy_sample_from_vocab_parallel_head"), (
        "DFlashWorker._greedy_sample_from_vocab_parallel_head missing"
    )

    orig_verify = DFlashVerifyInput.verify
    orig_sample = DFlashWorker._greedy_sample_from_vocab_parallel_head

    state: dict = {"pending_flat": None}  # flat [bs*(block-1)] confidence, row-major

    def _recompute_conf(hidden_states, lm_head, out_tokens) -> list:
        """softmax(hidden @ lm_head.weight[:num_org].T)[chosen], TP=1 fast path only."""
        shard = lm_head.shard_indices
        if int(shard.num_added_elements) != 0:
            raise RuntimeError("added-vocab shard; conf unsupported in v1")
        weight = lm_head.weight
        num_org = int(shard.num_org_elements)
        org_start = int(shard.org_vocab_start_index)
        n = int(hidden_states.shape[0])
        conf = [None] * n
        chunk = 256
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            hs = hidden_states[start:end]
            hs = hs if hs.dtype == weight.dtype else hs.to(weight.dtype)
            logits = torch.matmul(hs, weight[:num_org].T).float()
            probs = torch.softmax(logits, dim=-1)
            idx = (out_tokens[start:end].long() - org_start).clamp(min=0).view(-1, 1)
            p = probs.gather(1, idx).squeeze(1).detach().cpu().tolist()
            conf[start:end] = [float(x) for x in p]
        return conf

    def patched_sample(self, *, hidden_states, lm_head, chunk_size: int = 256):
        out = orig_sample(
            self, hidden_states=hidden_states, lm_head=lm_head, chunk_size=chunk_size
        )
        try:
            state["pending_flat"] = _recompute_conf(hidden_states, lm_head, out)
        except Exception as e:
            _warn_once("dflash_conf", f"dflash conf disabled (TP>1 / added vocab?): {e!r}")
            state["pending_flat"] = None
        return out

    def patched_verify(self, *, batch, logits_output, page_size):
        out = orig_verify(
            self, batch=batch, logits_output=logits_output, page_size=page_size
        )
        try:
            _capture_accept(self, batch, out)
        except Exception as e:
            _warn("dflash accept", e)
        return out

    def _capture_accept(verify_input, batch, out):
        _new_bonus, _commit_lens, _next_hidden, num_correct = out
        ntok = int(verify_input.draft_token_num)  # block_size verify positions per req
        width = ntok - 1  # block_size - 1 draft positions
        reqs = getattr(batch, "reqs", None) or []
        pending = state.get("pending_flat")
        state["pending_flat"] = None
        r_rids, r_rounds, r_accepts = [], [], []
        for i in range(min(len(reqs), len(num_correct))):
            req = reqs[i]
            accepted = max(0, int(num_correct[i]))
            rnd = int(getattr(req, "spec_verify_ct", 1)) - 1  # incremented in verify()
            mask = [1 if d < accepted else 0 for d in range(width)]
            buffer.add_accept(req.rid, rnd, accepted, mask)
            if pending is not None:
                row = pending[i * width : (i + 1) * width]
                if len(row) == width:
                    buffer.add_conf(req.rid, rnd, list(row))
            r_rids.append(req.rid)
            r_rounds.append(rnd)
            r_accepts.append(accepted + 1)  # committed positions incl. bonus
        from . import routing

        routing.emit_round(r_rids, r_rounds, ntok, r_accepts)

    DFlashVerifyInput.verify = patched_verify
    DFlashWorker._greedy_sample_from_vocab_parallel_head = patched_sample

    def uninstall():
        DFlashVerifyInput.verify = orig_verify
        DFlashWorker._greedy_sample_from_vocab_parallel_head = orig_sample

    return uninstall
