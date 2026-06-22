from specdec_calibration.hooks.buffer import CaptureBuffer


def test_merge_accept_and_conf():
    buf = CaptureBuffer(width=4)
    buf.register("r0", config="qualitative", category="coding", prompt_idx=0)
    # round 0: drafter proposed 4, two accepted
    buf.add_accept("r0", 0, accept=2, accept_mask=[1, 1, 0, 0])
    buf.add_conf("r0", 0, conf=[0.9, 0.8, 0.4, 0.2])

    acc = buf.acceptance_rows(model="M", speculator="eagle")
    spc = buf.speculator_rows(model="M", speculator="eagle")
    assert len(acc) == 1 and len(spc) == 1
    a = acc[0]
    assert a["accept"] == 2 and a["acc0"] == 1 and a["acc3"] == 0
    assert a["config"] == "qualitative" and a["prompt_idx"] == 0
    assert spc[0]["conf0"] == 0.9 and spc[0]["conf3"] == 0.2


def test_short_capture_is_null_padded():
    buf = CaptureBuffer(width=8)
    buf.register("r1", config="c", category="cat", prompt_idx=3)
    buf.add_accept("r1", 0, accept=1, accept_mask=[1, 0])  # only 2 positions captured
    buf.add_conf("r1", 0, conf=[0.7, 0.3])
    a = buf.acceptance_rows("M", "s")[0]
    s = buf.speculator_rows("M", "s")[0]
    assert a["acc1"] == 0 and a["acc2"] is None and a["acc7"] is None
    assert s["conf1"] == 0.3 and s["conf7"] is None


def test_accept_only_round_skipped_in_speculator_bank():
    buf = CaptureBuffer(width=2)
    buf.register("r2", config="c", category="cat", prompt_idx=0)
    buf.add_accept("r2", 0, accept=0, accept_mask=[0, 0])  # no conf captured
    assert len(buf.acceptance_rows("M", "s")) == 1
    assert len(buf.speculator_rows("M", "s")) == 0


def test_rows_sorted_by_request_and_round():
    buf = CaptureBuffer(width=1)
    # insert out of order; output must be deterministic, sorted by (rid, round_idx)
    for i, (rid, ridx) in enumerate([("r1", 1), ("r0", 0), ("r0", 1)]):
        buf.register(rid, config="c", category="cat", prompt_idx=i)
        buf.add_accept(rid, ridx, accept=1, accept_mask=[1])
    rows = buf.acceptance_rows("M", "s")
    assert len(rows) == 3
    # sorted by (rid, round_idx): ("r0",0), ("r0",1), ("r1",1)
    assert [r["round_idx"] for r in rows] == [0, 1, 1]
