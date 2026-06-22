import pyarrow.parquet as pq

from specdec_calibration import schema


def _accept_row(prompt_idx, round_idx, accept, mask):
    row = {
        "model": "M",
        "speculator": "eagle3:draft",
        "config": "qualitative",
        "category": "coding",
        "prompt_idx": prompt_idx,
        "turn": 0,
        "round_idx": round_idx,
        "accept": accept,
    }
    for k, v in enumerate(mask):
        row[f"acc{k}"] = v
    return row


def test_acceptance_roundtrip(tmp_path):
    rows = [
        _accept_row(0, 0, 2, [1, 1, 0, 0]),
        _accept_row(0, 1, 0, [0, 0, 0, 0]),
    ]
    path = tmp_path / "acceptance.parquet"
    n = schema.write_acceptance(rows, width=4, path=path)
    assert n == 2
    t = pq.read_table(path)
    assert t.column_names == schema.KEY_COLUMNS + ["accept"] + [f"acc{k}" for k in range(4)]
    d = t.to_pydict()
    assert d["accept"] == [2, 0]
    assert d["turn"] == [0, 0]
    assert d["acc0"] == [1, 0]


def test_speculator_roundtrip(tmp_path):
    rows = [
        {
            "model": "M", "speculator": "s", "config": "c", "category": "cat",
            "prompt_idx": 0, "turn": 0, "round_idx": 0,
            "conf0": 0.9, "conf1": 0.5, "conf2": 0.1, "conf3": 0.05,
        }
    ]
    path = tmp_path / "speculator.parquet"
    schema.write_speculator(rows, width=4, path=path)
    t = pq.read_table(path)
    conf_cols = [c for c in t.column_names if c.startswith("conf") and c[4:].isdigit()]
    assert conf_cols == [f"conf{k}" for k in range(4)]
    assert t.to_pydict()["turn"] == [0]
    assert t.to_pydict()["conf0"] == [0.9]


def test_stats(tmp_path):
    rows = [_accept_row(0, i, c, [1] * c + [0] * (4 - c)) for i, c in enumerate([0, 2, 2, 4])]
    stats = schema.write_stats(rows, "eagle3:draft", width=4, path=tmp_path / "stats.json")
    assert stats["total_rounds"] == 4
    assert stats["mean_commits_overall"] == 2.0
    assert stats["accept_hist"] == {"0": 1, "2": 2, "4": 1}
    # P(>=1 committed) = 3/4; P(>=4 committed) = 1/4
    assert stats["accept_by_position"][0] == 0.75
    assert stats["accept_by_position"][3] == 0.25
