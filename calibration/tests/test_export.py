import csv
import json

import pyarrow as pa
import pyarrow.parquet as pq

from specdec_calibration import schema
from specdec_calibration.export import TraceSignal, export_trace_bank


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_export_confidence_from_two_bank_run(tmp_path):
    run = tmp_path / "run"
    schema.write_acceptance(
        [
            {
                "model": "M",
                "speculator": "s",
                "config": "qualitative",
                "category": "coding",
                "prompt_idx": 0,
                "turn": 0,
                "round_idx": 0,
                "accept": 2,
                "acc0": 1,
                "acc1": 1,
                "acc2": 0,
            }
        ],
        width=3,
        path=run / "acceptance.parquet",
    )
    schema.write_speculator(
        [
            {
                "model": "M",
                "speculator": "s",
                "config": "qualitative",
                "category": "coding",
                "prompt_idx": 0,
                "turn": 0,
                "round_idx": 0,
                "conf0": 0.9,
                "conf1": 0.7,
                "conf2": None,
            }
        ],
        width=3,
        path=run / "speculator.parquet",
    )

    out = tmp_path / "trace.csv"
    meta = tmp_path / "trace.meta.json"
    manifest = export_trace_bank(
        run_dir=run,
        output_path=out,
        signal=TraceSignal.CONFIDENCE,
        metadata_path=meta,
    )

    assert manifest.rows == 1
    assert manifest.depth == 2
    assert _read_csv(out) == [
        ["commits", "category", "a0", "a1"],
        ["2", "coding", "0.9", "0.7"],
    ]
    assert json.loads(meta.read_text())["signal"] == "confidence"


def test_export_oracle_from_raw_parquet_without_acc_columns(tmp_path):
    raw = tmp_path / "raw.parquet"
    pq.write_table(
        pa.table(
            {
                "category": ["math"],
                "accept": [1],
                "conf0": [0.2],
                "conf1": [0.8],
                "conf2": [0.6],
            }
        ),
        raw,
    )
    out = tmp_path / "oracle.csv"

    manifest = export_trace_bank(raw_path=raw, output_path=out, signal=TraceSignal.ORACLE)

    assert manifest.depth == 3
    assert _read_csv(out) == [
        ["commits", "category", "a0", "a1", "a2"],
        ["1", "math", "1.0", "0.0", "0.0"],
    ]


def test_export_drops_all_null_trailing_signal_column(tmp_path):
    raw = tmp_path / "raw.parquet"
    pq.write_table(
        pa.table(
            {
                "category": ["coding", "coding"],
                "accept": [2, 0],
                "conf0": [0.9, 0.3],
                "conf1": [0.8, 0.2],
                "conf2": [None, None],
            }
        ),
        raw,
    )
    out = tmp_path / "trace.csv"

    manifest = export_trace_bank(raw_path=raw, output_path=out)

    assert manifest.depth == 2
    assert _read_csv(out)[0] == ["commits", "category", "a0", "a1"]
