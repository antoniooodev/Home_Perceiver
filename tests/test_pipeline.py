"""Smoke‑tests for core modules.

Run with:
    python -m pytest -q   # ensures venv interpreter is used

The tests avoid heavy models or webcam access and finish in <1 s.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tracker.tracker import Tracker
from core_utils.exporter import Exporter
import tools.analyze_run as analyze_run


# -----------------------------------------------------------------------------
# Tracker – ID persistence
# -----------------------------------------------------------------------------

def test_tracker_id_persistence() -> None:
    trk = Tracker(iou_threshold=0.3, max_lost=1)
    det1 = [[10, 10, 50, 50, 0.9, 0]]
    det2 = [[12, 12, 52, 52, 0.8, 0]]

    ids1 = [tid for tid, _ in trk.update(det1)]
    ids2 = [tid for tid, _ in trk.update(det2)]

    assert ids1 and ids1 == ids2, "Tracker should keep the same ID across frames"


# -----------------------------------------------------------------------------
# Exporter – file creation
# -----------------------------------------------------------------------------

def test_exporter_creates_files(tmp_path: Path) -> None:
    # redirect Exporter output to temp dir
    from core_utils import exporter as exp_mod

    exp_mod.Exporter.root = tmp_path  # type: ignore[attr-defined]
    for sub in ("json", "csv", "videos"):
        (tmp_path / sub).mkdir(parents=True)

    run_id = "testrun"
    exp = Exporter(run_id, save_video=False, fps=30, res=(640, 480))
    for i in range(3):
        exp.log_frame({"frame": i, "mode": "test", "boxes": [], "class_labels": []})
    exp.close()

    assert (tmp_path / "json" / f"{run_id}.jsonl").exists()
    csv_file = tmp_path / "csv" / f"{run_id}.summary.csv"
    assert csv_file.exists()

    csv = pd.read_csv(csv_file, header=None)
    assert int(csv.loc[csv[0] == "frames_total", 1].iat[0]) == 3


# -----------------------------------------------------------------------------
# analyze_run – summary & plot generation
# -----------------------------------------------------------------------------

def test_analyze_run_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jsonl = tmp_path / "mini.jsonl"
    frames = [
        {"frame": 0, "mode": "X", "boxes": [], "class_labels": ["cat", "dog"]},
        {"frame": 1, "mode": "X", "boxes": [], "class_labels": ["cat"]},
    ]
    jsonl.write_text("\n".join(json.dumps(f) for f in frames))

    monkeypatch.setattr(sys, "argv", ["analyze_run", "--run", str(jsonl)])
    analyze_run.main()

    csv_out = jsonl.with_suffix(".summary.csv")
    png_out = jsonl.with_suffix(".classes.png")

    assert csv_out.exists() and png_out.exists()
    df = pd.read_csv(csv_out)
    assert set(df["class"]) == {"cat", "dog"}
    assert df.loc[df["class"] == "cat", "count"].iat[0] == 2
