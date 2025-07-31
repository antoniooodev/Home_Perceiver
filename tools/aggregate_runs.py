#!/usr/bin/env python
"""
Merge all per-run *.summary.csv files into one wide table:
  output/runs_summary_all.csv
Columns:
  run, frames, duration, avg_fps, <class_1>, <class_2>, …
Assumes each run has:
  output/csv/<run_id>.summary.csv
  output/json/<run_id>.jsonl
"""
from __future__ import annotations
import json, glob, collections, pandas as pd, pathlib

ROOT   = pathlib.Path.cwd()
CSV_IN = ROOT / "output/csv"
JSON_IN = ROOT / "output/json"
CSV_OUT = ROOT / "output/runs_summary_all.csv"

rows = []
for csv_path in CSV_IN.glob("*.summary.csv"):
    run_id = csv_path.stem.replace(".summary", "")        # <- strip suffix

    # per-class counts
    cls_counts = (
        pd.read_csv(csv_path)
          .set_index("class")["count"]
          .to_dict()
    )

    # global stats (first line of JSONL, written by Exporter.close())
    meta = {}
    jsonl_path = JSON_IN / f"{run_id}.jsonl"
    if jsonl_path.exists():
        with jsonl_path.open() as fh:
            first_line = json.loads(fh.readline())
            meta = {
                "frames":   first_line.get("frames_total", 0),
                "duration": first_line.get("duration_sec", 0.0),
                "avg_fps":  first_line.get("avg_fps", 0.0),
            }

    rows.append({"run": run_id, **meta, **cls_counts})

pd.DataFrame(rows).fillna(0).to_csv(CSV_OUT, index=False)
print("Wrote", CSV_OUT.relative_to(ROOT))