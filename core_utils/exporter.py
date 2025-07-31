"""Frame-wise exporter for JSONL / CSV (+ optional MP4)."""

from __future__ import annotations

import csv
import json
import time
import uuid
from pathlib import Path
from typing import Mapping, Any, Optional

import cv2
import numpy as np


class Exporter:
    """Write one JSON record per frame and a summary CSV at close."""

    root = Path("output")
    (root / "json").mkdir(parents=True, exist_ok=True)
    (root / "csv").mkdir(parents=True, exist_ok=True)
    (root / "videos").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Construction / cleanup
    # ------------------------------------------------------------------
    def __init__(
        self,
        run_id: str,
        *,
        save_video: bool = False,
        fps: float = 30.0,
        res: tuple[int, int] = (640, 480),
    ) -> None:
        self.run_id = run_id
        self.fp_json = (self.root / "json" / f"{run_id}.jsonl").open("w")
        self.fp_csv  = (self.root / "csv"  / f"{run_id}.summary.csv").open("w", newline="")
        self.csv_writer = csv.writer(self.fp_csv)
        self.save_video = save_video
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w, h = res
            self.vw = cv2.VideoWriter(
                str(self.root / "videos" / f"{run_id}.mp4"), fourcc, fps, (w, h)
            )
        else:
            self.vw = None
        self.start = time.time()
        self.frame_count = 0

    def __enter__(self) -> "Exporter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def log_frame(
        self,
        record: Mapping[str, Any],
        *,
        vis_frame: Optional[np.ndarray] = None,
    ) -> None:
        """Write one JSON line and (optionally) one video frame."""
        self.fp_json.write(json.dumps(record) + "\n")
        self.frame_count += 1
        if self.save_video and vis_frame is not None and self.vw is not None:
            self.vw.write(vis_frame)

    def close(self) -> None:
        """Flush files and write a minimal CSV summary."""
        if not self.fp_json.closed:
            duration = time.time() - self.start
            avg_fps  = self.frame_count / duration if duration else 0.0
            # simple CSV: frames, duration, fps
            self.csv_writer.writerow(["frames_total", self.frame_count])
            self.csv_writer.writerow(["duration_sec", f"{duration:.2f}"])
            self.csv_writer.writerow(["avg_fps",      f"{avg_fps:.2f}"])
            self.fp_csv.close()
            self.fp_json.close()
        if self.vw is not None:
            self.vw.release()