"""Cross-platform video input helper.

Attributes
----------
cap : cv2.VideoCapture
fps : float
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class VideoStream:
    """Thin wrapper around ``cv2.VideoCapture``."""

    def __init__(self, source: int | str = 0) -> None:
        """Open webcam (0/1/…) or video file/URL."""
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._ts0 = time.time()

    # ------------------------------------------------------------------
    # Context-manager helpers
    # ------------------------------------------------------------------
    def __enter__(self) -> "VideoStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def read_frame(self, to_tensor: bool = False) -> Tuple[Optional[np.ndarray], float]:
        """Return (BGR frame, timestamp [s])."""
        ok, frame = self.cap.read()
        if not ok:
            return None, 0.0
        ts = time.time() - self._ts0
        if to_tensor:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.ascontiguousarray(frame).transpose(2, 0, 1)  # C,H,W
        return frame, ts

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()

def get_video_stream(source: int | str = 0) -> VideoStream:
    """Back-compat wrapper used by demo_modeA/B."""
    return VideoStream(source)