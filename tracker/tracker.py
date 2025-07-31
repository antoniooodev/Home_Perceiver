"""IoU-based multi-object tracker (lightweight, no Kalman).

----------
Tracker.update(detections) → list[tuple[int, list[float]]]

"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

__all__ = ["Track", "Tracker"]


class Track:
    """Single object track with ID and bounding-box history."""

    def __init__(self, track_id: int, initial_box: List[float], *, max_history: int = 30):
        self.id: int = track_id
        self.boxes: List[List[float]] = [initial_box]  # [[x1,y1,x2,y2], …]
        self.lost_frames: int = 0
        self.max_history: int = max_history

    # ------------------------------------------------------------------
    def predict_box(self) -> List[float]:
        """Return the last box (no motion model)."""
        return self.boxes[-1]

    def update(self, new_box: List[float]) -> None:
        self.boxes.append(new_box)
        if len(self.boxes) > self.max_history:
            self.boxes.pop(0)
        self.lost_frames = 0

    def mark_missed(self) -> None:
        self.lost_frames += 1


# -----------------------------------------------------------------------------
# Tracker wrapper
# -----------------------------------------------------------------------------

class Tracker:
    """Greedy IoU association tracker (CPU-only, no re-ID)."""

    def __init__(self, *, iou_threshold: float = 0.3, max_lost: int = 5):
        self.iou_threshold: float = iou_threshold
        self.max_lost: int = max_lost
        self.next_id: int = 0
        self.tracks: List[Track] = []

    # ------------------------------------------------------------------
    @staticmethod
    def _iou(boxA: List[float], boxB: List[float]) -> float:
        """Intersection-over-Union between two boxes [x1,y1,x2,y2]."""
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interW, interH = max(0, xB - xA), max(0, yB - yA)
        inter_area = interW * interH
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union_area = areaA + areaB - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    # ------------------------------------------------------------------
    def update(self, detections: List[List[float]]) -> List[Tuple[int, List[float]]]:
        """Associate *detections* and return active tracks.

        Parameters
        ----------
        detections
            ``[[x1,y1,x2,y2,conf,cls], …]`` from detector.
        """
        # 1. Predict positions
        predicted = [t.predict_box() for t in self.tracks]

        # 2. IoU matrix
        iou_mat = np.zeros((len(predicted), len(detections)), dtype=np.float32)
        for t_idx, p in enumerate(predicted):
            for d_idx, det in enumerate(detections):
                iou_mat[t_idx, d_idx] = self._iou(p, det[:4])

        matched_tracks, matched_dets = set[int](), set[int]()
        # 3. Greedy matching
        while iou_mat.size:
            t_idx, d_idx = divmod(iou_mat.argmax(), iou_mat.shape[1])
            if iou_mat[t_idx, d_idx] < self.iou_threshold:
                break
            self.tracks[t_idx].update(detections[d_idx][:4])
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)
            iou_mat[t_idx, :] = -1
            iou_mat[:, d_idx] = -1

        # 4. Unmatched tracks
        for idx, track in enumerate(self.tracks):
            if idx not in matched_tracks:
                track.mark_missed()

        # 5. New tracks
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                self.tracks.append(Track(self.next_id, det[:4]))
                self.next_id += 1

        # 6. Drop stale tracks
        self.tracks = [t for t in self.tracks if t.lost_frames <= self.max_lost]

        # 7. Return state
        return [(t.id, t.predict_box()) for t in self.tracks]
