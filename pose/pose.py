#!/usr/bin/env python
"""Realtime Keypoint-R CNN demo with 17-pt skeleton overlay.

This is a *demo script* (not a library module):
  • opens webcam / URL via :pyfunc:`core_utils.data_loader.get_video_stream`
  • loads Keypoint R-CNN via :pymod:`pose.pose_utils`
  • renders joints + synthetic neck in an OpenCV window
Quit with *q*.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from core_utils.data_loader import get_video_stream
from pose.pose_utils import load_pose_model, run_pose_inference

__all__ = ["main"]

# -----------------------------------------------------------------------------
# Constants – COCO 17-pt mapping & skeleton edges
# -----------------------------------------------------------------------------

COCO_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

SKELETON = [
    # head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # shoulders & arms
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    # torso
    (5, 11), (6, 12), (11, 12),
    # legs
    (11, 13), (13, 15), (12, 14), (14, 16),
]


# -----------------------------------------------------------------------------
# Drawing helpers
# -----------------------------------------------------------------------------

def draw_skeleton(
    frame: np.ndarray,
    poses: List[dict[str, np.ndarray]],
    *,
    kp_radius: int = 3,
    kp_color: Tuple[int, int, int] = (0, 0, 255),
    line_color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> None:
    """Render keypoints + skeleton into *frame* (in-place)."""
    h, w = frame.shape[:2]
    for person in poses:
        kps = person["keypoints"]  # (17,3)

        # joints
        for x, y, v in kps:
            if v > 0 and 0 <= int(x) < w and 0 <= int(y) < h:
                cv2.circle(frame, (int(x), int(y)), kp_radius, kp_color, -1)

        # synthetic neck: nose → midpoint shoulders
        v0, v5, v6 = kps[0, 2], kps[5, 2], kps[6, 2]
        if v0 > 0 and v5 > 0 and v6 > 0:
            x0, y0 = kps[0, :2]
            mx, my = (kps[5, 0] + kps[6, 0]) / 2, (kps[5, 1] + kps[6, 1]) / 2
            cv2.line(frame, (int(x0), int(y0)), (int(mx), int(my)), line_color, thickness)

        # skeleton edges
        for i, j in SKELETON:
            if kps[i, 2] > 0 and kps[j, 2] > 0:
                cv2.line(
                    frame,
                    (int(kps[i, 0]), int(kps[i, 1])),
                    (int(kps[j, 0]), int(kps[j, 1])),
                    line_color,
                    thickness,
                )


# -----------------------------------------------------------------------------
# Main CLI entry
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    # add repo root to path for editable-install safety
    proj_root = Path(__file__).resolve().parents[1]
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))

    vs = get_video_stream()
    model, min_score = load_pose_model(pretrained=True, min_score=0.5)

    try:
        while True:
            frame, _ = vs.read_frame()
            if frame is None:
                break
            poses = run_pose_inference(model, frame, min_score)
            draw_skeleton(frame, poses)
            cv2.imshow("Pose", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        vs.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
