"""Realtime YOLOv5-nano demo (legacy script).

* Launches webcam/RTSP via :pyfunc:`core_utils.data_loader.get_video_stream`.
* Uses helper wrappers in :pymod:`detector.yolo_utils` to load/run YOLOv5n.
* Shows live preview with bounding boxes; quits on ``q``.

"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import torch

from core_utils.data_loader import get_video_stream
from detector.yolo_utils import load_yolo_model, run_yolo_inference

__all__ = ["main"]

# Ensure project root is on sys.path for editable installs
proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))


# -----------------------------------------------------------------------------
# Drawing helper
# -----------------------------------------------------------------------------

def draw_boxes(
    frame: "np.ndarray",
    dets: List[Tuple[float, float, float, float, float]],
    *,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """Draw rectangles in *frame* for each ``(x1,y1,x2,y2,conf)`` tuple."""
    for x1, y1, x2, y2, _ in dets:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


# -----------------------------------------------------------------------------
# Main CLI entry
# -----------------------------------------------------------------------------

def main() -> None:
    print("Detection – CUDA available:", torch.cuda.is_available())
    print("Detection – default device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # 1. VideoStream (webcam by default; change env in get_video_stream)
    vs = get_video_stream()

    # 2. YOLOv5-nano via Ultralytics hub
    model = load_yolo_model("yolov5n")
    print("YOLOv5-nano model loaded.")

    try:
        while True:
            frame, _ = vs.read_frame()
            if frame is None:
                break
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)

            # 3. Detection
            dets = run_yolo_inference(model, frame)
            draw_boxes(frame, dets)

            cv2.imshow("YOLOv5n", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        vs.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
