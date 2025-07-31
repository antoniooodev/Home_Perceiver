"""Headless (optionally live) multi-class detection.
Exports JSONL/CSV (+MP4) into output/json, output/csv, output/videos.
Use --no-display to disable the live preview window.
"""
from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path
from typing import List

import cv2
import torch
from ultralytics import YOLO

from core_utils.data_loader import VideoStream
from core_utils.exporter import Exporter

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def draw_boxes(img, boxes: List[List[float]], labels: List[str]):
    """Draw simple red boxes + label."""
    for box, lab in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, lab, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser("Multiclass detection")
    ap.add_argument("--source", required=True, help="0=webcam or path/URL")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--no-display", action="store_true", help="disable live preview window")
    args = ap.parse_args()

    src = int(args.source) if str(args.source).isdigit() else args.source
    stream = VideoStream(src)
    if not stream.cap.isOpened():
        raise RuntimeError(f"Cannot open video source {args.source}")

    dev = torch.device(args.device)
    fps = stream.cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(stream.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(stream.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO("yolov5n.pt", task="detect")
    model.model.to(dev).eval().fuse()
    names = model.names

    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    exporter: Exporter | None = None
    if not args.no_export:
        exporter = Exporter(run_id, save_video=args.save_video, fps=fps, res=(w, h))

    frame_idx = 0
    try:
        while True:
            frame, _ = stream.read_frame(to_tensor=False)
            if frame is None:
                break

            res = model(frame)[0]
            boxes = res.boxes.xyxy.cpu().numpy().tolist()
            labels = [names[int(c)] for c in res.boxes.cls.cpu().numpy()]

            vis = frame.copy() if (args.save_video or not args.no_display) else None
            if vis is not None:
                draw_boxes(vis, boxes, labels)

            if exporter:
                exporter.log_frame({
                    "frame": frame_idx,
                    "mode": "Multi",
                    "boxes": boxes,
                    "ids": [],
                    "class_labels": labels,
                }, vis_frame=vis if args.save_video else None)

            if not args.no_display:
                cv2.imshow("Multi", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_idx += 1
    finally:
        if exporter:
            exporter.close()
        stream.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
