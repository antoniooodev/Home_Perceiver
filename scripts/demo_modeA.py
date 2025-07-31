"""Demo Mode A – person tight-mask with IoU tracking.
Exports one JSONL and one CSV per run (output/json, output/csv) and
optionally an annotated MP4 (output/videos) when --save-video is used.
Pipeline logic is unchanged; only minimal CLI and export hooks added.
"""
import argparse
import time
import uuid

import cv2
import torch
from ultralytics import YOLO

from core_utils.data_loader import get_video_stream
from core_utils.exporter import Exporter
from tracker.tracker import Tracker


def load_model(device: torch.device) -> YOLO:
    """COCO-pretrained YOLOv8 nano segmentation model."""
    model = YOLO("yolov8n-seg.pt", task="segment")
    model.model.to(device).eval().fuse()
    return model


def filter_person(result):
    """Return list[[x1,y1,x2,y2], conf] where class == person (cls 0)."""
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    return [([*map(float, xyxy[i])], float(conf[i])) for i in range(len(cls)) if cls[i] == 0]


def main() -> None:
    parser = argparse.ArgumentParser("Mode-A demo – mask + tracking")
    parser.add_argument("--source", default=0, help="0 = webcam or path/URL")
    parser.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    parser.add_argument("--save-video", action="store_true", help="store annotated MP4")
    parser.add_argument("--no-export", action="store_true", help="disable JSON/CSV export")
    args = parser.parse_args()

    device = torch.device(args.device)
    stream = get_video_stream()
    fps = stream.cap.get(cv2.CAP_PROP_FPS) or 30.0

    model = load_model(device)
    tracker = Tracker(iou_threshold=0.3, max_lost=5)

    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    exporter = None

    idx = 0
    try:
        while True:
            frame, _ = stream.read_frame(to_tensor=False)
            if frame is None:
                break

            if idx == 0 and not args.no_export:
                h, w = frame.shape[:2]
                exporter = Exporter(run_id, save_video=args.save_video, fps=fps, res=(w, h))

            res = model(frame, imgsz=320)[0]
            dets = filter_person(res)
            tracks = tracker.update([[*d[0], d[1], 0] for d in dets])

            annotated = frame.copy()
            for tid, bbox in tracks:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"ID {tid}", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if exporter:
                boxes = [b for _, b in tracks]
                ids = [t for t, _ in tracks]
                exporter.log_frame(
                    {
                        "frame": idx,
                        "mode": "A",
                        "boxes": boxes,
                        "ids": ids,
                        "class_labels": ["person"] * len(boxes),
                    },
                    vis_frame=annotated if args.save_video else None,
                )

            cv2.imshow("Mode A", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            idx += 1

    finally:
        if exporter:
            exporter.close()
        stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()