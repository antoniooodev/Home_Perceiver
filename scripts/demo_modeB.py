"""Demo Mode B – multiclass detection + optional person mask.
Exports JSONL/CSV (+MP4) into output/json, output/csv, output/videos.
"""
import argparse
import time
import uuid

import cv2
import torch
from ultralytics import YOLO

from core_utils.data_loader import get_video_stream
from core_utils.exporter import Exporter


def draw_boxes(img, boxes, labels):
    for box, lab in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, lab, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


def main() -> None:
    ap = argparse.ArgumentParser("Mode-B demo – multiclass detection")
    ap.add_argument("--source", default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    stream = get_video_stream()
    fps = stream.cap.get(cv2.CAP_PROP_FPS) or 30.0

    model = YOLO("yolov8n-seg.pt", task="detect")
    model.model.to(device).eval().fuse()
    names = model.names

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
                exporter = Exporter(run_id, save_video=args.save_video,
                                    fps=fps, res=(w, h))

            res = model(frame, imgsz=640)[0]
            boxes = res.boxes.xyxy.cpu().numpy().tolist()
            labels = [names[int(c)] for c in res.boxes.cls.cpu().numpy()]
            draw_boxes(frame, boxes, labels)

            if exporter:
                exporter.log_frame(
                    {
                        "frame": idx,
                        "mode": "B",
                        "boxes": boxes,
                        "ids": [],
                        "class_labels": labels,
                    },
                    vis_frame=frame if args.save_video else None,
                )

            cv2.imshow("Mode B", frame)
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