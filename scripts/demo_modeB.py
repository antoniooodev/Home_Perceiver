# File: scripts/demo_modeB.py
# Mode B: Person Mask + Multiclass Bounding Boxes
# – YOLOv8n‐seg for segmentation
# – Tight person outline with “person” label
# – Standard bounding boxes for all other COCO classes

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# project root on PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core_utils.data_loader import get_video_stream

# device selection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model():
    """
    Load YOLOv8‐nano segmentation model pretrained on COCO.
    """
    model = YOLO('yolov8n-seg.pt', task='segment')
    model.model.to(DEVICE).fuse()  # fuse conv+bn for speed
    return model


def draw_multiclass(frame, result, coco_names):
    """
    Draw tight person mask + label for class=0,
    and standard bounding boxes + labels for all other classes.
    """
    Hf, Wf = frame.shape[:2]
    kernel = np.ones((5, 5), np.uint8)

    # extract boxes [N,4], confidences [N], classes [N], masks [N,Hm,Wm]
    boxes_xy   = result.boxes.xyxy.cpu().numpy()            # [N,4]
    confs      = result.boxes.conf.cpu().numpy()            # [N]
    classes    = result.boxes.cls.cpu().numpy().astype(int) # [N]
    masks_data = getattr(result.masks, 'data', None)        # [N,Hm,Wm] or None

    for idx, (xy, conf, cls) in enumerate(zip(boxes_xy, confs, classes)):
        x1, y1, x2, y2 = map(int, xy)
        label = coco_names[cls] if cls < len(coco_names) else str(cls)

        if cls == 0 and masks_data is not None:
            # PERSON: extract that instance mask
            mask_small = masks_data[idx].cpu().numpy().astype(np.uint8) * 255
            full_mask  = cv2.resize(mask_small, (Wf, Hf), interpolation=cv2.INTER_NEAREST)
            clean      = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)
            cnts, _    = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue

            # draw the tight outline
            c = max(cnts, key=cv2.contourArea)
            cv2.polylines(frame, [c], True, (0, 255, 0), 2)

            # bounding rect to position the label
            bx, by, bw, bh = cv2.boundingRect(c)
            font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
            tx = bx + (bw - tw) // 2
            ty = max(by - 10, th + 5)

            # background box + text
            cv2.rectangle(frame, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, label, (tx, ty), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

        else:
            # OTHER CLASSES: standard bounding box + label
            color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            txt = f"{label} {conf:.2f}"
            font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            (tw, th), _ = cv2.getTextSize(txt, font, scale, thickness)
            ty = max(y1 - 10, th + 5)
            cv2.rectangle(frame, (x1, ty - th - 3), (x1 + tw + 3, ty + 3), color, cv2.FILLED)
            cv2.putText(frame, txt, (x1, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main():
    vs    = get_video_stream(use_ffmpeg=True)
    model = load_model()

    # grab the list of COCO class names from the model
    coco_names = model.names

    prev = time.time()
    print("Models loaded. Starting Mode B demo...")

    while True:
        frame, _ = vs.read_frame(to_tensor=False)
        if frame is None:
            break

        # inference (single frame)
        res = model(frame, imgsz=640, conf=0.15)[0]

        # overlay mask+boxes
        draw_multiclass(frame, res, coco_names)

        # compute & display FPS
        now = time.time()
        fps = 1.0 / (now - prev) if prev else 0.0
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Mode B: Person Mask + Multiclass Boxes", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()