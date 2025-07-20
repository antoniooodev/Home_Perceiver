# File: scripts/demo_modeA.py
# Mode A: Person Detection + Tight Semantic Segmentation Outline + Tracking IDs
# – YOLOv8-nano-seg for ultra-fast person detection+segmentation
# – No skeleton, only tight polygonal outline
# – IoU‐based tracker assigns persistent IDs

from core_utils.data_loader import get_video_stream
from tracker.tracker import Tracker
from pathlib import Path
from ultralytics import YOLO

import sys
import time
import cv2
import numpy as np
import torch

# project root on PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """
    Load YOLOv8-nano segmentation model (yolov8n-seg.pt) pretrained on COCO.
    """
    model = YOLO("yolov8n-seg.pt", task="segment")
    model.model.to(DEVICE).fuse()  # fuse conv+bn for speed
    return model


def create_person_mask_and_contour(result, frame_shape):
    """
    From a YOLOv8 'segment' result, merge all person-instance masks into one
    binary mask and extract its largest contour.

    Returns:
        contour (np.ndarray) or None
    """
    # guard: no masks
    if result.masks is None or result.masks.data is None:
        return None

    # raw masks [N, Hm, Wm], classes [N]
    masks = result.masks.data  # BoolTensor on DEVICE
    classes = result.boxes.data[:, 5].long()

    # select only person masks (class == 0)
    person_idxs = (classes == 0).nonzero(as_tuple=True)[0]
    if person_idxs.numel() == 0:
        return None

    # merge into a single mask
    merged = masks[person_idxs].any(dim=0).to(torch.uint8) * 255  # [Hm, Wm]
    small = merged.cpu().numpy()

    # upsample to full resolution
    Hf, Wf = frame_shape
    full_mask = cv2.resize(small, (Wf, Hf), interpolation=cv2.INTER_NEAREST)

    # clean + find largest contour
    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    return max(cnts, key=cv2.contourArea)


def main():
    # 1) initialize video stream (remote or local)
    vs = get_video_stream(use_ffmpeg=True)

    # 2) load segmentation model
    model = load_model()

    # 3) initialize tracker
    tracker = Tracker(iou_threshold=0.3, max_lost=5)

    print("Models loaded. Starting Mode A demo...")

    prev_time = time.time()
    while True:
        frame, _ = vs.read_frame(to_tensor=False)
        if frame is None:
            break

        # 4) run detection+segmentation (yolov8n-seg)
        result = model(frame, imgsz=320)[0]

        # 5) extract bounding boxes + classes for IoU‐based tracking
        #    result.boxes.xyxy: [N,4], result.boxes.conf: [N], result.boxes.cls: [N]
        bxy = result.boxes.xyxy.cpu().numpy()  # [N,4]
        confs = result.boxes.conf.cpu().numpy()  # [N]
        classes = result.boxes.cls.cpu().numpy()  # [N]
        dets = []
        for (x1, y1, x2, y2), conf, cls in zip(bxy, confs, classes):
            dets.append(
                [float(x1), float(y1), float(x2), float(y2), float(conf), int(cls)]
            )

        # 6) update tracker only on person detections
        person_dets = [d for d in dets if d[5] == 0]
        tracks = tracker.update(person_dets)

        # 7) for each active track, redraw tight outline + ID
        #    we approximate matching by IoU of box centroids
        for track_id, box in tracks:
            # get contour for this frame (common for all persons)
            cnt = create_person_mask_and_contour(result, frame.shape[:2])
            if cnt is not None:
                # draw polygonal outline
                cv2.polylines(frame, [cnt], True, (0, 255, 0), 2)

                # compute bounding rect of contour
                bx, by, bw, bh = cv2.boundingRect(cnt)

                # prepare ID text centered above
                text = f"ID {track_id}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale, thickness = 0.7, 2
                (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                tx = bx + (bw - tw) // 2
                ty = max(by - 10, th + 5)

                # draw background rectangle for readability
                cv2.rectangle(
                    frame,
                    (tx - 2, ty - th - 2),
                    (tx + tw + 2, ty + 2),
                    (0, 255, 0),
                    cv2.FILLED,
                )
                # draw ID text in black
                cv2.putText(
                    frame,
                    text,
                    (tx, ty),
                    font,
                    scale,
                    (0, 0, 0),
                    thickness,
                    cv2.LINE_AA,
                )

        # 8) compute & display true FPS
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Mode A: Tight Outline + Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
