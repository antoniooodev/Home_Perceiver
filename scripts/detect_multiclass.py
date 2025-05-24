# File: scripts/detect_multiclass.py
# Mode B: COCO Multi‐Class Detection Only
# – YOLOv5-nano for ultra-fast multi-class detection via PyTorch Hub
# – Standard bounding boxes + labels for all COCO classes

import sys
import time
from pathlib import Path

import cv2
import torch

# ensure project root on PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core_utils.data_loader import get_video_stream
from detector.yolo_utils    import load_yolo_model, run_yolo_inference

# select device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def draw_multiclass(frame, detections, class_names):
    """
    Draw a bounding box + label for every detection.
    Args:
        frame (np.ndarray): BGR image.
        detections (List[tuple]): each tuple = (x1, y1, x2, y2, conf, cls).
        class_names (List[str]): COCO class names by index.
    """
    for x1, y1, x2, y2, conf, cls in detections:
        cls = int(cls)
        color = (0, 0, 255)
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # prepare label text
        label = f"{class_names[cls] if cls < len(class_names) else cls} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # draw label background
        cv2.rectangle(frame,
                      (x1, y1 - th - 4),
                      (x1 + tw, y1),
                      color,
                      cv2.FILLED)

        # draw label text (white on red)
        cv2.putText(frame,
                    label,
                    (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA)


def main():
    # 1) initialize video stream (remote or local)
    vs = get_video_stream(use_ffmpeg=True)

    # 2) load YOLOv5-nano model
    model = load_yolo_model('yolov5n', pretrained=True)
    class_names = model.names

    print(f"Loaded YOLO model on {DEVICE}")

    # 3) create a resizable window at 1280×720
    window_name = "Mode B: Multi‐Class Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    prev_time = time.time()
    while True:
        # 4) grab full-resolution frame
        frame, _ = vs.read_frame(to_tensor=False)
        if frame is None:
            break

        # 5) run inference on that frame
        detections = run_yolo_inference(model, frame,
                                        conf_thresh=0.25,
                                        iou_thresh=0.45)

        # 6) draw boxes + labels
        draw_multiclass(frame, detections, class_names)

        # 7) compute & overlay true FPS
        now = time.time()
        fps = 1.0 / (now - prev_time) if prev_time else 0.0
        prev_time = now
        cv2.putText(frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)

        # 8) display
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 9) cleanup
    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()