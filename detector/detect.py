# File: detector/detect.py
# Real-time object detection demo using YOLOv5-nano via PyTorch Hub

import sys
from pathlib import Path
import cv2
import torch

# Ensure project root is on sys.path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core_utils.data_loader import get_video_stream, VideoStream
from detector.yolo_utils import load_yolo_model, run_yolo_inference

print("Detection – CUDA available:", torch.cuda.is_available())
print("Detection – default device:", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

def draw_boxes(frame, dets, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on the frame.
    dets: list of (x1, y1, x2, y2, conf)
    """
    for x1, y1, x2, y2, conf in dets:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def main():
    # 1. Initialize video stream (remote URL or camera index)
    vs = get_video_stream(use_ffmpeg=True)

    # 2. Load the YOLOv5-nano model via torch.hub
    model = load_yolo_model('yolov5n')
    print("YOLOv5-nano model loaded via torch.hub.")

    # 3. Loop over frames
    while True:
        frame, _ = vs.read_frame(to_tensor=True)
        if frame is None:
            break

        # Downscale for speed
        frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)

        # 4. Run detection
        detections = run_yolo_inference(model, frame)

        # 5. Draw boxes and show
        draw_boxes(frame, detections)
        cv2.imshow("YOLOv5-nano Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()