# detect.py
# Real-time object detection demo using YOLOv5 via PyTorch Hub

import sys
from pathlib import Path
import cv2

# Ensure project root is on sys.path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core_utils.data_loader import VideoStream
from detector.yolo_utils import load_yolo_model, run_yolo_inference

def draw_boxes(frame, dets, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on the frame.
    Args:
        frame: BGR image (numpy array)
        dets: list of detections [x1, y1, x2, y2, conf]
    """
    for x1, y1, x2, y2, conf in dets:
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            thickness
        )

def main():
    # 1. Initialize video stream
    vs = VideoStream(source=0)

    # 2. Load the YOLOv5s model via torch.hub
    model = load_yolo_model('yolov5s')
    print("YOLOv5s model loaded via torch.hub.")

    # 3. Loop over frames
    while True:
        # --- grab frame first
        frame, _ = vs.read_frame(to_tensor=True)
        # --- break if no frame
        if frame is None:
            break

        # --- downscale for speed
        frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_LINEAR)

        # 4. Run detection on the downscaled frame
        detections = run_yolo_inference(model, frame)

        # 5. Draw boxes
        draw_boxes(frame, detections)

        # 6. Display annotated frame
        cv2.imshow("YOLOv5 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 7. Cleanup
    vs.release()

if __name__ == "__main__":
    main()