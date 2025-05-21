# File: detector/detect.py
# Real-time object detection demo using YOLOv5 via PyTorch Hub

import sys
from pathlib import Path
import cv2
import torch
from core_utils.data_loader import get_video_stream

# Ensure project root is on sys.path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from detector.yolo_utils import load_yolo_model, run_yolo_inference

print("Detection – CUDA available:", torch.cuda.is_available())
print("Detection – default device:", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# COCO class names (index 0: person, others accordingly)
COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella',
    'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
    'baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork',
    'knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog',
    'pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv',
    'laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair dryer','toothbrush'
]

def draw_boxes(frame, dets, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on the frame.
    Args:
        frame: BGR image (numpy array)
        dets: list of detections [x1, y1, x2, y2, conf, cls]
    """
    for x1, y1, x2, y2, conf, cls in dets:
        # Draw rectangle
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            thickness
        )
        # Label with class name and confidence
        label = f"{COCO_CLASSES[cls]} {conf:.2f}"
        cv2.putText(
            frame, label, (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
    
def run_detection_on_frame(frame, conf_thresh=0.3):
    return run_yolo_inference(model, frame, conf_thresh)


def main():
    # 1. Initialize video stream (remote or fallback local)
    vs = get_video_stream(use_ffmpeg=True)
    
    # 2. Load the YOLOv5s model via torch.hub
    model = load_yolo_model('yolov5s')
    print("YOLOv5s model loaded via torch.hub.")

    # 3. Loop over frames
    while True:
        # --- grab frame
        frame, _ = vs.read_frame(to_tensor=False)
        if frame is None:
            break

        # --- optional downscale for speed
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

        # 4. Run detection
        detections = run_yolo_inference(model, frame)

        # 5. Draw boxes and labels
        draw_boxes(frame, detections)

        # 6. Display annotated frame
        cv2.imshow("YOLOv5 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 7. Cleanup
    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
