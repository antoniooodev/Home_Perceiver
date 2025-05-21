# File: scripts/demo_modeA.py
# Combined Mode A: Person Detection + Pose Estimation + Tracking

import sys
import time
from pathlib import Path
import cv2
import torch

# Ensure project root is on PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core_utils.data_loader import get_video_stream
from detector.yolo_utils import load_yolo_model, run_yolo_inference
from detector.detect import draw_boxes
from pose.pose_utils import load_pose_model, run_pose_inference
from pose.pose import draw_skeleton


def main():
    # 1) Initialize video stream with fallback
    vs = get_video_stream(use_ffmpeg=True)

    # 2) Load YOLO detection model
    yolo_model = load_yolo_model('yolov5s')
    print("Detection – CUDA available:", torch.cuda.is_available())
    print("Detection – default device:", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # 3) Load Pose estimation model
    pose_model, min_score = load_pose_model(pretrained=True, min_score=0.5)
    print("Keypoint R-CNN model loaded. Device:", next(pose_model.parameters()).device)

    torch.set_grad_enabled(False)
    prev_time = time.time()

    print("Starting Mode A demo...")
    while True:
        frame, _ = vs.read_frame(to_tensor=False)
        if frame is None:
            break

        # Detection on frame
        dets = run_yolo_inference(yolo_model, frame, conf_thresh=0.3)
        # Filter to person class id 0
        person_boxes = [d for d in dets if d[5] == 0]
        draw_boxes(frame, person_boxes)

        # Pose estimation
        poses = run_pose_inference(pose_model, frame, min_score)
        # Filter poses inside person boxes
        filtered_poses = []
        for p in poses:
            kps = p['keypoints']
            # check if any visible keypoint is inside any person box
            if any((b[0] <= x <= b[2] and b[1] <= y <= b[3])
                   for b in person_boxes
                   for x, y, v in kps if v > 0):
                filtered_poses.append(p)
        # Draw all skeletons at once
        draw_skeleton(frame, filtered_poses)

        # Compute and display FPS
        now = time.time()
        fps = 1.0 / (now - prev_time) if prev_time else 0.0
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.imshow("Mode A: Detection+Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
