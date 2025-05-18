# pose.py
# Real-time human pose estimation demo

import sys
from pathlib import Path
# Ensure project root is on sys.path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
from core_utils.data_loader import VideoStream
from pose.pose_utils import load_pose_model, run_pose_inference

# Define skeleton connections for COCO keypoints
SKELETON = [
    (5, 7), (7, 9),    # left arm
    (6, 8), (8, 10),   # right arm
    (11, 13), (13, 15),# left leg
    (12, 14), (14, 16),# right leg
    (5, 6),            # shoulders
    (11, 12)           # hips
]


def draw_skeleton(frame, poses, kp_radius=3, kp_color=(0,0,255), line_color=(255,0,0), thickness=2):
    """
    Draw keypoints and skeleton on the frame.
    Args:
        frame: BGR image
        poses: list of pose dicts with 'keypoints'
    """
    for person in poses:
        kps = person['keypoints']
        # draw keypoints
        for x, y, v in kps:
            if v > 0:
                cv2.circle(frame, (int(x), int(y)), kp_radius, kp_color, -1)
        # draw skeleton lines
        for i, j in SKELETON:
            x1, y1, v1 = kps[i]
            x2, y2, v2 = kps[j]
            if v1 > 0 and v2 > 0:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), line_color, thickness)


def main():
    # 1. Initialize Video Stream
    vs = VideoStream(source=0)

    # 2. Load pose model
    model, min_score = load_pose_model(pretrained=True, min_score=0.5)
    print("Keypoint R-CNN model loaded.")

    # 3. Loop over frames
    while True:
        # --- read_frame must come first
        frame, _ = vs.read_frame(to_tensor=True)

        # --- break out if no more frames
        if frame is None:
            break

        # --- now it's safe to downscale
        frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_LINEAR)

        # 4. Run pose inference
        poses = run_pose_inference(model, frame, min_score)

        # 5. Draw skeletons
        draw_skeleton(frame, poses)

        # 6. Display result
        cv2.imshow("Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.release()
    
if __name__ == "__main__":
    main()