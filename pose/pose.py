# File: pose/pose.py
# Real-time human pose estimation demo with debug prints

import sys
from pathlib import Path
# Ensure project root is on sys.path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
from core_utils.data_loader import get_video_stream
from pose.pose_utils import load_pose_model, run_pose_inference

# Complete 17-point COCO skeleton with synthetic neck handled in draw_skeleton
COCO_KEYPOINTS = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16
}
COCO_IDX2NAME = {v: k for k, v in COCO_KEYPOINTS.items()}

SKELETON = [
    # head
    (COCO_KEYPOINTS["nose"], COCO_KEYPOINTS["left_eye"]),
    (COCO_KEYPOINTS["nose"], COCO_KEYPOINTS["right_eye"]),
    (COCO_KEYPOINTS["left_eye"], COCO_KEYPOINTS["left_ear"]),
    (COCO_KEYPOINTS["right_eye"], COCO_KEYPOINTS["right_ear"]),
    # shoulders & arms
    (COCO_KEYPOINTS["left_shoulder"], COCO_KEYPOINTS["right_shoulder"]),
    (COCO_KEYPOINTS["left_shoulder"], COCO_KEYPOINTS["left_elbow"]),
    (COCO_KEYPOINTS["left_elbow"], COCO_KEYPOINTS["left_wrist"]),
    (COCO_KEYPOINTS["right_shoulder"], COCO_KEYPOINTS["right_elbow"]),
    (COCO_KEYPOINTS["right_elbow"], COCO_KEYPOINTS["right_wrist"]),
    # torso
    (COCO_KEYPOINTS["left_shoulder"], COCO_KEYPOINTS["left_hip"]),
    (COCO_KEYPOINTS["right_shoulder"], COCO_KEYPOINTS["right_hip"]),
    (COCO_KEYPOINTS["left_hip"], COCO_KEYPOINTS["right_hip"]),
    # legs
    (COCO_KEYPOINTS["left_hip"], COCO_KEYPOINTS["left_knee"]),
    (COCO_KEYPOINTS["left_knee"], COCO_KEYPOINTS["left_ankle"]),
    (COCO_KEYPOINTS["right_hip"], COCO_KEYPOINTS["right_knee"]),
    (COCO_KEYPOINTS["right_knee"], COCO_KEYPOINTS["right_ankle"]),
]


def draw_skeleton(frame, poses, kp_radius=3, kp_color=(0,0,255), line_color=(255,0,0), thickness=2):
    """
    Draw keypoints and full 17-point skeleton with synthetic neck.
    Args:
        frame: BGR image
        poses: list of pose dicts with 'keypoints'
    """
    h, w = frame.shape[:2]

    for person in poses:
        kps = person['keypoints']

        # draw joints
        for idx, (x, y, v) in enumerate(kps):
            if v > 0 and 0 <= int(x) < w and 0 <= int(y) < h:
                cv2.circle(frame, (int(x), int(y)), kp_radius, kp_color, -1)

        # synthetic neck line: nose → midpoint of shoulders
        v0 = kps[COCO_KEYPOINTS['nose']][2]
        v5 = kps[COCO_KEYPOINTS['left_shoulder']][2]
        v6 = kps[COCO_KEYPOINTS['right_shoulder']][2]
        if v0 > 0 and v5 > 0 and v6 > 0:
            x0, y0 = kps[COCO_KEYPOINTS['nose']][:2]
            x5, y5 = kps[COCO_KEYPOINTS['left_shoulder']][:2]
            x6, y6 = kps[COCO_KEYPOINTS['right_shoulder']][:2]
            mx, my = (x5 + x6) / 2, (y5 + y6) / 2
            cv2.line(frame, (int(x0), int(y0)), (int(mx), int(my)), line_color, thickness)

        # draw skeleton lines
        for i, j in SKELETON:
            x1, y1, v1 = kps[i]
            x2, y2, v2 = kps[j]
            if v1 > 0 and v2 > 0:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), line_color, thickness)

def run_pose_on_frame(frame, min_score=0.5):
    return run_pose_inference(model, frame, min_score)

def main():
    # Use the HTTP MJPEG stream (with fallback to local)
    vs = get_video_stream(use_ffmpeg=True)

    # Load pose model (Keypoint-RCNN)
    model, min_score = load_pose_model(pretrained=True, min_score=0.5)
    print("Keypoint R-CNN model loaded.")
    print("Pose – model device:", next(model.parameters()).device)

    while True:
        frame, _ = vs.read_frame(to_tensor=False)
        if frame is None:
            break

        poses = run_pose_inference(model, frame, min_score)
        draw_skeleton(frame, poses)

        cv2.imshow("Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()