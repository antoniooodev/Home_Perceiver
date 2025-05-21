# File: pose/pose_utils.py
# Real-time human pose estimation with confidence filtering and reduced smoothing.

import torch
import cv2
import numpy as np
from torchvision.transforms import functional as F
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from collections import deque

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# History buffer for smoothing: only last 2 frames now
_pose_history = deque(maxlen=2)

def load_pose_model(pretrained: bool = True, min_score: float = 0.7):
    """
    Load Keypoint R-CNN from torchvision.models.detection.
    """
    weights = 'KeypointRCNN_ResNet50_FPN_Weights.DEFAULT' if pretrained else None
    model = keypointrcnn_resnet50_fpn(weights=weights)
    model.to(DEVICE).eval()
    return model, min_score

def run_pose_inference(model, frame: np.ndarray, min_score: float = 0.7):
    """
    Run pose inference with:
      - 640×640 letterbox (lower resolution)
      - box & keypoint confidence filtering
      - 2-frame temporal smoothing (shorter buffer)
    Returns a list of dicts: {'keypoints': np.ndarray[17,3]}
    """
    # 1) Letterbox to 640×640
    h0, w0 = frame.shape[:2]
    size = 640
    r = min(size/h0, size/w0)
    new_unpad = (int(w0*r), int(h0*r))
    pad_w = (size - new_unpad[0]) / 2
    pad_h = (size - new_unpad[1]) / 2
    resized = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(pad_h-0.1), int(pad_h+0.1)
    left, right = int(pad_w-0.1), int(pad_w+0.1)
    img = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114,114,114))

    # 2) To tensor (make contiguous to avoid negative strides)
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    tensor = F.to_tensor(rgb).unsqueeze(0).to(DEVICE)

    # 3) Inference
    with torch.no_grad():
        outputs = model(tensor)[0]

    # 4) Filter boxes and keypoints by score
    keep = outputs['scores'] > min_score
    kps = outputs['keypoints'][keep].cpu().numpy()            # [M,17,3]
    kp_scores = outputs['keypoints_scores'][keep].cpu().numpy()

    poses = []
    for person_kps, person_kp_scores in zip(kps, kp_scores):
        # zero-out low-confidence joints
        person_kps[:,2] = np.where(person_kp_scores < min_score, 0, person_kps[:,2])
        # unpad & unscale to original frame
        person_kps[:,:2] = (person_kps[:,:2] - np.array([left, top])) / r
        # clip coords
        person_kps[:,0] = person_kps[:,0].clip(0, w0-1)
        person_kps[:,1] = person_kps[:,1].clip(0, h0-1)
        poses.append({'keypoints': person_kps})

    # 5) Temporal smoothing over last 2 frames
    _pose_history.append(poses)
    smoothed = []
    for idx in range(len(poses)):
        # collect matching person across history
        stacks = [epoch[idx]['keypoints'] for epoch in _pose_history if len(epoch)>idx]
        arr = np.stack(stacks, axis=0)               # [T,17,3]
        avg_xy = arr[:,:,:2].mean(axis=0)            # [17,2]
        avg_v  = arr[:,:,2].mean(axis=0)             # [17]
        kp = np.concatenate([avg_xy, avg_v[:,None]], axis=1)
        smoothed.append({'keypoints': kp})

    return smoothed