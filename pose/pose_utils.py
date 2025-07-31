"""
Utilities for Keypoint R-CNN human-pose inference.
"""
from __future__ import annotations

from collections import deque
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.transforms import functional as F

__all__ = [
    "DEVICE",
    "load_pose_model",
    "run_pose_inference",
]

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# keep only the last two frames for lightweight smoothing
_pose_history: deque[List[dict[str, np.ndarray]]] = deque(maxlen=2)

# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def load_pose_model(*, pretrained: bool = True, min_score: float = 0.7):
    """Return a Keypoint R-CNN model and its score threshold.

    Parameters
    ----------
    pretrained : bool, default True
        Download weights from torchvision if *True*.
    min_score : float, default 0.7
        Per-keypoint confidence threshold.
    """
    weights = "KeypointRCNN_ResNet50_FPN_Weights.DEFAULT" if pretrained else None
    model = keypointrcnn_resnet50_fpn(weights=weights)
    model.to(DEVICE).eval()
    return model, min_score


# -----------------------------------------------------------------------------
# Inference wrapper
# -----------------------------------------------------------------------------

def run_pose_inference(
    model: torch.nn.Module,
    frame: np.ndarray,
    *,
    min_score: float = 0.7,
) -> List[dict[str, np.ndarray]]:
    """Detect 17-point COCO skeletons in *frame*.

    Returns a list like ``[{"keypoints": (17,3) array}, …]`` where each row is
    ``(x, y, v)`` and *v* is the averaged visibility score after 2-frame
    smoothing.
    """
    h0, w0 = frame.shape[:2]
    size = 640  # letterbox target
    r = min(size / h0, size / w0)
    new_hw: Tuple[int, int] = (int(w0 * r), int(h0 * r))
    pad_w, pad_h = (size - new_hw[0]) / 2, (size - new_hw[1]) / 2

    resized = cv2.resize(frame, new_hw, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(pad_h - 0.1), int(pad_h + 0.1)
    left, right = int(pad_w - 0.1), int(pad_w + 0.1)
    img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    rgb = np.ascontiguousarray(img[:, :, ::-1])
    tensor = F.to_tensor(rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)[0]

    keep = outputs["scores"] > min_score
    kps = outputs["keypoints"][keep].cpu().numpy()        # (M,17,3)
    kp_scores = outputs["keypoints_scores"][keep].cpu().numpy()

    poses: List[dict[str, np.ndarray]] = []
    for person_kps, person_kp_scores in zip(kps, kp_scores):
        # mask low-confidence joints
        person_kps[:, 2] = np.where(person_kp_scores < min_score, 0, person_kps[:, 2])
        # revert letterbox
        person_kps[:, :2] = (person_kps[:, :2] - np.array([left, top])) / r
        person_kps[:, 0] = person_kps[:, 0].clip(0, w0 - 1)
        person_kps[:, 1] = person_kps[:, 1].clip(0, h0 - 1)
        poses.append({"keypoints": person_kps})

    # ---- 2-frame smoothing ----
    _pose_history.append(poses)
    smoothed: List[dict[str, np.ndarray]] = []
    for idx in range(len(poses)):
        stacks = [epoch[idx]["keypoints"] for epoch in _pose_history if len(epoch) > idx]
        arr = np.stack(stacks, axis=0)  # (T,17,3)
        avg_xy = arr[:, :, :2].mean(axis=0)
        avg_v = arr[:, :, 2].mean(axis=0)
        kp = np.concatenate([avg_xy, avg_v[:, None]], axis=1)
        smoothed.append({"keypoints": kp})

    return smoothed
