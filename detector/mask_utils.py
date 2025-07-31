"""Light helpers for Mask R‑CNN instance segmentation.

Only cosmetic clean‑up: docstrings, type hints, import order. Runtime logic
unchanged.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

__all__ = [
    "DEVICE",
    "load_mask_model",
    "run_mask_inference",
]

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Loading / warm‑up
# -----------------------------------------------------------------------------

def load_mask_model(*, pretrained: bool = True, min_score: float = 0.5):
    """Return a ready‑to‑run Mask R‑CNN model and its score threshold."""
    model = maskrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)
    model.to(DEVICE).eval()
    return model, min_score


# -----------------------------------------------------------------------------
# Inference wrapper
# -----------------------------------------------------------------------------

def run_mask_inference(
    model: torch.nn.Module,
    frame: np.ndarray,
    *,
    min_score: float = 0.5,
) -> Tuple[List[List[float]], List[np.ndarray]]:
    """Run Mask R‑CNN on a BGR *frame*.

    Parameters
    ----------
    model : torch.nn.Module
        Mask R‑CNN network returned by :pyfunc:`load_mask_model`.
    frame : np.ndarray
        BGR image (H×W×3, uint8).
    min_score : float, default 0.5
        Min confidence to keep a detection.

    Returns
    -------
    dets
        ``[[x1,y1,x2,y2,score,cls], …]`` in original image coords.
    masks
        List of boolean masks aligned with *frame* (True = object).
    """
    rgb = frame[..., ::-1].copy()              # BGR → RGB
    img = F.to_tensor(rgb).to(DEVICE)          # CHW float32 [0,1]

    with torch.no_grad():
        outputs = model([img])[0]

    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()
    masks_t = outputs["masks"].cpu().numpy()  # (N,1,H,W)

    H, W = frame.shape[:2]
    dets: List[List[float]] = []
    masks: List[np.ndarray] = []
    for box, score, lab, m in zip(boxes, scores, labels, masks_t):
        if score < min_score:
            continue
        x1, y1, x2, y2 = map(float, box)
        dets.append([x1, y1, x2, y2, float(score), int(lab)])
        mask = (m[0] > 0.5).astype(np.uint8)
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        masks.append(mask.astype(bool))

    return dets, masks
