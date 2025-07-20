# File: detector/mask_utils.py

import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mask_model(pretrained: bool = True, min_score: float = 0.5):
    """
    Load a Mask R-CNN model for instance segmentation.
    Returns: (model, min_score_threshold)
    """
    model = maskrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)
    model.to(DEVICE).eval()
    return model, min_score


def run_mask_inference(model, frame, min_score: float = 0.5):
    """
    Run Mask R-CNN on a BGR frame.
    Returns:
      dets: list of [x1,y1,x2,y2,score,cls]
      masks: list of H×W boolean masks
    """
    # 1) Convert BGR→RGB and ensure contiguous memory
    rgb = frame[..., ::-1].copy()
    # 2) To tensor CHW [0,1]
    img = F.to_tensor(rgb).to(DEVICE)

    # 3) Inference
    with torch.no_grad():
        outputs = model([img])[0]

    boxes = outputs["boxes"].cpu().numpy()  # [N,4]
    scores = outputs["scores"].cpu().numpy()  # [N]
    labels = outputs["labels"].cpu().numpy()  # [N]
    masks_t = outputs["masks"].cpu().numpy()  # [N,1,H,W]

    H, W = frame.shape[:2]
    dets, masks = [], []
    for box, score, lab, m in zip(boxes, scores, labels, masks_t):
        if score < min_score:
            continue
        x1, y1, x2, y2 = box
        dets.append([x1, y1, x2, y2, score, lab])
        mask = (m[0] > 0.5).astype(np.uint8)
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        masks.append(mask.astype(bool))

    return dets, masks
