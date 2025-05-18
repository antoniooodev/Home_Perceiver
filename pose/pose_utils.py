# pose_utils.py
# Helper functions for human pose estimation using PyTorch Keypoint R-CNN

import torch
import numpy as np
from pathlib import Path
from torchvision.models.detection import keypointrcnn_resnet50_fpn

# Select device (CUDA if available, otherwise CPU or MPS)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global model cache
_model = None

def load_pose_model(pretrained: bool = True, min_score: float = 0.5):
    """
    Load a Keypoint R-CNN model from torchvision.
    Args:
        pretrained (bool): whether to use pretrained COCO weights
        min_score (float): minimum person detection score to keep
    Returns:
        model: the loaded model in eval mode
        min_score: the filtering threshold
    """
    global _model
    if _model is None:
        # Initialize Keypoint R-CNN with ResNet-50 FPN backbone
        model = keypointrcnn_resnet50_fpn(pretrained=pretrained)
        model.to(DEVICE)
        model.eval()
        _model = model
    return _model, min_score

def run_pose_inference(model, frame, min_score: float):
    """
    Run pose estimation on a single BGR frame.
    Args:
        model: Keypoint R-CNN model
        frame: BGR image (numpy array) as read by OpenCV
        min_score: confidence threshold for person detections
    Returns:
        poses: list of dicts, each with 'keypoints': [(x,y,score), ...]
    """
    # Convert BGR to RGB and ensure positive-stride layout
    rgb = frame[:, :, ::-1].copy()
    # Normalize to [0,1] and convert to float32 numpy array
    arr = rgb.astype(np.float32) / 255.0  # H x W x 3
    # Convert to torch.Tensor: 1 x 3 x H x W
    img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        outputs = model(img)[0]

    poses = []
    # Iterate over detected persons
    scores = outputs['scores'].cpu().numpy()
    keypoints = outputs['keypoints'].cpu().numpy()  # shape [N,17,3]
    for idx, score in enumerate(scores):
        if score < min_score:
            continue
        kps = keypoints[idx]  # 17 x 3
        pose = {
            'keypoints': [
                (float(x), float(y), float(v)) for x, y, v in kps
            ]
        }
        poses.append(pose)

    return poses