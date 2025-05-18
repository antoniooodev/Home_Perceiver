# yolo_utils.py

import torch
from pathlib import Path

# Device: use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model once globally
_model = None

def load_yolo_model(model_name: str = 'yolov5s', pretrained: bool = True):
    """
    Load a YOLOv5 model via PyTorch Hub.
    Args:
        model_name (str): 'yolov5s', 'yolov5m', 'yolov5l', etc.
        pretrained (bool): use pretrained weights from ultralytics repository.
    Returns:
        model: YOLOv5 model loaded onto DEVICE.
    """
    global _model
    if _model is None:
        # Load model from ultralytics/yolov5 repository
        _model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=pretrained)
        _model.to(DEVICE)
        _model.eval()
    return _model


def run_yolo_inference(model, frame):
    """
    Run YOLOv5 inference on a single BGR image frame.
    Args:
        model: YOLOv5 model from torch.hub
        frame: BGR image (numpy array) as read by OpenCV
    Returns:
        dets (list): List of detections, each as (x1, y1, x2, y2, confidence).
    """
    # Inference via AutoShape: handles resizing and padding internally
    results = model(frame)
    # results.xyxy[0]: numpy array N x 6 (x1,y1,x2,y2,conf,cls)
    dets = []
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2 = box
        dets.append((x1, y1, x2, y2, float(conf)))
    return dets
