# File: detector/yolo_utils.py

import torch
from pathlib import Path

# Device selection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_model = None

def load_yolo_model(model_name: str = 'yolov5n', pretrained: bool = True):
    """
    Load a YOLOv5 model via torch.hub, cache it, send to DEVICE.
    """
    global _model
    if _model is None:
        _model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=pretrained)
        _model.to(DEVICE).eval()
    return _model

def run_yolo_inference(model, frame, conf_thresh: float = 0.25, iou_thresh: float = 0.45):
    """
    Run YOLOv5 AutoShape inference on a BGR frame.
    Returns list of (x1,y1,x2,y2,conf,cls_id).
    """
    # apply thresholds on the model itself
    model.conf = conf_thresh  # confidence threshold
    model.iou  = iou_thresh   # NMS IoU threshold

    # run
    results = model(frame)    # AutoShape
    xyxy = results.xyxy[0]    # tensor N×6

    dets = xyxy.cpu().numpy()  # to numpy
    return [tuple(d.tolist()) for d in dets]