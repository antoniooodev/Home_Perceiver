import torch

# Device: GPU if available else CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_model = None

def load_yolo_model(model_name: str = 'yolov5s', pretrained: bool = True):
    """
    Load a YOLOv5 model via torch.hub (AutoShape wrapper).
    """
    global _model
    if _model is None:
        # 👇 This loads the Ultralyics YOLOv5 from GitHub directly
        _model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=pretrained)
        _model.to(DEVICE).eval()
    return _model

def run_yolo_inference(model, frame, conf_thresh: float = 0.25):
    """
    Run inference on a BGR frame, returning list of detections:
      [x1, y1, x2, y2, confidence, class_id]
    """
    # AutoShape lets you pass in an ndarray BGR directly
    results = model(frame, size=640)  # you can adjust `size` if you like
    dets = []
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2 = box
        if conf >= conf_thresh:
            dets.append((float(x1), float(y1), float(x2), float(y2),
                         float(conf), int(cls)))
    return dets