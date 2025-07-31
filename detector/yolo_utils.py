#!/usr/bin/env python
"""
detector/yolo_utils.py
======================

Utility functions to load Ultralytics-YOLO v8 models for **detection** or
**instance-segmentation** and, optionally, to create an *ensemble* that merges
the COCO baseline with a custom HomeObjects-3K weight.

Key features
------------
* **Variants**
  • ``coco``  – standard YOLOv8-nano (80 COCO classes)  
  • ``home``  – fine-tuned weight on HomeObjects-3K (12 indoor classes)  
  • ``mixed`` – ensemble of the two (default)

* **Task selector** – ``task="seg"`` (instance masks) or ``task="detect"``  
* **Seamless API** – ``load_yolo_model(...)`` returns ``(model, min_score)``
  so existing demo scripts work unchanged.

The ensemble wrapper re-applies Non-Max Suppression across models to remove
duplicate predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torchvision.ops as ops
from ultralytics import YOLO
from ultralytics.engine.results import Results

# --------------------------------------------------------------------------- #
# Weight paths
# --------------------------------------------------------------------------- #
_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

_SEG_WEIGHTS = {
    "coco": _WEIGHTS_DIR / "yolov8n-seg.pt",          # 80 COCO classes
    "home": _WEIGHTS_DIR / "yolov8n-home-seg.pt",     # 12 HomeObjects classes
}
_DET_WEIGHTS = {
    "coco": _WEIGHTS_DIR / "yolov8n.pt",
    "home": _WEIGHTS_DIR / "yolov8n-home-det.pt",     # bbox-only
}

_DEFAULT_SCORE = 0.25


# --------------------------------------------------------------------------- #
# Ensemble helper
# --------------------------------------------------------------------------- #
class _EnsembleYolo:
    """Callable that runs several YOLO models and merges their outputs."""

    def __init__(self, models: List[YOLO]) -> None:
        self.models = models
        self.device = next(models[0].model.parameters()).device  # type: ignore[arg-type]

    # Ultralytics models are callable; we mimic the same signature
    def __call__(self, img, *args, **kwargs):  # type: ignore[override]
        outputs: List[Results] = []
        for model in self.models:
            outputs.extend(model(img, *args, **kwargs))
        return _apply_nms(outputs, iou=0.5)

    def fuse(self):
        """Fuse Conv-BN layers for every backbone (ignored by demo scripts)."""
        for m in self.models:
            m.fuse()
        return self


def _apply_nms(results: List[Results], iou: float = 0.5) -> List[Results]:
    """Run torchvision NMS across results from multiple models."""
    merged: List[Results] = []
    for res in results:
        boxes, conf, cls = res.boxes.xyxy, res.boxes.conf, res.boxes.cls
        keep = ops.nms(boxes, conf, iou)
        res.boxes.xyxy = boxes[keep]
        res.boxes.conf = conf[keep]
        res.boxes.cls = cls[keep]
        merged.append(res)
    return merged


# --------------------------------------------------------------------------- #
# Public loader
# --------------------------------------------------------------------------- #
def load_yolo_model(
    variant: str = "mixed",
    task: str = "seg",
    min_score: float = _DEFAULT_SCORE,
) -> Tuple[YOLO, float]:
    """
    Load a YOLO model (or an ensemble) and return ``(model, min_score)``.

    Parameters
    ----------
    variant : {'coco', 'home', 'mixed'}, optional
        Which weight to load. ``mixed`` builds an ensemble of COCO + HomeObjects.
    task : {'seg', 'detect'}, optional
        Segmentation (instance masks) or pure bounding-box detection.
    min_score : float, optional
        Confidence threshold propagated to calling scripts.

    Raises
    ------
    ValueError
        If *variant* or *task* is not supported.
    """
    variant = variant.lower()
    if variant not in {"coco", "home", "mixed"}:
        raise ValueError("variant must be 'coco', 'home' or 'mixed'")
    if task not in {"seg", "detect"}:
        raise ValueError("task must be 'seg' or 'detect'")

    weight_map = _SEG_WEIGHTS if task == "seg" else _DET_WEIGHTS

    if variant == "mixed":
        models = [
            YOLO(str(weight_map["coco"]), task=task),
            YOLO(str(weight_map["home"]), task=task),
        ]
        model: YOLO = _EnsembleYolo(models)  # type: ignore[assignment]
    else:
        model = YOLO(str(weight_map[variant]), task=task)

    # Fuse Conv-BN pairs when available (ensemble wrapper safely ignores)
    try:
        model.fuse()
    except AttributeError:
        pass

    return model, min_score