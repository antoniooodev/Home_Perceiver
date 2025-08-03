# Home Perceiver – Real‑Time Privacy‑Aware Perception Module

> **TL;DR** A cross‑platform stack (PyTorch → OpenCV) that runs
> in real time on GPU/CPU/MPS, detects objects & people, segments persons for
> privacy, tracks IDs, estimates poses, exports structured logs, and builds
> PDF reports – all via three demo scripts and a modular Python API.

---

\## Features

| Capability                            | Mode      | Model             | Notes                     |
| ------------------------------------- | --------- | ----------------- | ------------------------- |
| 🟥 Object detection (80 COCO classes) | **Multi** | YOLOv5‑nano       | 35 FPS @ 640 px           |
| 🟦 Person instance segmentation       | **A / B** | YOLOv8‑nano‑seg   | tight privacy mask        |
| 🟨 Pose estimation (17‑pt)            | **A**     | Keypoint R‑CNN    | optional skeleton overlay |
| 🟩 Multi‑person tracking              | **A**     | IoU tracker       | persistent IDs            |
| 📈 Structured export                  | all       | JSONL + CSV + MP4 | one line per frame        |
| 📄 Report builder                     | tools     | LaTeX             | summary PDF with plots    |

Tested on **macOS 14 (M2 Pro)**, **Windows 11 (RTX 3060 Ti)** and **Ubuntu 22.04
(CUDA 12)**.

---

\## Directory layout

```text
visual_detection/
│  main.py                ← entry‑point selector (optional)
│  pyproject.toml         ← editable install
│
├─ core_utils/            ← VideoStream + Exporter (logging)
├─ detector/              ← YOLO helpers + demo detect.py (legacy)
├─ pose/                  ← pose utils + demo script
├─ tracker/               ← lightweight IoU tracker
├─ scripts/               ← high‑level demos
│   ├─ demo_modeA.py      ← mask + pose + tracking (privacy)
│   ├─ demo_modeB.py      ← mask only, multi‑class detection
│   └─ detect_multiclass.py ← bbox only, fastest
├─ tools/                 ← analyze_run.py, build_report.py, …
├─ tests/                 ← pytest smoke suite
└─ weights/               ← pre‑downloaded *.pt (git LFS)
```

---

\## Installation

\### 1 · Clone & create venv (🐍 ≥ 3.11)

```bash
git clone https://github.com/antoniooodev/Home_Perceiver.git
cd Home_Perceiver
python3.11 -m venv .venv
source .venv/bin/activate
```

\### 2 · Install project & deps

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt      # full stack (~1 GB incl. Torch)
# or: pip install -e .               # editable install (uses pyproject)
```

*Apple Silicon ✓* – the PyTorch wheels include the **MPS** backend; no extra
steps required.

\### 3 · Retrieve model weights

```bash
# already under weights/, but you can refresh:
python - <<'PY'
from torchvision.models import get_weight
from ultralytics import YOLO
YOLO('yolov8n-seg.pt')        # auto‑download to ~/.cache/torch/hub
PY
```

_(Large files in `weights/` are tracked via **git LFS** – run
`brew install git‑lfs && git lfs install` if needed.)_

\### 4 · Verify

```bash
pytest -q          # 3 ✓ fast smoke tests
```

---

\## Usage

\### Mode A – Privacy mask + pose + tracking

```bash
python scripts/demo_modeA.py \
       --source 0            # 0=webcam / path / rtsp:// …
       --device mps          # cpu | cuda | mps
       --save-video          # optional MP4 in output/videos/
```

\### Mode B – Multi‑class detection + mask persona

```bash
python scripts/demo_modeB.py --source sample.mp4 --no-display
```

\### Multi‑class only (fastest)

```bash
python scripts/detect_multiclass.py \
       --source https://…/stream.mjpg --save-video
```

All demos create a **run‑ID** timestamp (e.g. `20250726‑1631xx‑abcdef`) and
write:

```
output/json/<run>.jsonl    ← one line per frame
output/csv/<run>.summary.csv
output/videos/<run>.mp4    ← if --save-video
```

---

\## Post‑processing

\### 1 · Aggregate a single run

```bash
python tools/analyze_run.py --run output/json/<run>.jsonl
# → <run>.summary.csv + <run>.classes.png
```

\### 2 · Generate PDF report

```bash
python tools/build_report.py \
       --run-id <run> \
       --title "Mode‑A – Webcam (Mac M2)"
# → output/reports/<run>.pdf (needs pdflatex in PATH)
```

\### 3 · Batch aggregate

```bash
python tools/aggregate_runs.py --glob "output/json/*.jsonl"
# merges into output/csv/_aggregate.csv
```

---

\## Training on custom data  (optional)

Fine‑tune **YOLOv8‑nano** on your dataset (COCO‑style and Homeobj3k) :

```bash
# data.yaml lists train/val and class names
ultralytics yolo detect train \
    model=yolov8n.pt data=data.yaml imgsz=640 epochs=50 batch=16 device=mps
cp runs/detect/train/weights/best.pt weights/yolov8n-custom.pt
```

Update `detector/yolo_utils.py` to load the new weight.

---

\## Testing & CI

```bash
pytest -q                  # 3 fast tests (<1 s)
ruff check .                # lint (PEP‑8 + import order)
```

Set up GitHub Actions with matrix `{macos‑14, ubuntu‑latest}` to run the
smoke suite on push.

---

\## Credits & License

- Code © 2025 Visual Detection Team – MIT License (see LICENSE).<br>
- YOLOv5/YOLOv8 © Ultralytics – GPLv3 weights redistributed via LFS.<br>
- Keypoint R‑CNN weights © TorchVision (BSD‑3).<br>
- COCO, Objects‑365 datasets under original licenses.

> For academic use please cite “Visual Detection Project – Real‑Time
> Privacy‑Aware Vision Module, 2025”.
