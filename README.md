### Updated README (drop-in replacement)

# YOLO Fruit Object + Color Detection (CBD)

### Final Year Project – Reconfigurable Conveyor Sorting System with AI-Based Object Classification (CBD)

This repository implements a **real-time fruit detection and color estimation system** using **Ultralytics YOLO**.
It is the vision + benchmarking core of the **Reconfigurable Conveyor Sorting System with AI-Based Object Classification (CBD)**.

Unlike generic datasets/models (e.g., COCO / Objects365), this project is designed to run with a **fruit-specific YOLO model** to eliminate “donut/vase/sink” style confusion and provide reliable classifications for conveyor sorting use-cases.

The system is designed to run on:

* **Laptop** (development + benchmarking)
* **Raspberry Pi 5** (embedded deployment)

with an emphasis on **modularity, reproducibility, and experimental validity**.

---

## Objectives

* Detect **fruits** in real time using a YOLO model trained specifically for fruit classes.
* Estimate dominant color for the detected fruit using a lightweight, mask-aware color pipeline.
* Run efficiently on low-power hardware (Raspberry Pi 5).
* Provide a **frame-matched benchmarking framework** for fair laptop vs Pi comparisons.
* Record **what was detected**, **confidence**, **color**, and **timings**, per trial and overall.

---

## System Architecture

Single entry-point with modular packages:

```
CBD/
├── main.py                          # single entry point
├── models/
│   └── fruit_yolo.pt                # fruit-specific YOLO weights (auto-downloaded or user-provided)
│
├── app/
│   ├── config/
│   │   └── settings.py              # experiment + camera defaults
│   ├── camera/
│   │   └── camera.py                # camera handling & backends
│   ├── vision/
│   │   ├── yolo_runner.py           # YOLO wrapper + fruit allowlist filter
│   │   ├── color_utils.py           # mask-aware color naming + white balance
│   │   └── mask_utils.py            # background masking + fallback bbox
│   ├── benchmark/
│   │   └── frame_benchmark.py       # frame-matched benchmark engine
│   └── utils/
│       └── stats.py                 # running stats + detection aggregation
│
├── requirements.txt
└── README.md
```

Only **`main.py`** is executed directly.

---

## Per-Frame Processing Pipeline

Each frame is processed in three stages:

1. **Capture** frame from camera / stream
2. **YOLO inference** (fruit-specific model)
3. **Post-processing**

   * choose a single primary detection per frame (largest bbox)
   * background masking (belt removal) + fallback bbox when YOLO misses
   * dominant color estimation (mask-aware)
   * per-frame timing + per-trial aggregation

Each stage is timed independently (capture / inference / post / end-to-end).

---

## Benchmarking Methodology

### Frame-Matched, Fixed-Trial Design

To ensure a fair comparison between laptop and Raspberry Pi:

* Fixed total frames per run (default: **600**)
* Fixed number of trials: **12**
* Frames split evenly: **12 × 50**

Why frame-matched?
Time-matched benchmarks allow faster devices to process more frames → biased averages.
Frame-matched ensures both platforms process the **same number of observations**.

---

## Recorded Metrics (Per Trial & Overall)

### Performance

* capture time (ms)
* inference time (ms)
* post-processing time (ms)
* end-to-end latency (ms)
* derived FPS

### Environment

* brightness %
* HSV V mean
* grayscale luminance mean

### Detections

* detections per trial
* mean confidence (proxy accuracy for live video)
* top classes / colors / class-color pairs
* mean confidence per class

---

## Installation

```bash
# 1) Create venv
python -m venv venv

# 2) Activate
venv\Scripts\activate          # Windows
# or
source venv/bin/activate       # macOS/Linux

# 3) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Example `requirements.txt`:

```txt
ultralytics
opencv-python
numpy
torch
torchvision
```

---

## Running

```bash
python main.py --source 0
```

Flow:

* READY screen appears
* Press **R** to run benchmark (auto runs 600 frames, 12 trials)
* Prints per-trial + overall summaries in terminal
* Final frame freezes
* Press **R** to rerun, **Q/ESC** to quit

### Optional Flags

* `--source 0/1/2` camera index OR a file/URL path
* `--backend auto|any|dshow|msmf|v4l2|avfoundation|gstreamer`
* `--mjpg` (helps some USB cams / Pi setups)
* `--no_draw` disables boxes + color naming (speed testing)

---

## Fruit Model Notes (Important)

This project expects a **fruit-specific YOLO weights file** (e.g., `models/fruit_yolo.pt`).

To protect the system from “COCO junk detections”, the pipeline includes an optional **fruit allowlist filter** inside:

* `app/vision/yolo_runner.py`

So even if the model outputs extra labels, only allowed fruit classes are kept.

---

## Applications

* conveyor fruit sorting
* robotics perception + automation
* embedded AI benchmarking and optimisation
* industrial inspection pipelines

---

## References

* Ultralytics YOLO Docs (official)
* OpenCV Docs