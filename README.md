# YOLO Object and Color Detection

### Final Year Project – Reconfigurable Conveyor Sorting System with AI-Based Object Classification (CBD)

This repository implements a **real-time object and color detection system** using **Ultralytics YOLO** deep-learning models.
The system forms the vision and benchmarking core of the **Reconfigurable Conveyor Sorting System with AI-Based Object Classification (CBD)**.

It detects everyday objects, estimates their dominant color using a lightweight HSV-based pipeline, and evaluates real-time performance using a **frame-matched benchmarking methodology** designed for fair comparison between heterogeneous platforms.

The system is designed to run on both:

* a **laptop** (development, testing, and benchmarking), and
* a **Raspberry Pi 5** (low-power embedded deployment),

with a strong emphasis on **modularity, reproducibility, and experimental validity**.

---

## Objectives

* Detect multiple common objects in real time using a pretrained YOLO model.
* Estimate the dominant color of each detected object using HSV color analysis.
* Run efficiently on low-power hardware such as the Raspberry Pi 5.
* Provide a **robust, frame-matched benchmarking framework** for fair laptop vs Raspberry Pi comparison.
* Record **what was detected**, **how confidently**, and **how fast**, per trial.
* Serve as an extensible base for conveyor sorting, robotics, and intelligent automation systems.

---

## System Architecture

The project follows a **single-entry-point, modular package-based design**.

```
CBD/
├── main.py                  # single entry point
├── yolo11n.pt
│
├── config/                  # experiment & camera defaults
│   └── settings.py
│
├── camera/                  # camera handling & backends
│   └── camera.py
│
├── vision/                  # vision utilities
│   ├── yolo_runner.py       # YOLO wrapper
│   └── color_utils.py       # HSV color estimation
│
├── benchmark/               # benchmarking engine
│   └── frame_benchmark.py
│
├── utils/                   # statistics & aggregation
│   └── stats.py
```

Only **`main.py`** is executed directly.
All other components are imported as modules, improving maintainability and extensibility.

---

## Per-Frame Processing Pipeline

Each frame is processed in three clearly separated stages:

1. **Frame capture** from a USB or CSI camera
2. **YOLO inference** using Ultralytics’ `predict()` pipeline
3. **Post-processing**, including:

   * bounding-box rendering
   * dominant color estimation
   * metric and detection aggregation

Each stage is timed independently to identify performance bottlenecks.

---

## Benchmarking Methodology

### Frame-Matched, Fixed-Trial Design

To ensure a **fair and scientifically valid comparison** between the laptop and Raspberry Pi, the system uses a **frame-matched benchmark design**.

### Key design choices

* **Fixed total number of frames per run** (default: 600 frames)
* **Fixed number of trials: 12** (non-configurable by design)
* Frames are split evenly:

  * 600 frames → **12 trials × 50 frames per trial**

### Why frame-matched?

In time-matched benchmarks, faster devices process more frames, leading to unequal sample sizes and biased averages.
By fixing the number of frames, both platforms process the **same number of observations**, enabling a clean, per-frame efficiency comparison.

---

## Recorded Metrics (Per Trial & Overall)

For **each trial** and for the **entire run**, the system reports:

### Performance metrics

* Capture time (ms)
* Inference time (ms)
* Post-processing time (ms)
* End-to-end latency (ms)
* Derived FPS

### Environmental metrics

* Scene brightness (%)
* HSV V-channel mean
* Grayscale luminance mean

### Detection metrics

* Total detections per trial
* **Mean confidence** (used as an *accuracy proxy* for live video)
* Top detected object classes
* Top detected colors
* Top object–color pairs
* Mean confidence per object class

> **Note:** True accuracy (precision / recall / mAP) requires ground-truth labels and is therefore not computed for live camera input. Mean confidence is used as a practical proxy.

---

## Installation (Laptop Setup)

```bash
# 1. Create and activate virtual environment
python -m venv yolo-pi
yolo-pi\Scripts\activate        # Windows
# or
source yolo-pi/bin/activate    # macOS / Linux

# 2. Install dependencies
pip install --upgrade pip
pip install ultralytics opencv-python torch torchvision torchaudio
```

---

## Running the System

### Detection + Visualization Mode

```bash
python main.py --source 0
```

The display window shows:

* Object name
* Dominant color
* Detection confidence

Press **Q** or **ESC** to exit.

---

### Frame-Matched Benchmark Mode (Recommended)

```bash
python main.py --source 0 
```

Benchmark flow:

* A **READY** screen appears
* Press **R** to start the benchmark
* The system processes exactly 600 frames
* 12 trials are executed automatically
* Per-trial and overall summaries are printed to the terminal
* The final frame freezes on screen
* Press **R** to run again, **Q** to quit

---

### Optional Flags

* ``--source (0, 1, 2)``: to access main device cam or external.
* ``--backend (auto|any|dshow|msmf|v4l2|avfoundation|gstreamer)``: for different devices (windows, mac, rasberry)
* ``--mjpg``: mainly used for external cam on rasberry pi
* ``--no_draw``: doesn't draw boxes

---

## Applications

* Reconfigurable conveyor sorting systems
* Robotics and autonomous platforms
* Smart manufacturing and industrial inspection
* Embedded AI benchmarking and optimisation studies

---

## References

* Ultralytics YOLO Documentation: [https://docs.ultralytics.com](https://docs.ultralytics.com)
* Objects365 Dataset: [https://www.objects365.org](https://www.objects365.org)
* OpenCV Documentation: [https://docs.opencv.org](https://docs.opencv.org)
