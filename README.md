# YOLO Object and Color Detection

### Final Year Project – Intelligent Vision System

This repository implements a **real-time object and color detection system** using **Ultralytics YOLO** deep-learning models and will be used for the final Reconfigurable Conveyer Sorting System with AI-Based Object Classification (CBD).

It detects everyday objects and estimates their dominant color using a lightweight HSV-based post-processing pipeline.

The system is designed to run on both:

* a **laptop** (development and benchmarking), and
* a **Raspberry Pi 5** (low-power deployment),

with a strong emphasis on **fair, reproducible performance evaluation**.

---

## 🎯 Objectives

* Detect multiple common objects in real time using a pretrained YOLO model.
* Estimate the dominant color of each detected object using HSV color analysis.
* Run efficiently on low-power hardware such as the Raspberry Pi 5.
* Provide a **robust benchmarking framework** to fairly compare laptop vs Raspberry Pi performance.
* Serve as an extendable base for IoT, robotics, and intelligent automation systems.

---

## 🧠 System Overview

The system consists of three main stages per frame:

1. **Frame capture** from a USB or CSI camera
2. **YOLO inference** using Ultralytics’ `predict()` pipeline
3. **Post-processing**, including:

   * bounding-box rendering
   * dominant color estimation
   * metric collection (timing and lighting)

Each stage is measured independently to understand where performance bottlenecks occur.

---

## 🧪 Benchmarking Methodology (Frame-Matched, Fixed Trials)

To ensure a **fair and scientifically valid comparison** between the laptop and Raspberry Pi, the system uses a **frame-matched benchmark design**.

### Key design choices

* **Fixed total number of frames per run** (default: 600 frames)
* **Fixed number of trials: 12**
* Frames are split evenly:

  * 600 frames → **12 trials × 50 frames per trial**
* Each frame records:

  * capture time
  * inference time
  * post-processing time
  * end-to-end latency
  * derived FPS
  * scene light level (brightness)

---

## 🧰 Requirements

### Hardware

* Laptop or Desktop (development and benchmarking)
* Raspberry Pi 5 (4 GB RAM or higher recommended)
* USB webcam or CSI camera module

### Software

* **Python 3.11+**
* **Ultralytics YOLO** (v8 / v10 / v11 compatible)
* **OpenCV** (camera input and image processing)
* **PyTorch (CPU)** for laptop inference
* Optional: NCNN / ONNX export for Raspberry Pi optimisation

---

## ⚙️ Installation (Laptop Setup)

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

## ▶️ Running the System (Detection Mode)

```bash
python demo_color.py --model yolo11n.pt --source 0 --imgsz 512
```

The display window shows:

* Object name
* Dominant color
* Detection confidence

Press **Q** or **ESC** to exit.

---

## ▶️ Running the Benchmark (Frame-Matched)

```bash
python demo_color.py --model yolo11n.pt --source 0 --frames 600
```

During benchmarking:

* Press **R** to start a benchmark run
* The system automatically stops after the specified number of frames
* Per-trial and overall averages are printed to the terminal
* The final frame freezes on screen
* Press **R** to run again, **Q** to quit

### Optional flags

* `--no_draw` → disables drawing and color detection (pure inference benchmark)
* `--mjpg` → forces MJPG codec (recommended for USB webcams on Raspberry Pi)
* `--backend v4l2` → recommended capture backend on Linux / Raspberry Pi

---

## 📊 Reported Metrics

Per-trial and overall averages are reported for:

* Capture time (ms)
* Inference time (ms)
* Post-processing time (ms)
* End-to-end latency (ms)
* Derived FPS
* Scene brightness (light level)

This allows precise identification of performance bottlenecks between platforms.

---

## 🧩 Applications

* Intelligent object sorting systems
* Robotics and autonomous platforms
* Smart home and IoT vision nodes
* Embedded AI benchmarking and optimisation studies

---

## 📚 References

* Ultralytics YOLO Documentation: [https://docs.ultralytics.com](https://docs.ultralytics.com)
* Objects365 Dataset: [https://www.objects365.org](https://www.objects365.org)
* OpenCV Documentation: [https://docs.opencv.org](https://docs.opencv.org)