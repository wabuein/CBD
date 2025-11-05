# YOLO Object and Color Detection

### Final Year Project – Intelligent Vision System

This project implements a **real-time object and color detection system** using **YOLO (You Only Look Once)** deep-learning models.
It identifies everyday household objects and estimates their dominant color through a lightweight post-processing step.
The system runs on both a **laptop** (for development) and a **Raspberry Pi 5** (for deployment).

---

## 🎯 Objectives

* Detect multiple common objects in real time using a pretrained YOLO model.
* Estimate the dominant color of each detected object using HSV color analysis.
* Run efficiently on low-power hardware such as Raspberry Pi 5.
* Provide an extendable base for IoT, home-automation, and robotics vision tasks.

---

## 🧰 Requirements

### Hardware

* Laptop or Desktop (for development / testing)
* Raspberry Pi 5 (4 GB RAM + 32 GB SSD recommended)
* USB or CSI camera module

### Software

* **Python 3.11+**
* **Ultralytics YOLO** (v8/10/11 compatible)
* **OpenCV** for camera input and image handling
* **PyTorch CPU** (for inference on laptop; optional on Pi when exporting to NCNN)

---

## ⚙️ Installation (Laptop Setup)

```bash
# 1. create and activate virtual environment
python -m venv yolo-pi
yolo-pi\Scripts\activate  # (Windows)
# or
source yolo-pi/bin/activate  # (macOS/Linux)

# 2. install dependencies
pip install --upgrade pip
pip install ultralytics opencv-python torch torchvision torchaudio
```

---

## ▶️ Running the System (on Laptop)

1. Place the file **detect_color.py** in your working directory.
2. Run YOLO with your webcam:

   ```bash
   python detect_color.py --model yolo11n.pt --source 0 --imgsz 512 --show_fps
   ```
3. The window will display bounding boxes with:

   * **Object name**
   * **Dominant color**
   * **Confidence score**

Press **Q** or **ESC** to exit.

---

## 🧩 Applications

* Object sorting and counting systems
* Robotics and automation
* Assistive vision for accessibility

---

## 📚 References

* [Ultralytics YOLO Documentation](https://docs.ultralytics.com)
* [Objects365 Dataset](https://www.objects365.org)
* [OpenCV Documentation](https://docs.opencv.org)

### 🏁 Summary

This project demonstrates a portable, efficient, and extendable vision system capable of identifying everyday objects and estimating their color in real time.
By combining pretrained YOLO models with optimized inference frameworks like NCNN, it delivers high accuracy even on low-power d
