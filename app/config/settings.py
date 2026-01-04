# config/settings.py
"""
Central place for default experiment and camera settings.
You can override these via CLI in main.py, but these are the defaults.
"""

# Fixed experimental design
TRIALS = 12

# Benchmark defaults
DEFAULT_TOTAL_FRAMES = 600   # frame-matched run length
DEFAULT_IMGSZ = 512
DEFAULT_CONF = 0.25
VERSION = "3.0.1"

# Camera defaults
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_CAM_FPS = 30

# UI / Window titles
WINDOW_READY_TITLE = "READY - press R to run, Q to quit"
WINDOW_RUN_TITLE = "YOLO Benchmark (press Q to abort run)"
