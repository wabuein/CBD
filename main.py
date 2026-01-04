# main.py
import argparse
import numpy as np
import cv2

from app.config.settings import (
    TRIALS,
    DEFAULT_TOTAL_FRAMES,
    DEFAULT_IMGSZ,
    DEFAULT_CONF,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_CAM_FPS,
    WINDOW_READY_TITLE,
    VERSION,
)

from app.camera.camera import open_capture, apply_camera_settings
from app.vision.yolo_runner import YoloRunner
from app.benchmark.frame_benchmark import FrameMatchedBenchmark


def _overlay_text(img, lines, x=12, y=50, line_h=34):
    for i, s in enumerate(lines):
        yy = y + i * line_h
        cv2.putText(img, s, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)


def main():
    print("CBD By Waseem Abuein")
    print(f"VERSION: {VERSION}\n")

    ap = argparse.ArgumentParser(description="YOLO object+color + frame-matched benchmark (12 trials fixed)")
    ap.add_argument("--model", default="yolo11n.pt", help="Path to YOLO weights.")
    ap.add_argument("--source", default="0", help="Camera index '0','1',... or file/stream path.")
    ap.add_argument("--backend", default="auto", help="auto|any|dshow|msmf|v4l2|avfoundation|gstreamer")

    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)

    ap.add_argument("--frames", type=int, default=DEFAULT_TOTAL_FRAMES, help="Total frames per run (frame-matched).")

    # Camera settings
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--cam_fps", type=int, default=DEFAULT_CAM_FPS)
    ap.add_argument("--mjpg", action="store_true", help="Force MJPG codec (often helps USB webcams).")

    # Optional: isolate inference by disabling drawing + colour
    ap.add_argument("--no_draw", action="store_true", help="Disable drawing and colour naming.")
    args = ap.parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be > 0")

    frames_per_trial = max(1, args.frames // TRIALS)
    print(f"Design: {args.frames} frames split into {TRIALS} trials (~{frames_per_trial} frames/trial).")

    # Create components
    yolo = YoloRunner(args.model)
    benchmark = FrameMatchedBenchmark(yolo=yolo, imgsz=args.imgsz, conf_thres=args.conf)

    cap = open_capture(args.source, args.backend)

    # Apply capture settings only for camera sources
    if args.source.isdigit():
        apply_camera_settings(cap, args.width, args.height, args.cam_fps, args.mjpg)

    # Main UI loop: press R to run, auto-stops, press R again to rerun
    cv2.namedWindow(WINDOW_READY_TITLE, cv2.WINDOW_NORMAL)

    while True:
        idle = np.zeros((300, 1200, 3), dtype=np.uint8)
        _overlay_text(idle, [
            "READY.",
            f"Press R to run: {args.frames} frames | fixed {TRIALS} trials (~{frames_per_trial} frames each).",
            "Per-trial output includes: performance, lighting, detections, top objects/colors, mean confidence (proxy accuracy).",
            "Press Q or ESC to quit."
        ])

        cv2.imshow(WINDOW_READY_TITLE, idle)
        key = cv2.waitKey(30) & 0xFF

        if key in (27, ord("q")):
            break

        if key == ord("r"):
            last_frame = benchmark.run(
                cap=cap,
                total_frames=args.frames,
                draw=(not args.no_draw),
            )

            if last_frame is None:
                last_frame = idle.copy()

            _overlay_text(last_frame, [
                "BENCHMARK COMPLETE.",
                "Check terminal output for per-trial + overall means.",
                "Press R to run again, Q to quit."
            ], x=12, y=180, line_h=34)

            while True:
                cv2.imshow(WINDOW_READY_TITLE, last_frame)
                k2 = cv2.waitKey(30) & 0xFF
                if k2 in (27, ord("q")):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                if k2 == ord("r"):
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()