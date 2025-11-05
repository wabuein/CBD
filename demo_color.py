import argparse
import os
import sys
import platform
import cv2
import numpy as np
from ultralytics import YOLO

def colour_name_from_bgr(bgr_crop):
    if bgr_crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    s_mean = float(np.mean(s))
    v_mean = float(np.mean(v))
    if s_mean < 25: 
        if v_mean > 200:
            return "white"
        if v_mean < 60:
            return "black"
        return "gray"

    mask = (v > 40) & (v < 230) & (s > 40)
    if mask.sum() < 50:
        return "unknown"

    hue = h[mask].astype(np.float32)
    med = float(np.median(hue))

    if med < 10 or med >= 170: return "red"
    if med < 25:  return "orange"
    if med < 35:  return "yellow"
    if med < 85:  return "green"
    if med < 100: return "cyan"
    if med < 135:return "blue"
    if med < 160:return "purple"
    return "magenta"

def open_capture(src_str):
    source = 0 if src_str.isdigit() else src_str

    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        return cap
    
    if platform.system() == "Windows" and isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap

    raise RuntimeError(f"Could not open source: {src_str}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt",
                    help="Path to YOLO .pt or NCNN folder. Start with yolo11n.pt")
    ap.add_argument("--source", default="0",
                    help="Camera index or video file. '0' = default webcam")
    ap.add_argument("--imgsz", type=int, default=512,
                    help="Inference image size (lower = faster)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Confidence threshold")
    args = ap.parse_args()

    model = YOLO(args.model)
    cap = open_capture(args.source)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    win = "YOLO (object + colour) - press Q or ESC to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

        annotated = frame.copy()
        for r in results:
            names = r.names
            boxes = r.boxes
            for (xyxy, cls, conf) in zip(boxes.xyxy, boxes.cls, boxes.conf):
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                crop = frame[y1:y2, x1:x2]
                obj = names[int(cls)]
                colour = colour_name_from_bgr(crop)
                label = f"{obj} | {colour} | {conf:.2f}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, label, (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(win, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
