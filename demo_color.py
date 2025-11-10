import argparse
import time
import platform
import cv2
import numpy as np
from ultralytics import YOLO


def colour_name_from_bgr(bgr_crop: np.ndarray) -> str:
    if bgr_crop is None or bgr_crop.size == 0:
        return "unknown"

    h_img, w_img = bgr_crop.shape[:2]
    if h_img < 4 or w_img < 4:
        return "unknown"

    py = int(0.2 * h_img)
    px = int(0.2 * w_img)
    cx = bgr_crop[py:h_img - py, px:w_img - px]
    if cx.size == 0:
        cx = bgr_crop

    cx = cv2.medianBlur(cx, 3)

    hsv = cv2.cvtColor(cx, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    s_mean = float(np.mean(S))
    v_mean = float(np.mean(V))
    if s_mean < 20:  
        if v_mean > 200:
            return "white"
        if v_mean < 55:
            return "black"
        return "gray"

    mask = (S > 30) & (V > 40) & (V < 230)
    if mask.sum() < 50:
        if v_mean > 200:
            return "white"
        if v_mean < 55:
            return "black"
        return "gray"

    Hm = H[mask].astype(np.int32)
    Sm = S[mask].astype(np.float32) / 255.0
    Vm = V[mask].astype(np.float32) / 255.0
    weights = Sm * Vm  

    hist = np.bincount(Hm, weights=weights, minlength=180).astype(np.float32)

    peak = int(np.argmax(hist))
    red_wrap = hist[0:10].sum() + hist[170:180].sum()

    def hue_to_name(h):
        if h < 10 or h >= 170:
            return "red"
        if h < 25:
            return "orange"
        if h < 35:
            return "yellow"
        if h < 85:
            return "green"
        if h < 100:
            return "cyan"
        if h < 135:
            return "blue"
        if h < 160:
            return "purple"
        return "magenta"

    name = hue_to_name(peak)
    if red_wrap > hist[peak] * 1.2:
        name = "red"

    return name


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


def draw_labelled_box(img, x1, y1, x2, y2, text):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 255, 0), -1)
    cv2.putText(img, text, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="YOLO object + colour demo")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="Path to YOLO .pt weights or NCNN folder (for Pi later).")
    ap.add_argument("--source", default="0",
                    help="Camera index ('0') or path to a video/image.")
    ap.add_argument("--imgsz", type=int, default=512,
                    help="Inference image size. Lower = faster.")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Confidence threshold.")
    ap.add_argument("--show_fps", action="store_true",
                    help="Overlay FPS in the window.")
    args = ap.parse_args()

    model = YOLO(args.model)
    cap = open_capture(args.source)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    win = "YOLO (object + colour)  —  press Q or ESC to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    t_prev = time.time()
    fps = 0.0

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

                ch = y2 - y1
                cw = x2 - x1
                cy1 = y1 + int(0.2 * ch)
                cy2 = y2 - int(0.2 * ch)
                cx1 = x1 + int(0.2 * cw)
                cx2 = x2 - int(0.2 * cw)
                cy1, cy2 = max(0, cy1), max(cy1 + 1, cy2)
                cx1, cx2 = max(0, cx1), max(cx1 + 1, cx2)

                crop = frame[cy1:cy2, cx1:cx2]
                colour = colour_name_from_bgr(crop)

                label = f"{names[int(cls)]} | {colour} | {conf:.2f}"
                draw_labelled_box(annotated, x1, y1, x2, y2, label)

        if args.show_fps:
            now = time.time()
            dt = now - t_prev
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
            t_prev = now
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(win, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
