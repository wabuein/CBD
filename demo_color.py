import argparse
import time
import platform
import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# FIXED EXPERIMENT DESIGN
# ============================================================
TRIALS = 12  # Always 12 trials (e.g., 60s -> 12 x 5s)


# ============================================================
# COLOUR DETECTION (your original logic kept; only comments added)
# ============================================================
def colour_name_from_bgr(bgr_crop: np.ndarray) -> str:
    """Estimate a simple colour name from a cropped BGR region."""
    if bgr_crop is None or bgr_crop.size == 0:
        return "unknown"

    h_img, w_img = bgr_crop.shape[:2]
    if h_img < 4 or w_img < 4:
        return "unknown"

    # Use the center region to avoid edge/background contamination
    py = int(0.2 * h_img)
    px = int(0.2 * w_img)
    cx = bgr_crop[py:h_img - py, px:w_img - px]
    if cx.size == 0:
        cx = bgr_crop

    # Reduce sensor noise
    cx = cv2.medianBlur(cx, 3)

    # Convert to HSV for more stable colour naming
    hsv = cv2.cvtColor(cx, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # Low-saturation logic for grayscale/black/white
    s_mean = float(np.mean(S))
    v_mean = float(np.mean(V))
    if s_mean < 20:
        if v_mean > 200:
            return "white"
        if v_mean < 55:
            return "black"
        return "gray"

    # Mask out pixels that are too dark/bright to be reliable
    mask = (S > 30) & (V > 40) & (V < 230)
    if mask.sum() < 50:
        if v_mean > 200:
            return "white"
        if v_mean < 55:
            return "black"
        return "gray"

    # Weighted hue histogram (weight = saturation * brightness)
    Hm = H[mask].astype(np.int32)
    Sm = S[mask].astype(np.float32) / 255.0
    Vm = V[mask].astype(np.float32) / 255.0
    weights = Sm * Vm

    hist = np.bincount(Hm, weights=weights, minlength=180).astype(np.float32)
    peak = int(np.argmax(hist))

    # Special-case red wrap-around (0° and 180° are both red-ish)
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


# ============================================================
# LIGHT / DARKNESS METRIC (NEW)
# ============================================================
def compute_light_level(frame_bgr: np.ndarray):
    """
    Returns:
      - brightness_pct: 0..100 (based on grayscale mean)
      - v_mean: HSV V mean (0..255)
      - luma_mean: grayscale mean (0..255)
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    luma_mean = float(np.mean(gray))

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v_mean = float(np.mean(hsv[:, :, 2]))

    brightness_pct = (luma_mean / 255.0) * 100.0
    return brightness_pct, v_mean, luma_mean


# ============================================================
# RUNNING STATS (NEW) - online mean/min/max without storing all frames
# ============================================================
class RunningStats:
    """Tracks mean/min/max online (streaming)."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.min = float("inf")
        self.max = float("-inf")

    def add(self, x: float):
        x = float(x)
        self.n += 1
        self.mean += (x - self.mean) / self.n
        if x < self.min:
            self.min = x
        if x > self.max:
            self.max = x

    def summary(self, unit=""):
        if self.n == 0:
            return "n=0"
        return f"avg={self.mean:.2f}{unit}  min={self.min:.2f}{unit}  max={self.max:.2f}{unit}  n={self.n}"


# ============================================================
# CAMERA OPENING (CHANGED) - better for external webcams + Pi
# ============================================================
def open_capture(src_str: str, preferred_backend: str = "auto"):
    """
    Supports:
      --source 0 / 1 / 2  (camera indices incl. external USB webcams)
      --source /path/video.mp4
      --source rtsp://...
    Backend selection helps external webcams behave properly across OS.
    """
    source = int(src_str) if src_str.isdigit() else src_str
    system = platform.system()

    backend = cv2.CAP_ANY
    pb = preferred_backend.lower()

    if isinstance(source, int):
        # Auto choose best backend per OS for webcams
        if pb == "auto":
            if system == "Windows":
                backend = cv2.CAP_DSHOW
            elif system == "Linux":
                backend = cv2.CAP_V4L2
            else:
                backend = cv2.CAP_AVFOUNDATION
        else:
            backend_map = {
                "any": cv2.CAP_ANY,
                "dshow": cv2.CAP_DSHOW,
                "msmf": cv2.CAP_MSMF,
                "v4l2": cv2.CAP_V4L2,
                "avfoundation": cv2.CAP_AVFOUNDATION,
                "gstreamer": cv2.CAP_GSTREAMER,
            }
            if pb not in backend_map:
                raise ValueError(f"Unknown backend '{preferred_backend}'.")
            backend = backend_map[pb]

        cap = cv2.VideoCapture(source, backend)
    else:
        # File/URL streams
        if pb == "gstreamer":
            cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {src_str}")

    return cap


def apply_camera_settings(cap: cv2.VideoCapture, width: int, height: int, fps: int, mjpg: bool):
    """
    Camera settings help benchmark consistency:
    - Force MJPG for many USB webcams (often improves FPS on Pi)
    - Set resolution + target FPS (if supported by the webcam/driver)
    """
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(fps))


# ============================================================
# DRAW HELPERS (same as you had; added tiny helper for metrics text)
# ============================================================
def draw_labelled_box(img, x1, y1, x2, y2, text):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 255, 0), -1)
    cv2.putText(img, text, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def overlay_text(img, lines, x=10, y=30, line_h=24):
    for i, s in enumerate(lines):
        yy = y + i * line_h
        cv2.putText(img, s, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


# ============================================================
# BENCHMARK RUN (NEW)
# ============================================================
def run_benchmark(cap, model, imgsz, conf, duration_s, draw_detections=True):
    """
    Runs benchmark for duration_s total, split into TRIALS fixed time windows.
    Measures:
      - Light level (brightness % + raw means)
      - Capture time (ms)
      - Inference time (ms)
      - Post time (ms)
      - End-to-end time (ms)
      - FPS
    Prints per-trial averages and overall averages, then returns a final frame.
    """
    # Fixed trial design (12 always)
    num_trials = TRIALS
    trial_s = duration_s / num_trials

    # Overall stats
    st_fps = RunningStats()
    st_cap = RunningStats()
    st_inf = RunningStats()
    st_post = RunningStats()
    st_e2e = RunningStats()
    st_light = RunningStats()
    st_vmean = RunningStats()
    st_luma = RunningStats()

    # Trial stats (reset each trial)
    tr_fps = RunningStats()
    tr_cap = RunningStats()
    tr_inf = RunningStats()
    tr_post = RunningStats()
    tr_e2e = RunningStats()
    tr_light = RunningStats()

    trial_results = []
    trial_index = 1

    bench_start = time.perf_counter()
    next_trial_cut = bench_start + trial_s

    last_annotated = None

    while True:
        now = time.perf_counter()
        elapsed = now - bench_start
        if elapsed >= duration_s:
            break

        # Use perf_counter for accurate benchmarking timings
        t0 = time.perf_counter()

        # --- Capture timing ---
        tcap0 = time.perf_counter()
        ok, frame = cap.read()
        tcap1 = time.perf_counter()
        if not ok or frame is None:
            break

        # --- Light metric ---
        brightness_pct, v_mean, luma_mean = compute_light_level(frame)

        # --- Inference timing ---
        tinf0 = time.perf_counter()
        results = model.predict(source=frame, imgsz=imgsz, conf=conf, verbose=False)
        tinf1 = time.perf_counter()

        # --- Post timing (drawing + colour) ---
        annotated = frame.copy()
        if draw_detections:
            for r in results:
                names = r.names
                boxes = r.boxes
                for (xyxy, cls, cconf) in zip(boxes.xyxy, boxes.cls, boxes.conf):
                    x1, y1, x2, y2 = map(int, xyxy.tolist())
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)

                    # Keep crop safe
                    crop = frame[y1:y2, x1:x2]
                    colour = colour_name_from_bgr(crop)

                    label = f"{names[int(cls)]} | {colour} | {cconf:.2f}"
                    draw_labelled_box(annotated, x1, y1, x2, y2, label)

        tpost1 = time.perf_counter()
        t1 = tpost1

        # Compute metrics in ms
        cap_ms = (tcap1 - tcap0) * 1000.0
        inf_ms = (tinf1 - tinf0) * 1000.0
        post_ms = (tpost1 - tinf1) * 1000.0
        e2e_ms = (t1 - t0) * 1000.0
        fps = 1000.0 / max(1e-6, e2e_ms)

        # Update overall stats
        st_fps.add(fps)
        st_cap.add(cap_ms)
        st_inf.add(inf_ms)
        st_post.add(post_ms)
        st_e2e.add(e2e_ms)
        st_light.add(brightness_pct)
        st_vmean.add(v_mean)
        st_luma.add(luma_mean)

        # Update trial stats
        tr_fps.add(fps)
        tr_cap.add(cap_ms)
        tr_inf.add(inf_ms)
        tr_post.add(post_ms)
        tr_e2e.add(e2e_ms)
        tr_light.add(brightness_pct)

        # Live overlay during benchmarking
        overlay_text(annotated, [
            f"Benchmark: {elapsed:5.1f}/{duration_s:.0f}s | Trial {trial_index}/{num_trials} ({trial_s:.2f}s each)",
            f"Light: {brightness_pct:5.1f}% (V={v_mean:5.1f}/255, Luma={luma_mean:5.1f}/255)",
            f"FPS: {fps:5.1f} | cap {cap_ms:5.1f}ms | inf {inf_ms:5.1f}ms | post {post_ms:5.1f}ms | e2e {e2e_ms:5.1f}ms",
        ])

        last_annotated = annotated

        # Show frame while running
        cv2.imshow("YOLO Benchmark (press Q to abort run)", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        # Trial boundary
        if time.perf_counter() >= next_trial_cut and trial_index <= num_trials:
            trial_results.append({
                "trial": trial_index,
                "frames": tr_fps.n,
                "fps_avg": tr_fps.mean,
                "cap_ms_avg": tr_cap.mean,
                "inf_ms_avg": tr_inf.mean,
                "post_ms_avg": tr_post.mean,
                "e2e_ms_avg": tr_e2e.mean,
                "light_pct_avg": tr_light.mean,
            })

            # Reset trial stats
            tr_fps = RunningStats()
            tr_cap = RunningStats()
            tr_inf = RunningStats()
            tr_post = RunningStats()
            tr_e2e = RunningStats()
            tr_light = RunningStats()

            trial_index += 1
            next_trial_cut += trial_s

    # Print summary
    print("\n" + "=" * 78)
    print(f"BENCHMARK RESULTS | duration={duration_s:.1f}s | trials={TRIALS} | trial_len={duration_s/TRIALS:.2f}s")
    print(f"Device: {platform.system()} ({platform.machine()}) | imgsz={imgsz} | conf={conf}")
    print("-" * 78)
    for tr in trial_results:
        print(
            f"Trial {tr['trial']:02d} | frames={tr['frames']:4d} | "
            f"FPS {tr['fps_avg']:.2f} | "
            f"cap {tr['cap_ms_avg']:.2f}ms | inf {tr['inf_ms_avg']:.2f}ms | post {tr['post_ms_avg']:.2f}ms | e2e {tr['e2e_ms_avg']:.2f}ms | "
            f"light {tr['light_pct_avg']:.2f}%"
        )

    print("-" * 78)
    print("OVERALL AVERAGES (all frames in run)")
    print(f"FPS:        {st_fps.summary('')}")
    print(f"Capture ms: {st_cap.summary(' ms')}")
    print(f"Infer ms:   {st_inf.summary(' ms')}")
    print(f"Post ms:    {st_post.summary(' ms')}")
    print(f"End2End ms: {st_e2e.summary(' ms')}")
    print(f"Light %:    {st_light.summary('%')}")
    print(f"HSV V mean: {st_vmean.summary('')}")
    print(f"Luma mean:  {st_luma.summary('')}")
    print("=" * 78 + "\n")

    return last_annotated


# ============================================================
# MAIN LOOP (CHANGED) - idle screen, press R to run, auto-stop, press R again
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="YOLO + colour + fixed-12-trial benchmark (auto-stop + rerun)")
    ap.add_argument("--model", default="yolo11n.pt", help="YOLO weights path (e.g., yolo11n.pt).")
    ap.add_argument("--source", default="0", help="Camera index ('0','1',...) or video/stream path.")
    ap.add_argument("--backend", default="auto", help="auto|any|dshow|msmf|v4l2|avfoundation|gstreamer")
    ap.add_argument("--imgsz", type=int, default=512, help="Inference image size (lower -> faster).")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")

    # Benchmark duration only (trials fixed at 12)
    ap.add_argument("--duration", type=float, default=60.0, help="Benchmark duration in seconds (default 60).")

    # Camera settings
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--cam_fps", type=int, default=30)
    ap.add_argument("--mjpg", action="store_true", help="Force MJPG codec (often improves USB webcam FPS).")

    # Optional: disable drawing to isolate inference performance
    ap.add_argument("--no_draw", action="store_true", help="Disable drawing/colour (more pure speed test).")

    args = ap.parse_args()

    if args.duration <= 0:
        raise ValueError("duration must be > 0")

    trial_len = args.duration / TRIALS
    print(f"Configured experimental design: duration={args.duration:.1f}s split into {TRIALS} trials "
          f"({trial_len:.2f}s each).")

    model = YOLO(args.model)
    cap = open_capture(args.source, args.backend)

    # Apply webcam settings only if source is an integer camera index
    if args.source.isdigit():
        apply_camera_settings(cap, args.width, args.height, args.cam_fps, args.mjpg)

    win = "READY - press R to run benchmark, Q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        # Idle screen
        idle = np.zeros((260, 980, 3), dtype=np.uint8)
        overlay_text(idle, [
            "READY.",
            f"Press R to run benchmark: {args.duration:.0f}s total, fixed {TRIALS} trials ({trial_len:.2f}s each).",
            "Press Q or ESC to quit.",
        ], x=12, y=50, line_h=34)

        cv2.imshow(win, idle)
        key = cv2.waitKey(30) & 0xFF

        if key in (27, ord("q")):
            break

        if key == ord("r"):
            # Run benchmark; it will auto-stop at duration
            last_frame = run_benchmark(
                cap=cap,
                model=model,
                imgsz=args.imgsz,
                conf=args.conf,
                duration_s=args.duration,
                draw_detections=(not args.no_draw),
            )

            # Freeze last frame and wait for user to rerun or quit
            if last_frame is None:
                last_frame = idle.copy()

            overlay_text(last_frame, [
                "BENCHMARK COMPLETE.",
                "Printed per-trial + overall averages to terminal.",
                "Press R to run again, Q to quit.",
            ], x=12, y=160, line_h=30)

            while True:
                cv2.imshow(win, last_frame)
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
