import argparse
import time
import platform
import cv2
import numpy as np
from ultralytics import YOLO
import psutil
from collections import Counter, defaultdict


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


def open_capture(src_str: str, backend: str = "auto"):
    source = int(src_str) if src_str.isdigit() else src_str

    backend_map = {
        "auto": 0,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "v4l2": cv2.CAP_V4L2,
        "gstreamer": cv2.CAP_GSTREAMER,
        "ffmpeg": cv2.CAP_FFMPEG,
    }

    if backend not in backend_map:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {', '.join(backend_map.keys())}")

    if backend != "auto":
        cap = cv2.VideoCapture(source, backend_map[backend])
        if cap.isOpened():
            return cap

    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        return cap

    if platform.system() == "Windows" and isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap

    raise RuntimeError(f"Could not open source: {src_str} (backend={backend})")


def draw_labelled_box(img, x1, y1, x2, y2, text):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 255, 0), -1)
    cv2.putText(img, text, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def light_percent_from_bgr(frame: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    return float(np.mean(v) / 255.0 * 100.0)


def safe_cpu_temp_c():
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None
        for key in ("cpu_thermal", "coretemp", "soc_thermal"):
            if key in temps and temps[key]:
                vals = [t.current for t in temps[key] if t.current is not None]
                if vals:
                    return float(np.mean(vals))
        all_vals = []
        for group in temps.values():
            for t in group:
                if t.current is not None:
                    all_vals.append(t.current)
        return float(np.mean(all_vals)) if all_vals else None
    except Exception:
        return None


def mean_std(x):
    x = np.array(x, dtype=np.float64)
    if x.size == 0:
        return 0.0, 0.0
    return float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else 0.0


def top_k_counter(counter: Counter, k: int = 5):
    return counter.most_common(k)


def main():
    ap = argparse.ArgumentParser(description="YOLO benchmark: FPS + light% + CPU + detections + confidence + colour (with live view)")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--source", default="1", help="External camera index (often 1) or a video path.")
    ap.add_argument("--backend", default="auto", help="auto, dshow, msmf, v4l2, gstreamer, ffmpeg")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)

    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--frames_per_trial", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)

    ap.add_argument("--topk", type=int, default=5, help="How many top classes/colours to print per trial.")
    ap.add_argument("--show_fps", action="store_true", help="Overlay rolling FPS on the video.")
    args = ap.parse_args()

    total_frames = args.trials * args.frames_per_trial

    model = YOLO(args.model)
    cap = open_capture(args.source, backend=args.backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    proc = psutil.Process()
    psutil.cpu_percent(interval=None)
    proc.cpu_percent(interval=None)

    win = "YOLO Benchmark (live) — press Q or ESC to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("\n=== BENCHMARK CONFIG ===")
    print(f"Model:            {args.model}")
    print(f"Source:           {args.source} (backend={args.backend})")
    print(f"Resolution:       {args.width}x{args.height}")
    print(f"imgsz/conf:       {args.imgsz} / {args.conf}")
    print(f"Trials:           {args.trials}")
    print(f"Frames/trial:     {args.frames_per_trial}")
    print(f"Total frames:     {total_frames}")
    print(f"Warmup frames:    {args.warmup}")
    print("Colour pipeline:  ON (always)")
    print("========================\n")

    # Warmup (includes detection + colour path)
    for _ in range(args.warmup):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Camera/video ended during warmup.")
        results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        for r in results:
            boxes = r.boxes
            if boxes is None or boxes.xyxy is None:
                continue
            for xyxy in boxes.xyxy:
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                ch = y2 - y1
                cw = x2 - x1
                if ch < 2 or cw < 2:
                    continue
                cy1 = y1 + int(0.2 * ch)
                cy2 = y2 - int(0.2 * ch)
                cx1 = x1 + int(0.2 * cw)
                cx2 = x2 - int(0.2 * cw)
                cy1, cy2 = max(0, cy1), max(cy1 + 1, cy2)
                cx1, cx2 = max(0, cx1), max(cx1 + 1, cx2)
                crop = frame[cy1:cy2, cx1:cx2]
                _ = colour_name_from_bgr(crop)

        _ = light_percent_from_bgr(frame)
        psutil.cpu_percent(interval=None)
        proc.cpu_percent(interval=None)

    # per-trial aggregates
    trial_fps, trial_light = [], []
    trial_cpu, trial_proc_cpu = [], []
    trial_ram_mb, trial_proc_ram_mb = [], []
    trial_freq_mhz, trial_temp_c = [], []
    trial_mean_conf = []  # confidence as a proxy for "accuracy"
    trial_total_dets = []

    # global aggregates
    global_class_counts = Counter()
    global_colour_counts = Counter()
    global_conf_list = []

    rolling_fps = 0.0
    t_prev = time.perf_counter()

    frames_done = 0

    for t in range(1, args.trials + 1):
        # trial samples
        light_samples = []
        cpu_samples = []
        proc_cpu_samples = []
        ram_samples = []
        proc_ram_samples = []
        freq_samples = []
        temp_samples = []

        class_counts = Counter()
        colour_counts = Counter()
        conf_list = []
        class_conf_sum = defaultdict(float)
        class_conf_count = defaultdict(int)

        t0 = time.perf_counter()

        for fidx in range(1, args.frames_per_trial + 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Camera/video ended early at frame {frames_done}/{total_frames}.")

            results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

            annotated = frame.copy()

            # annotate + compute colour + detection stats
            for r in results:
                names = r.names
                boxes = r.boxes
                if boxes is None or boxes.xyxy is None:
                    continue

                for (xyxy, cls, conf) in zip(boxes.xyxy, boxes.cls, boxes.conf):
                    x1, y1, x2, y2 = map(int, xyxy.tolist())
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)

                    ch = y2 - y1
                    cw = x2 - x1
                    if ch < 2 or cw < 2:
                        continue

                    cy1 = y1 + int(0.2 * ch)
                    cy2 = y2 - int(0.2 * ch)
                    cx1 = x1 + int(0.2 * cw)
                    cx2 = x2 - int(0.2 * cw)
                    cy1, cy2 = max(0, cy1), max(cy1 + 1, cy2)
                    cx1, cx2 = max(0, cx1), max(cx1 + 1, cx2)

                    crop = frame[cy1:cy2, cx1:cx2]
                    colour = colour_name_from_bgr(crop)

                    cls_i = int(cls)
                    cls_name = names.get(cls_i, str(cls_i)) if isinstance(names, dict) else names[cls_i]

                    c = float(conf)
                    conf_list.append(c)
                    class_counts[cls_name] += 1
                    colour_counts[colour] += 1
                    class_conf_sum[cls_name] += c
                    class_conf_count[cls_name] += 1

                    label = f"{cls_name} | {colour} | {c*100:.1f}%"
                    draw_labelled_box(annotated, x1, y1, x2, y2, label)

            # light %
            light_samples.append(light_percent_from_bgr(frame))

            # CPU
            cpu_samples.append(psutil.cpu_percent(interval=None))
            proc_cpu_samples.append(proc.cpu_percent(interval=None))

            # memory
            vm = psutil.virtual_memory()
            trial_used_mb = vm.used / (1024 * 1024)
            ram_samples.append(trial_used_mb)

            pm = proc.memory_info()
            proc_ram_samples.append(pm.rss / (1024 * 1024))

            # freq
            try:
                f = psutil.cpu_freq()
                freq_samples.append(float(f.current) if f and f.current else 0.0)
            except Exception:
                freq_samples.append(0.0)

            # temp
            temp = safe_cpu_temp_c()
            if temp is not None:
                temp_samples.append(temp)

            # rolling FPS overlay (optional)
            if args.show_fps:
                now = time.perf_counter()
                dt = now - t_prev
                if dt > 0:
                    inst = 1.0 / dt
                    rolling_fps = 0.9 * rolling_fps + 0.1 * inst if rolling_fps > 0 else inst
                t_prev = now
                cv2.putText(annotated, f"Rolling FPS: {rolling_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

            # trial/frame overlay
            cv2.putText(annotated, f"Trial {t}/{args.trials}  Frame {fidx}/{args.frames_per_trial}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(win, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                cap.release()
                cv2.destroyAllWindows()
                print("\nStopped by user.\n")
                return

            frames_done += 1

        t1 = time.perf_counter()
        secs = max(1e-9, (t1 - t0))
        fps = args.frames_per_trial / secs

        avg_light = float(np.mean(light_samples)) if light_samples else 0.0
        avg_cpu = float(np.mean(cpu_samples)) if cpu_samples else 0.0
        avg_proc_cpu = float(np.mean(proc_cpu_samples)) if proc_cpu_samples else 0.0
        avg_ram = float(np.mean(ram_samples)) if ram_samples else 0.0
        avg_proc_ram = float(np.mean(proc_ram_samples)) if proc_ram_samples else 0.0
        avg_freq = float(np.mean(freq_samples)) if freq_samples else 0.0
        avg_temp = float(np.mean(temp_samples)) if temp_samples else None

        mean_conf = float(np.mean(conf_list)) if conf_list else 0.0
        total_dets = int(sum(class_counts.values()))

        # save trial metrics
        trial_fps.append(fps)
        trial_light.append(avg_light)
        trial_cpu.append(avg_cpu)
        trial_proc_cpu.append(avg_proc_cpu)
        trial_ram_mb.append(avg_ram)
        trial_proc_ram_mb.append(avg_proc_ram)
        trial_freq_mhz.append(avg_freq)
        if avg_temp is not None:
            trial_temp_c.append(avg_temp)
        trial_mean_conf.append(mean_conf)
        trial_total_dets.append(total_dets)

        # update global aggregates
        global_class_counts.update(class_counts)
        global_colour_counts.update(colour_counts)
        global_conf_list.extend(conf_list)

        # prepare top-k strings
        top_classes = top_k_counter(class_counts, args.topk)
        top_colours = top_k_counter(colour_counts, args.topk)

        def class_line():
            if not top_classes:
                return "None"
            parts = []
            for name, cnt in top_classes:
                mconf = (class_conf_sum[name] / class_conf_count[name]) if class_conf_count[name] else 0.0
                parts.append(f"{name}({cnt}, {mconf*100:.1f}%)")
            return ", ".join(parts)

        def colour_line():
            if not top_colours:
                return "None"
            return ", ".join([f"{c}({n})" for c, n in top_colours])

        print(
            f"Trial {t:02d}/{args.trials} | "
            f"FPS: {fps:7.2f} | "
            f"Light: {avg_light:6.2f}% | "
            f"MeanConf: {mean_conf*100:6.2f}% | "
            f"Dets: {total_dets:4d} | "
            f"CPU: {avg_cpu:6.2f}% | "
            f"ProcCPU: {avg_proc_cpu:7.2f}% | "
            f"RAM: {avg_ram:8.1f} MB | "
            f"ProcRAM: {avg_proc_ram:7.1f} MB | "
            f"Freq: {avg_freq:7.0f} MHz"
            + (f" | Temp: {avg_temp:5.1f} C" if avg_temp is not None else "")
        )
        print(f"  Top classes: {class_line()}")
        print(f"  Top colours: {colour_line()}")

    cap.release()
    cv2.destroyAllWindows()

    # Summary
    fps_m, fps_s = mean_std(trial_fps)
    light_m, light_s = mean_std(trial_light)
    cpu_m, cpu_s = mean_std(trial_cpu)
    pcpu_m, pcpu_s = mean_std(trial_proc_cpu)
    ram_m, ram_s = mean_std(trial_ram_mb)
    pram_m, pram_s = mean_std(trial_proc_ram_mb)
    freq_m, freq_s = mean_std(trial_freq_mhz)
    conf_m, conf_s = mean_std(np.array(trial_mean_conf) * 100.0)
    det_m, det_s = mean_std(trial_total_dets)

    print("\n=== SUMMARY (mean ± std over trials) ===")
    print(f"FPS:         {fps_m:.2f} ± {fps_s:.2f}")
    print(f"Light %:     {light_m:.2f} ± {light_s:.2f}")
    print(f"MeanConf %:  {conf_m:.2f} ± {conf_s:.2f}")
    print(f"Dets/trial:  {det_m:.1f} ± {det_s:.1f}")
    print(f"CPU %:       {cpu_m:.2f} ± {cpu_s:.2f}")
    print(f"ProcCPU %:   {pcpu_m:.2f} ± {pcpu_s:.2f}")
    print(f"RAM MB:      {ram_m:.1f} ± {ram_s:.1f}")
    print(f"ProcRAM MB:  {pram_m:.1f} ± {pram_s:.1f}")
    print(f"Freq MHz:    {freq_m:.0f} ± {freq_s:.0f}")
    if trial_temp_c:
        temp_m, temp_s = mean_std(trial_temp_c)
        print(f"Temp C:      {temp_m:.1f} ± {temp_s:.1f}")

    print("\nTop classes overall:")
    for name, cnt in global_class_counts.most_common(args.topk):
        print(f"  {name}: {cnt}")

    print("\nTop colours overall:")
    for name, cnt in global_colour_counts.most_common(args.topk):
        print(f"  {name}: {cnt}")

    if global_conf_list:
        print(f"\nOverall mean confidence: {np.mean(global_conf_list)*100:.2f}%")

    print("=======================================\n")


if __name__ == "__main__":
    main()
