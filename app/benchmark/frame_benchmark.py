import time
import platform
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from app.config.settings import TRIALS, WINDOW_RUN_TITLE
from app.vision.color_utils import colour_name_from_bgr, compute_light_level, white_balance_gray_world
from app.vision.mask_utils import BackgroundMasker
from app.utils.stats import RunningStats, DetectionSummary
from app.vision.yolo_runner import YoloRunner


@dataclass
class TrialReport:
    trial_idx: int
    frames: int

    # performance metrics (means/min/max tracked in RunningStats)
    cap_ms: RunningStats
    inf_ms: RunningStats
    post_ms: RunningStats
    e2e_ms: RunningStats
    fps: RunningStats

    # lighting metrics
    light_pct: RunningStats
    v_mean: RunningStats
    luma_mean: RunningStats

    # detections (object, color, confidence)
    det: DetectionSummary


def _overlay_text(img, lines, x=10, y=30, line_h=24):
    for i, s in enumerate(lines):
        yy = y + i * line_h
        cv2.putText(img, s, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def _draw_labelled_box(img, x1, y1, x2, y2, text):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 255, 0), -1)
    cv2.putText(img, text, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int):
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    return x1, y1, x2, y2


def _choose_primary_detection(dets, w: int, h: int):
    """
    Conveyor-style: choose a single "primary" object per frame.
    Largest area wins; tie-break by confidence.
    """
    best = None
    best_area = 0
    best_conf = -1.0

    for d in dets:
        x1, y1, x2, y2 = _clip_box(int(d.x1), int(d.y1), int(d.x2), int(d.y2), w, h)
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2 - x1) * (y2 - y1)
        if (area > best_area) or (area == best_area and float(d.conf) > best_conf):
            best = (x1, y1, x2, y2, d)
            best_area = area
            best_conf = float(d.conf)

    return best


class FrameMatchedBenchmark:
    """
    Frame-matched benchmark:
      - process exactly N frames
      - fixed TRIALS=12
      - per frame: capture/infer/post/e2e + light level
      - per trial: object counts, color counts, object-color pairs, mean confidence (proxy accuracy)
    """

    def __init__(self, yolo: YoloRunner, imgsz: int, conf_thres: float):
        self.yolo = yolo
        self.imgsz = imgsz
        self.conf_thres = conf_thres

    def run(self, cap: cv2.VideoCapture, total_frames: int, draw: bool = True) -> Optional[np.ndarray]:
        if total_frames <= 0:
            raise ValueError("total_frames must be > 0")

        frames_per_trial = max(1, total_frames // TRIALS)

        # Overall stats
        st_cap = RunningStats()
        st_inf = RunningStats()
        st_post = RunningStats()
        st_e2e = RunningStats()
        st_fps = RunningStats()

        st_light = RunningStats()
        st_v = RunningStats()
        st_luma = RunningStats()

        st_det = DetectionSummary()

        # Trial stats
        tr_cap = RunningStats()
        tr_inf = RunningStats()
        tr_post = RunningStats()
        tr_e2e = RunningStats()
        tr_fps = RunningStats()

        tr_light = RunningStats()
        tr_v = RunningStats()
        tr_luma = RunningStats()

        tr_det = DetectionSummary()

        trial_reports: List[TrialReport] = []

        frame_idx = 0
        trial_idx = 1

        run_start = time.perf_counter()
        last_annotated = None

        # ---- Capture background reference once (empty belt recommended) ----
        masker = BackgroundMasker()
        ok_ref, ref = cap.read()
        if not ok_ref or ref is None:
            return None
        ref = white_balance_gray_world(ref)
        masker.set_reference(ref)
        print("[MASK] Background reference captured. (Ensure belt was empty for best results)")

        # Main loop
        while frame_idx < total_frames:
            t0 = time.perf_counter()

            # ---- Capture ----
            tcap0 = time.perf_counter()
            ok, frame = cap.read()
            tcap1 = time.perf_counter()
            if not ok or frame is None:
                break

            # White balance frame (stabilizes color naming)
            frame = white_balance_gray_world(frame)

            # Foreground mask (belt/background removal)
            fg = masker.mask(frame)

            # ---- Lighting ----
            light_pct, v_mean, luma_mean = compute_light_level(frame)

            # ---- Inference ----
            tinf0 = time.perf_counter()
            dets = self.yolo.predict(frame, imgsz=self.imgsz, conf_thres=self.conf_thres)
            tinf1 = time.perf_counter()

            # ---- Post (draw + colour) ----
            annotated = frame.copy()
            h, w = frame.shape[:2]

            # Choose a single primary object to log per frame (reduces random COCO junk)
            primary = _choose_primary_detection(dets, w, h) if dets else None

            # Fallback: if YOLO found nothing, try using the foreground mask bbox as the "object"
            fallback_cls = "object"
            fallback_conf = 0.0
            if primary is None:
                bb = masker.largest_component_bbox(fg)
                if bb is not None:
                    x1, y1, x2, y2 = _clip_box(bb[0], bb[1], bb[2], bb[3], w, h)
                    if x2 > x1 and y2 > y1:
                        primary = (x1, y1, x2, y2, None)

            tpost_start = time.perf_counter()

            if primary is not None:
                x1, y1, x2, y2, d = primary
                crop = frame[y1:y2, x1:x2]
                crop_mask = fg[y1:y2, x1:x2]

                if draw:
                    if d is not None:
                        color = colour_name_from_bgr(crop, mask=crop_mask)
                        tr_det.add_detection(d.cls_name, color, d.conf)
                        st_det.add_detection(d.cls_name, color, d.conf)

                        label = f"{d.cls_name} | {color} | {float(d.conf):.2f}"
                        _draw_labelled_box(annotated, x1, y1, x2, y2, label)
                    else:
                        # fallback object (mask-only)
                        color = colour_name_from_bgr(crop, mask=crop_mask)
                        tr_det.add_detection(fallback_cls, color, fallback_conf)
                        st_det.add_detection(fallback_cls, color, fallback_conf)

                        label = f"{fallback_cls} | {color}"
                        _draw_labelled_box(annotated, x1, y1, x2, y2, label)
                else:
                    # Still record object+confidence, but color is not computed
                    if d is not None:
                        tr_det.add_detection(d.cls_name, "n/a", d.conf)
                        st_det.add_detection(d.cls_name, "n/a", d.conf)
                    else:
                        tr_det.add_detection(fallback_cls, "n/a", fallback_conf)
                        st_det.add_detection(fallback_cls, "n/a", fallback_conf)

            tpost1 = time.perf_counter()
            t1 = tpost1

            # ---- Metrics ----
            cap_ms = (tcap1 - tcap0) * 1000.0
            inf_ms = (tinf1 - tinf0) * 1000.0
            post_ms = (tpost1 - tpost_start) * 1000.0
            e2e_ms = (t1 - t0) * 1000.0
            fps = 1000.0 / max(1e-6, e2e_ms)

            # overall
            st_cap.add(cap_ms); st_inf.add(inf_ms); st_post.add(post_ms); st_e2e.add(e2e_ms); st_fps.add(fps)
            st_light.add(light_pct); st_v.add(v_mean); st_luma.add(luma_mean)

            # trial
            tr_cap.add(cap_ms); tr_inf.add(inf_ms); tr_post.add(post_ms); tr_e2e.add(e2e_ms); tr_fps.add(fps)
            tr_light.add(light_pct); tr_v.add(v_mean); tr_luma.add(luma_mean)

            frame_idx += 1
            current_trial = min(TRIALS, (frame_idx - 1) // frames_per_trial + 1)

            # Live overlay
            _overlay_text(annotated, [
                f"Frame-matched: {frame_idx}/{total_frames} | Trial {current_trial}/{TRIALS} | {frames_per_trial} f/trial",
                f"Light: {light_pct:5.1f}% (V={v_mean:5.1f}/255, Luma={luma_mean:5.1f}/255)",
                f"FPS: {fps:5.1f} | cap {cap_ms:5.1f}ms | inf {inf_ms:5.1f}ms | post {post_ms:5.1f}ms | e2e {e2e_ms:5.1f}ms",
                f"Dets(trial): {tr_det.det_count} | mean conf(proxy): {tr_det.det_conf.mean:.3f}",
                "Press Q to abort.",
            ])

            last_annotated = annotated

            cv2.imshow(WINDOW_RUN_TITLE, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            # ---- Trial boundary ----
            if frame_idx % frames_per_trial == 0 and trial_idx <= TRIALS:
                trial_reports.append(
                    TrialReport(
                        trial_idx=trial_idx,
                        frames=tr_fps.n,
                        cap_ms=tr_cap, inf_ms=tr_inf, post_ms=tr_post, e2e_ms=tr_e2e, fps=tr_fps,
                        light_pct=tr_light, v_mean=tr_v, luma_mean=tr_luma,
                        det=tr_det,
                    )
                )

                # reset trial accumulators
                tr_cap = RunningStats(); tr_inf = RunningStats(); tr_post = RunningStats()
                tr_e2e = RunningStats(); tr_fps = RunningStats()
                tr_light = RunningStats(); tr_v = RunningStats(); tr_luma = RunningStats()
                tr_det = DetectionSummary()

                trial_idx += 1

        total_time_s = time.perf_counter() - run_start
        self._print_report(
            trial_reports=trial_reports,
            frames_done=frame_idx,
            frames_target=total_frames,
            frames_per_trial=frames_per_trial,
            total_time_s=total_time_s,
            st_cap=st_cap, st_inf=st_inf, st_post=st_post, st_e2e=st_e2e, st_fps=st_fps,
            st_light=st_light, st_v=st_v, st_luma=st_luma,
            st_det=st_det,
        )

        return last_annotated

    def _print_report(
        self,
        trial_reports: List[TrialReport],
        frames_done: int,
        frames_target: int,
        frames_per_trial: int,
        total_time_s: float,
        st_cap: RunningStats, st_inf: RunningStats, st_post: RunningStats, st_e2e: RunningStats, st_fps: RunningStats,
        st_light: RunningStats, st_v: RunningStats, st_luma: RunningStats,
        st_det: DetectionSummary,
    ) -> None:
        print("\n" + "=" * 110)
        print(f"FRAME-MATCHED BENCHMARK | frames={frames_done}/{frames_target} | trials={TRIALS} | ~{frames_per_trial} frames/trial")
        print(f"Device: {platform.system()} ({platform.machine()}) | imgsz={self.imgsz} | conf_thres={self.conf_thres}")
        print(f"Total wall time: {total_time_s:.2f}s | Throughput: {frames_done / max(1e-9, total_time_s):.2f} FPS")
        print("-" * 110)

        for tr in trial_reports:
            print(f"Trial {tr.trial_idx:02d} | frames={tr.frames:4d} | "
                  f"FPS {tr.fps.mean:.2f} | cap {tr.cap_ms.mean:.2f}ms | inf {tr.inf_ms.mean:.2f}ms | post {tr.post_ms.mean:.2f}ms | e2e {tr.e2e_ms.mean:.2f}ms | "
                  f"light {tr.light_pct.mean:.2f}% | "
                  f"mean conf(proxy) {tr.det.det_conf.mean:.3f} | dets {tr.det.det_count}")

            top_cls = ", ".join([f"{c}({n})" for c, n in tr.det.top_classes(5)]) or "none"
            top_col = ", ".join([f"{c}({n})" for c, n in tr.det.top_colors(5)]) or "none"
            top_pair = ", ".join([f"{k[0]}-{k[1]}({n})" for k, n in tr.det.top_class_color_pairs(5)]) or "none"
            print(f"  Top objects: {top_cls}")
            print(f"  Top colors:  {top_col}")
            print(f"  Top pairs:   {top_pair}")

            cmc = tr.det.class_mean_conf()
            top_classes = [c for c, _ in tr.det.top_classes(5)]
            if top_classes:
                per_cls_conf = ", ".join([f"{c}:{cmc.get(c, 0.0):.3f}" for c in top_classes])
                print(f"  Mean conf by object (top): {per_cls_conf}")

        print("-" * 110)
        print("OVERALL PERFORMANCE (all frames)")
        print(f"Capture ms: {st_cap.summary(' ms')}")
        print(f"Infer ms:   {st_inf.summary(' ms')}")
        print(f"Post ms:    {st_post.summary(' ms')}")
        print(f"End2End ms: {st_e2e.summary(' ms')}")
        print(f"FPS:        {st_fps.summary('')}")
        print("OVERALL LIGHTING")
        print(f"Light %:    {st_light.summary('%')}")
        print(f"HSV V mean: {st_v.summary('')}")
        print(f"Luma mean:  {st_luma.summary('')}")

        print("OVERALL DETECTIONS")
        print(f"Total detections: {st_det.det_count}")
        print(f"Mean confidence (proxy): {st_det.det_conf.summary('')}")
        print(f"Top objects: {', '.join([f'{c}({n})' for c, n in st_det.top_classes(10)]) or 'none'}")
        print(f"Top colors:  {', '.join([f'{c}({n})' for c, n in st_det.top_colors(10)]) or 'none'}")
        top_pairs = st_det.top_class_color_pairs(10)
        if top_pairs:
            print("Top object-color pairs:")
            for (cls_name, color), n in top_pairs:
                print(f"  - {cls_name} | {color}: {n}")

        print("=" * 110 + "\n")
