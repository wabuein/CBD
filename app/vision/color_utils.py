# vision/color_utils.py
import cv2
import numpy as np


def colour_name_from_bgr(bgr_crop: np.ndarray) -> str:
    """
    Your original HSV-based colour naming logic.
    Kept the same, only reorganized into a module.
    """
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

    def hue_to_name(h: int) -> str:
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


def compute_light_level(frame_bgr: np.ndarray):
    """
    Measures brightness/darkness:
    Returns:
      brightness_pct (0..100), v_mean (0..255), luma_mean (0..255)
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    luma_mean = float(np.mean(gray))

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v_mean = float(np.mean(hsv[:, :, 2]))

    brightness_pct = (luma_mean / 255.0) * 100.0
    return brightness_pct, v_mean, luma_mean
