import cv2
import numpy as np


# -----------------------------
# White balance (cheap + strong)
# -----------------------------
def white_balance_gray_world(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Gray-world white balance to reduce lighting color cast.
    Works well for indoor LED/warm light which often causes yellow/orange confusion.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr

    img = frame_bgr.astype(np.float32)
    mean_b = float(np.mean(img[:, :, 0]))
    mean_g = float(np.mean(img[:, :, 1]))
    mean_r = float(np.mean(img[:, :, 2]))
    mean = (mean_b + mean_g + mean_r) / 3.0

    # Avoid divide by 0
    scale_b = mean / (mean_b + 1e-6)
    scale_g = mean / (mean_g + 1e-6)
    scale_r = mean / (mean_r + 1e-6)

    img[:, :, 0] *= scale_b
    img[:, :, 1] *= scale_g
    img[:, :, 2] *= scale_r

    return np.clip(img, 0, 255).astype(np.uint8)


# -----------------------------
# Dominant color (mask-aware)
# -----------------------------
def _dominant_bgr_from_pixels(pixels_bgr: np.ndarray) -> np.ndarray | None:
    """
    Robust dominant colour estimate from Nx3 pixels in BGR.
    Uses median (robust to highlights/shadows).
    """
    if pixels_bgr is None or pixels_bgr.size == 0:
        return None
    if pixels_bgr.shape[0] < 200:
        return None
    return np.median(pixels_bgr, axis=0).astype(np.uint8)


def _pixels_from_crop(bgr_crop: np.ndarray, mask: np.ndarray | None) -> np.ndarray | None:
    """
    Extract valid pixels from crop, optionally using a mask.
    Mask must be same HxW as crop; non-zero = keep.
    Also removes extreme dark/bright pixels that distort color naming.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return None

    h, w = bgr_crop.shape[:2]
    if h < 8 or w < 8:
        return None

    # Center crop to reduce edge/background leakage (still useful even with mask)
    py = int(0.15 * h)
    px = int(0.15 * w)
    cx = bgr_crop[py:h - py, px:w - px]
    if cx.size == 0:
        cx = bgr_crop

    if mask is not None:
        # Center-crop the mask the same way
        cm = mask[py:h - py, px:w - px]
        if cm.size == 0:
            cm = mask
        keep = (cm > 0)
    else:
        keep = np.ones((cx.shape[0], cx.shape[1]), dtype=bool)

    # Convert to HSV to filter extremes (speculars / deep shadows)
    hsv = cv2.cvtColor(cx, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # Keep pixels that are not too dark and not blown-out highlights
    good = keep & (V > 35) & (V < 245)

    # If we have enough saturated pixels, prefer them (helps distinguish colored objects)
    sat = good & (S > 35)
    if int(np.sum(sat)) > 250:
        good = sat

    if int(np.sum(good)) < 200:
        return None

    pixels = cx[good]
    return pixels


# -----------------------------
# LAB name mapping (stable)
# -----------------------------
# LAB references (OpenCV LAB: L 0-255, a 0-255, b 0-255)
# You can tune these later with your belt lighting, but these work well as a baseline.
_NAMED_LAB = {
    "red":    np.array([136, 208, 195], dtype=np.float32),
    "orange": np.array([170, 171, 200], dtype=np.float32),
    "yellow": np.array([224,  42, 211], dtype=np.float32),
    "green":  np.array([140,  80, 170], dtype=np.float32),
    "cyan":   np.array([190,  80, 120], dtype=np.float32),
    "blue":   np.array([ 82, 207,  20], dtype=np.float32),
    "purple": np.array([ 80, 180,  90], dtype=np.float32),
    "magenta":np.array([120, 185, 120], dtype=np.float32),

    "brown":  np.array([ 95, 145, 155], dtype=np.float32),
    "white":  np.array([245, 128, 128], dtype=np.float32),
    "gray":   np.array([170, 128, 128], dtype=np.float32),
    "black":  np.array([ 20, 128, 128], dtype=np.float32),
}


def _bgr_to_lab(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    return lab


def _name_from_lab(lab: np.ndarray) -> tuple[str, float]:
    """
    Returns (name, distance). Smaller distance = more confident.
    Uses Euclidean distance in LAB (ΔE-ish). Good enough for robust naming.
    """
    L, A, B = lab

    # quick grayscale rules (very reliable)
    if L < 45:
        return "black", 0.0
    if abs(A - 128) < 8 and abs(B - 128) < 8:
        if L > 230:
            return "white", 0.0
        return "gray", 0.0

    best_name = "unknown"
    best_d = 1e9
    for name, ref in _NAMED_LAB.items():
        # skip grayscale refs in the general match (handled above)
        if name in ("white", "gray", "black"):
            continue
        d = float(np.linalg.norm(lab - ref))
        if d < best_d:
            best_d = d
            best_name = name

    return best_name, best_d


# -----------------------------
# Public API (drop-in)
# -----------------------------
def colour_name_from_bgr(bgr_crop: np.ndarray, mask: np.ndarray | None = None, *, do_white_balance: bool = True) -> str:
    """
    Determines the dominant colour name from a BGR crop, optionally using a mask.
    Returns:
        A color name string.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return "unknown"

    if do_white_balance:
        bgr_crop = white_balance_gray_world(bgr_crop)

    pixels = _pixels_from_crop(bgr_crop, mask)
    if pixels is None:
        return "unknown"

    dom_bgr = _dominant_bgr_from_pixels(pixels)
    if dom_bgr is None:
        return "unknown"

    lab = _bgr_to_lab(dom_bgr)
    name, dist = _name_from_lab(lab)

    # Confidence gating: if too far from all references, don't guess.
    # Tune threshold for your lighting; 35-55 is a decent range.
    if name != "unknown" and dist > 50.0:
        return "unknown"

    return name


def compute_light_level(frame_bgr: np.ndarray):
    """
    Measures brightness/darkness:
    Returns:
      brightness_pct (0..100), v_mean (0..255), luma_mean (0..255)

    Kept compatible with your existing pipeline.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    luma_mean = float(np.mean(gray))

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    v_mean = float(np.mean(hsv[:, :, 2]))

    brightness_pct = (luma_mean / 255.0) * 100.0
    return brightness_pct, v_mean, luma_mean
