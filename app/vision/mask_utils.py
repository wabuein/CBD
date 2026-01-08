import cv2
import numpy as np

class BackgroundMasker:
    def __init__(self, blur=5, diff_thresh=25, min_area=800):
        self.ref = None
        self.blur = blur
        self.diff_thresh = diff_thresh
        self.min_area = min_area

    def set_reference(self, frame_bgr: np.ndarray):
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.blur > 0:
            g = cv2.GaussianBlur(g, (self.blur, self.blur), 0)
        self.ref = g

    def mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.ref is None:
            raise RuntimeError("Background reference not set. Call set_reference() on empty belt frame.")

        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.blur > 0:
            g = cv2.GaussianBlur(g, (self.blur, self.blur), 0)

        diff = cv2.absdiff(g, self.ref)
        _, fg = cv2.threshold(diff, self.diff_thresh, 255, cv2.THRESH_BINARY)

        fg = cv2.medianBlur(fg, 5)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
        return fg

    def largest_component_bbox(self, fg_mask: np.ndarray):
        cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < self.min_area:
            return None
        x, y, w, h = cv2.boundingRect(c)
        return x, y, x + w, y + h
