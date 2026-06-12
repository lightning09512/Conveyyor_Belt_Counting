from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: float


class ForegroundSegmenter:
    def __init__(
        self,
        mode: str = "bgsub",
        bg_history: int = 300,
        bg_var_threshold: int = 36,
        bg_detect_shadows: bool = True,
    ):
        self.mode = mode
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=int(bg_history),
            varThreshold=float(bg_var_threshold),
            detectShadows=bool(bg_detect_shadows),
        )

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def reset(self) -> None:
        # Re-create subtractor to reset its model
        self._bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=36, detectShadows=True)

    def segment(
        self,
        frame_bgr: np.ndarray,
        use_otsu: bool,
        threshold_value: int,
        invert: bool = False,
    ) -> np.ndarray:
        """Return binary mask (uint8 0/255) where foreground is 255."""
        if self.mode == "bgsub":
            fg = self._bg.apply(frame_bgr)
            # Remove shadows (MOG2 shadow value is 127)
            _, mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
            return mask

        # threshold mode
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.2)

        thresh_type = cv2.THRESH_BINARY_INV if bool(invert) else cv2.THRESH_BINARY
        if use_otsu:
            _, mask = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)
        else:
            t = int(np.clip(threshold_value, 0, 255))
            _, mask = cv2.threshold(gray, t, 255, thresh_type)
        return mask


def postprocess_mask(mask: np.ndarray, kernel_size: int = 5, iters: int = 2) -> np.ndarray:
    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), dtype=np.uint8)

    x = mask
    # Connect fragmented pieces of the same object first
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel, iterations=max(1, int(iters)))
    # Remove small scattered background noise
    x = cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel, iterations=max(1, int(iters)))
    return x


def detect_products(
    mask_255: np.ndarray,
    min_area: int,
    max_area: int,
) -> List[Detection]:
    """Detect blobs from a binary mask using contours."""
    min_a = float(max(0, int(min_area)))
    max_a = float(max(min_a + 1.0, int(max_area)))

    contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets: List[Detection] = []

    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_a or area > max_a:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w <= 1 or h <= 1:
            continue
        cx = x + w / 2.0
        cy = y + h / 2.0
        dets.append(Detection(centroid=(cx, cy), bbox=(int(x), int(y), int(w), int(h)), area=area))

    # Sort for more stable behavior
    dets.sort(key=lambda d: d.centroid[0])
    return dets


def crop_roi(frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> tuple[np.ndarray, Tuple[int, int]]:
    """Crop ROI; returns (cropped, offset_xy)."""
    if roi is None:
        return frame, (0, 0)
    x, y, w, h = roi
    h0, w0 = frame.shape[:2]
    x = int(np.clip(x, 0, w0 - 1))
    y = int(np.clip(y, 0, h0 - 1))
    w = int(np.clip(w, 1, w0 - x))
    h = int(np.clip(h, 1, h0 - y))
    return frame[y : y + h, x : x + w], (x, y)
