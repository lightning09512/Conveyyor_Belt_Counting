"""Computer-vision helpers for conveyor product detection.

Pipeline:
  1. ForegroundSegmenter — background subtraction or thresholding
  2. postprocess_mask   — morphological cleanup
  3. detect_products    — hybrid detection: color segmentation + bg mask refinement
"""
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
    color_label: str = "unknown"


# ── ForegroundSegmenter ──────────────────────────────────────────────

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
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=36, detectShadows=True
        )

    def segment(
        self,
        frame_bgr: np.ndarray,
        use_otsu: bool,
        threshold_value: int,
        invert: bool = False,
    ) -> np.ndarray:
        """Return binary mask (uint8 0/255) where foreground is 255."""
        if self.mode == "bgsub":
            blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 1.0)
            fg = self._bg.apply(blurred, learningRate=0.002)
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


# ── Morphological helpers ────────────────────────────────────────────

_MORPH_KERNELS: dict[int, np.ndarray] = {}


def _get_kernel(size: int) -> np.ndarray:
    if size not in _MORPH_KERNELS:
        _MORPH_KERNELS[size] = np.ones((size, size), dtype=np.uint8)
    return _MORPH_KERNELS[size]


def postprocess_mask(mask: np.ndarray, kernel_size: int = 5, iters: int = 2) -> np.ndarray:
    """Clean mask: remove noise, fill small holes."""
    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    kernel = _get_kernel(k)
    small_k = _get_kernel(3)
    it = max(1, int(iters))

    x = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_k, iterations=it)
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel, iterations=it)
    return x


# ── Color segmentation (the key improvement) ─────────────────────────

# HSV ranges for each color class.
# Each entry: list of (lower, upper) tuples in HSV space.
_COLOR_RANGES: dict[str, list[tuple[tuple[int, ...], tuple[int, ...]]]] = {
    "Red": [
        ((0, 70, 50), (10, 255, 255)),
        ((160, 70, 50), (180, 255, 255)),
    ],
    "Yellow": [
        ((15, 70, 50), (35, 255, 255)),
    ],
    "Green": [
        ((35, 60, 40), (85, 255, 255)),
    ],
    "Blue": [
        ((85, 60, 40), (130, 255, 255)),
    ],
}


def _build_color_masks(frame_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Segment the frame into per-color binary masks using HSV ranges."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    masks: dict[str, np.ndarray] = {}
    for color_name, ranges in _COLOR_RANGES.items():
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            combined = cv2.bitwise_or(
                combined, cv2.inRange(hsv, np.array(lower), np.array(upper))
            )
        masks[color_name] = combined
    return masks


def _color_segment_and_detect(
    frame_bgr: np.ndarray,
    bg_mask: np.ndarray,
    min_area: float,
    max_area: float,
) -> list[Detection]:
    """Detect objects per color channel using HSV segmentation.

    For each color:
      1. Create HSV-based mask for that color
      2. AND with the bg-subtraction mask to remove false positives
         from static colored objects (like cables, equipment)
      3. Morphological cleanup per-color channel
      4. Find contours → each contour is one object of that color
    """
    color_masks = _build_color_masks(frame_bgr)
    k3 = _get_kernel(3)

    dets: list[Detection] = []
    detected_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)

    # Lightly dilate bg mask to compensate for edge noise,
    # but keep it tight to reject static colored objects
    bg_dilated = cv2.dilate(bg_mask, k3, iterations=2)

    for color_name, cmask in color_masks.items():
        # Intersection: colored AND foreground-moving
        refined = cv2.bitwise_and(cmask, bg_dilated)

        # Morphological cleanup per color channel
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, k3, iterations=1)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, k3, iterations=2)

        contours, _ = cv2.findContours(
            refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for c in contours:
            area = float(cv2.contourArea(c))
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if w <= 2 or h <= 2:
                continue

            # Reject very irregular shapes (noise / reflections)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < 0.35:
                continue

            # Require meaningful overlap with original bg mask
            roi_bg = bg_mask[y:y+h, x:x+w]
            overlap_ratio = float(np.count_nonzero(roi_bg)) / (w * h)
            if overlap_ratio < 0.15:
                continue

            # ── Large same-color cluster → split with watershed ──
            cluster_thresh = max(min_area * 1.8, 3500.0)
            if area > cluster_thresh:
                sub = _watershed_split(
                    frame_bgr[y:y+h, x:x+w],
                    refined[y:y+h, x:x+w],
                    min_area, max_area, color_name,
                )
                if sub:
                    for d in sub:
                        d.centroid = (d.centroid[0] + x, d.centroid[1] + y)
                        d.bbox = (d.bbox[0] + x, d.bbox[1] + y,
                                  d.bbox[2], d.bbox[3])
                        dets.append(d)
                    cv2.drawContours(detected_mask, [c], -1, 255, cv2.FILLED)
                    continue
                # Watershed failed → still report as single detection
                # (fallthrough to normal detection below)

            cx = x + w / 2.0
            cy = y + h / 2.0
            dets.append(Detection(
                centroid=(cx, cy),
                bbox=(x, y, w, h),
                area=area,
                color_label=color_name,
            ))

            # Mark this region as already detected
            cv2.drawContours(detected_mask, [c], -1, 255, cv2.FILLED)

    return dets, detected_mask


def _watershed_split(
    roi_bgr: np.ndarray,
    roi_mask: np.ndarray,
    min_area: float,
    max_area: float,
    color_name: str,
) -> list[Detection] | None:
    """Split a large same-color blob into individual objects via watershed.

    Uses distance transform to find object centers, then watershed
    to determine boundaries between them.
    """
    h_roi, w_roi = roi_mask.shape[:2]
    if h_roi < 10 or w_roi < 10:
        return None

    dist_transform = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
    dt_max = float(dist_transform.max())
    if dt_max <= 3.0:
        return None

    # 1. Smooth distance transform to remove local noise/ripples
    dist_smoothed = cv2.GaussianBlur(dist_transform, (9, 9), 2.0)
    
    # 2. Find local maxima (peaks) using morphological dilation
    # Neighborhood size 25 is ideal for typical block sizes
    peak_kernel_size = 25
    peak_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (peak_kernel_size, peak_kernel_size))
    dilated = cv2.dilate(dist_smoothed, peak_kernel)
    
    threshold = max(3.0, 0.2 * dist_smoothed.max())
    peaks = (dist_smoothed == dilated) & (dist_smoothed > threshold)
    peaks = np.uint8(peaks * 255)
    
    # 3. Dilate peaks slightly to merge any multi-peaks inside the same object
    peaks_merged = cv2.dilate(peaks, np.ones((7, 7), np.uint8))
    
    # Background is where roi_mask is 0
    # Border region (unknown) is where roi_mask is 1 but peaks_merged is 0
    unknown = cv2.subtract(roi_mask, peaks_merged)
    
    # 4. Markers labeling
    _, markers = cv2.connectedComponents(peaks_merged)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # 5. Watershed segmentation
    markers = cv2.watershed(roi_bgr, markers)

    dets: list[Detection] = []
    for label in range(2, markers.max() + 1):
        obj_mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
        obj_mask[markers == label] = 255

        cnts, _ = cv2.findContours(
            obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not cnts:
            continue
        sc = max(cnts, key=cv2.contourArea)
        sub_area = float(cv2.contourArea(sc))
        if sub_area < min_area or sub_area > max_area:
            continue

        sx, sy, sw, sh = cv2.boundingRect(sc)
        if sw <= 2 or sh <= 2:
            continue

        cx = sx + sw / 2.0
        cy = sy + sh / 2.0
        dets.append(Detection(
            centroid=(cx, cy),
            bbox=(sx, sy, sw, sh),
            area=sub_area,
            color_label=color_name,
        ))

    return dets if len(dets) >= 2 else None

def classify_color(roi_bgr: np.ndarray, roi_mask: np.ndarray) -> str:
    """Classify dominant color in BGR ROI given a mask."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mean_val = cv2.mean(hsv, mask=roi_mask)
    h, s, v = mean_val[0], mean_val[1], mean_val[2]

    if s < 60 or v < 60:
        return "unknown"

    if (h < 12) or (h > 165):
        return "Red"
    elif 15 < h < 35:
        return "Yellow"
    elif 35 <= h < 85:
        return "Green"
    elif 85 <= h <= 130:
        return "Blue"
    return "unknown"


# ── Main detection entry point ────────────────────────────────────────

def detect_products(
    mask_255: np.ndarray,
    min_area: int,
    max_area: int,
    frame_bgr: Optional[np.ndarray] = None,
) -> List[Detection]:
    """Detect objects using hybrid color segmentation + bg subtraction.

    Strategy:
      1. Color segmentation: detect objects per-color via HSV ranges,
         intersected with bg mask. This handles touching objects of
         different colors perfectly (the key improvement).
      2. Fallback for unclassified blobs: any remaining foreground
         blobs not covered by color detection are processed normally.
    """
    min_a = float(max(0, int(min_area)))
    max_a = float(max(min_a + 1.0, int(max_area)))

    # If no color frame is available, fall back to simple contour detection
    if frame_bgr is None:
        return _detect_from_mask_only(mask_255, min_a, max_a)

    # ── Stage 1: Color-based detection (primary) ──
    dets, detected_mask = _color_segment_and_detect(
        frame_bgr, mask_255, min_a, max_a
    )

    # ── Stage 2: Catch any remaining foreground blobs ──
    # Subtract already-detected regions from bg mask
    remaining = cv2.bitwise_and(mask_255, cv2.bitwise_not(detected_mask))
    # Clean up
    remaining = cv2.morphologyEx(remaining, cv2.MORPH_OPEN, _get_kernel(3), iterations=2)

    contours, _ = cv2.findContours(
        remaining, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Fallback detections use a higher min area to avoid noise
    fallback_min = max(min_a, 1500.0)
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < fallback_min or area > max_a:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w <= 2 or h <= 2:
            continue

        # Reject irregular shapes
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < 0.5:
            continue

        roi_bgr = frame_bgr[y:y+h, x:x+w]
        roi_mask = mask_255[y:y+h, x:x+w]
        color = classify_color(roi_bgr, roi_mask)
        
        if color == "unknown":
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0
        dets.append(Detection(
            centroid=(cx, cy),
            bbox=(x, y, w, h),
            area=area,
            color_label=color,
        ))

    dets.sort(key=lambda d: d.centroid[0])
    return dets


def _detect_from_mask_only(
    mask_255: np.ndarray, min_a: float, max_a: float
) -> List[Detection]:
    """Simple contour detection without color info."""
    dets: List[Detection] = []
    contours, _ = cv2.findContours(
        mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_a or area > max_a:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w <= 1 or h <= 1:
            continue
        cx = x + w / 2.0
        cy = y + h / 2.0
        dets.append(Detection(
            centroid=(cx, cy),
            bbox=(x, y, w, h),
            area=area,
        ))
    dets.sort(key=lambda d: d.centroid[0])
    return dets


# ── ROI cropping ──────────────────────────────────────────────────────

def crop_roi(
    frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]]
) -> tuple[np.ndarray, Tuple[int, int]]:
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
