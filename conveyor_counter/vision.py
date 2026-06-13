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


_MORPH_KERNELS: dict[int, np.ndarray] = {}
WATERSHED_KERNEL = np.ones((3, 3), dtype=np.uint8)


def get_morph_kernel(size: int) -> np.ndarray:
    """Get or create cached 2D numpy kernel for morphology operations."""
    if size not in _MORPH_KERNELS:
        _MORPH_KERNELS[size] = np.ones((size, size), dtype=np.uint8)
    return _MORPH_KERNELS[size]


def postprocess_mask(mask: np.ndarray, kernel_size: int = 5, iters: int = 2) -> np.ndarray:
    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1
    kernel = get_morph_kernel(k)

    x = mask
    # Connect fragmented pieces of the same object first
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel, iterations=max(1, int(iters)))
    # Remove small scattered background noise
    x = cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel, iterations=max(1, int(iters)))
    return x


def classify_color(roi_bgr: np.ndarray, roi_mask: np.ndarray) -> str:
    """Classify dominant color in BGR ROI given a mask."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mean_val = cv2.mean(hsv, mask=roi_mask)
    h, s, v = mean_val[0], mean_val[1], mean_val[2]
    
    if s < 40 or v < 40:
        return "unknown"
        
    if (h < 12) or (h > 165):
        return "Red"
    elif 15 < h < 35:
        return "Yellow"
    elif 35 <= h < 85:
        return "Green"
    elif 90 <= h <= 130:
        return "Blue"
    return "unknown"


def detect_products(
    mask_255: np.ndarray,
    min_area: int,
    max_area: int,
    frame_bgr: Optional[np.ndarray] = None,
) -> List[Detection]:
    """Detect blobs from a binary mask using contours and Watershed."""
    min_a = float(max(0, int(min_area)))
    max_a = float(max(min_a + 1.0, int(max_area)))

    dets: List[Detection] = []
    
    # Find raw contours from the global mask
    contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        area = float(cv2.contourArea(c))
        # We can be a bit lenient with max_area because multiple objects might be joined
        if area < min_a:
            continue
            
        x, y, w, h = cv2.boundingRect(c)
        if w <= 1 or h <= 1:
            continue
            
        # If no frame_bgr, just return standard detection
        if frame_bgr is None:
            if area > max_a:
                continue
            cx = x + w / 2.0
            cy = y + h / 2.0
            dets.append(Detection(centroid=(cx, cy), bbox=(int(x), int(y), int(w), int(h)), area=area))
            continue
            
        # LOCAL WATERSHED
        # Extract ROI
        roi_mask = mask_255[y:y+h, x:x+w]
        roi_bgr = frame_bgr[y:y+h, x:x+w].copy()
        
        dist_transform = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
        dt_max = dist_transform.max()
        if dt_max <= 0:
            continue
        
        # Dynamically size the local peak search window based on block size (dt_max)
        k_size = int(dt_max * 0.55)
        if k_size % 2 == 0:
            k_size += 1
        k_size = max(5, min(k_size, 31))
        
        local_max_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        dilated = cv2.dilate(dist_transform, local_max_kernel)
        
        # Find peaks: local maxima that are sufficiently far from the edges and background
        peaks = (dist_transform >= dilated) & (dist_transform > 0.25 * dt_max) & (dist_transform > 3)
        
        sure_fg = np.zeros_like(roi_mask, dtype=np.uint8)
        sure_fg[peaks] = 255
        
        # Fallback if no peaks were found
        if np.max(sure_fg) == 0:
            _, sure_fg = cv2.threshold(dist_transform, 0.5 * dt_max, 255, 0)
            sure_fg = np.uint8(sure_fg)
            num_labels, markers = cv2.connectedComponents(sure_fg)
        else:
            num_labels, markers = cv2.connectedComponents(sure_fg)
            if num_labels > 2:
                # Extract centroids of components
                centroids = []
                for idx in range(1, num_labels):
                    pts = np.argwhere(markers == idx)
                    if len(pts) > 0:
                        cy, cx = np.mean(pts, axis=0)
                        centroids.append((cx, cy, idx))
                
                # Union Find grouping
                parent = {idx: idx for idx in range(1, num_labels)}
                def find_root(idx):
                    if parent[idx] == idx:
                        return idx
                    parent[idx] = find_root(parent[idx])
                    return parent[idx]
                
                def union_roots(idx1, idx2):
                    r1 = find_root(idx1)
                    r2 = find_root(idx2)
                    if r1 != r2:
                        parent[r1] = r2
                
                # Merge peaks that are closer than 1.4 * dt_max (since they are within the same single cube)
                merge_dist = 1.4 * dt_max
                for i_a in range(len(centroids)):
                    for i_b in range(i_a + 1, len(centroids)):
                        c1 = centroids[i_a]
                        c2 = centroids[i_b]
                        dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
                        if dist < merge_dist:
                            union_roots(c1[2], c2[2])
                
                # Sequential mapping of unique roots to new labels
                unique_roots = set(find_root(idx) for idx in range(1, num_labels))
                root_to_label = {root: new_idx for new_idx, root in enumerate(unique_roots, start=1)}
                
                # Rebuild markers with merged labels
                new_markers = np.zeros_like(markers)
                for idx in range(1, num_labels):
                    root = find_root(idx)
                    new_markers[markers == idx] = root_to_label[root]
                
                markers = new_markers
                num_labels = len(unique_roots) + 1
        
        # If there is only 1 component (plus background), it's a single isolated object.
        # Avoid running watershed to prevent unnecessary over-segmentation/splitting.
        if num_labels <= 2:
            cx = x + w / 2.0
            cy = y + h / 2.0
            color_label = classify_color(roi_bgr, roi_mask)
            # Make sure to enforce min_area and max_area constraints
            if min_a <= area <= max_a:
                dets.append(Detection(
                    centroid=(cx, cy),
                    bbox=(int(x), int(y), int(w), int(h)),
                    area=area,
                    color_label=color_label
                ))
            continue
            
        sure_bg = cv2.dilate(roi_mask, WATERSHED_KERNEL, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        markers = markers + 1
        markers[unknown == 255] = 0
        
        markers = cv2.watershed(roi_bgr, markers)
        
        # Collect sub-components
        for label in range(2, markers.max() + 1):
            obj_mask = np.zeros_like(roi_mask, dtype=np.uint8)
            obj_mask[markers == label] = 255
            
            sub_contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not sub_contours:
                continue
            sc = max(sub_contours, key=cv2.contourArea)
            sub_area = float(cv2.contourArea(sc))
            
            # Check size against global min/max
            if sub_area < min_a or sub_area > max_a:
                continue
                
            sx, sy, sw, sh = cv2.boundingRect(sc)
            if sw <= 1 or sh <= 1:
                continue
                
            cx = x + sx + sw / 2.0
            cy = y + sy + sh / 2.0
            
            # Color classification
            sub_roi_bgr = roi_bgr[sy:sy+sh, sx:sx+sw]
            sub_roi_mask = obj_mask[sy:sy+sh, sx:sx+sw]
            color_label = classify_color(sub_roi_bgr, sub_roi_mask)
            
            dets.append(Detection(
                centroid=(cx, cy), 
                bbox=(int(x + sx), int(y + sy), int(sw), int(sh)), 
                area=sub_area,
                color_label=color_label
            ))

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
