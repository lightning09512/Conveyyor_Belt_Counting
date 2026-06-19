"""YOLOv8 object detection wrapper for conveyor product detection.

Provides the same Detection interface as vision.py, allowing seamless
switching between traditional CV and YOLO-based detection in the app.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .vision import Detection, crop_roi

logger = logging.getLogger(__name__)

# Lazy import flag — ultralytics is only imported when actually needed
_ULTRALYTICS_AVAILABLE: bool | None = None


def _check_ultralytics() -> bool:
    """Check if ultralytics is installed. Cached after first call."""
    global _ULTRALYTICS_AVAILABLE
    if _ULTRALYTICS_AVAILABLE is None:
        try:
            import ultralytics  # noqa: F401
            _ULTRALYTICS_AVAILABLE = True
        except ImportError:
            _ULTRALYTICS_AVAILABLE = False
    return _ULTRALYTICS_AVAILABLE


class YOLODetector:
    """Wrapper around YOLOv8 for object detection on conveyor belt frames.

    Usage:
        detector = YOLODetector("path/to/model.pt")
        detections = detector.detect(frame_bgr, confidence=0.5)

    The detector returns Detection objects with the same format as
    vision.detect_products(), so the tracker and counter work unchanged.
    """

    # Map common YOLO class names to our color labels.
    # If a custom model is trained with color-based class names,
    # those are used directly. Otherwise we fall back to the class name.
    _COLOR_ALIASES: dict[str, str] = {
        "red": "Red",
        "yellow": "Yellow",
        "green": "Green",
        "blue": "Blue",
        "Red": "Red",
        "Yellow": "Yellow",
        "Green": "Green",
        "Blue": "Blue",
        # Common YOLO pretrained classes
        "box": "unknown",
        "bottle": "unknown",
        "cup": "unknown",
    }

    def __init__(self, model_path: str = ""):
        self._model = None
        self._model_path = model_path
        self._loaded = False
        self._class_names: dict[int, str] = {}

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def class_names(self) -> dict[int, str]:
        """Return {class_id: class_name} mapping from loaded model."""
        return self._class_names

    def load(self, model_path: str = "") -> bool:
        """Load a YOLO model from the given path.

        Args:
            model_path: Path to .pt model file. If empty, uses self._model_path.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if not _check_ultralytics():
            logger.error(
                "ultralytics is not installed. "
                "Run: pip install ultralytics"
            )
            return False

        path = model_path or self._model_path
        if not path:
            logger.error("No model path specified.")
            return False

        p = Path(path)
        if not p.exists():
            logger.error(f"Model file not found: {p}")
            return False

        try:
            from ultralytics import YOLO
            self._model = YOLO(str(p))
            self._model_path = str(p)
            self._loaded = True

            # Extract class names from the model
            if hasattr(self._model, "names"):
                self._class_names = dict(self._model.names)
            else:
                self._class_names = {}

            logger.info(
                f"YOLO model loaded: {p.name} "
                f"({len(self._class_names)} classes: "
                f"{list(self._class_names.values())[:10]})"
            )
            return True

        except Exception as exc:
            logger.error(f"Failed to load YOLO model: {exc}")
            self._model = None
            self._loaded = False
            return False

    def unload(self) -> None:
        """Unload the current model to free memory."""
        self._model = None
        self._loaded = False
        self._class_names = {}

    def detect(
        self,
        frame_bgr: np.ndarray,
        confidence: float = 0.5,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Detection]:
        """Run YOLO inference on a frame and return Detection objects.

        Args:
            frame_bgr: BGR image (numpy array).
            confidence: Minimum confidence threshold (0.0 - 1.0).
            roi: Optional (x, y, w, h) region of interest. If provided,
                 only detections whose centers fall within the ROI are returned,
                 and coordinates are relative to the ROI (to match traditional CV).

        Returns:
            List of Detection objects, sorted by x-coordinate.
        """
        if not self.is_loaded:
            return []

        if roi is not None:
            rx, ry, rw, rh = roi
            inference_frame = frame_bgr[ry:ry+rh, rx:rx+rw]
        else:
            inference_frame = frame_bgr

        try:
            # Run inference on the cropped frame to speed up processing
            results = self._model(
                inference_frame,
                conf=float(confidence),
                verbose=False,
            )
        except Exception as exc:
            logger.error(f"YOLO inference error: {exc}")
            return []

        dets: List[Detection] = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                # Bounding box (xyxy format) in the inference frame
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf_val = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                w = x2 - x1
                h = y2 - y1
                if w <= 2 or h <= 2:
                    continue

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                area = float(w * h)

                # Map class to color label
                class_name = self._class_names.get(cls_id, f"class_{cls_id}")
                color_label = self._COLOR_ALIASES.get(class_name, class_name)

                # Since we cropped the inference frame, coordinates are already ROI-relative
                dets.append(Detection(
                    centroid=(cx, cy),
                    bbox=(x1, y1, w, h),
                    area=area,
                    color_label=color_label,
                ))

        dets.sort(key=lambda d: d.centroid[0])
        return dets

    def detect_with_mask(
        self,
        frame_bgr: np.ndarray,
        confidence: float = 0.5,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[List[Detection], np.ndarray]:
        """Run detection and also return a synthetic binary mask.

        The mask is generated from YOLO bounding boxes for visualization
        compatibility with the existing UI (mask view).

        Returns:
            (detections, mask_255) where mask is uint8 0/255.
        """
        import cv2

        dets = self.detect(frame_bgr, confidence, roi)

        # Create a synthetic mask from detections
        if roi is not None:
            rx, ry, rw, rh = roi
            mask = np.zeros((rh, rw), dtype=np.uint8)
        else:
            h, w = frame_bgr.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

        for d in dets:
            x, y, bw, bh = d.bbox
            cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)

        return dets, mask
