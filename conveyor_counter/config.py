from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int


@dataclass
class Line:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class AppConfig:
    # Source
    source_type: str = "video"  # 'video' | 'webcam' | 'images'
    video_path: str = ""
    webcam_index: int = 0
    images_dir: str = ""

    # Region of interest
    roi: Optional[ROI] = None

    # Counting line (in ROI coordinates if ROI is set)
    line: Optional[Line] = None

    # Counting mode
    # - 'line': count each object once when crossing a line (best for video/webcam)
    # - 'blob': count number of detected blobs in the current frame/image (best for still images)
    counting_mode: str = "line"  # 'line' | 'blob'

    # Segmentation mode
    seg_mode: str = "bgsub"  # 'bgsub' | 'threshold'

    # Parameters
    min_area: int = 600
    max_area: int = 60000
    morph_kernel: int = 11
    morph_iters: int = 3

    # Threshold mode
    threshold_value: int = 80
    use_otsu: bool = True
    threshold_invert: bool = False

    # Background subtractor params
    bg_history: int = 300
    bg_var_threshold: int = 36
    bg_detect_shadows: bool = True

    # Tracking
    max_match_distance: float = 60.0
    max_missing_frames: int = 15

    # Output / UI
    show_mask: bool = True
    save_overlay: bool = False
    out_dir: str = "outputs"
    playback_speed: float = 1.0


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    roi = data.get("roi")
    line = data.get("line")

    cfg = AppConfig(**{k: v for k, v in data.items() if k not in {"roi", "line"}})
    if roi:
        cfg.roi = ROI(**roi)
    if line:
        cfg.line = Line(**line)
    return cfg


def save_config(path: str | Path, cfg: AppConfig) -> None:
    p = Path(path)
    payload = asdict(cfg)
    # dataclasses in optional fields are already dicts by asdict
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
