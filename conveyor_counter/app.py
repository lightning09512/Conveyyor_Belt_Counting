from __future__ import annotations

import os
import time
import tkinter as tk
import customtkinter as ctk
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from .config import AppConfig, Line, ROI, load_config, save_config
from .geometry import Line2D, Point
from .tracker import CentroidTracker, Track, LineCrossingCounter
from .vision import ForegroundSegmenter, crop_roi, detect_products, postprocess_mask


def _bgr_to_tk_image(img_bgr: np.ndarray, max_w: int = 760) -> tuple[ImageTk.PhotoImage, float]:
    """Convert BGR frame to a Tkinter-compatible image (RGB)."""
    if img_bgr is None:
        raise ValueError("img_bgr is None")
    h, w = img_bgr.shape[:2]
    scale = 1.0
    if w > max_w:
        scale = max_w / float(w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil), scale


def _gray_to_tk_image(img_gray: np.ndarray, max_w: int = 760) -> ImageTk.PhotoImage:
    if img_gray is None:
        raise ValueError("img_gray is None")
    if img_gray.ndim == 2:
        x = img_gray
    else:
        x = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    h, w = x.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        x = cv2.resize(x, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(x)
    return ImageTk.PhotoImage(pil)


class ConveyorCounterApp:
    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("Conveyor Product Counter (CV) - Live Dashboard")
        self.root.geometry("1280x760")

        self.cfg = AppConfig()

        self.cap: cv2.VideoCapture | None = None
        self.image_paths: list[str] = []
        self.image_index: int = 0
        self.last_raw_frame: np.ndarray | None = None

        self.running = False
        self.last_frame_time = 0.0

        self.segmenter = ForegroundSegmenter(
            mode=self.cfg.seg_mode,
            bg_history=self.cfg.bg_history,
            bg_var_threshold=self.cfg.bg_var_threshold,
            bg_detect_shadows=self.cfg.bg_detect_shadows,
        )
        self.tracker = CentroidTracker(max_distance=self.cfg.max_match_distance, max_missing=self.cfg.max_missing_frames)
        self.counter = LineCrossingCounter()
        self.prev_centroids: dict[int, Point] = {}

        self.ui_state = "idle" # "idle", "roi", "line"
        self.view_scale = 1.0
        self.roi_start_pt = None
        self.roi_end_pt = None
        self.line_pts = []
        self.mouse_pos = None

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self) -> None:
        # Left Sidebar (Scrollable)
        self.sidebar = ctk.CTkScrollableFrame(self.root, width=320, label_text="CONTROLS & PARAMETERS")
        self.sidebar.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(10, 5), pady=10)

        # Right Main Panel
        frm_main = ctk.CTkFrame(self.root, fg_color="transparent")
        frm_main.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)

        # Top Metrics Row
        frm_metrics = ctk.CTkFrame(frm_main, fg_color="transparent")
        frm_metrics.pack(fill=tk.X, padx=0, pady=(0, 10))

        # Metrics cards
        card_total = ctk.CTkFrame(frm_metrics, fg_color=("#f0f0f5", "#1e1e24"), corner_radius=8, border_width=1, border_color=("#d0d0d5", "#2e2e38"))
        card_total.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.lbl_total_title = ctk.CTkLabel(card_total, text="TOTAL COUNTED", font=("Helvetica", 10, "bold"), text_color="gray")
        self.lbl_total_title.pack(pady=(8, 2))
        self.lbl_total_val = ctk.CTkLabel(card_total, text="0", font=("Helvetica", 22, "bold"))
        self.lbl_total_val.pack(pady=(0, 8))

        card_red = ctk.CTkFrame(frm_metrics, fg_color=("#fef0f0", "#2d1f1f"), corner_radius=8, border_width=1, border_color=("#fbdcdd", "#4a2425"))
        card_red.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ctk.CTkLabel(card_red, text="RED", font=("Helvetica", 10, "bold"), text_color=("#d9534f", "#f35a58")).pack(pady=(8, 2))
        self.lbl_red_val = ctk.CTkLabel(card_red, text="0", font=("Helvetica", 18, "bold"))
        self.lbl_red_val.pack(pady=(0, 8))

        card_yellow = ctk.CTkFrame(frm_metrics, fg_color=("#fefdf0", "#2d2a1f"), corner_radius=8, border_width=1, border_color=("#fbf5dc", "#4a4224"))
        card_yellow.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ctk.CTkLabel(card_yellow, text="YELLOW", font=("Helvetica", 10, "bold"), text_color=("#f0ad4e", "#f0ad4e")).pack(pady=(8, 2))
        self.lbl_yellow_val = ctk.CTkLabel(card_yellow, text="0", font=("Helvetica", 18, "bold"))
        self.lbl_yellow_val.pack(pady=(0, 8))

        card_green = ctk.CTkFrame(frm_metrics, fg_color=("#f0fef0", "#1f2d1f"), corner_radius=8, border_width=1, border_color=("#dcfbdc", "#244a24"))
        card_green.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ctk.CTkLabel(card_green, text="GREEN", font=("Helvetica", 10, "bold"), text_color=("#5cb85c", "#5cb85c")).pack(pady=(8, 2))
        self.lbl_green_val = ctk.CTkLabel(card_green, text="0", font=("Helvetica", 18, "bold"))
        self.lbl_green_val.pack(pady=(0, 8))

        card_blue = ctk.CTkFrame(frm_metrics, fg_color=("#f0f7fe", "#1f272d"), corner_radius=8, border_width=1, border_color=("#dcecfb", "#24374a"))
        card_blue.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ctk.CTkLabel(card_blue, text="BLUE", font=("Helvetica", 10, "bold"), text_color=("#0275d8", "#42a5f5")).pack(pady=(8, 2))
        self.lbl_blue_val = ctk.CTkLabel(card_blue, text="0", font=("Helvetica", 18, "bold"))
        self.lbl_blue_val.pack(pady=(0, 8))

        card_unknown = ctk.CTkFrame(frm_metrics, fg_color=("#f5f5f5", "#242424"), corner_radius=8, border_width=1, border_color=("#e0e0e0", "#333333"))
        card_unknown.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        ctk.CTkLabel(card_unknown, text="UNKNOWN", font=("Helvetica", 10, "bold"), text_color="gray").pack(pady=(8, 2))
        self.lbl_unknown_val = ctk.CTkLabel(card_unknown, text="0", font=("Helvetica", 18, "bold"))
        self.lbl_unknown_val.pack(pady=(0, 8))

        # Views Container
        self.frm_views_container = ctk.CTkFrame(frm_main, fg_color="black", corner_radius=8)
        self.frm_views_container.pack(fill=tk.BOTH, expand=True)

        self.lbl_view = tk.Label(self.frm_views_container, bg="black")
        self.lbl_view.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.lbl_view.bind("<Button-1>", self._on_view_click)
        self.lbl_view.bind("<B1-Motion>", self._on_view_drag)
        self.lbl_view.bind("<ButtonRelease-1>", self._on_view_release)
        self.lbl_view.bind("<Motion>", self._on_view_motion)
        self.lbl_view.bind("<Button-3>", self._on_view_cancel)

        self.lbl_mask = tk.Label(self.frm_views_container, bg="black")
        self.lbl_mask.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Footer Panel (under views)
        frm_footer = ctk.CTkFrame(frm_main, fg_color="transparent")
        frm_footer.pack(fill=tk.X, pady=(10, 0))

        self.lbl_fps = ctk.CTkLabel(frm_footer, text="FPS: --", font=("Helvetica", 12))
        self.lbl_fps.pack(side=tk.LEFT, padx=10)

        ctk.CTkLabel(frm_footer, text="Conveyor Belt Counting Dashboard v1.1", font=("Helvetica", 10, "italic"), text_color="gray").pack(side=tk.RIGHT, padx=10)

        # Build Sidebar Content
        # Section 1: Connection & Control
        frm_src_grp = ctk.CTkFrame(self.sidebar, fg_color=("#f0f0f5", "#20202a"), corner_radius=8)
        frm_src_grp.pack(fill=tk.X, padx=5, pady=5)
        ctk.CTkLabel(frm_src_grp, text="INPUT SOURCE", font=("Helvetica", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 5))

        self.var_source = tk.StringVar(value="video")
        frm_radio = ctk.CTkFrame(frm_src_grp, fg_color="transparent")
        frm_radio.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkRadioButton(frm_radio, text="Video", variable=self.var_source, value="video", command=self._on_source_changed).pack(side=tk.LEFT, expand=True)
        ctk.CTkRadioButton(frm_radio, text="Webcam", variable=self.var_source, value="webcam", command=self._on_source_changed).pack(side=tk.LEFT, expand=True)
        ctk.CTkRadioButton(frm_radio, text="Images", variable=self.var_source, value="images", command=self._on_source_changed).pack(side=tk.LEFT, expand=True)

        frm_path = ctk.CTkFrame(frm_src_grp, fg_color="transparent")
        frm_path.pack(fill=tk.X, padx=10, pady=5)
        self.entry_video = ctk.CTkEntry(frm_path, placeholder_text="Video file path or Image directory")
        self.entry_video.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ctk.CTkButton(frm_path, text="Browse", command=self._browse_source_path, width=60).pack(side=tk.RIGHT)

        frm_cam = ctk.CTkFrame(frm_src_grp, fg_color="transparent")
        frm_cam.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(frm_cam, text="Webcam Index:").pack(side=tk.LEFT)
        self.entry_cam = ctk.CTkEntry(frm_cam, width=60)
        self.entry_cam.insert(0, "0")
        self.entry_cam.pack(side=tk.RIGHT)

        frm_actions = ctk.CTkFrame(frm_src_grp, fg_color="transparent")
        frm_actions.pack(fill=tk.X, padx=10, pady=(5, 10))
        frm_actions.columnconfigure((0, 1), weight=1)
        
        ctk.CTkButton(frm_actions, text="Open Source", command=self.open_capture, fg_color="#1f538d", hover_color="#184270").grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        ctk.CTkButton(frm_actions, text="Start Video", command=self.start, fg_color="#2ecc71", hover_color="#27ae60", text_color="white").grid(row=0, column=1, padx=2, pady=2, sticky="ew")
        ctk.CTkButton(frm_actions, text="Pause Stream", command=self.pause, fg_color="#f39c12", hover_color="#d35400", text_color="white").grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        ctk.CTkButton(frm_actions, text="Reset Counter", command=self.reset_count, fg_color="#e74c3c", hover_color="#c0392b", text_color="white").grid(row=1, column=1, padx=2, pady=2, sticky="ew")

        # Section 2: Area of Interest
        frm_roi_grp = ctk.CTkFrame(self.sidebar, fg_color=("#f0f0f5", "#20202a"), corner_radius=8)
        frm_roi_grp.pack(fill=tk.X, padx=5, pady=5)
        ctk.CTkLabel(frm_roi_grp, text="DETECTION ZONES", font=("Helvetica", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 5))

        ctk.CTkButton(frm_roi_grp, text="Set ROI Window", command=self.select_roi, fg_color=("#7f8c8d", "#5c6a6b")).pack(fill=tk.X, padx=10, pady=5)
        self.lbl_roi = ctk.CTkLabel(frm_roi_grp, text="ROI: (none)", font=("Helvetica", 11))
        self.lbl_roi.pack(anchor="w", padx=10, pady=(0, 5))

        ctk.CTkButton(frm_roi_grp, text="Draw Counting Line", command=self.select_line, fg_color=("#7f8c8d", "#5c6a6b")).pack(fill=tk.X, padx=10, pady=5)
        self.lbl_line = ctk.CTkLabel(frm_roi_grp, text="Line: (none)", font=("Helvetica", 11))
        self.lbl_line.pack(anchor="w", padx=10, pady=(0, 10))

        # Section 3: Vision Parameters
        frm_params_grp = ctk.CTkFrame(self.sidebar, fg_color=("#f0f0f5", "#20202a"), corner_radius=8)
        frm_params_grp.pack(fill=tk.X, padx=5, pady=5)
        ctk.CTkLabel(frm_params_grp, text="VISION CONFIGURATION", font=("Helvetica", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 5))

        frm_drop1 = ctk.CTkFrame(frm_params_grp, fg_color="transparent")
        frm_drop1.pack(fill=tk.X, padx=10, pady=3)
        ctk.CTkLabel(frm_drop1, text="Seg Mode:").pack(side=tk.LEFT)
        self.var_seg = tk.StringVar(value=self.cfg.seg_mode)
        ctk.CTkOptionMenu(frm_drop1, variable=self.var_seg, values=["bgsub", "threshold"], command=lambda _v: self._sync_params_to_cfg(), width=130).pack(side=tk.RIGHT)

        frm_drop2 = ctk.CTkFrame(frm_params_grp, fg_color="transparent")
        frm_drop2.pack(fill=tk.X, padx=10, pady=3)
        ctk.CTkLabel(frm_drop2, text="Count Mode:").pack(side=tk.LEFT)
        self.var_count_mode = tk.StringVar(value=self.cfg.counting_mode)
        ctk.CTkOptionMenu(frm_drop2, variable=self.var_count_mode, values=["line", "blob"], command=lambda _v: self._sync_params_to_cfg(), width=130).pack(side=tk.RIGHT)

        frm_area = ctk.CTkFrame(frm_params_grp, fg_color="transparent")
        frm_area.pack(fill=tk.X, padx=10, pady=3)
        
        frm_min = ctk.CTkFrame(frm_area, fg_color="transparent")
        frm_min.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ctk.CTkLabel(frm_min, text="Min Area:", font=("Helvetica", 11)).pack(anchor="w")
        self.entry_min_area = ctk.CTkEntry(frm_min)
        self.entry_min_area.insert(0, str(self.cfg.min_area))
        self.entry_min_area.pack(fill=tk.X)

        frm_max = ctk.CTkFrame(frm_area, fg_color="transparent")
        frm_max.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        ctk.CTkLabel(frm_max, text="Max Area:", font=("Helvetica", 11)).pack(anchor="w")
        self.entry_max_area = ctk.CTkEntry(frm_max)
        self.entry_max_area.insert(0, str(self.cfg.max_area))
        self.entry_max_area.pack(fill=tk.X)

        frm_morph = ctk.CTkFrame(frm_params_grp, fg_color="transparent")
        frm_morph.pack(fill=tk.X, padx=10, pady=3)

        frm_kern = ctk.CTkFrame(frm_morph, fg_color="transparent")
        frm_kern.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ctk.CTkLabel(frm_kern, text="Kernel Size:", font=("Helvetica", 11)).pack(anchor="w")
        self.entry_kernel = ctk.CTkEntry(frm_kern)
        self.entry_kernel.insert(0, str(self.cfg.morph_kernel))
        self.entry_kernel.pack(fill=tk.X)

        frm_iter = ctk.CTkFrame(frm_morph, fg_color="transparent")
        frm_iter.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        ctk.CTkLabel(frm_iter, text="Morph Iters:", font=("Helvetica", 11)).pack(anchor="w")
        self.entry_iters = ctk.CTkEntry(frm_iter)
        self.entry_iters.insert(0, str(self.cfg.morph_iters))
        self.entry_iters.pack(fill=tk.X)

        frm_thresh = ctk.CTkFrame(frm_params_grp, fg_color="transparent")
        frm_thresh.pack(fill=tk.X, padx=10, pady=3)
        ctk.CTkLabel(frm_thresh, text="Threshold (Static):").pack(side=tk.LEFT)
        self.entry_thresh = ctk.CTkEntry(frm_thresh, width=80)
        self.entry_thresh.insert(0, str(self.cfg.threshold_value))
        self.entry_thresh.pack(side=tk.RIGHT)

        frm_chk = ctk.CTkFrame(frm_params_grp, fg_color="transparent")
        frm_chk.pack(fill=tk.X, padx=10, pady=(5, 5))
        frm_chk.columnconfigure((0, 1), weight=1)

        self.var_otsu = ctk.StringVar(value="1" if self.cfg.use_otsu else "0")
        ctk.CTkCheckBox(frm_chk, text="Use Otsu", variable=self.var_otsu, onvalue="1", offvalue="0", command=self._sync_params_to_cfg, font=("Helvetica", 11)).grid(row=0, column=0, sticky="w", pady=2)

        self.var_invert = ctk.StringVar(value="1" if self.cfg.threshold_invert else "0")
        ctk.CTkCheckBox(frm_chk, text="Invert Thresh", variable=self.var_invert, onvalue="1", offvalue="0", command=self._sync_params_to_cfg, font=("Helvetica", 11)).grid(row=0, column=1, sticky="w", pady=2)

        self.var_show_mask = ctk.StringVar(value="1" if self.cfg.show_mask else "0")
        ctk.CTkCheckBox(frm_chk, text="Show Mask View", variable=self.var_show_mask, onvalue="1", offvalue="0", command=self._sync_params_to_cfg, font=("Helvetica", 11)).grid(row=1, column=0, sticky="w", pady=2)

        self.var_save_overlay = ctk.StringVar(value="1" if self.cfg.save_overlay else "0")
        ctk.CTkCheckBox(frm_chk, text="Save Overlay", variable=self.var_save_overlay, onvalue="1", offvalue="0", command=self._sync_params_to_cfg, font=("Helvetica", 11)).grid(row=1, column=1, sticky="w", pady=2)

        ctk.CTkButton(frm_params_grp, text="Apply Config Params", command=self._sync_params_to_cfg, fg_color=("#34495e", "#2c3e50")).pack(fill=tk.X, padx=10, pady=(5, 10))

        # Section 4: Configuration Presets
        frm_cfg_grp = ctk.CTkFrame(self.sidebar, fg_color=("#f0f0f5", "#20202a"), corner_radius=8)
        frm_cfg_grp.pack(fill=tk.X, padx=5, pady=5)
        ctk.CTkLabel(frm_cfg_grp, text="PRESETS & STATUS", font=("Helvetica", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 5))

        frm_presets = ctk.CTkFrame(frm_cfg_grp, fg_color="transparent")
        frm_presets.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkButton(frm_presets, text="Load Config", command=self.load_cfg_dialog, width=110, fg_color=("#7f8c8d", "#555f60")).pack(side=tk.LEFT, expand=True, padx=(0, 2))
        ctk.CTkButton(frm_presets, text="Save Config", command=self.save_cfg_dialog, width=110, fg_color=("#7f8c8d", "#555f60")).pack(side=tk.RIGHT, expand=True, padx=(2, 0))

        self.lbl_status = ctk.CTkLabel(frm_cfg_grp, text="System Ready", font=("Helvetica", 11, "bold"), text_color="gray")
        self.lbl_status.pack(fill=tk.X, padx=10, pady=(5, 10))

        # Setup source change configurations
        self._on_source_changed()

        # Bind Return/Enter key on all entry fields to trigger synchronization
        for entry in [self.entry_video, self.entry_cam, self.entry_min_area, self.entry_max_area,
                      self.entry_kernel, self.entry_iters, self.entry_thresh]:
            entry.bind("<Return>", lambda event: self._sync_params_to_cfg())

    def _update_metrics_ui(self, total: int, red: int, yellow: int, green: int, blue: int, unknown: int, is_blob: bool = False) -> None:
        if is_blob:
            self.lbl_total_title.configure(text="IN-FRAME")
        else:
            self.lbl_total_title.configure(text="TOTAL COUNTED")
        
        self.lbl_total_val.configure(text=str(total))
        self.lbl_red_val.configure(text=str(red))
        self.lbl_yellow_val.configure(text=str(yellow))
        self.lbl_green_val.configure(text=str(green))
        self.lbl_blue_val.configure(text=str(blue))
        self.lbl_unknown_val.configure(text=str(unknown))

    def _on_source_changed(self, preserve_count_mode: bool = False) -> None:
        src = self.var_source.get()
        is_video = src == "video"
        is_images = src == "images"
        self.entry_video.configure(state=("normal" if (is_video or is_images) else "disabled"))
        self.entry_cam.configure(state=("normal" if src == "webcam" else "disabled"))

        if not preserve_count_mode:
            if is_images:
                self.var_count_mode.set("blob")
            else:
                self.var_count_mode.set("line")
        self._sync_params_to_cfg()

    # ---------------- Config ----------------
    def _sync_params_to_cfg(self) -> None:
        try:
            self.cfg.source_type = self.var_source.get()
            path_text = self.entry_video.get().strip()
            if self.cfg.source_type == "images":
                self.cfg.images_dir = path_text
            else:
                self.cfg.video_path = path_text
            self.cfg.webcam_index = int(self.entry_cam.get().strip() or "0")

            self.cfg.seg_mode = self.var_seg.get()
            self.cfg.show_mask = self.var_show_mask.get() == "1"
            self.cfg.save_overlay = self.var_save_overlay.get() == "1"

            self.cfg.min_area = int(self.entry_min_area.get().strip() or self.cfg.min_area)
            self.cfg.max_area = int(self.entry_max_area.get().strip() or self.cfg.max_area)
            self.cfg.morph_kernel = int(self.entry_kernel.get().strip() or self.cfg.morph_kernel)
            self.cfg.morph_iters = int(self.entry_iters.get().strip() or self.cfg.morph_iters)

            self.cfg.threshold_value = int(self.entry_thresh.get().strip() or self.cfg.threshold_value)
            self.cfg.use_otsu = self.var_otsu.get() == "1"
            self.cfg.threshold_invert = self.var_invert.get() == "1"

            self.cfg.counting_mode = self.var_count_mode.get()

            # Keep metrics cards consistent with mode (when not running)
            if self.cfg.counting_mode == "blob":
                if not self.running:
                    self._update_metrics_ui(0, 0, 0, 0, 0, 0, is_blob=True)
            else:
                if not self.running:
                    c = self.counter.counts
                    self._update_metrics_ui(
                        total=self.counter.total,
                        red=c.get("Red", 0),
                        yellow=c.get("Yellow", 0),
                        green=c.get("Green", 0),
                        blue=c.get("Blue", 0),
                        unknown=c.get("unknown", 0),
                        is_blob=False
                    )

            self.segmenter.set_mode(self.cfg.seg_mode)

            self.tracker.max_distance = float(self.cfg.max_match_distance)
            self.tracker.max_missing = int(self.cfg.max_missing_frames)

            self.lbl_status.configure(text="Params applied successfully", text_color=("#2ecc71", "#2ecc71"))
        except Exception as exc:
            self.lbl_status.configure(text=f"Params Error: {exc}", text_color=("#e74c3c", "#e74c3c"))

    def save_cfg_dialog(self) -> None:
        self._sync_params_to_cfg()
        path = filedialog.asksaveasfilename(
            title="Save config",
            defaultextension=".json",
            filetypes=(("JSON", "*.json"), ("All files", "*.*")),
            initialfile="conveyor_config.json",
        )
        if not path:
            return
        save_config(path, self.cfg)
        self.lbl_status.configure(text=f"Saved config: {Path(path).name}", text_color="gray")

    def load_cfg_dialog(self) -> None:
        path = filedialog.askopenfilename(
            title="Load config",
            filetypes=(("JSON", "*.json"), ("All files", "*.*")),
        )
        if not path:
            return
        self.cfg = load_config(path)

        # Push to UI
        self.var_source.set(self.cfg.source_type)
        self.entry_video.configure(state="normal")
        self.entry_video.delete(0, tk.END)
        if self.cfg.source_type == "images":
            self.entry_video.insert(0, self.cfg.images_dir)
        else:
            self.entry_video.insert(0, self.cfg.video_path)

        self.entry_cam.configure(state="normal")
        self.entry_cam.delete(0, tk.END)
        self.entry_cam.insert(0, str(self.cfg.webcam_index))

        self.var_seg.set(self.cfg.seg_mode)
        self.var_show_mask.set("1" if self.cfg.show_mask else "0")
        self.var_save_overlay.set("1" if self.cfg.save_overlay else "0")

        self.entry_min_area.delete(0, tk.END)
        self.entry_min_area.insert(0, str(self.cfg.min_area))
        self.entry_max_area.delete(0, tk.END)
        self.entry_max_area.insert(0, str(self.cfg.max_area))
        self.entry_kernel.delete(0, tk.END)
        self.entry_kernel.insert(0, str(self.cfg.morph_kernel))
        self.entry_iters.delete(0, tk.END)
        self.entry_iters.insert(0, str(self.cfg.morph_iters))
        self.entry_thresh.delete(0, tk.END)
        self.entry_thresh.insert(0, str(self.cfg.threshold_value))
        self.var_otsu.set("1" if self.cfg.use_otsu else "0")
        self.var_invert.set("1" if self.cfg.threshold_invert else "0")

        self._on_source_changed(preserve_count_mode=True)
        self.var_count_mode.set(self.cfg.counting_mode)

        self._update_roi_label()
        self._update_line_label()
        self._sync_params_to_cfg()
        self.lbl_status.configure(text=f"Loaded config: {Path(path).name}", text_color="gray")

    # ---------------- Source handling ----------------
    def _browse_source_path(self) -> None:
        src = self.var_source.get()
        if src == "images":
            path = filedialog.askdirectory(
                title="Choose image folder",
                initialdir=str(Path(__file__).resolve().parents[1] / "assets"),
            )
        else:
            path = filedialog.askopenfilename(
                title="Choose video",
                filetypes=(("Video", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")),
                initialdir=str(Path(__file__).resolve().parents[1] / "assets"),
            )
        if path:
            self.entry_video.delete(0, tk.END)
            self.entry_video.insert(0, path)

    def open_capture(self) -> None:
        self._sync_params_to_cfg()
        self.pause()

        # Release old
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.image_paths = []
        self.image_index = 0
        self.last_raw_frame = None

        if self.cfg.source_type == "images":
            if not self.cfg.images_dir:
                messagebox.showwarning("Missing", "Please choose an image folder.")
                return
            p = Path(self.cfg.images_dir)
            if not p.exists() or not p.is_dir():
                messagebox.showerror("Error", "Image folder not found")
                return

            exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
            self.image_paths = [str(x) for x in sorted(p.rglob("*")) if x.suffix.lower() in exts]
            if not self.image_paths:
                messagebox.showerror("Error", "No images found in folder")
                return

            # Reset models
            self.segmenter = ForegroundSegmenter(
                mode=self.cfg.seg_mode,
                bg_history=self.cfg.bg_history,
                bg_var_threshold=self.cfg.bg_var_threshold,
                bg_detect_shadows=self.cfg.bg_detect_shadows,
            )
            self.tracker.reset()
            self.counter.reset()
            self.prev_centroids.clear()
            self.lbl_fps.configure(text="FPS: --")

            self.lbl_status.configure(text=f"Opened images: {len(self.image_paths)}", text_color="gray")
            self._grab_and_show_single_frame()
            return

        if self.cfg.source_type == "webcam":
            self.cap = cv2.VideoCapture(int(self.cfg.webcam_index))
        else:
            if not self.cfg.video_path:
                messagebox.showwarning("Missing", "Please choose a video file.")
                return
            self.cap = cv2.VideoCapture(self.cfg.video_path)

        if self.cap is None or not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open source")
            self.cap = None
            return

        self.segmenter = ForegroundSegmenter(
            mode=self.cfg.seg_mode,
            bg_history=self.cfg.bg_history,
            bg_var_threshold=self.cfg.bg_var_threshold,
            bg_detect_shadows=self.cfg.bg_detect_shadows,
        )
        self.tracker.reset()
        self.counter.reset()
        self.prev_centroids.clear()

        self.lbl_status.configure(text="Opened source successfully", text_color="gray")
        self._grab_and_show_single_frame()

    def _grab_and_show_single_frame(self) -> None:
        frame = self._read_current_frame_peek()
        if frame is None:
            self.lbl_status.configure(text="No frame preview available", text_color="gray")
            return

        if self.ui_state in ("roi", "line"):
            self._redraw_interactive()
            self.lbl_mask.configure(image="")
            self.lbl_mask.image = None
        else:
            vis, mask = self._process_frame(frame)
            self._update_views(vis, mask)

    def _read_current_frame_peek(self) -> np.ndarray | None:
        """Read a frame for preview without advancing the file stream too much."""
        if self.cfg.source_type == "images":
            if not self.image_paths:
                return None
            frame = cv2.imread(self.image_paths[self.image_index])
            if frame is None:
                return None
            self.last_raw_frame = frame
            return frame

        if self.cap is None:
            return None

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        self.last_raw_frame = frame

        # Seek back one frame if file
        if self.cfg.source_type == "video":
            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 1))
        return frame

    # ---------------- ROI & Line selection ----------------
    def _update_roi_label(self) -> None:
        if self.cfg.roi is None:
            self.lbl_roi.configure(text="ROI: (none)")
        else:
            r = self.cfg.roi
            self.lbl_roi.configure(text=f"ROI: x={r.x}, y={r.y}, w={r.w}, h={r.h}")

    def _update_line_label(self) -> None:
        if self.cfg.line is None:
            self.lbl_line.configure(text="Line: (none)")
        else:
            ln = self.cfg.line
            self.lbl_line.configure(text=f"Line: ({ln.x1},{ln.y1})-({ln.x2},{ln.y2})")

    def select_roi(self) -> None:
        if self.cap is None and not self.image_paths:
            messagebox.showwarning("Missing", "Open a source first.")
            return
        self.pause()
        self.ui_state = "roi"
        self.roi_start_pt = None
        self.roi_end_pt = None
        self.lbl_status.configure(text="Drag on video to select ROI. Right-click to cancel.", text_color="#f39c12")
        self._grab_and_show_single_frame()

    def select_line(self) -> None:
        if self.cap is None and not self.image_paths:
            messagebox.showwarning("Missing", "Open a source first.")
            return
        self.pause()
        self.ui_state = "line"
        self.line_pts = []
        self.mouse_pos = None
        self.lbl_status.configure(text="Click 2 points on video to draw Line. Right-click to cancel.", text_color="#f39c12")
        self._grab_and_show_single_frame()

    def _get_orig_pt(self, event) -> tuple[int, int]:
        lbl_w = self.lbl_view.winfo_width()
        lbl_h = self.lbl_view.winfo_height()
        
        img = self.lbl_view.image
        if not img:
            return int(event.x / self.view_scale), int(event.y / self.view_scale)
            
        img_w = img.width()
        img_h = img.height()
        
        offset_x = (lbl_w - img_w) // 2
        offset_y = (lbl_h - img_h) // 2
        
        img_x = event.x - offset_x
        img_y = event.y - offset_y
        
        img_x = max(0, min(img_x, img_w - 1))
        img_y = max(0, min(img_y, img_h - 1))
        
        return int(img_x / self.view_scale), int(img_y / self.view_scale)

    def _on_view_click(self, event) -> None:
        if self.ui_state == "idle":
            return
        pt = self._get_orig_pt(event)
        
        if self.ui_state == "roi":
            self.roi_start_pt = pt
            self.roi_end_pt = pt
            self._redraw_interactive()
            
        elif self.ui_state == "line":
            off_x, off_y = 0, 0
            if self.cfg.roi is not None:
                off_x, off_y = self.cfg.roi.x, self.cfg.roi.y
            roi_pt = (pt[0] - off_x, pt[1] - off_y)
            self.line_pts.append(roi_pt)
            if len(self.line_pts) == 2:
                (x1, y1), (x2, y2) = self.line_pts
                self.cfg.line = Line(x1=x1, y1=y1, x2=x2, y2=y2)
                self.ui_state = "idle"
                self._update_line_label()
                self.lbl_status.configure(text="Line updated successfully", text_color=("#2ecc71", "#2ecc71"))
                self._grab_and_show_single_frame()
            else:
                self._redraw_interactive()

    def _on_view_drag(self, event) -> None:
        if self.ui_state == "roi" and self.roi_start_pt is not None:
            self.roi_end_pt = self._get_orig_pt(event)
            self._redraw_interactive()

    def _on_view_release(self, event) -> None:
        if self.ui_state == "roi" and self.roi_start_pt is not None:
            x1, y1 = self.roi_start_pt
            x2, y2 = self.roi_end_pt
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            
            if w <= 2 or h <= 2:
                self.cfg.roi = None
                self.lbl_status.configure(text="ROI cleared", text_color="gray")
            else:
                self.cfg.roi = ROI(x=x, y=y, w=w, h=h)
                self.lbl_status.configure(text="ROI updated successfully", text_color=("#2ecc71", "#2ecc71"))
            
            self.cfg.line = None
            self.ui_state = "idle"
            self._update_roi_label()
            self._update_line_label()
            self._grab_and_show_single_frame()

    def _on_view_motion(self, event) -> None:
        if self.ui_state == "line" and len(self.line_pts) == 1:
            self.mouse_pos = self._get_orig_pt(event)
            self._redraw_interactive()

    def _on_view_cancel(self, event) -> None:
        if self.ui_state != "idle":
            self.ui_state = "idle"
            self.lbl_status.configure(text="Selection cancelled", text_color="gray")
            self._grab_and_show_single_frame()

    def _redraw_interactive(self) -> None:
        if self.last_raw_frame is None:
            return
        frame = self.last_raw_frame.copy()
        
        # Determine current label width for scaling
        w_view = self.lbl_view.winfo_width()
        max_w = w_view if w_view > 10 else 520

        if self.ui_state == "roi":
            if self.roi_start_pt is not None and self.roi_end_pt is not None:
                cv2.rectangle(frame, self.roi_start_pt, self.roi_end_pt, (255, 0, 0), 2)
            im1, scale = _bgr_to_tk_image(frame, max_w=max_w)
            self.view_scale = scale
            self.lbl_view.configure(image=im1)
            self.lbl_view.image = im1
            
        elif self.ui_state == "line":
            view = frame.copy()
            off_x, off_y = 0, 0
            if self.cfg.roi is not None:
                rx, ry, rw, rh = self.cfg.roi.x, self.cfg.roi.y, self.cfg.roi.w, self.cfg.roi.h
                off_x, off_y = rx, ry
                # Draw semi-transparent ROI
                overlay = view.copy()
                cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (128, 0, 128), -1)
                cv2.addWeighted(overlay, 0.3, view, 0.7, 0, view)
                cv2.rectangle(view, (rx, ry), (rx + rw, ry + rh), (255, 0, 255), 2)
            
            for p in self.line_pts:
                cv2.circle(view, (p[0] + off_x, p[1] + off_y), 4, (0, 255, 255), -1)
            if len(self.line_pts) == 1 and self.mouse_pos is not None:
                p0 = (self.line_pts[0][0] + off_x, self.line_pts[0][1] + off_y)
                cv2.line(view, p0, self.mouse_pos, (0, 255, 255), 2)
                
            im1, scale = _bgr_to_tk_image(view, max_w=max_w)
            self.view_scale = scale
            self.lbl_view.configure(image=im1)
            self.lbl_view.image = im1

    # ---------------- Run loop ----------------
    def start(self) -> None:
        if self.cap is None and not self.image_paths:
            messagebox.showwarning("Missing", "Open a source first.")
            return

        self._sync_params_to_cfg()
        if self.cfg.counting_mode == "line" and self.cfg.line is None:
            messagebox.showwarning("Missing", "Please select a counting line first (Count mode=line).")
            return
        self.running = True
        self.lbl_status.configure(text="Running...", text_color="#2ecc71")
        self._loop()

    def pause(self) -> None:
        self.running = False
        self.lbl_status.configure(text="Paused", text_color="#f39c12")

    def reset_count(self) -> None:
        self.counter.reset()
        for tr in self.tracker.tracks.values():
            tr.counted = False
        self.prev_centroids.clear()
        
        if self.cfg.counting_mode == "blob":
            self._update_metrics_ui(0, 0, 0, 0, 0, 0, is_blob=True)
        else:
            c = self.counter.counts
            self._update_metrics_ui(
                total=self.counter.total,
                red=c.get("Red", 0),
                yellow=c.get("Yellow", 0),
                green=c.get("Green", 0),
                blue=c.get("Blue", 0),
                unknown=c.get("unknown", 0),
                is_blob=False
            )
        self.lbl_status.configure(text="Count reset successfully", text_color="gray")

    def _loop(self) -> None:
        loop_start = time.time()
        if not self.running:
            return
        frame = None
        if self.cfg.source_type == "images":
            if not self.image_paths or self.image_index >= len(self.image_paths):
                self.running = False
                self.lbl_status.configure(text="End of images folder", text_color="gray")
                return
            frame = cv2.imread(self.image_paths[self.image_index])
            self.image_index += 1
            if frame is None:
                # skip unreadable
                self.root.after(1, self._loop)
                return
            self.last_raw_frame = frame
        else:
            if self.cap is None:
                self.running = False
                return
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.running = False
                self.lbl_status.configure(text="End of video stream", text_color="gray")
                return
            self.last_raw_frame = frame

        vis, mask = self._process_frame(frame)
        self._update_views(vis, mask)

        # FPS calculation
        if self.cfg.source_type != "images":
            now = time.time()
            if self.last_frame_time > 0:
                fps = 1.0 / max(1e-6, (now - self.last_frame_time))
                self.lbl_fps.configure(text=f"FPS: {fps:.1f}")
            self.last_frame_time = now

        delay_ms = 1
        if self.cfg.source_type == "images":
            delay_ms = 150
        elif self.cfg.source_type == "video" and self.cap is not None:
            vid_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if vid_fps > 0:
                ideal_delay = 1000.0 / vid_fps
                process_time = (time.time() - loop_start) * 1000.0
                delay_ms = int(max(1, ideal_delay - process_time))

        self.root.after(delay_ms, self._loop)

    # ---------------- Processing ----------------
    def _process_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Optimization: We do not sync params from UI every frame.
        # UI controls and bindings handle updating `self.cfg` when they change.

        roi_tuple = None
        if self.cfg.roi is not None:
            r = self.cfg.roi
            roi_tuple = (r.x, r.y, r.w, r.h)

        roi_frame, offset = crop_roi(frame_bgr, roi_tuple)

        mask = self.segmenter.segment(
            roi_frame,
            use_otsu=self.cfg.use_otsu,
            threshold_value=self.cfg.threshold_value,
            invert=self.cfg.threshold_invert,
        )
        mask = postprocess_mask(mask, kernel_size=self.cfg.morph_kernel, iters=self.cfg.morph_iters)

        dets = detect_products(mask, min_area=self.cfg.min_area, max_area=self.cfg.max_area, frame_bgr=roi_frame)

        ln = self.cfg.line
        tracks = {}
        if self.cfg.counting_mode == "line":
            # Convert detections to tracker format (Point, bbox, color_label)
            detections = []
            for d in dets:
                cx, cy = d.centroid
                detections.append((Point(cx, cy), d.bbox, d.color_label))

            # Keep previous centroids for crossing
            prev = {tid: tr.centroid for tid, tr in self.tracker.tracks.items()}
            tracks = self.tracker.update(detections)

            # Counting
            if ln is not None:
                line = Line2D(ln.x1, ln.y1, ln.x2, ln.y2)
                self.counter.update_counts(tracks, prev, line)
                c = self.counter.counts
                self._update_metrics_ui(
                    total=self.counter.total,
                    red=c.get("Red", 0),
                    yellow=c.get("Yellow", 0),
                    green=c.get("Green", 0),
                    blue=c.get("Blue", 0),
                    unknown=c.get("unknown", 0),
                    is_blob=False
                )
        else:
            # Blob counting
            c = {"Red": 0, "Yellow": 0, "Green": 0, "Blue": 0, "unknown": 0}
            for d in dets:
                label = d.color_label if d.color_label in c else "unknown"
                c[label] += 1
            self._update_metrics_ui(
                total=len(dets),
                red=c["Red"],
                yellow=c["Yellow"],
                green=c["Green"],
                blue=c["Blue"],
                unknown=c["unknown"],
                is_blob=True
            )

        # Visualization on FULL frame
        vis = frame_bgr.copy()
        off_x, off_y = offset

        # Draw semi-transparent ROI
        if roi_tuple is not None:
            rx, ry, rw, rh = roi_tuple
            overlay = vis.copy()
            cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (128, 0, 128), -1)
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
            cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (255, 0, 255), 2)

        # Draw line (only when line mode and line is set)
        if self.cfg.counting_mode == "line" and ln is not None:
            cv2.line(vis, (ln.x1 + off_x, ln.y1 + off_y), (ln.x2 + off_x, ln.y2 + off_y), (0, 255, 255), 2)

        color_map = {
            "Red": (0, 0, 255),
            "Yellow": (0, 255, 255),
            "Green": (0, 255, 0),
            "Blue": (255, 0, 0),
            "unknown": (255, 255, 255)
        }

        if self.cfg.counting_mode == "line":
            # Draw tracks
            for tid, tr in tracks.items():
                if tr.missing > 0:
                    continue
                x, y, w, h = tr.bbox
                color = color_map.get(tr.color, (0, 200, 0)) if not tr.counted else (128, 128, 128)
                cv2.rectangle(vis, (x + off_x, y + off_y), (x + w + off_x, y + h + off_y), color, 2)
                label_text = f"ID:{tid} {tr.color}" if tr.color != "unknown" else f"ID:{tid}"
                cv2.putText(
                    vis,
                    label_text,
                    (x + off_x, max(0, y + off_y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        else:
            # Draw detections directly
            for i, d in enumerate(dets, start=1):
                x, y, w, h = d.bbox
                color = color_map.get(d.color_label, (0, 200, 0))
                cv2.rectangle(vis, (x + off_x, y + off_y), (x + w + off_x, y + h + off_y), color, 2)
                label_text = f"#{i} {d.color_label}" if d.color_label != "unknown" else f"#{i}"
                cv2.putText(
                    vis,
                    label_text,
                    (x + off_x, max(0, y + off_y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        # Draw stats overlay on vis frame
        if self.cfg.counting_mode == "line":
            c = self.counter.counts
            stats_text = f"Total Counted: {self.counter.total}"
            color_stats = f"R:{c.get('Red',0)} Y:{c.get('Yellow',0)} G:{c.get('Green',0)} B:{c.get('Blue',0)}"
        else:
            c = {"Red": 0, "Yellow": 0, "Green": 0, "Blue": 0, "unknown": 0}
            for d in dets:
                label = d.color_label if d.color_label in c else "unknown"
                c[label] += 1
            stats_text = f"In-frame: {len(dets)}"
            color_stats = f"R:{c['Red']} Y:{c['Yellow']} G:{c['Green']} B:{c['Blue']} U:{c['unknown']}"

        cv2.putText(vis, stats_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, color_stats, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Optional: save overlay
        if self.cfg.save_overlay:
            out_dir = Path(self.cfg.out_dir)
            if not out_dir.is_absolute():
                out_dir = Path(__file__).resolve().parents[1] / out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / "last_overlay.png"), vis)
            cv2.imwrite(str(out_dir / "last_mask.png"), mask)

        return vis, mask

    def _update_views(self, vis_bgr: np.ndarray, mask: np.ndarray) -> None:
        try:
            # Query the actual display width dynamically to match parent container size
            w_view = self.lbl_view.winfo_width()
            max_w = w_view if w_view > 10 else 520

            im1, scale = _bgr_to_tk_image(vis_bgr, max_w=max_w)
            self.view_scale = scale
            self.lbl_view.configure(image=im1)
            self.lbl_view.image = im1

            if self.cfg.show_mask:
                self.lbl_mask.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
                w_mask = self.lbl_mask.winfo_width()
                max_w_mask = w_mask if w_mask > 10 else 520
                im2 = _gray_to_tk_image(mask, max_w=max_w_mask)
                self.lbl_mask.configure(image=im2)
                self.lbl_mask.image = im2
            else:
                self.lbl_mask.pack_forget()
                self.lbl_mask.configure(image="")
                self.lbl_mask.image = None
        except Exception as exc:
            self.lbl_status.configure(text=f"View error: {exc}", text_color=("#e74c3c", "#e74c3c"))


def main() -> None:
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("dark-blue")
    root = ctk.CTk()
    app = ConveyorCounterApp(root)

    def _on_close():
        try:
            app.pause()
            if app.cap is not None:
                app.cap.release()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()
