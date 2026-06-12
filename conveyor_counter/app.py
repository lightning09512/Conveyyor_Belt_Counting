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
from .tracker import CentroidTracker, LineCrossingCounter
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
        self.root.title("Conveyor Product Counter (CV)")
        self.root.geometry("1180x720")
        # self.root.eval("tk::PlaceWindow . center")

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
        frm_top = ctk.CTkFrame(self.root)
        frm_top.pack(fill=tk.X, padx=10, pady=8)

        # Source controls
        ctk.CTkLabel(frm_top, text="Source:").pack(side=tk.LEFT, padx=(10, 4))

        self.var_source = tk.StringVar(value="video")
        ctk.CTkRadioButton(frm_top, text="Video", variable=self.var_source, value="video", command=self._on_source_changed).pack(side=tk.LEFT, padx=6)
        ctk.CTkRadioButton(frm_top, text="Webcam", variable=self.var_source, value="webcam", command=self._on_source_changed).pack(side=tk.LEFT)
        ctk.CTkRadioButton(frm_top, text="Images", variable=self.var_source, value="images", command=self._on_source_changed).pack(side=tk.LEFT, padx=6)

        self.entry_video = ctk.CTkEntry(frm_top, width=300)
        self.entry_video.pack(side=tk.LEFT, padx=8)
        ctk.CTkButton(frm_top, text="Browse...", command=self._browse_source_path, width=80).pack(side=tk.LEFT)

        ctk.CTkLabel(frm_top, text="Cam idx:").pack(side=tk.LEFT, padx=(16, 4))
        self.entry_cam = ctk.CTkEntry(frm_top, width=40)
        self.entry_cam.insert(0, "0")
        self.entry_cam.pack(side=tk.LEFT)

        ctk.CTkButton(frm_top, text="Open", command=self.open_capture, width=60).pack(side=tk.LEFT, padx=10)
        ctk.CTkButton(frm_top, text="Start", command=self.start, width=60).pack(side=tk.LEFT, padx=4)
        ctk.CTkButton(frm_top, text="Pause", command=self.pause, width=60).pack(side=tk.LEFT, padx=4)
        ctk.CTkButton(frm_top, text="Reset", command=self.reset_count, width=60).pack(side=tk.LEFT, padx=4)

        # ROI & Line
        frm_roi = ctk.CTkFrame(self.root)
        frm_roi.pack(fill=tk.X, padx=10, pady=6)

        ctk.CTkButton(frm_roi, text="Select ROI", command=self.select_roi, width=100).pack(side=tk.LEFT, padx=10, pady=6)
        self.lbl_roi = ctk.CTkLabel(frm_roi, text="ROI: (none)")
        self.lbl_roi.pack(side=tk.LEFT, padx=10)

        ctk.CTkButton(frm_roi, text="Select Line", command=self.select_line, width=100).pack(side=tk.LEFT, padx=(20, 10))
        self.lbl_line = ctk.CTkLabel(frm_roi, text="Line: (none)")
        self.lbl_line.pack(side=tk.LEFT, padx=10)

        # Params
        frm_params = ctk.CTkFrame(self.root)
        frm_params.pack(fill=tk.X, padx=10, pady=6)
        
        frm_params_grid = ctk.CTkFrame(frm_params, fg_color="transparent")
        frm_params_grid.pack(padx=10, pady=10, anchor="w")

        ctk.CTkLabel(frm_params_grid, text="Seg mode:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.var_seg = tk.StringVar(value=self.cfg.seg_mode)
        ctk.CTkOptionMenu(frm_params_grid, variable=self.var_seg, values=["bgsub", "threshold"], command=lambda _v: self._sync_params_to_cfg(), width=110).grid(row=0, column=1, sticky="w", padx=6, pady=4)

        self.var_show_mask = ctk.StringVar(value="1" if self.cfg.show_mask else "0")
        ctk.CTkCheckBox(frm_params_grid, text="Show mask", variable=self.var_show_mask, onvalue="1", offvalue="0", command=self._sync_params_to_cfg).grid(row=0, column=2, sticky="w", padx=16, pady=4)

        self.var_save_overlay = ctk.StringVar(value="1" if self.cfg.save_overlay else "0")
        ctk.CTkCheckBox(frm_params_grid, text="Save overlay", variable=self.var_save_overlay, onvalue="1", offvalue="0", command=self._sync_params_to_cfg).grid(row=0, column=3, sticky="w", padx=16, pady=4)

        ctk.CTkLabel(frm_params_grid, text="Count mode:").grid(row=0, column=4, sticky="w", padx=(16, 6), pady=4)
        self.var_count_mode = tk.StringVar(value=self.cfg.counting_mode)
        ctk.CTkOptionMenu(frm_params_grid, variable=self.var_count_mode, values=["line", "blob"], command=lambda _v: self._sync_params_to_cfg(), width=110).grid(row=0, column=5, sticky="w", padx=6, pady=4)

        ctk.CTkLabel(frm_params_grid, text="Min area:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.entry_min_area = ctk.CTkEntry(frm_params_grid, width=110)
        self.entry_min_area.insert(0, str(self.cfg.min_area))
        self.entry_min_area.grid(row=1, column=1, sticky="w", padx=6, pady=4)

        ctk.CTkLabel(frm_params_grid, text="Max area:").grid(row=1, column=2, sticky="w", padx=16, pady=4)
        self.entry_max_area = ctk.CTkEntry(frm_params_grid, width=110)
        self.entry_max_area.insert(0, str(self.cfg.max_area))
        self.entry_max_area.grid(row=1, column=3, sticky="w", padx=6, pady=4)

        ctk.CTkLabel(frm_params_grid, text="Kernel:").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.entry_kernel = ctk.CTkEntry(frm_params_grid, width=110)
        self.entry_kernel.insert(0, str(self.cfg.morph_kernel))
        self.entry_kernel.grid(row=2, column=1, sticky="w", padx=6, pady=4)

        ctk.CTkLabel(frm_params_grid, text="Iters:").grid(row=2, column=2, sticky="w", padx=16, pady=4)
        self.entry_iters = ctk.CTkEntry(frm_params_grid, width=110)
        self.entry_iters.insert(0, str(self.cfg.morph_iters))
        self.entry_iters.grid(row=2, column=3, sticky="w", padx=6, pady=4)

        ctk.CTkLabel(frm_params_grid, text="Threshold:").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        self.entry_thresh = ctk.CTkEntry(frm_params_grid, width=110)
        self.entry_thresh.insert(0, str(self.cfg.threshold_value))
        self.entry_thresh.grid(row=3, column=1, sticky="w", padx=6, pady=4)

        self.var_otsu = ctk.StringVar(value="1" if self.cfg.use_otsu else "0")
        ctk.CTkCheckBox(frm_params_grid, text="Use Otsu", variable=self.var_otsu, onvalue="1", offvalue="0", command=self._sync_params_to_cfg).grid(row=3, column=2, sticky="w", padx=16, pady=4)

        self.var_invert = ctk.StringVar(value="1" if self.cfg.threshold_invert else "0")
        ctk.CTkCheckBox(frm_params_grid, text="Invert threshold", variable=self.var_invert, onvalue="1", offvalue="0", command=self._sync_params_to_cfg).grid(row=3, column=3, sticky="w", padx=16, pady=4)

        ctk.CTkButton(frm_params_grid, text="Apply params", command=self._sync_params_to_cfg, width=110).grid(row=3, column=4, sticky="w", padx=(16, 6), pady=4)

        # Config load/save
        frm_cfg = ctk.CTkFrame(self.root)
        frm_cfg.pack(fill=tk.X, padx=10, pady=6)

        ctk.CTkButton(frm_cfg, text="Save config...", command=self.save_cfg_dialog, width=110).pack(side=tk.LEFT, padx=(10, 8), pady=6)
        ctk.CTkButton(frm_cfg, text="Load config...", command=self.load_cfg_dialog, width=110).pack(side=tk.LEFT, padx=8)
        self.lbl_status = ctk.CTkLabel(frm_cfg, text="Ready", text_color="gray")
        self.lbl_status.pack(side=tk.LEFT, padx=12)

        # Views
        frm_views = ctk.CTkFrame(self.root)
        frm_views.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.lbl_view = tk.Label(frm_views, bg="black")
        self.lbl_view.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 4), pady=8)

        self.lbl_view.bind("<Button-1>", self._on_view_click)
        self.lbl_view.bind("<B1-Motion>", self._on_view_drag)
        self.lbl_view.bind("<ButtonRelease-1>", self._on_view_release)
        self.lbl_view.bind("<Motion>", self._on_view_motion)
        self.lbl_view.bind("<Button-3>", self._on_view_cancel)

        self.lbl_mask = tk.Label(frm_views, bg="black")
        self.lbl_mask.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 8), pady=8)

        # Stats footer
        frm_footer = ctk.CTkFrame(self.root)
        frm_footer.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.lbl_count = ctk.CTkLabel(frm_footer, text="Count: 0", font=("Arial", 20, "bold"))
        self.lbl_count.pack(side=tk.LEFT, padx=10, pady=6)

        self.lbl_fps = ctk.CTkLabel(frm_footer, text="FPS: --", font=("Arial", 14))
        self.lbl_fps.pack(side=tk.LEFT, padx=16)

        self._on_source_changed()

    def _on_source_changed(self, preserve_count_mode: bool = False) -> None:
        src = self.var_source.get()
        is_video = src == "video"
        is_images = src == "images"
        # Entry is used for both: video path or image folder.
        self.entry_video.configure(state=("normal" if (is_video or is_images) else "disabled"))
        self.entry_cam.configure(state=("normal" if src == "webcam" else "disabled"))

        # Suggest a sensible default counting mode
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

            # Keep footer label consistent with mode (when not running)
            if self.cfg.counting_mode == "blob":
                if not self.running:
                    self.lbl_count.configure(text="In-frame: 0")
            else:
                if not self.running:
                    c = self.counter.counts
                    self.lbl_count.configure(text=f"Count: {self.counter.total} (R:{c.get('Red',0)} Y:{c.get('Yellow',0)} G:{c.get('Green',0)} B:{c.get('Blue',0)})")

            self.segmenter.set_mode(self.cfg.seg_mode)

            self.tracker.max_distance = float(self.cfg.max_match_distance)
            self.tracker.max_missing = int(self.cfg.max_missing_frames)

            self.lbl_status.configure(text="Params applied")
        except Exception as exc:
            messagebox.showerror("Error", f"Invalid parameters: {exc}")

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
        self.lbl_status.configure(text=f"Saved config: {Path(path).name}")

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
        self.lbl_status.configure(text=f"Loaded config: {Path(path).name}")

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

            self.lbl_status.configure(text=f"Opened images: {len(self.image_paths)}")
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

        self.lbl_status.configure(text="Opened source")
        self._grab_and_show_single_frame()

    def _grab_and_show_single_frame(self) -> None:
        frame = self._read_current_frame_peek()
        if frame is None:
            self.lbl_status.configure(text="No frame")
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
        self.lbl_status.configure(text="Drag on video to select ROI. Right-click to cancel.")
        self._grab_and_show_single_frame()

    def select_line(self) -> None:
        if self.cap is None and not self.image_paths:
            messagebox.showwarning("Missing", "Open a source first.")
            return
        self.pause()
        self.ui_state = "line"
        self.line_pts = []
        self.mouse_pos = None
        self.lbl_status.configure(text="Click 2 points on video to draw Line. Right-click to cancel.")
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
                self.lbl_status.configure(text="Line updated")
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
                self.lbl_status.configure(text="ROI cleared")
            else:
                self.cfg.roi = ROI(x=x, y=y, w=w, h=h)
                self.lbl_status.configure(text="ROI updated")
            
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
            self.lbl_status.configure(text="Selection cancelled")
            self._grab_and_show_single_frame()

    def _redraw_interactive(self) -> None:
        if self.last_raw_frame is None:
            return
        frame = self.last_raw_frame.copy()
        
        if self.ui_state == "roi":
            if self.roi_start_pt is not None and self.roi_end_pt is not None:
                cv2.rectangle(frame, self.roi_start_pt, self.roi_end_pt, (255, 0, 0), 2)
            im1, scale = _bgr_to_tk_image(frame)
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
                
            im1, scale = _bgr_to_tk_image(view)
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
        self.lbl_status.configure(text="Running...")
        self._loop()

    def pause(self) -> None:
        self.running = False
        self.lbl_status.configure(text="Paused")

    def reset_count(self) -> None:
        self.counter.reset()
        for tr in self.tracker.tracks.values():
            tr.counted = False
        self.prev_centroids.clear()
        if self.cfg.counting_mode == "blob":
            self.lbl_count.configure(text="In-frame: 0")
        else:
            c = self.counter.counts
            self.lbl_count.configure(text=f"Count: {self.counter.total} (R:{c.get('Red',0)} Y:{c.get('Yellow',0)} G:{c.get('Green',0)} B:{c.get('Blue',0)})")
        self.lbl_status.configure(text="Count reset")

    def _loop(self) -> None:
        loop_start = time.time()
        if not self.running:
            return
        frame = None
        if self.cfg.source_type == "images":
            if not self.image_paths or self.image_index >= len(self.image_paths):
                self.running = False
                self.lbl_status.configure(text="End of images")
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
                self.lbl_status.configure(text="End of stream")
                return
            self.last_raw_frame = frame

        vis, mask = self._process_frame(frame)
        self._update_views(vis, mask)

        # FPS (only meaningful for streaming video/webcam)
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
        self._sync_params_to_cfg()

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
                self.lbl_count.configure(text=f"Count: {self.counter.total} (R:{c.get('Red',0)} Y:{c.get('Yellow',0)} G:{c.get('Green',0)} B:{c.get('Blue',0)})")
        else:
            # Blob counting
            c = {"Red": 0, "Yellow": 0, "Green": 0, "Blue": 0}
            for d in dets:
                if d.color_label in c:
                    c[d.color_label] += 1
            self.lbl_count.configure(text=f"In-frame: {len(dets)} (R:{c['Red']} Y:{c['Yellow']} G:{c['Green']} B:{c['Blue']})")

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

        # Optional: save overlay
        if self.cfg.save_overlay:
            out_dir = Path(self.cfg.out_dir)
            if not out_dir.is_absolute():
                out_dir = Path(__file__).resolve().parents[1] / out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            # Save every N frames could be better; keep simple: save last overlay only
            cv2.imwrite(str(out_dir / "last_overlay.png"), vis)
            cv2.imwrite(str(out_dir / "last_mask.png"), mask)

        return vis, mask

    def _update_views(self, vis_bgr: np.ndarray, mask: np.ndarray) -> None:
        try:
            im1, scale = _bgr_to_tk_image(vis_bgr)
            self.view_scale = scale
            self.lbl_view.configure(image=im1)
            self.lbl_view.image = im1

            if self.cfg.show_mask:
                im2 = _gray_to_tk_image(mask)
                self.lbl_mask.configure(image=im2)
                self.lbl_mask.image = im2
            else:
                self.lbl_mask.configure(image="")
                self.lbl_mask.image = None
        except Exception as exc:
            self.lbl_status.configure(text=f"View error: {exc}")


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
