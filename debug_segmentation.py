"""Diagnostic script to analyze segmentation quality on 2.mp4.

Extracts frames, applies the detection pipeline, and saves annotated images
showing contours, watershed markers, and detection results for visual debugging.
"""
import cv2
import numpy as np
from pathlib import Path

VIDEO_PATH = r"d:\hoc\Xu Ly Anh\Conveyyor_Belt_Counting\assets\2.mp4"
OUT_DIR = Path(r"d:\hoc\Xu Ly Anh\Conveyyor_Belt_Counting\debug_output")
OUT_DIR.mkdir(exist_ok=True)

# Open video and extract a few frames after background model stabilizes
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {fps:.1f} FPS, {total_frames} frames")

# Create background subtractor
bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=36, detectShadows=True)

# Let background model stabilize by feeding first 60 frames
frame_idx = 0
while frame_idx < 60:
    ok, frame = cap.read()
    if not ok:
        break
    bg.apply(frame)
    frame_idx += 1

print(f"Background model warmed up with {frame_idx} frames")

# Now process and save diagnostic images for next 200 frames
KERNEL_SIZE = 11
MORPH_ITERS = 3
MIN_AREA = 600
MAX_AREA = 60000

kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), dtype=np.uint8)

saved = 0
while saved < 200:
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1
    
    # Only save every 5th frame  
    if frame_idx % 5 != 0:
        continue
    
    # 1. Background subtraction
    fg = bg.apply(frame)
    _, raw_mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    
    # 2. Morphological cleanup
    cleaned_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERS)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITERS)
    
    # 3. Find contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area
    valid_contours = []
    large_contours = []  # Potentially touching clusters
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        if area > MAX_AREA:
            large_contours.append((c, area))
        else:
            valid_contours.append((c, area))
    
    # Skip frames with no detections
    if not valid_contours and not large_contours:
        continue
    
    # Create diagnostic visualization
    h, w = frame.shape[:2]
    
    # Top-left: original frame with contours
    vis_frame = frame.copy()
    for c, area in valid_contours:
        x, y, cw, ch = cv2.boundingRect(c)
        cv2.rectangle(vis_frame, (x, y), (x+cw, y+ch), (0, 255, 0), 2)
        cv2.putText(vis_frame, f"A:{int(area)}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    for c, area in large_contours:
        x, y, cw, ch = cv2.boundingRect(c)
        cv2.rectangle(vis_frame, (x, y), (x+cw, y+ch), (0, 0, 255), 2)
        cv2.putText(vis_frame, f"CLUSTER A:{int(area)}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Top-right: mask
    mask_bgr = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)
    
    # Bottom-left: distance transform visualization for large contours
    dist_vis = np.zeros_like(frame)
    all_contours = valid_contours + large_contours
    for c, area in all_contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw <= 1 or ch <= 1:
            continue
        roi_mask = cleaned_mask[y:y+ch, x:x+cw]
        dist = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
        dt_max = dist.max()
        if dt_max > 0:
            dist_norm = (dist / dt_max * 255).astype(np.uint8)
            dist_color = cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET)
            # Only show where mask is active
            dist_color[roi_mask == 0] = 0
            dist_vis[y:y+ch, x:x+cw] = dist_color
            
            # Show dt_max value
            cv2.putText(dist_vis, f"dt:{dt_max:.1f}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Bottom-right: watershed result
    ws_vis = frame.copy()
    for c, area in all_contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw <= 1 or ch <= 1:
            continue
        roi_mask = cleaned_mask[y:y+ch, x:x+cw]
        roi_bgr = frame[y:y+ch, x:x+cw].copy()
        
        dist = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
        dt_max = dist.max()
        if dt_max <= 0:
            continue
        
        # Current approach: local max
        local_max_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilated = cv2.dilate(dist, local_max_kernel)
        peaks = (dist >= dilated) & (dist > 5.0)
        
        n_peaks = np.sum(peaks)
        
        sure_fg = np.zeros_like(roi_mask, dtype=np.uint8)
        sure_fg[peaks] = 255
        
        if np.max(sure_fg) == 0:
            _, sure_fg = cv2.threshold(dist, 0.5 * dt_max, 255, 0)
            sure_fg = np.uint8(sure_fg)
        
        num_labels, markers = cv2.connectedComponents(sure_fg)
        
        cv2.putText(ws_vis, f"peaks:{n_peaks} labels:{num_labels-1}", (x, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        
        if num_labels <= 2:
            cv2.rectangle(ws_vis, (x, y), (x+cw, y+ch), (0, 255, 0), 2)
            continue
        
        # Run watershed
        sure_bg = cv2.dilate(roi_mask, np.ones((3, 3), np.uint8), iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)
        markers2 = markers + 1
        markers2[unknown == 255] = 0
        markers2 = cv2.watershed(roi_bgr, markers2)
        
        # Draw each segment in different color
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for label in range(2, markers2.max() + 1):
            obj_mask = np.zeros_like(roi_mask, dtype=np.uint8)
            obj_mask[markers2 == label] = 255
            sub_contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if sub_contours:
                sc = max(sub_contours, key=cv2.contourArea)
                sub_area = cv2.contourArea(sc)
                sx, sy, sw, sh = cv2.boundingRect(sc)
                color = colors[(label - 2) % len(colors)]
                cv2.rectangle(ws_vis, (x+sx, y+sy), (x+sx+sw, y+sy+sh), color, 2)
                cv2.putText(ws_vis, f"s{label}:{int(sub_area)}", (x+sx, y+sy-3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # Combine into 2x2 grid
    top = np.hstack([vis_frame, mask_bgr])
    bottom = np.hstack([dist_vis, ws_vis])
    combined = np.vstack([top, bottom])
    
    # Resize for saving
    scale = 1920.0 / combined.shape[1]
    if scale < 1.0:
        combined = cv2.resize(combined, (1920, int(combined.shape[0] * scale)))
    
    out_path = OUT_DIR / f"diag_{frame_idx:05d}.png"
    cv2.imwrite(str(out_path), combined)
    saved += 1
    if saved % 10 == 0:
        print(f"  Saved {saved} diagnostic frames...")

cap.release()
print(f"Done! Saved {saved} diagnostic images to {OUT_DIR}")
