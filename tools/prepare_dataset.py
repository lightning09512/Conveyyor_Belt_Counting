import argparse
import cv2
import os
import sys
from pathlib import Path

# Add project root to sys.path so we can import config
sys.path.append(str(Path(__file__).resolve().parents[1]))
from conveyor_counter.config import load_config
from conveyor_counter.vision import crop_roi

def main():
    parser = argparse.ArgumentParser(description="Extract frames from video for YOLO dataset.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--out_dir", type=str, default="datasets/images", help="Output directory for frames")
    parser.add_argument("--interval", type=int, default=15, help="Extract 1 frame every N frames")
    parser.add_argument("--config", type=str, default="", help="Path to config file to apply ROI cropping")
    
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return

    # Load ROI if config provided
    roi_tuple = None
    if args.config:
        cfg = load_config(args.config)
        if cfg.roi:
            roi_tuple = (cfg.roi.x, cfg.roi.y, cfg.roi.w, cfg.roi.h)
            print(f"Loaded ROI from config: {roi_tuple}")

    frame_count = 0
    saved_count = 0

    print(f"Extracting frames from {args.video}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % args.interval == 0:
            if roi_tuple:
                frame, _ = crop_roi(frame, roi_tuple)
            
            out_path = out_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_count += 1
            
            if saved_count % 10 == 0:
                print(f"Saved {saved_count} frames...")

        frame_count += 1

    cap.release()
    print(f"Done! Saved {saved_count} images to {out_dir}")

if __name__ == "__main__":
    main()
