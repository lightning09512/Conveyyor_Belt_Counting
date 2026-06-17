"""Convert Edge Impulse bounding-box exports to YOLOv8 dataset layout."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2


def convert_to_yolo(src_dir: Path, dest_split_dir: Path, class_map: dict[str, int]) -> None:
    labels_file = src_dir / "bounding_boxes.labels"
    if not labels_file.exists():
        print(f"Warning: {labels_file} not found.")
        return

    dest_images_dir = dest_split_dir / "images"
    dest_labels_dir = dest_split_dir / "labels"
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(labels_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    bounding_boxes = data.get("boundingBoxes", {})
    for filename, boxes in bounding_boxes.items():
        src_img_path = src_dir / filename
        if not src_img_path.exists():
            continue

        img = cv2.imread(str(src_img_path))
        if img is None:
            continue
        h, w, _ = img.shape

        shutil.copy(src_img_path, dest_images_dir / filename)

        txt_filename = (
            filename.rsplit(".", 1)[0] + ".txt"
            if filename.count(".") > 1
            else Path(filename).with_suffix(".txt").name
        )
        dest_label_path = dest_labels_dir / txt_filename

        with open(dest_label_path, "w", encoding="utf-8") as f:
            for box in boxes:
                label = box["label"]
                if label not in class_map:
                    class_map[label] = len(class_map)
                class_id = class_map[label]

                box_x = box["x"]
                box_y = box["y"]
                box_w = box["width"]
                box_h = box["height"]

                center_x = (box_x + box_w / 2.0) / w
                center_y = (box_y + box_h / 2.0) / h
                norm_w = box_w / w
                norm_h = box_h / h
                f.write(
                    f"{class_id} {center_x:.6f} {center_y:.6f} "
                    f"{norm_w:.6f} {norm_h:.6f}\n"
                )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Convert Edge Impulse dataset to YOLOv8 format."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=repo_root / "assets",
        help="Directory containing training/ and testing/ folders",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dataset directory (default: <base-dir>/yolo_dataset)",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    train_src = base_dir / "training"
    val_src = base_dir / "testing"
    out_dir = (args.out_dir or base_dir / "yolo_dataset").resolve()

    if not train_src.exists() and not val_src.exists():
        print(
            f"Error: expected {train_src} or {val_src} with bounding_boxes.labels",
            file=sys.stderr,
        )
        sys.exit(1)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    class_map: dict[str, int] = {}
    print("Converting training data...")
    convert_to_yolo(train_src, out_dir / "train", class_map)
    print("Converting validation data...")
    convert_to_yolo(val_src, out_dir / "val", class_map)

    classes = [k for k, v in sorted(class_map.items(), key=lambda item: item[1])]
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("path: .\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n\n")
        f.write("names:\n")
        for i, cls_name in enumerate(classes):
            f.write(f"  {i}: {cls_name.capitalize()}\n")

    print(f"Dataset converted successfully! Saved to {out_dir}")
    print(f"Classes: {classes}")
    print(f'Train with: python tools/train_yolo.py --data "{yaml_path}"')


if __name__ == "__main__":
    main()
