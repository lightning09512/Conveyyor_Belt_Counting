import json
import os
import shutil
import cv2
from pathlib import Path

def convert_to_yolo(src_dir, dest_split_dir, class_map):
    src_dir = Path(src_dir)
    labels_file = src_dir / "bounding_boxes.labels"
    
    if not labels_file.exists():
        print(f"Warning: {labels_file} not found.")
        return

    dest_images_dir = dest_split_dir / "images"
    dest_labels_dir = dest_split_dir / "labels"
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_labels_dir.mkdir(parents=True, exist_ok=True)

    with open(labels_file, "r") as f:
        data = json.load(f)

    bounding_boxes = data.get("boundingBoxes", {})
    
    for filename, boxes in bounding_boxes.items():
        src_img_path = src_dir / filename
        if not src_img_path.exists():
            continue

        # Get image dimensions for normalization
        img = cv2.imread(str(src_img_path))
        if img is None:
            continue
        h, w, _ = img.shape

        # Copy image
        dest_img_path = dest_images_dir / filename
        shutil.copy(src_img_path, dest_img_path)

        # Write labels
        txt_filename = Path(filename).with_suffix(".txt").name
        # If the original filename has multiple dots, ensure we just change the very last to .txt
        if filename.count('.') > 1:
            txt_filename = filename.rsplit('.', 1)[0] + '.txt'

        dest_label_path = dest_labels_dir / txt_filename

        with open(dest_label_path, "w") as f:
            for box in boxes:
                label = box["label"]
                if label not in class_map:
                    class_map[label] = len(class_map)
                
                class_id = class_map[label]
                
                # Edge Impulse bounding boxes format: x, y is top-left
                box_x = box["x"]
                box_y = box["y"]
                box_w = box["width"]
                box_h = box["height"]

                # YOLO format: center_x, center_y, width, height (normalized 0-1)
                center_x = (box_x + box_w / 2.0) / w
                center_y = (box_y + box_h / 2.0) / h
                norm_w = box_w / w
                norm_h = box_h / h

                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")

if __name__ == "__main__":
    base_dir = Path(r"d:\hoc\Xu Ly Anh\Conveyyor_Belt_Counting\assets")
    train_src = base_dir / "training"
    val_src = base_dir / "testing"

    out_dir = base_dir / "yolo_dataset"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    class_map = {}
    
    print("Converting training data...")
    convert_to_yolo(train_src, out_dir / "train", class_map)
    
    print("Converting validation data...")
    convert_to_yolo(val_src, out_dir / "val", class_map)

    # Sort class map by ID to write yaml
    classes = [k for k, v in sorted(class_map.items(), key=lambda item: item[1])]

    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_dir.absolute().as_posix()}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n\n")
        f.write("names:\n")
        for i, cls_name in enumerate(classes):
            # Capitalize to match existing logic
            f.write(f"  {i}: {cls_name.capitalize()}\n")

    print(f"Dataset converted successfully! Saved to {out_dir}")
    print(f"Classes: {classes}")
    print(f"You can now train using: python tools/train_yolo.py --data \"{yaml_path}\"")
