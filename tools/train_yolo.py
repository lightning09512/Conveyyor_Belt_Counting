import argparse
from pathlib import Path
import sys

def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Train YOLOv8 on custom dataset.")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model to use (default: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    print(f"Loading base model: {args.model}")
    model = YOLO(args.model)

    print(f"Starting training on dataset: {data_path}")
    print(f"Epochs: {args.epochs}, ImgSz: {args.imgsz}, Batch: {args.batch}")
    
    # Train the model
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs/train",
        name="conveyor_model",
        exist_ok=True,
    )

    print("\nTraining completed!")
    print(f"Model saved at: runs/train/conveyor_model/weights/best.pt")

if __name__ == "__main__":
    main()
