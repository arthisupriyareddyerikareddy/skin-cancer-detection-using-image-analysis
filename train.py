import argparse
from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLO model.')
    parser.add_argument('--data', type=str, required=True, help='Path to the data used for train and validate the model.')
    args = parser.parse_args()
    # Load a model
    model = YOLO('yolov8s-cls.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data=args.data,
                          epochs=100,
                          imgsz=480,
                          patience=20,
                          batch=16,
                          dropout=0.25,
                          device=0)
