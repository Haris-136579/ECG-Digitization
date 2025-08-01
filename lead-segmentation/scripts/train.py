import os
from ultralytics import YOLO
from dotenv import load_dotenv

# === Load paths from .env file ===
load_dotenv()

DATASET_YAML = os.getenv("YOLO_DATASET_YAML", "./data/ecg.yaml")
MODEL_NAME = os.getenv("YOLO_MODEL", "yolov11s.pt")
EPOCHS = int(os.getenv("YOLO_EPOCHS", 100))
IMG_SIZE = int(os.getenv("YOLO_IMG_SIZE", 640))
BATCH_SIZE = int(os.getenv("YOLO_BATCH", 16))
PROJECT_NAME = os.getenv("YOLO_PROJECT", "yolo-ecg-project")
EXPERIMENT_NAME = os.getenv("YOLO_EXPERIMENT", "exp")

# === Load model (fine-tune from pretrained .pt file) ===
model = YOLO(MODEL_NAME)

# === Start training ===
model.train(
    data=DATASET_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    project=PROJECT_NAME,
    name=EXPERIMENT_NAME,
    pretrained=True
)