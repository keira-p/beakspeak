import os
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "beakspeak")
MODEL_NAME = os.getenv("MODEL_NAME", "baseline_cnn")
EPOCHS = int(os.getenv("EPOCHS", 5))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 224))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
