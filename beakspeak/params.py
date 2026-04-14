import os
from dotenv import load_dotenv

load_dotenv()

# --- image ---
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 224))
IMG_HEIGHT = IMAGE_SIZE
IMG_WIDTH = IMAGE_SIZE

# --- training ---
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
EPOCHS = int(os.getenv("EPOCHS", 5))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))
SEED = int(os.getenv("SEED", 42))

# --- dataset ---
NUM_CLASSES = int(os.getenv("NUM_CLASSES", 200))

# --- tf performance ---
AUTOTUNE = None  # set in notebook as tf.data.AUTOTUNE

# --- mlflow ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///../mlruns.db")
