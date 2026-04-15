import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

from pathlib import Path
from beakspeak.data import load_metadata, preprocess_uploaded_image
from beakspeak.params import DATA_DIR

# --- Load model ---
MODEL_PATH = Path("models/efficientnetb0_fine_tuned.keras")
model = tf.keras.models.load_model(MODEL_PATH)

# --- Load class mapping ---
metadata_df = load_metadata(DATA_DIR)

class_map = (
    metadata_df[["class_id", "class_name"]]
    .drop_duplicates()
    .sort_values("class_id")
)

def clean_name(name):
    name = name.split(".", 1)[-1]
    name = name.replace("_", " ")
    return name

class_id_to_name = {
    class_id - 1: clean_name(name)
    for class_id, name in zip(class_map["class_id"], class_map["class_name"])
}


# --- UI ---
st.title("🐦 Bird Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = preprocess_uploaded_image(image)

    # Predict
    preds = model.predict(input_tensor)
    probs = tf.nn.softmax(preds[0]).numpy()

    # Top 3
    top3_idx = np.argsort(probs)[-3:][::-1]

    st.subheader("Top Predictions")

    for i in top3_idx:
        st.write(f"{class_id_to_name[i]} — {probs[i]:.2%}")
