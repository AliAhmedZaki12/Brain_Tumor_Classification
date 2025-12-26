# ===============================
# 🧠 Brain Tumor Detection App
# Compatible with legacy .h5 models
# ===============================

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import pandas as pd

st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered"
)

# ===============================
# 🔹 Load Model
# ===============================
@st.cache_resource
def load_trained_model():
    return load_model("brain_tumor_model.h5", compile=False)

model = load_trained_model()

# ===============================
# 🔹 Class Names (FIXED)
# ===============================
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

NUM_CLASSES = len(CLASS_NAMES)

# ===============================
# 🔹 Detect Model Input Type
# ===============================
INPUT_SHAPE = model.input_shape  # (None, N) OR (None, H, W, C)

if len(INPUT_SHAPE) == 2:
    MODEL_TYPE = "VECTOR"
    VECTOR_SIZE = INPUT_SHAPE[1]

elif len(INPUT_SHAPE) == 4:
    MODEL_TYPE = "IMAGE"
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = INPUT_SHAPE[1:]

else:
    st.error(f"❌ Unsupported model input shape: {INPUT_SHAPE}")
    st.stop()

# ===============================
# 🔹 Preprocess Image
# ===============================
def preprocess_image(image: Image.Image):
    image = np.array(image.convert("RGB"))

    # ===== Image-based model =====
    if MODEL_TYPE == "IMAGE":
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = image.astype("float32") / 255.0
        return np.expand_dims(image, axis=0)

    # ===== Vector-based model =====
    image = cv2.resize(image, (224, 224))
    image = image.astype("float32") / 255.0
    vector = image.flatten()

    if vector.shape[0] > VECTOR_SIZE:
        vector = vector[:VECTOR_SIZE]
    else:
        vector = np.pad(vector, (0, VECTOR_SIZE - vector.shape[0]))

    return np.expand_dims(vector, axis=0)

# ===============================
# 🔹 UI
# ===============================
st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI image to get prediction probabilities")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed = preprocess_image(image)
    preds = model.predict(processed, verbose=0)[0]

    # ===== Safety: match output size =====
    preds = preds[:NUM_CLASSES]

    if preds.ndim != 1 or preds.shape[0] != NUM_CLASSES:
        st.error(
            f"❌ Model output shape mismatch: expected {NUM_CLASSES}, got {preds.shape}"
        )
        st.stop()

    df = pd.DataFrame({
        "Tumor Type": CLASS_NAMES,
        "Probability (%)": np.round(preds * 100, 2)
    }).sort_values(by="Probability (%)", ascending=False)

    st.subheader("📊 Prediction Probabilities")
    st.dataframe(df, use_container_width=True)

    top = df.iloc[0]
    st.success(
        f"Model suggests **{top['Tumor Type']}** "
        f"with confidence **{top['Probability (%)']}%**"
    )

st.caption("Legacy model compatible | No retraining required")
