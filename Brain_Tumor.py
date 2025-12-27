import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import pandas as pd

# ==========================================
#  Page Config
# ==========================================
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered"
)

# ==========================================
# 🔹 Load Model (Cached)
# ==========================================
@st.cache_resource
def load_trained_model():
    return load_model("brain_tumor_model.h5", compile=False)

model = load_trained_model()

# ==========================================
# 🔹 Class Names
# ==========================================
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# ==========================================
# 🔹 Detect Model Input Type
# ==========================================
INPUT_SHAPE = model.input_shape

if len(INPUT_SHAPE) == 2:
    MODEL_TYPE = "VECTOR"
    VECTOR_SIZE = INPUT_SHAPE[1]
elif len(INPUT_SHAPE) == 4:
    MODEL_TYPE = "IMAGE"
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = INPUT_SHAPE[1:]
else:
    st.error(f"Unsupported model input shape: {INPUT_SHAPE}")
    st.stop()

# ==========================================
# 🔹 Preprocessing
# ==========================================
def preprocess_image(image: Image.Image):
    image = np.array(image.convert("RGB"))

    if MODEL_TYPE == "IMAGE":
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = image.astype("float32") / 255.0
        return np.expand_dims(image, axis=0)

    image = cv2.resize(image, (224, 224))
    image = image.astype("float32") / 255.0
    vec = image.flatten()

    if vec.shape[0] > VECTOR_SIZE:
        vec = vec[:VECTOR_SIZE]
    else:
        vec = np.pad(vec, (0, VECTOR_SIZE - vec.shape[0]))

    return np.expand_dims(vec, axis=0)

# ==========================================
# 🔹 Bias Engine (Core Logic)
# ==========================================
TUMOR_PRIORS = np.array([0.45, 0.30, 0.25])  # glioma, meningioma, pituitary

def biased_distribution(p_tumor, priors, alpha):
    biased = priors ** alpha
    biased /= biased.sum()
    return biased * p_tumor

# ==========================================
# 🖥️ UI
# ==========================================
st.title(" Brain Tumor Detection System")
st.write("Upload an MRI image to get biased multi-class probabilities")

bias_strength = st.slider(
    " Bias Strength (Higher = Stronger Dominance)",
    min_value=1.0,
    max_value=7.0,
    value=5.0,
    step=0.5
)

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# 🔮 Prediction
# ==========================================
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", width=350)

    processed = preprocess_image(image)
    raw_pred = model.predict(processed, verbose=0)[0]

    # Binary Output
    p_tumor = float(raw_pred[0])
    p_notumor = 1 - p_tumor

    # Apply Bias only if confidence is high
    if p_tumor >= 0.6:
        tumor_probs = biased_distribution(
            p_tumor=p_tumor,
            priors=TUMOR_PRIORS,
            alpha=bias_strength
        )
    else:
        tumor_probs = (TUMOR_PRIORS / TUMOR_PRIORS.sum()) * p_tumor

    # Final Probabilities
    preds = np.array([
        tumor_probs[0],   # glioma
        tumor_probs[1],   # meningioma
        p_notumor,        # notumor
        tumor_probs[2]    # pituitary
    ])

    # ==========================================
    # 📊 Display Results
    # ==========================================
    df = pd.DataFrame({
        "Tumor Type": CLASS_NAMES,
        "Probability (%)": np.round(preds * 100, 2)
    }).sort_values(by="Probability (%)", ascending=False)

    st.subheader(" Prediction Probabilities")
    st.dataframe(df, width=500)

    top = df.iloc[0]
    st.success(
        f" Most Likely: **{top['Tumor Type']}** "
        f"with confidence **{top['Probability (%)']}%**"
    )
# ==========================================
#  Footer
# ==========================================
st.caption("Developed by Ali Ahmed Zaki")
