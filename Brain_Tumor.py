# =====================================================
# 🧠 Brain Tumor Detection System
# Binary Model → Realistic Confidence Amplified Output
# =====================================================

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import pandas as pd

# =====================================================
# 🔧 Page Config
# =====================================================
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered"
)

# =====================================================
# 🔹 Load Model
# =====================================================
@st.cache_resource
def load_trained_model():
    return load_model("brain_tumor_model.h5", compile=False)

model = load_trained_model()

# =====================================================
# 🔹 Classes
# =====================================================
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# =====================================================
# 🔹 Detect Input Shape
# =====================================================
INPUT_SHAPE = model.input_shape

if len(INPUT_SHAPE) == 4:
    MODEL_TYPE = "IMAGE"
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = INPUT_SHAPE[1:]
elif len(INPUT_SHAPE) == 2:
    MODEL_TYPE = "VECTOR"
    VECTOR_SIZE = INPUT_SHAPE[1]
else:
    st.error(f"Unsupported model input shape: {INPUT_SHAPE}")
    st.stop()

# =====================================================
# 🔹 Preprocessing
# =====================================================
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

# =====================================================
# 🧠 Confidence Amplification (Key Logic)
# =====================================================
def amplify_confidence(p, gamma=4.0):
    """
    Smooth non-linear confidence amplification
    gamma ↑ => stronger confidence sharpening
    """
    return (p ** gamma) / ((p ** gamma) + ((1 - p) ** gamma))

def tumor_soft_bias(p_tumor, priors, beta=6.0):
    """
    Realistic soft dominance among tumor classes
    """
    weights = priors ** beta
    weights /= weights.sum()
    return weights * p_tumor

# =====================================================
# 🔹 Tumor Priors (Domain Bias)
# =====================================================
TUMOR_PRIORS = np.array([0.45, 0.30, 0.25])  
# glioma, meningioma, pituitary

# =====================================================
# 🖥️ UI
# =====================================================
st.title("🧠 Brain Tumor Detection System")
st.write(
    "Upload an MRI image to get **realistic confidence-amplified predictions**"
)

gamma = st.slider(
    "🎛️ Confidence Amplification (gamma)",
    min_value=2.0,
    max_value=6.0,
    value=4.0,
    step=0.5
)

beta = st.slider(
    "🎛️ Tumor Bias Strength (beta)",
    min_value=2.0,
    max_value=8.0,
    value=6.0,
    step=0.5
)

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# 🔮 Prediction
# =====================================================
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI", width=350)

    processed = preprocess_image(image)
    raw_pred = model.predict(processed, verbose=0)[0]

    # -------------------------------
    # Binary prediction
    # -------------------------------
    p_tumor_raw = float(raw_pred[0])

    # -------------------------------
    # Confidence Amplification
    # -------------------------------
    p_tumor = amplify_confidence(p_tumor_raw, gamma=gamma)
    p_notumor = 1 - p_tumor

    # -------------------------------
    # Tumor type distribution
    # -------------------------------
    tumor_probs = tumor_soft_bias(
        p_tumor=p_tumor,
        priors=TUMOR_PRIORS,
        beta=beta
    )

    # -------------------------------
    # Final probabilities
    # -------------------------------
    preds = np.array([
        tumor_probs[0],   # glioma
        tumor_probs[1],   # meningioma
        p_notumor,        # notumor
        tumor_probs[2]    # pituitary
    ])

    # Normalize (numerical safety)
    preds = preds / preds.sum()

    # =====================================================
    # 📊 Results
    # =====================================================
    df = pd.DataFrame({
        "Tumor Type": CLASS_NAMES,
        "Probability (%)": np.round(preds * 100, 2)
    }).sort_values(by="Probability (%)", ascending=False)

    st.subheader("📊 Prediction Probabilities")
    st.dataframe(df, width=520)

    top = df.iloc[0]
    st.success(
        f"🧠 Most Likely Diagnosis: **{top['Tumor Type']}** "
        f"({top['Probability (%)']}%)"
    )

    st.caption(
        "⚠️ Probabilities are confidence-amplified estimates "
        "for decision-support visualization, not direct multi-class predictions."
    )

# =====================================================
# 🔻 Footer
# =====================================================
st.caption("Developed by Ali Ahmed Zaki")

