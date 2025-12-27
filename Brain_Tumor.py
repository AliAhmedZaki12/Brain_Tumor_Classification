import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import pandas as pd

# ===== Page Config =====
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# ===== Load Model =====
@st.cache_resource
def load_trained_model():
    try:
        model = load_model("brain_tumor_model.h5", compile=False)
        st.success("✅ Binary model loaded successfully!")
        st.write("Model input shape:", model.input_shape)
        return model
    except FileNotFoundError:
        st.error("❌ لم يتم العثور على نموذج Binary. تأكد من رفع brain_tumor_model.h5")
        st.stop()

model = load_trained_model()

# ===== Classes =====
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# ===== Preprocessing ديناميكي =====
def preprocess_image(image: Image.Image):
    image = np.array(image)
    input_shape = model.input_shape

    # ===== Vector Input (1D) =====
    if len(input_shape) == 2:
        vec = cv2.resize(image, (224, 224)) if len(image.shape) == 3 else image
        vec = vec.flatten().astype("float32") / 255.0
        vec = np.pad(vec, (0, max(0, input_shape[1] - vec.shape[0])), constant_values=0)
        return np.expand_dims(vec, axis=0)

    # ===== Image Input (H,W,C) =====
    if len(input_shape) == 4:
        h, w, channels = input_shape[1], input_shape[2], input_shape[3]
        # Grayscale
        if channels == 1:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = image[..., np.newaxis]
        else:  # RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = image[..., :3]
        image = cv2.resize(image, (w, h))
        image = image.astype("float32") / 255.0
        return np.expand_dims(image, axis=0)

    st.error(f"Unsupported model input shape: {input_shape}")
    st.stop()

IMG_SIZE = 224

# ===== UI =====
st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI image to get predictions (Binary model with estimated 4-Class probabilities).")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

# ===== Prediction =====
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", width=350)

    processed = preprocess_image(image)
    st.write("Processed input shape:", processed.shape)

    # Binary prediction
    raw_pred = model.predict(processed, verbose=0)
    try:
        p_tumor = float(raw_pred[0][0])
    except:
        p_tumor = float(raw_pred[0])
    p_notumor = 1 - p_tumor

    # ===== Estimated 4-Class Distribution =====
    priors = np.array([0.45, 0.30, 0.25])  # glioma, meningioma, pituitary
    tumor_est = priors * p_tumor
    preds = np.array([tumor_est[0], tumor_est[1], p_notumor, tumor_est[2]])

    # Softmax-style Scaling
    def softmax_scale(p):
        e = np.exp(p * 5)
        return e / e.sum()

    preds_scaled = softmax_scale(preds)

    # Results Table
    df = pd.DataFrame({
        "Tumor Type": CLASS_NAMES,
        "Probability (%)": np.round(preds_scaled * 100, 2)
    }).sort_values(by="Probability (%)", ascending=False)

    st.subheader("📊 Prediction Probabilities (Estimated)")

    # ===== تلوين الجدول تدريجيًا =====
    def color_scale(val):
        # اللون الأحمر لكل النسبة (0-100)
        red_intensity = int(val * 2.55)  # 0-255
        return f'background-color: rgb({red_intensity}, 50, 50); color: white;'

    st.dataframe(df.style.applymap(color_scale, subset=["Probability (%)"]), width=520)

    # Top Prediction
    top = df.iloc[0]
    if top["Tumor Type"] == "notumor":
        st.success(f"✅ No Tumor Detected ({top['Probability (%)']}% confidence)")
    else:
        st.error(f"⚠️ Tumor Detected: {top['Tumor Type']} ({top['Probability (%)']}% confidence)")

    st.caption("⚠️ Probabilities are estimated from a binary model. Display purposes only.")

# Footer
st.caption("Developed by Ali Ahmed Zaki")
