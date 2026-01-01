import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import requests
import tensorflow as tf
import tempfile
import matplotlib.pyplot as plt

# ===============================
# 🔹 Streamlit Page Config
# ===============================
st.set_page_config(page_title="🧠 Brain Tumor Detection Lite", layout="centered")

st.title("🧠 Brain Tumor Detection System (Medical-Grade Lite TFLite)")
st.write("Upload an MRI image to get **exact prediction probabilities**")

# ===============================
# 🔹 Class Mapping
# ===============================
class_indices = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
idx_to_class = {v: k for k, v in class_indices.items()}

# ===============================
# 🔹 Load TFLite Model from GitHub
# ===============================
TFLITE_URL = "https://github.com/AliAhmedZaki12/Brain_Tumor11/raw/main/brain_tumor_model_lite.tflite"

@st.cache_resource
def load_tflite_model(url):
    r = requests.get(url)
    if r.status_code != 200:
        st.error("Failed to download the TFLite model!")
        return None
    # Save temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(r.content)
    tfile.close()
    # Load TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=tfile.name)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model(TFLITE_URL)
if interpreter is None:
    st.stop()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===============================
# 🔹 Image Upload & Preprocess
# ===============================
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","jpeg","png"])

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((input_details[0]['shape'][2], input_details[0]['shape'][1]))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=350)
    processed = preprocess_image(image)

    # ===============================
    # 🔹 Run TFLite Inference
    # ===============================
    interpreter.set_tensor(input_details[0]['index'], processed)
    interpreter.invoke()
    raw_pred = interpreter.get_tensor(output_details[0]['index'])[0]

    # ===============================
    # 🔹 Prepare DataFrame
    # ===============================
    df = pd.DataFrame({
        "Tumor Type": [idx_to_class[i] for i in range(len(raw_pred))],
        "Probability (%)": np.round(raw_pred * 100, 2)
    }).sort_values(by="Probability (%)", ascending=False)

    # Highlight the top prediction
    top_class = df.iloc[0]['Tumor Type']

    st.subheader("📊 Prediction Probabilities (Exact)")
    st.dataframe(df.style.highlight_max(axis=0, color='lightgreen'), width=500)

    # ===============================
    # 🔹 Bar Chart
    # ===============================
    st.subheader("📈 Probability Chart")
    fig, ax = plt.subplots(figsize=(8,5))
    colors = ['green' if cls==top_class else 'steelblue' for cls in df['Tumor Type']]
    ax.bar(df['Tumor Type'], df['Probability (%)'], color=colors)
    ax.set_ylabel("Probability (%)")
    ax.set_ylim(0, 100)
    for i, v in enumerate(df['Probability (%)']):
        ax.text(i, v + 1, f"{v}%", ha='center', fontweight='bold')
    st.pyplot(fig)

    st.success(
        f"Model suggests **{top_class}** "
        f"with confidence **{df.iloc[0]['Probability (%)']}%**"
    )

st.caption("Developed by Ali Ahmed Zaki ")
