import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

# ===============================
# 🔹 Load TFLite Model (Local)
# ===============================
TFLITE_MODEL_PATH = "brain_tumor_model_lite.tfliteA"

@st.cache_resource
def load_tflite_model_local(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model_local(TFLITE_MODEL_PATH)

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
class_labels   = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# ===============================
# 🔹 Preprocess Image
# ===============================
def preprocess_image(image: Image.Image, target_size=(299, 299)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# ===============================
# 🔹 Prediction Function with Reject Option
# ===============================
def predict_tflite(image: Image.Image, threshold=0.8):
    img = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    # ❌ Reject if max probability < threshold
    max_prob = np.max(preds)
    if max_prob < threshold:
        return None, preds

    predicted_class = class_labels[np.argmax(preds)]
    return predicted_class, preds

# ===============================
# 🔹 Streamlit App
# ===============================
st.set_page_config(page_title="Brain Tumor Classifier", layout="wide")
st.title("🧠 Brain Tumor Classifier (TFLite Lite)")

uploaded_file = st.file_uploader("Upload MRI Brain Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    predicted_class, preds = predict_tflite(image, threshold=0.8)

    if predicted_class is None:
        st.warning("❌ Image rejected: not similar to brain MRI patterns.")
    else:
        st.success(f"✅ Predicted Class: {predicted_class}")

        # 🔹 Display Probabilities Table
        df_probs = pd.DataFrame({
            "Class": class_labels,
            "Probability": np.round(preds, 3)
        }).sort_values(by="Probability", ascending=False)
        st.table(df_probs)

        # 🔹 Display Bar Chart
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(df_probs["Class"], df_probs["Probability"], color="skyblue")
        ax.set_ylim([0,1])
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Probabilities")
        st.pyplot(fig)
