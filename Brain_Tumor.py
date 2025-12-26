# app.py
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import json

# -----------------------------
# 1️⃣ تحميل النموذج وملف الفئات
# -----------------------------
model = load_model("brain_tumor_model.h5")

with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# استخراج حجم الإدخال من النموذج تلقائيًا
input_shape = model.input_shape[1:3]  # (height, width)

# -----------------------------
# 2️⃣ دالة المعالجة المسبقة
# -----------------------------
def preprocess_image(image_file, target_size=input_shape):
    """
    تقرأ الصورة، تحولها إلى RGB، تغير حجمها لتتناسب مع النموذج،
    وتعيدها على شكل numpy array مع batch dimension.
    """
    image = Image.open(image_file).convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # إضافة batch dimension
    return image_array

# -----------------------------
# 3️⃣ واجهة Streamlit
# -----------------------------
st.title("Brain Tumor Classification")
st.write("Upload any brain MRI image and the model will predict the tumor type.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # معالجة الصورة
    processed_image = preprocess_image(uploaded_file)
    
    # عرض الصورة
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # التنبؤ
    predictions = model.predict(processed_image, verbose=0)[0]
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    # عرض النتيجة
    st.markdown(f"### Predicted Class: **{predicted_class}**")
    st.markdown(f"### Confidence: **{confidence:.2f}%**")
