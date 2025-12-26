# brain_tumor_app.py

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json

# ==========================
# إعدادات الصفحة
# ==========================
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor Classifier")

# ==========================
# تحميل النموذج والفئات
# ==========================
@st.cache_resource
def load_model_and_labels():
    model = load_model("brain_tumor_model.h5")
    with open("class_labels.json", "r") as f:
        class_labels = json.load(f)
    return model, class_labels

model, class_labels = load_model_and_labels()

# ==========================
# دالة المعالجة المسبقة للصورة
# ==========================
def preprocess_image(uploaded_file, target_size=(299, 299)):
    if uploaded_file is None:
        return None

    try:
        # تحويل الصورة لـ RGB وضبط الحجم
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize(target_size)
        
        # تحويل لمصفوفة numpy وتطبيع القيم
        image_array = np.array(image) / 255.0
        
        # إضافة بعد batch
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# ==========================
# واجهة المستخدم
# ==========================
uploaded_file = st.file_uploader("Upload a Brain MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # معالجة الصورة
    processed_image = preprocess_image(uploaded_file)
    
    if processed_image is not None:
        # توقع النموذج
        predictions = model.predict(processed_image, verbose=0)
        pred_index = np.argmax(predictions[0])
        pred_label = class_labels[pred_index]
        confidence = predictions[0][pred_index] * 100
        
        # عرض النتيجة
        st.markdown(f"### Prediction: **{pred_label}**")
        st.markdown(f"### Confidence: **{confidence:.2f}%**")
