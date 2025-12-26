import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json

# ==========================
# تحميل النموذج والفئات
# ==========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("brain_tumor_model.h5")
    with open("class_labels.json", "r") as f:
        class_labels = json.load(f)
    return model, class_labels

model, class_labels = load_model()

# ==========================
# دالة لمعالجة الصورة لأي حجم
# ==========================
def preprocess_image(image: Image.Image):
    # تحويل الصورة إلى RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # تحويل الصورة إلى مصفوفة NumPy
    img_array = np.array(image) / 255.0
    # إضافة بُعد الدفعة
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==========================
# واجهة Streamlit
# ==========================
st.title("Brain Tumor Classification")
st.write("ارفع صورة MRI لأي حجم ليتم التنبؤ بالورم")

uploaded_file = st.file_uploader("اختر صورة", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    processed_image = preprocess_image(image)
    
    # التنبؤ
    predictions = model.predict(processed_image)
    pred_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    
    st.success(f"Predicted Class: {pred_class} ({confidence:.2f}%)")

