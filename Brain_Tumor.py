# brain_tumor_app.py
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import json

# ==========================
# 1️⃣ إعداد التطبيق
# ==========================
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("Brain Tumor Detection 🧠")
st.write("Upload an MRI image, and the model will predict the type of brain tumor.")

# ==========================
# 2️⃣ تحميل النموذج والفئات
# ==========================
model = load_model("brain_tumor_model.h5")
with open("class_labels.json", "r") as f:
    class_labels = json.load(f)

# ==========================
# 3️⃣ دالة تجهيز الصورة
# ==========================
def preprocess_image(uploaded_file, target_size=(299, 299)):
    """
    تقوم بتحويل الصورة إلى RGB، إعادة تحجيمها، تقسيم قيم البكسل على 255،
    وإضافة بعد الـ batch لتكون جاهزة للنموذج.
    """
    image = Image.open(uploaded_file)

    # تحويل الصورة إلى RGB إذا لم تكن
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # إعادة التحجيم لتتناسب مع النموذج
    image = image.resize(target_size)

    # تحويل الصورة إلى array وتقسيمها على 255
    img_array = np.array(image) / 255.0

    # إضافة بعد الـ batch
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ==========================
# 4️⃣ رفع الصورة وعرض النتيجة
# ==========================
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # تجهيز الصورة
    processed_image = preprocess_image(uploaded_file)

    # عرض الصورة
    st.image(processed_image[0], caption="Uploaded Image", use_column_width=True)

    # التنبؤ
    predictions = model.predict(processed_image, verbose=0)
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    # عرض النتيجة
    st.success(f"Predicted Tumor Type: {predicted_label}")
    st.info(f"Confidence: {confidence:.2f}%")

# ==========================
# 5️⃣ ملاحظات
# ==========================
st.write("""
**Notes:**  
- This app automatically resizes any uploaded image to 299x299 pixels for the Xception model.  
- Ensure images are clear MRI scans for accurate predictions.
""")
