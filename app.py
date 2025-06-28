import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications import EfficientNetV2B2

# Page setup
st.set_page_config(page_title="♻️ Garbage Classifier", layout="centered")

# Load the trained EfficientNetV2B2 model
model = tf.keras.models.load_model(
    "garbage_classifier_efficientnetv2b2.keras",
    custom_objects={'EfficientNetV2B2': EfficientNetV2B2}
)

# Class labels and icons
labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
icons = {
    'cardboard': '📦',
    'glass': '🥛',
    'metal': '🛢️',
    'paper': '📄',
    'plastic': '🧴',
    'trash': '🗑️'
}

# Title
st.markdown("""
    <div style="text-align: center;">
        <h1>♻️ Garbage Classifier</h1>
        <p style="font-size:18px;">Upload a waste image or use your webcam to classify trash types!</p>
    </div>
""", unsafe_allow_html=True)

# Upload or capture section
st.markdown("### 📤 Upload an image or take a photo")

uploaded_file = st.file_uploader("🖼️ Choose an image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("📷 Or take a live photo")

# Prefer camera if both are used
if camera_image:
    uploaded_file = camera_image

# If an image is available
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Clearer and larger display
    st.markdown("### 🔍 Preview of Uploaded Image")
    st.image(image, caption="✅ Clear View", width=450)

    # Preprocess for EfficientNetV2B2
    image = image.resize((224, 224))
    img_array = np.array(image)  # ✅ Do NOT normalize again
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds)
    pred_label = labels[pred_idx]
    confidence = np.max(preds) * 100

    # Result
    st.markdown(f"""
        <div style="text-align: center; padding: 20px; border-radius: 10px;
                    background-color: #f1f3f6; margin-top: 20px;">
            <h2 style="font-size: 28px;">{icons[pred_label]} Predicted: <span style="color: #4CAF50;">{pred_label.capitalize()}</span></h2>
            <p style="font-size: 18px;">Confidence: <strong>{confidence:.2f}%</strong></p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.info("⬆️ Please upload an image or take a photo to classify")
