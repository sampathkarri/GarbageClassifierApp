import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load the saved Keras model
model = tf.keras.models.load_model("garbage_classifier_efficientnetv2b2.keras")

# Class labels
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

st.set_page_config(page_title="Garbage Classifier", layout="centered")

st.title("♻️ AI Garbage Classification System")
st.markdown("Upload a garbage image and I’ll tell you what type it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.markdown(f"### 🧠 Predicted: **{class_names[class_index].capitalize()}**")
    st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
