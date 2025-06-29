import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Sidebar: show TensorFlow version
st.sidebar.write(f"TensorFlow version: {tf.__version__}")

# Load model with error handling
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("garbage_classifier_efficientnetv2b2.keras")
        st.sidebar.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        return None

model = load_model()
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Title and instructions
st.title("🗑️ Garbage Classifier with Camera")
st.write("Upload an image or take a photo to classify the type of garbage.")

# Upload OR Camera input
uploaded_file = st.file_uploader("📂 Upload an image...", type=["jpg", "jpeg", "png"])
camera_file = st.camera_input("📸 Or take a photo")
image_input = camera_file if camera_file is not None else uploaded_file

# Prediction
if image_input is not None and model is not None:
    img = Image.open(image_input).convert("RGB")
    st.image(img, caption="Input Image", use_column_width=True)

    with st.spinner("🔍 Analyzing..."):
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions)
        predicted_class = class_names[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])

    st.markdown(f"### 🧠 Prediction: **{predicted_class.title()}**")
    st.markdown(f"**Confidence:** {confidence:.2%}")

    if confidence < 0.75:
        st.warning("🤔 Confidence is low. This might be a mixed or unclear image.")
        st.info("Tip: Make sure the image is clear and focused.")

    # Show all class probabilities
    st.markdown("### 📊 All Class Probabilities:")
    for name, prob in zip(class_names, predictions[0]):
        st.write(f"**{name.title()}**: {prob:.2%}")

    # Feedback form
    with st.expander("📝 Give Feedback"):
        feedback = st.text_area("Was the prediction correct? If not, tell us the correct type or leave a suggestion.")
        if st.button("Submit Feedback"):
            st.success("💌 Thank you for your feedback!")

elif image_input is not None and model is None:
    st.error("⚠️ Model could not be loaded. Please check logs.")

# About Section
with st.expander("ℹ️ About this App"):
    st.markdown("""
    This app uses a pre-trained **EfficientNetV2-B2** model to classify waste into six categories:
    
    - 📦 Cardboard
    - 🥃 Glass
    - 🔩 Metal
    - 📄 Paper
    - 🥤 Plastic
    - 🗑️ Trash

    Built with ❤️ by Sampu using TensorFlow and Streamlit.
    """)

