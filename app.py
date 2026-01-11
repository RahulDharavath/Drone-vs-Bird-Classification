import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
from keras.applications.mobilenet_v2 import preprocess_input

# ------------------------------------
# Streamlit Page Config
# ------------------------------------
st.set_page_config(
    page_title="Drone vs Bird Classifier",
    page_icon="🚁",
    layout="centered"
)

# ------------------------------------
# Load Model (Cached)
# ------------------------------------
@st.cache_resource
def load_model():
    model = keras.models.load_model(
        "Drone_Vs_Bird_Model.keras",
        compile=False
    )
    return model

model = load_model()

# ------------------------------------
# Constants
# ------------------------------------
IMG_SIZE = (224, 224)
THRESHOLD = 0.5  # sigmoid threshold

# ------------------------------------
# Image Preprocessing (MATCHES TRAINING)
# ------------------------------------
def prepare_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)

# ------------------------------------
# UI
# ------------------------------------
st.title("🚁 Drone vs 🦅 Bird Classification")
st.markdown("Upload an image to classify it as **Drone** or **Bird**")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# ------------------------------------
# Prediction
# ------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    if st.button("Predict 🤖"):
        with st.spinner("Analyzing image..."):
            img = prepare_image(image)
            prediction = model.predict(img, verbose=0)[0][0]

            if prediction >= THRESHOLD:
                label = "🚁 Drone"
                confidence = prediction
            else:
                label = "🦅 Bird"
                confidence = 1 - prediction

        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: **{confidence:.2%}**")

