import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("digit_model.keras", compile=False)

st.title("🤖 AI Digit Recognizer")

uploaded_file = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image")

    image = image.resize((28, 28))
    img = np.array(image)

    img = 255 - img
    img = img / 255.0

    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    digit = np.argmax(prediction)

    st.success(f"Predicted Digit: {digit}")