import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import cv2

# Load model
model = tf.keras.models.load_model("digit_model.keras", compile=False)

st.set_page_config(page_title="AI Digit Studio", page_icon="✍️", layout="centered")

# ================= UI =================
st.markdown("""
<style>
body {
    background-color: #0b1220;
}

.title {
    text-align: center;
    font-size: 40px;
    color: #00f5d4;
    font-weight: 800;
}

.card {
    background: #111827;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #00f5d4, #3b82f6);
    color: black;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown('<div class="title">✍️ AI Digit Drawer</div>', unsafe_allow_html=True)

# MODE SELECT
mode = st.radio("Choose Input Method:", ["✍️ Draw Digit", "📤 Upload Image"])

img = None

# ================= DRAW MODE =================
if mode == "✍️ Draw Digit":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if canvas_result.image_data is not None:
        img = canvas_result.image_data

# ================= UPLOAD MODE =================
else:
    uploaded_file = st.file_uploader("Upload digit image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)

        st.image(img, caption="Input Image", width=200)

# ================= PREDICTION =================
if img is not None:

    if st.button("🚀 Predict Digit"):

        st.write("🧠 AI Processing...")

        # convert to grayscale
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

        # resize
        img = cv2.resize(img, (28, 28))

        # invert (IMPORTANT for drawing)
        img = 255 - img

        # normalize
        img = img / 255.0

        img = img.reshape(1, 28, 28, 1)

        prediction = model.predict(img)
        digit = np.argmax(prediction)

        st.success(f"🎯 Predicted Digit: {digit}")

else:
    st.info("✍️ Draw or upload a digit first")

# FOOTER
st.markdown("""
<div style="text-align:center; color:#64748b; margin-top:30px;">
Built by <b>bhatmoshin</b> • <b>amina ahad</b> • <b>huzaifa nazir</b>
</div>
""", unsafe_allow_html=True)
