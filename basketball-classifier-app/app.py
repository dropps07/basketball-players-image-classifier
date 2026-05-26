import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import base64
from io import BytesIO

# ── Page config ────────────────────────────────────────────
st.set_page_config(page_title="Basketball Legend Classifier", layout="wide")

# ── Load model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# ── Class names — must match your training folder order ────
CLASS_NAMES = ['kobe_bryant', 'lebron_james', 'michael_jordan', 'shaquille_oneal', 'stephen_curry']

PLAYER_EMOJI = {
    'kobe_bryant':      '🐍',
    'lebron_james':     '👑',
    'michael_jordan':   '🐐',
    'shaquille_oneal':  '💪',
    'stephen_curry':    '🎯',
}

# ── CSS ────────────────────────────────────────────────────
page_style = """
<style>
    .stApp {
        background-image: url("https://wallpapercave.com/wp/wp7614422.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }
    .glass-header {
        backdrop-filter: blur(10px);
        background-color: rgba(0, 0, 0, 0.45);
        padding: 1.5rem 1rem;
        border-radius: 25px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    .glass-header h1 {
        font-size: 2.5rem;
        font-weight: 900;
        margin-bottom: 0.2rem;
        color: white;
    }
    .glass-header p {
        font-size: 1rem;
        margin-top: 0.5rem;
        color: #f0a500;
    }
    .instruction {
        background-color: rgba(0, 0, 0, 0.5);
        border-radius: 20px;
        padding: 1rem 1.5rem;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .player-name {
        position: absolute;
        top: 12px;
        left: 12px;
        background: rgba(255, 165, 0, 0.9);
        padding: 6px 14px;
        font-weight: bold;
        border-radius: 10px;
        color: black;
        font-size: 1rem;
    }
    .img-wrapper {
        position: relative;
        display: inline-block;
        border-radius: 15px;
        overflow: hidden;
        border: 3px solid #f0a500;
        margin-top: 20px;
    }
    .confidence {
        background-color: rgba(0,0,0,0.6);
        color: #f0a500;
        border-radius: 12px;
        padding: 0.5rem 1.2rem;
        margin-top: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        display: inline-block;
    }
</style>
"""

st.markdown(page_style, unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────
st.markdown("""
<div class="glass-header">
    <h1>🏀 Basketball Legend Classifier</h1>
    <p>Can the AI tell your GOAT from another? Upload a photo and find out.</p>
</div>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────
if 'predicted' not in st.session_state:
    st.session_state.predicted = False

# ── Upload / Sample selection screen ──────────────────────
if not st.session_state.predicted:
    st.markdown("""
    <div class="instruction">
        Upload a photo of <strong>Kobe, LeBron, Jordan, Shaq or Curry</strong> — or pick a sample below.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    sample_folder = "sample_images"
    sample_images = [f for f in os.listdir(sample_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if uploaded_file is None and sample_images:
        st.markdown("### Or pick a sample image:")
        cols = st.columns(len(sample_images))
        for i, img_name in enumerate(sample_images):
            img_path = os.path.join(sample_folder, img_name)
            img = Image.open(img_path).resize((250, 250))
            cols[i].image(img, use_container_width=True)
            if cols[i].button(f"Select", key=img_name):
                st.session_state['selected_image'] = img_path
                st.session_state.predicted = True
                st.rerun()

    elif uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB").resize((250, 250))
        st.session_state['selected_image'] = img
        st.session_state.predicted = True
        st.rerun()

# ── Result screen ──────────────────────────────────────────
else:
    img = st.session_state['selected_image']
    if isinstance(img, str):
        img = Image.open(img).convert("RGB").resize((250, 250))

    # preprocess
    img_model  = img.resize((100, 100))
    img_array  = np.array(img_model) / 255.0
    img_array  = np.expand_dims(img_array, axis=0)

    prediction     = model.predict(img_array)
    top_idx        = int(np.argmax(prediction))
    predicted_key  = CLASS_NAMES[top_idx]
    predicted_name = predicted_key.replace('_', ' ').title()
    confidence     = prediction[0][top_idx] * 100
    emoji          = PLAYER_EMOJI[predicted_key]

    # encode image for HTML display
    buffered    = BytesIO()
    img.save(buffered, format="PNG")
    img_base64  = base64.b64encode(buffered.getvalue()).decode()

    # result layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div style="text-align:center">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="img-wrapper" style="margin: 0 auto;">
            <div class="player-name">{emoji} {predicted_name}</div>
            <img src="data:image/png;base64,{img_base64}" width="300"/>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="confidence">Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### All predictions:")
        for i, name in enumerate(CLASS_NAMES):
            conf = prediction[0][i] * 100
            display = name.replace('_', ' ').title()
            emp = PLAYER_EMOJI[name]
            st.progress(int(conf), text=f"{emp} {display}: {conf:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn = st.columns([1, 1, 1])[1]
    with col_btn:
        if st.button("🔄 Try Another Image", use_container_width=True):
            st.session_state.predicted = False
            st.session_state.selected_image = None
            st.rerun()