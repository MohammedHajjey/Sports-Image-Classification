import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import base64

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Sports Image Classifier",
    page_icon="🏃",
    layout="wide"
)

# === BASE64 ENCODE BACKGROUND IMAGE ===
def get_base64_bg(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_path = r'C:/Users/User/Desktop/Universty Document/term8/Bulut-bilişim-Yapayzeka/Sports-Image-Classification/background.webp'
bg_base64 = get_base64_bg(bg_path)

# === BACKGROUND IMAGE & MODERN CSS ===
st.markdown(f"""
    <style>
    body, .stApp {{
        background: url('data:image/webp;base64,{bg_base64}') no-repeat center center fixed !important;
        background-size: cover !important;
    }}
    .header-bar {{
        text-align: center;
        padding: 1.2rem 0;
        background: linear-gradient(135deg, #28a5f5 0%, #1f8cd6 100%);
        border-radius: 0 0 18px 18px;
        color: white;
        margin-bottom: 2rem;
        font-size: 2rem;
        font-weight: bold;
        letter-spacing: 1px;
    }}
    .footer-bar {{
        text-align: center;
        color: #6c757d;
        font-size: 1rem;
        padding: 1.2rem 0 0.5rem 0;
    }}
    .result-title {{
        color: #28a5f5;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.7rem;
        text-shadow: 0 2px 8px #0002;
    }}
    .confidence-label {{
        color: #6c757d;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 0.3rem;
    }}
    .confidence-value {{
        color: #28a5f5;
        font-size: 1.3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }}
    .confidence-bar {{
        width: 100%;
        height: 12px;
        background: #181c20cc;
        border-radius: 6px;
        margin-bottom: 0.7rem;
        overflow: hidden;
    }}
    .confidence-bar-inner {{
        height: 100%;
        background: linear-gradient(90deg, #28a5f5 0%, #1f8cd6 100%);
        border-radius: 6px;
        transition: width 0.5s;
    }}
    .upload-btn-custom {{
        width: 100%;
        background: linear-gradient(135deg, #28a5f5 0%, #1f8cd6 100%);
        color: white;
        border: none;
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.2s;
        margin-bottom: 0.5rem;
    }}
    .upload-btn-custom:hover {{
        background: linear-gradient(135deg, #1f8cd6 0%, #28a5f5 100%);
        transform: translateY(-2px) scale(1.03);
        box-shadow: 0 4px 12px rgba(40,165,245,0.15);
    }}
    </style>
""", unsafe_allow_html=True)

# === MODEL & CLASS NAMES ===
@st.cache_resource
def load_trained_model():
    model_path = 'best_sports_model.keras'
    return load_model(model_path)

@st.cache_data
def get_class_names():
    return sorted(os.listdir(r'C:/Users/User/Desktop/Universty Document/term8/Bulut-bilişim-Yapayzeka/Sports-Image-Classification/train'))

model = load_trained_model()
class_names = get_class_names()

# === HEADER ===
st.markdown('<div class="header-bar">🏃 Sports Image Classifier</div>', unsafe_allow_html=True)

# === MAIN LAYOUT ===
left, right = st.columns([1, 1], gap="large")

with left:
    uploaded_image = None
    if 'uploaded_image' in st.session_state:
        uploaded_image = st.session_state['uploaded_image']

    # ✅ LABEL BOŞ VERİLMEK YERİNE ANLAMLI BİR METİNLE GİZLENDİ
    upload_file = st.file_uploader(
        "Upload a sports image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
        key="uploader"
    )

    if upload_file is not None:
        image = Image.open(upload_file)
        st.image(image, width=350)
        uploaded_image = image
    elif uploaded_image is not None:
        st.image(uploaded_image, width=350)

    start_btn_clicked = st.button("Start Prediction", key="upload_btn", help="Start prediction for the uploaded image")
    
    # Butona stil ekle (custom class eklemek için JS hilesi)
    st.markdown("""
        <script>
        const btn = window.parent.document.querySelector('button[data-testid="baseButton-upload_btn"]');
        if(btn){ btn.classList.add('upload-btn-custom'); }
        </script>
    """, unsafe_allow_html=True)

    if start_btn_clicked and upload_file is not None:
        st.session_state['uploaded_image'] = image
        img = image.resize((350, 350))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array, verbose=0)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        st.session_state['prediction'] = (predicted_class, confidence)

with right:
    prediction = st.session_state.get('prediction', None)
    if prediction is not None:
        predicted_class, confidence = prediction
        st.markdown(f'<div class="result-title">{predicted_class}</div>', unsafe_allow_html=True)
        st.markdown('<div class="confidence-label">Confidence</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence-value">{confidence*100:.2f}%</div>', unsafe_allow_html=True)
        st.markdown(f'''<div class="confidence-bar"><div class="confidence-bar-inner" style="width: {confidence*100}%;"></div></div>''', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-title" style="color:#6c757d;">No prediction yet</div>', unsafe_allow_html=True)
        st.markdown('<div class="confidence-label">Confidence</div>', unsafe_allow_html=True)
        st.markdown('<div class="confidence-value">-</div>', unsafe_allow_html=True)
        st.markdown(f'''<div class="confidence-bar"><div class="confidence-bar-inner" style="width: 0%;"></div></div>''', unsafe_allow_html=True)

# === FOOTER ===
st.markdown('<div class="footer-bar">&copy; 2024 Sports Image Classifier. All rights reserved.</div>', unsafe_allow_html=True)
