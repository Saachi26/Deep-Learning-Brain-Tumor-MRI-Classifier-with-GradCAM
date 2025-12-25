import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import requests
from streamlit_lottie import st_lottie

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #0e1117; }
    .block-container { padding-top: 2rem; }

    /* Card Styling */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: #1c212c;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border: 1px solid #2d333b;
    }

    /* Titles */
    h1 { font-family: 'Helvetica Neue', sans-serif; color: #e6e6e6; font-weight: 700; }
    h2, h3 { color: #4CAF50; font-family: 'Helvetica Neue', sans-serif; }

    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.4);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #2d333b;
    }
    
    /* --- STICKY FOOTER CSS --- */
    div[data-testid="stSidebar"] > div:first-child {
        height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---
@st.cache_resource
def load_model():
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    MODEL_PATH = "models/brain_tumor_efficientnet_augmented.pth"
    
    # Ensure model architecture matches training
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, DEVICE

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

model, DEVICE = load_model()
CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
lottie_brain = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_5njp3vgg.json")

# --- 4. SIDEBAR DASHBOARD ---
with st.sidebar:
    # --- TOP CONTENT ---
    with st.container():
        st.image("https://cdn-icons-png.flaticon.com/512/2040/2040946.png", width=80)
        st.title("NeuroScan AI")
        st.markdown("---")
        
        uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])
        
        st.markdown("---")
        st.markdown("""
            <div style="
                background-color: rgba(28, 131, 225, 0.1); 
                border: 1px solid rgba(28, 131, 225, 0.5); 
                padding: 10px; 
                border-radius: 5px; 
                color: rgb(230, 230, 230);
                font-size: 14px;
                white-space: nowrap; /* <--- This forces single-line text */
                overflow: hidden;
                text-overflow: ellipsis;
            ">
                <strong style="display: block; margin-bottom: 8px;">ℹ️ Model Info</strong>
                Architecture: EfficientNet-B0<br>
                Accuracy: 99.39%<br>
                Training: Augmented Data
            </div>
        """, unsafe_allow_html=True)
    # --- BOTTOM FOOTER (Sticky) ---
    st.markdown("---")
    st.markdown("""
<div style="margin-top: 5vh; padding-bottom: 20px;">
<div style="text-align: center; color: #8b949e; font-size: 14px;">
<p style="margin-bottom: 20px;">Created by <b style="color: #e6e6e6;">Saachi Badal</b></p>

<div style="display: flex; justify-content: center; gap: 20px; align-items: center;">
<a href="https://www.linkedin.com/in/saachi-badal/" target="_blank" style="text-decoration: none; opacity: 0.8;">
<img src="https://cdn-icons-png.flaticon.com/512/3536/3536505.png" width="24" style="filter: invert(1);">
</a>
<a href="https://github.com/Saachi26" target="_blank" style="text-decoration: none; opacity: 0.8;">
<img src="https://cdn-icons-png.flaticon.com/512/733/733553.png" width="24" style="filter: invert(1);">
</a>
<a href="mailto:saachibadal@gmail.com" target="_blank" style="text-decoration: none; opacity: 0.8;">
<img src="https://cdn-icons-png.flaticon.com/512/732/732200.png" width="24" style="filter: invert(1);">
</a>
</div>
<p style="margin-top: 20px; font-size: 10px; opacity: 0.4;">© 2025 NeuroScan AI</p>
</div>
</div>
""", unsafe_allow_html=True)

# --- 5. MAIN INTERFACE ---
st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>🧠 AI-Powered Brain Tumor Detection</h1>", unsafe_allow_html=True)

if uploaded_file is not None:
    # --- ANALYSIS MODE ---
    image = Image.open(uploaded_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown("### 📷 Original Scan")
        st.image(image, use_container_width=True, caption="Patient MRI")
        analyze_btn = st.button("🔍 Run Diagnostics", use_container_width=True)

    if analyze_btn:
        with st.spinner('🧬 Analyzing neural patterns...'):
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                conf, pred_idx = torch.max(probs, 1)
            
            prediction = CLASSES[pred_idx.item()]
            confidence = conf.item() * 100

        with col2:
            st.markdown("### 🩺 Diagnostic Results")
            if prediction == "No Tumor":
                result_color = "#4CAF50" # Green
                status_icon = "✅"
            else:
                result_color = "#FF4B4B" # Red
                status_icon = "⚠️"

            st.markdown(f"""
                <div style="background-color: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid {result_color};">
                    <h3 style="color: white; margin:0;">{status_icon} Prediction:</h3>
                    <h1 style="color: {result_color}; margin:0; font-size: 36px;">{prediction}</h1>
                    <p style="color: gray; margin-top: 5px;">Confidence Score: <b>{confidence:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Confidence Breakdown")
            probs_np = probs.cpu().numpy()[0]
            for i, class_name in enumerate(CLASSES):
                prob_val = probs_np[i]
                st.markdown(f"""
                    <div style="margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; color: #e6e6e6; font-size: 14px;">
                            <span>{class_name}</span>
                            <span>{prob_val*100:.1f}%</span>
                        </div>
                        <div style="width: 100%; background-color: #333; border-radius: 5px; height: 8px;">
                            <div style="width: {prob_val*100}%; background-color: #4CAF50; height: 8px; border-radius: 5px;"></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🔍 Explainable AI (Grad-CAM)")
        
        target_layers = [model.conv_head]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(pred_idx.item())]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        rgb_img = np.array(image.resize((224, 224))) / 255.0
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(visualization, caption=f"Heatmap Focus Area ({prediction})", use_container_width=True)

else:
    # --- EMPTY STATE (Animation) ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if lottie_brain:
            st_lottie(lottie_brain, height=300, key="brain")
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/2040/2040946.png", width=200)

    st.markdown("""
        <div style="text-align: center; margin-top: 20px;">
            <h3>👋 Welcome to NeuroScan</h3>
            <p style="color: #888;">
                This tool uses Deep Learning (EfficientNet) to detect brain tumors with 99.4% accuracy.<br>
                Please upload an MRI scan from the <b>sidebar</b> to begin diagnostics.
            </p>
        </div>
    """, unsafe_allow_html=True)