# 🧪 Blood Cell Classifier with ResNet34 + Grad-CAM using Streamlit

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.transforms.functional import to_pil_image

from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask

import streamlit as st
from PIL import Image
import io

# ────────────────────────────────────────────────
# 1️⃣  Setup
# ────────────────────────────────────────────────
st.set_page_config(page_title="Blood Cell Classifier", layout="centered")

CLASSES = ['eosinophil', 'erythroblast', 'lymphocyte', 'neutrophil', 'platelet']
MODEL_PATH = "models/best_resnet34_final.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────
# 2️⃣  Load Model
# ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    return model.to(device).eval()

model = load_model()
cam_extractor = GradCAM(model, target_layer="layer4")

# ────────────────────────────────────────────────
# 3️⃣  Image Transform
# ────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ────────────────────────────────────────────────
# 4️⃣  Prediction Function
# ────────────────────────────────────────────────
def classify(image):
    tensor = transform(image).unsqueeze(0).to(device)
    output = model(tensor)
    probs = F.softmax(output, dim=1)
    pred = probs.argmax(1).item()
    confidence = probs[0][pred].item()
    activation = cam_extractor(pred, output)[0].cpu()

    # Unnormalize for visualization
    original = to_pil_image(tensor.squeeze().cpu() * 0.5 + 0.5)
    cam_overlay = overlay_mask(original, to_pil_image(activation), alpha=0.5)
    return CLASSES[pred], confidence, cam_overlay

# ────────────────────────────────────────────────
# 5️⃣  Streamlit UI
# ────────────────────────────────────────────────
st.title("🔬 WBC Classifier with Grad-CAM")
st.caption("Upload a blood cell image to classify it into one of 5 categories and visualize the Grad-CAM activation map.")

uploaded = st.file_uploader("📤 Upload Blood Cell Image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="📸 Uploaded Image", use_column_width=True)

    with st.spinner("🔎 Classifying..."):
        label, confidence, heatmap = classify(img)

    st.success(f"🧠 **Prediction:** {label} ({confidence:.2%} confidence)")
    st.image(heatmap, caption="🔥 Grad-CAM Heatmap", use_column_width=True)
