# ────────────────────────────────────────────────
# 🔬 Blood Cell Classifier with ResNet34 + Grad-CAM
# ✨ Built using PyTorch + Gradio
# ────────────────────────────────────────────────

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.transforms.functional import to_pil_image

from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torch.nn.functional import softmax

from PIL import Image
import gradio as gr

# ────────────────────────────────────────────────
# 1️⃣  Setup and Configuration
# ────────────────────────────────────────────────

# Path to the trained model
MODEL_PATH = "models/best_resnet34_final.pth"

# Class names (ensure they match training order!)
CLASSES = ['eosinophil', 'erythroblast', 'lymphocyte', 'neutrophil', 'platelet']

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ────────────────────────────────────────────────
# 2️⃣  Load Pretrained ResNet34 Model
# ────────────────────────────────────────────────

# Load ResNet34 with ImageNet weights
model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

# Replace final FC layer for 5-class classification
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

# Load saved model weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device).eval()

# ────────────────────────────────────────────────
# 3️⃣  Define Image Preprocessing
# ────────────────────────────────────────────────

# Same transform used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ────────────────────────────────────────────────
# 4️⃣  Grad-CAM Setup
# ────────────────────────────────────────────────

# Grad-CAM extractor targeting final conv block of ResNet34
cam_extractor = GradCAM(model, target_layer="layer4")

# ────────────────────────────────────────────────
# 5️⃣  Inference Function (image → prediction + Grad-CAM)
# ────────────────────────────────────────────────

def predict(image):
    """
    Runs model inference on uploaded image and returns:
    - Predicted class name
    - Confidence score
    - Grad-CAM visualization
    """
    # Preprocess input image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Run model forward pass
    output = model(img_tensor)

    # Get class probabilities via softmax
    prob = softmax(output, dim=1)

    # Get top predicted class and confidence
    conf_score, pred_class = torch.max(prob, 1)
    class_name = CLASSES[pred_class.item()]
    confidence = conf_score.item()

    # Grad-CAM activation map for predicted class
    activation_map = cam_extractor(pred_class.item(), output)[0].cpu()

    # Convert input tensor back to PIL image for display
    original_img = to_pil_image(img_tensor.squeeze().cpu() * 0.5 + 0.5)

    # Overlay Grad-CAM map on image
    cam_overlay = overlay_mask(original_img, to_pil_image(activation_map), alpha=0.5)

    return class_name, f"{confidence:.4f}", cam_overlay

# ────────────────────────────────────────────────
# 6️⃣  Build Gradio Interface
# ────────────────────────────────────────────────

title = "🧪 WBC Classifier with ResNet34 + Grad-CAM"
description = "Upload a blood cell image to classify it into 5 categories and see a heatmap using Grad-CAM."

# Gradio UI setup
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Cell Image"),
    outputs=[
        gr.Label(num_top_classes=1, label="Predicted Class"),
        gr.Textbox(label="Confidence Score"),
        gr.Image(label="Grad-CAM Heatmap")
    ],
    title=title,
    description=description,
    allow_flagging="never"
)

# ────────────────────────────────────────────────
# 7️⃣  Launch App
# ────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch()
