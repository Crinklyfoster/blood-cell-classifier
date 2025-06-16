# 🧬 White Blood Cell Classifier with Grad-CAM

This project uses a ResNet34 model trained on microscopic blood cell images to classify:
- Eosinophil
- Lymphocyte
- Neutrophil
- Platelet
- Erythroblast

Built with **PyTorch** and **Gradio**, deployed on Hugging Face Spaces.

## 🚀 Try it now!
Upload a blood cell image and get:
- 🔍 Predicted class
- 🎯 Confidence score
- 🧠 Grad-CAM heatmap visualization

## 🛠️ Run Locally

```bash
pip install -r requirements.txt
python app.py
