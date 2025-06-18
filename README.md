# 🧪 Blood Cell Classifier using ResNet34 + Grad-CAM

A deep learning-based classifier for detecting types of blood cells from microscope images. Built with **PyTorch**, explained using **Grad-CAM**, and deployed with **Streamlit**.

---

## 🔍 Project Overview

This project aims to **automatically classify white blood cells (WBCs)** and other blood components using image data. The system uses a convolutional neural network (ResNet34) trained to distinguish between 5 types of blood cells:

- 🧪 Eosinophil
- 🧠 Erythroblast
- 🧬 Lymphocyte
- 🧲 Neutrophil
- 🧫 Platelet

It also visualizes **model attention** using Grad-CAM to enhance transparency and trust in predictions.

---

## 🧠 Model Details

- **Architecture**: ResNet34 (pretrained on ImageNet)
- **Framework**: PyTorch
- **Training Setup**:
  - Stratified 5-Fold Cross-Validation
  - Weighted CrossEntropyLoss
  - Adam optimizer + ReduceLROnPlateau scheduler
  - Data augmentations via `torchvision.transforms`
- **Explainability**: Grad-CAM (`torchcam`) for class activation visualization

---

## 🧪 Try it Live (Streamlit App)

🚀 **Streamlit Live App**:  
[🔗 Click here to launch the app](https://blood-cell-classifier-4dkswvwtf8imuyk8hef2z8.streamlit.app/)

- Upload any microscope image of a blood cell.
- Get the predicted label with confidence.
- See a **Grad-CAM heatmap** overlaid on the input image.

---

## 📊 Sample Grad-CAM Result

| Uploaded Image | Grad-CAM Overlay |
|----------------|------------------|
| ![Input](./assets/sample_cell.jpg) | ![GradCAM](./assets/sample_gradcam.jpg) |

> *The heatmap shows where the model focused while predicting.*

---

## 📁 Folder Structure

