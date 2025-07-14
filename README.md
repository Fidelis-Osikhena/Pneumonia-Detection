# Pneumonia Detection from Chest X-ray Images using Deep Learning

This repository contains the code and data pipeline for a deep learning-based pneumonia detection project. It compares transfer learning and custom CNNs, and evaluates model explainability using Grad-CAM.

## 🧪 Project Overview

**Objective:**  
To investigate whether deep learning models can accurately classify chest X-ray images as:
- Normal
- Pneumonia
- Other lung diseases

It also evaluates:
- The performance impact of transfer learning using ResNet50
- The interpretability of predictions via Grad-CAM heatmaps

---

## ✅ Project Stages

1. **Data Preparation**
   - Dataset: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
   - Preprocessing: Normalization, augmentation, and generator setup

2. **Model Development**
   - Transfer Learning: ResNet50 pretrained on ImageNet
   - Scratch Model: Custom CNN built from basic layers

3. **Training and Evaluation**
   - Both models trained and validated for 5 epochs
   - Evaluation using classification reports (precision, recall, F1-score)

4. **Explainability with Grad-CAM**
   - Grad-CAM visualizations highlight areas influencing model predictions

---

## 📊 Key Findings

- Transfer learning with ResNet50 performs better when fine-tuned appropriately.
- Grad-CAM highlights medically relevant regions, aiding model interpretability.
- Scratch models may initially appear competitive, but underperform over time.


---

## 📌 Dependencies

- Python 3.8+
- TensorFlow 2.x
- NumPy, Matplotlib, Seaborn
- scikit-learn
- OpenCV (for image overlays, optional)
