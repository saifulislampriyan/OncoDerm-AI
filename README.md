#  DermaVision-X: AI-Powered Skin Cancer Classification

> End-to-end deep learning system for multi-class skin cancer detection using VGG16, featuring training, evaluation, and real-time deployment via Streamlit.

---

##  Overview

DermaVision-X is a deep learning–based dermatological diagnosis system designed to classify skin lesions into multiple cancer categories using medical image analysis.  
The project implements a complete pipeline from **data preprocessing → model training → evaluation → deployment**, enabling real-time predictions through an interactive web interface.

---

##  Key Features

- 🔍 Multi-class Skin Lesion Classification (9 classes)  
- 🧠 Transfer Learning with VGG16 (ImageNet pretrained)  
- ⚙️ Data Augmentation for Robust Training  
- 📊 Model Evaluation (Confusion Matrix + Classification Report)  
- 🌐 Streamlit Web App for Real-Time Prediction  
- 🧪 Modular and Reproducible Pipeline  

---

##  Project Structure


---

##  Model Architecture

- Base Model: VGG16 (pretrained on ImageNet)  
- Custom Layers:
  - Flatten  
  - Dense (256, ReLU)  
  - Dropout (0.5)  
  - Softmax Output Layer  

---

##  Dataset Classes

- Actinic Keratosis  
- Basal Cell Carcinoma  
- Dermatofibroma  
- Melanoma  
- Nevus  
- Pigmented Benign Keratosis  
- Seborrheic Keratosis  
- Squamous Cell Carcinoma  
- Vascular Lesion  

---

