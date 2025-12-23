# 🕵️ Fake Image Detection using ELA & VGG16

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

A Deep Learning project that detects **digital image tampering** (splicing, copy-move manipulation) using **Error Level Analysis (ELA)** and a fine-tuned **VGG16 Convolutional Neural Network**.

## 📌 Project Overview
Digital image forgery is becoming increasingly sophisticated. This tool helps identify manipulated images by analyzing compression artifacts.
1.  **ELA Preprocessing:** The image is re-compressed at a specific quality (90%) and subtracted from the original. This highlights "alien" pixels that have different compression levels than the background.
2.  **CNN Classification:** A VGG16 model (pre-trained on ImageNet) analyzes the ELA noise map to classify the image as **Real** or **Fake**.

## 🚀 Features
* **Real-time Detection:** Upload an image and get an instant verdict (Real vs. Fake).
* **Visual Analysis:** Displays the ELA (Error Level Analysis) view alongside the original image, showing exactly *where* the tampering might have happened.
* **Confidence Score:** Provides a percentage probability for the prediction.
* **User-Friendly Interface:** Built with Streamlit for easy drag-and-drop testing.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow / Keras (VGG16 Architecture)
* **Web Framework:** Streamlit
* **Image Processing:** Pillow (PIL), NumPy

## 📂 Project Structure
```text
Fake-Image-Detection/
│
├── app.py                  # Main Streamlit Web Application
├── train_full.py           # Script to train the VGG16 model
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
│
├── dataset_ela/            # (Not included in repo) Preprocessed ELA images
└── models/                 # (Not included in repo) Saved .h5 model files