import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Fake Image Detector", page_icon="🕵️", layout="wide")

st.title("🕵️ Fake Image Detector")
st.markdown("Upload an image to test if it's Real or Fake!")

# --- ELA FUNCTION ---
def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    resaved_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, resaved_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image

# --- LOAD MODEL (Matches your Training Code EXACTLY) ---
@st.cache_resource
def load_my_model():
    # 1. Rebuild the VGG16 Base
    base_model = VGG16(
        weights=None, 
        include_top=False, 
        input_shape=(224, 224, 3)
    )
    
    # 2. Rebuild the Classifier Head (The crucial fix!)
    # We use GlobalAveragePooling2D because that is what your train code uses.
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax")
    ])

    # 3. Load the weights from your fine-tuned file
    model_path = r"D:\FakeImageDetection\models\fake_image_detector_finetuned.h5"
    
    try:
        # We load weights (compile=False makes it safer/faster)
        model.load_weights(model_path)
    except OSError:
        st.error("❌ Model file not found! Did you finish training?")
        return None
    except Exception as e:
        # Fallback: Try loading the full model if weights fail
        try:
             from tensorflow.keras.models import load_model
             model = load_model(model_path, compile=False)
        except Exception as e2:
             st.error(f"❌ Critical Error: {e2}")
             return None

    return model

# --- MAIN APP LOGIC ---
model = load_my_model()

if model:
    st.success("✅ Model System Online")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        # Original Image
        image = Image.open(uploaded_file).convert('RGB')
        with col1:
            st.image(image, caption='Original', use_column_width=True)
        
        # ELA Process
        temp_path = "temp_upload.jpg"
        image.save(temp_path)
        ela_image = convert_to_ela_image(temp_path, 90)
        
        with col2:
            st.image(ela_image, caption='ELA Analysis', use_column_width=True)
            
        # Prediction
        if ela_image:
            # Resize to 224x224 (Standard for VGG16)
            ela_image = ela_image.resize((224, 224))
            img_array = np.array(ela_image).flatten() / 255.0
            img_array = img_array.reshape(-1, 224, 224, 3)
            
            prediction = model.predict(img_array)
            
            # Note: In flow_from_directory, classes are usually sorted alphabetically.
            # fake = 0, real = 1
            fake_prob = prediction[0][0] * 100
            real_prob = prediction[0][1] * 100
            
            st.divider()
            
            # Logic to display result
            if real_prob > fake_prob:
                st.success(f"✅ REAL IMAGE DETECTED (Confidence: {real_prob:.1f}%)")
            else:
                st.error(f"🚨 FAKE / TAMPERED IMAGE (Confidence: {fake_prob:.1f}%)")
                st.warning("⚠️ Note: The AI detected high error levels (noise) inconsistent with the rest of the image.")