import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
# Path to your trained model
model_path = r"D:\FakeImageDetection\models\fake_image_detector_full.h5"

# Path to the image you want to test
# (Replace this with any image path from your computer)
image_path = r"D:\FakeImageDetection\dataset\fake\Tp_D_NRN_S_N_ani10171_ani00001_12458.jpg"

# --- 1. ELA CONVERSION FUNCTION ---
# This must match exactly what we used for training
def convert_to_ela_image(path, quality):
    try:
        im = Image.open(path).convert('RGB')
        
        # Save to memory buffer to compress
        from io import BytesIO
        buffer = BytesIO()
        im.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        resaved_im = Image.open(buffer)
        
        # Calculate difference
        ela_im = ImageChops.difference(im, resaved_im)
        
        # Enhance brightness
        extrema = ela_im.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
        
        return ela_im
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# --- 2. LOAD MODEL ---
print("⏳ Loading the model... (This may take a moment)")
model = load_model(model_path)

# --- 3. PREPARE IMAGE ---
print(f"🔍 Inspecting: {image_path}")
ela_img = convert_to_ela_image(image_path, 90)

if ela_img:
    # Resize to match the training input (224x224)
    # If you changed image_size in training, change it here too!
    ela_img_resized = ela_img.resize((224, 224))
    
    # Convert to array and normalize (0-1)
    img_array = np.array(ela_img_resized).flatten() / 255.0
    img_array = img_array.reshape(-1, 224, 224, 3)
    
    # --- 4. PREDICT ---
    prediction = model.predict(img_array)
    
    # Class 0 = Fake, Class 1 = Real (Based on alphabetical folder order)
    # We look at the probabilities
    fake_confidence = prediction[0][0] * 100
    real_confidence = prediction[0][1] * 100
    
    print("\n" + "="*40)
    print("           RESULT           ")
    print("="*40)
    
    label = ""
    if real_confidence > fake_confidence:
        label = "REAL"
        print(f"✅ Verdict: REAL Image")
        print(f"📊 Confidence: {real_confidence:.2f}%")
    else:
        label = "FAKE"
        print(f"🚨 Verdict: FAKE / TAMPERED Image")
        print(f"📊 Confidence: {fake_confidence:.2f}%")
    print("="*40 + "\n")

    # --- 5. VISUALIZE (Bonus) ---
    # Show the Original and the ELA version side-by-side
    plt.figure(figsize=(10, 5))
    
    # Original
    plt.subplot(1, 2, 1)
    original = Image.open(image_path)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis('off')
    
    # ELA (What the AI sees)
    plt.subplot(1, 2, 2)
    plt.imshow(ela_img)
    plt.title(f"ELA View (AI sees: {label})")
    plt.axis('off')
    
    plt.show()

else:
    print("❌ Could not load image. Check the path.")