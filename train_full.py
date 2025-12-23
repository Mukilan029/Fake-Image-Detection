import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

# =========================
# CONFIGURATION
# =========================
DATASET_PATH = r"D:\FakeImageDetection\dataset_ela"
MODEL_SAVE_PATH = r"D:\FakeImageDetection\models\fake_image_detector_finetuned.h5"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 96

EPOCHS_STAGE_1 = 8     # Train classifier head
EPOCHS_STAGE_2 = 6     # Fine-tuning

LR_STAGE_1 = 1e-4
LR_STAGE_2 = 1e-5      # Very low LR for safety

# =========================
# DEVICE CHECK
# =========================
print("🔍 TensorFlow Version:", tf.__version__)
print("🧠 Devices:", tf.config.list_physical_devices())

# =========================
# DATA GENERATORS
# =========================
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=10,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# =========================
# MODEL DEFINITION
# =========================
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# -------- STAGE 1: FREEZE ALL VGG16 --------
for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(2, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=LR_STAGE_1),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# STAGE 1 TRAINING
# =========================
print("\n🚀 Stage 1: Training classifier head...\n")
start_time = time.time()

history_stage_1 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS_STAGE_1,
    verbose=1
)

# =========================
# STAGE 2: SAFE FINE-TUNING
# =========================
print("\n🔧 Stage 2: Fine-tuning last VGG16 block...\n")

# Unfreeze ONLY last VGG16 block
for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=LR_STAGE_2),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_stage_2 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS_STAGE_2,
    verbose=1
)

end_time = time.time()
print(f"\n⏱️ Total training time: {(end_time - start_time)/60:.2f} minutes")

# =========================
# SAVE MODEL
# =========================
model.save(MODEL_SAVE_PATH)
print(f"\n✅ Fine-tuned model saved at:\n{MODEL_SAVE_PATH}")
