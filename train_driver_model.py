
import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = "AlcoholDetectionDataset"  # root dataset folder
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
MODEL_SAVE_PATH = "driver_alcoholism_model.h5"
CLASS_NAMES = ["Alcoholic", "Non-Alcoholic"]

# -----------------------------
# Load Dataset with Face Detection
# -----------------------------
def load_images(dataset_dir):
    images = []
    labels = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(dataset_dir, class_name.lower().replace("-", "_"))
        if not os.path.exists(class_dir):
            continue

        for filename in os.listdir(class_dir):
            filepath = os.path.join(class_dir, filename)
            img = cv2.imread(filepath)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            # Skip if no face found
            if len(faces) == 0:
                continue

            # Take only the first detected face
            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, IMG_SIZE)
                face = face.astype("float32") / 255.0
                images.append(face)
                labels.append(idx)
                break

    return np.array(images), np.array(labels)

print("📂 Loading dataset...")
X, y = load_images(DATASET_DIR)
print(f"✅ Loaded {len(X)} images.")

# -----------------------------
# Train/Validation Split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Data Augmentation
# -----------------------------
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

# -----------------------------
# Build CNN Model
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(CLASS_NAMES), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# Train the Model
# -----------------------------
print("🚀 Training model...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# -----------------------------
# Save the Model
# -----------------------------
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")
