# train_hydenet.py
# (Reusing filename for backward compatibility, now trains EfficientNet-B0)
import os
import numpy as np
import tensorflow as tf
from hyde_net import build_efficientnet_model

# EfficientNetB0 expects 224x224 input
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
DATASET_DIR = "dataset/train"

print("=== EfficientNet-B0 (MC Dropout) Training Script Started ===")

# --- Load dataset using tf.data ---
print(f"Loading dataset from: {DATASET_DIR}")

# We use image_dataset_from_directory which handles labels and resizing
# It returns a tf.data.Dataset yielding (images, labels)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical' # one-hot encoded labels
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

# Optimize dataset loading 
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- Data Augmentation ---
print("Configuring Data Augmentation pipeline...")
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# Apply augmentation only to training data
train_ds = train_ds.map(
  lambda x, y: (data_augmentation(x, training=True), y),
  num_parallel_calls=AUTOTUNE
)

# --- Build model ---
print("\nBuilding EfficientNet-B0 model with MC Dropout...")
model = build_efficientnet_model(num_classes=num_classes, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
print("Model built successfully.")
model.summary()

# --- Train ---
print("\nStarting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)
print("Training finished!")

# --- Save model and label encoder ---
print("Saving model weights and label encoder...")
model.save_weights("efficientnet_asl.h5")
# Save the class names as a numpy array for inference
np.save("label_classes.npy", np.array(class_names))

print(" Training complete. Model weights saved as 'efficientnet_asl.h5' and labels as 'label_classes.npy'")
print("=== Training Script Finished ===")
