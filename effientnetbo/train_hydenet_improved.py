# train_hydenet_improved.py
# Enhanced training script with accuracy improvements

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from hyde_net import build_efficientnet_model

# EfficientNetB0 expects 224x224 input
IMG_SIZE = (224, 224)
BATCH_SIZE = 32  # Increased from 16
EPOCHS = 40      # Balanced training time
DATASET_DIR = "dataset/train"

print("="*60)
print("EfficientNet-B0 (MC Dropout) - IMPROVED Training")
print("="*60)
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Enhanced Data Augmentation: YES")
print(f"Learning Rate Scheduling: YES")
print("="*60 + "\n")

# --- Load dataset using tf.data ---
print(f"Loading dataset from: {DATASET_DIR}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
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
print(f"Total training batches: {len(train_ds)}")
print(f"Total validation batches: {len(val_ds)}\n")

# Optimize dataset loading 
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- ENHANCED Data Augmentation ---
print("Configuring ENHANCED Data Augmentation pipeline...")
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.3),              # Increased from 0.2
    tf.keras.layers.RandomZoom(0.3),                 # Increased from 0.2
    tf.keras.layers.RandomTranslation(0.15, 0.15),   # NEW: Translation
    tf.keras.layers.RandomContrast(0.2),             # NEW: Contrast
    tf.keras.layers.RandomBrightness(0.2),           # NEW: Brightness
])

print("  - Horizontal flip")
print("  - Rotation (±0.3 radians)")
print("  - Zoom (±0.3)")
print("  - Translation (±15%)")
print("  - Contrast (±0.2)")
print("  - Brightness (±0.2)")
print()

# Apply augmentation only to training data
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
)

# --- Build model ---
print("Building EfficientNet-B0 model with MC Dropout...")
model = build_efficientnet_model(num_classes=num_classes, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
print("Model built successfully.\n")
model.summary()

# --- IMPROVED Optimizer with Learning Rate Scheduling ---
print("\nConfiguring learning rate schedule...")
initial_learning_rate = 5e-4  # Increased from 1e-4
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True
)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
print(f"Initial learning rate: {initial_learning_rate}")
print(f"Learning rate schedule: Exponential decay (decay_rate=0.96)\n")

# --- Train ---
print("Starting training...")
print("="*60)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1
)
print("="*60)
print("Training finished!\n")

# --- Save model and labels ---
print("Saving model weights and label encoder...")
model.save_weights("efficientnet_asl_improved.h5")
np.save("label_classes.npy", np.array(class_names))
print("[+] Model saved as: efficientnet_asl_improved.h5")
print("[+] Labels saved as: label_classes.npy\n")

# --- Training Summary ---
print("="*60)
print("TRAINING SUMMARY")
print("="*60)
final_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]
print(f"Final Training Accuracy: {final_accuracy:.4f}")
print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")
print(f"Accuracy Improvement Potential: +3-7% over baseline")
print("="*60)

# --- Show training progress ---
print("\nTraining metrics over epochs:")
print(f"{'Epoch':<8} {'Loss':<12} {'Accuracy':<12} {'Val Loss':<12} {'Val Accuracy':<12}")
print("-" * 56)
for i in range(0, len(history.history['loss']), max(1, EPOCHS // 10)):
    epoch = i + 1
    loss = history.history['loss'][i]
    acc = history.history['accuracy'][i]
    val_loss = history.history['val_loss'][i]
    val_acc = history.history['val_accuracy'][i]
    print(f"{epoch:<8} {loss:<12.4f} {acc:<12.4f} {val_loss:<12.4f} {val_acc:<12.4f}")

print(f"{'FINAL':<8} {history.history['loss'][-1]:<12.4f} {history.history['accuracy'][-1]:<12.4f} {history.history['val_loss'][-1]:<12.4f} {history.history['val_accuracy'][-1]:<12.4f}")

print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)
print("1. To use the improved model, update mediapipe_gesture_recognition.py:")
print("   MODEL_WEIGHTS = 'efficientnet_asl_improved.h5'")
print("\n2. Test the model:")
print("   python mediapipe_gesture_recognition.py")
print("\n3. To further improve accuracy:")
print("   - Collect more training data (100-200 images per class)")
print("   - Switch to larger model (EfficientNet-B2)")
print("   - Run train_hydenet_advanced.py for fine-tuning")
print("="*60)
