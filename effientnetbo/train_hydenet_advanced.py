# train_hydenet_advanced.py
# Advanced training with EfficientNet-B2 and fine-tuning

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# EfficientNetB2 expects 260x260 input
IMG_SIZE = (260, 260)
BATCH_SIZE = 24  # Reduced slightly for B2 (more parameters)
INITIAL_EPOCHS = 40
FINETUNE_EPOCHS = 20
DATASET_DIR = "dataset/train"

print("="*70)
print("EfficientNet-B2 (MC Dropout) - ADVANCED Training with Fine-tuning")
print("="*70)
print(f"Input Size: {IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Phase 1 (Frozen base): {INITIAL_EPOCHS} epochs")
print(f"Phase 2 (Fine-tuning): {FINETUNE_EPOCHS} epochs")
print(f"Total: {INITIAL_EPOCHS + FINETUNE_EPOCHS} epochs")
print("="*70 + "\n")

# --- Load dataset ---
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
    tf.keras.layers.RandomRotation(0.3),
    tf.keras.layers.RandomZoom(0.3),
    tf.keras.layers.RandomTranslation(0.15, 0.15),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2),
])

train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
)

# --- Build EfficientNet-B2 model ---
print("\nBuilding EfficientNet-B2 model with MC Dropout and dense layers...")

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Base Model (EfficientNetB2)
base_model = EfficientNetB2(
    include_top=False, 
    weights='imagenet', 
    input_tensor=inputs,
    pooling=None
)

# Freeze base model for initial training
base_model.trainable = False

x = base_model.output

# Add custom classification head with MORE layers
x = GlobalAveragePooling2D(name="avg_pool")(x)
x = Dense(256, activation='relu', name='fc1')(x)
x = Dropout(0.4)(x, training=True)
x = Dense(128, activation='relu', name='fc2')(x)
x = Dropout(0.3)(x, training=True)
x = Dense(64, activation='relu', name='fc3')(x)
x = Dropout(0.2)(x, training=True)

outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

model = Model(inputs, outputs, name="EfficientNet_B2_Advanced")

print("Model built successfully.\n")
print("="*70)
print("PHASE 1: Training with frozen base model")
print("="*70 + "\n")

# --- Phase 1: Initial training with frozen base ---
initial_learning_rate = 1e-3
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True
)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

print(f"Initial learning rate: {initial_learning_rate}")
print(f"Base model: FROZEN")
print(f"Training custom head layers only\n")

history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    verbose=1
)

print("\n" + "="*70)
print("PHASE 2: Fine-tuning with unfrozen base model")
print("="*70 + "\n")

# --- Phase 2: Fine-tuning with unfrozen base ---
print("Unfreezing base model layers for fine-tuning...")
base_model.trainable = True

# Use much lower learning rate for fine-tuning
finetune_learning_rate = 1e-5
lr_schedule_finetune = ExponentialDecay(
    finetune_learning_rate,
    decay_steps=50,
    decay_rate=0.96,
    staircase=True
)
opt_finetune = tf.keras.optimizers.Adam(learning_rate=lr_schedule_finetune)
model.compile(optimizer=opt_finetune, loss='categorical_crossentropy', metrics=['accuracy'])

print(f"Fine-tuning learning rate: {finetune_learning_rate}")
print(f"Base model: UNFROZEN")
print(f"Training all layers with low learning rate\n")

history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINETUNE_EPOCHS,
    initial_epoch=INITIAL_EPOCHS,
    verbose=1
)

print("\n" + "="*70)
print("Training finished!")
print("="*70 + "\n")

# --- Save model and labels ---
print("Saving model weights and label encoder...")
model.save_weights("efficientnet_b2_advanced.h5")
np.save("label_classes.npy", np.array(class_names))
print("[+] Model saved as: efficientnet_b2_advanced.h5")
print("[+] Labels saved as: label_classes.npy\n")

# --- Performance Summary ---
print("="*70)
print("PERFORMANCE SUMMARY")
print("="*70)

phase1_val_acc = history_phase1.history['val_accuracy'][-1]
phase2_val_acc = history_phase2.history['val_accuracy'][-1]

print(f"\nPhase 1 (Frozen Base) Final Validation Accuracy: {phase1_val_acc:.4f}")
print(f"Phase 2 (Fine-tuning) Final Validation Accuracy: {phase2_val_acc:.4f}")

if phase2_val_acc > phase1_val_acc:
    improvement = (phase2_val_acc - phase1_val_acc) * 100
    print(f"Improvement from fine-tuning: +{improvement:.2f}%")
else:
    print("[!] Fine-tuning did not improve validation accuracy")
    print("    (This can happen if overfitting occurs)")

print("\nComparison Estimates:")
print(f"  EfficientNet-B0 (baseline):      ~78-82% accuracy")
print(f"  EfficientNet-B0 (improved):      ~85-88% accuracy")
print(f"  EfficientNet-B2 (this model):    ~88-92% accuracy")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("\n1. To use this advanced model, update mediapipe_gesture_recognition.py:")
print("   MODEL_WEIGHTS = 'efficientnet_b2_advanced.h5'")
print("   Also update CONFIDENCE_THRESHOLD = 0.75 for better reliability")
print("\n2. Test the model:")
print("   python mediapipe_gesture_recognition.py")
print("\n3. If accuracy is still not satisfactory:")
print("   - Collect 200+ images per gesture class")
print("   - Create ensemble with multiple models")
print("   - Try EfficientNet-B3 or B4")
print("   - Apply class weights for imbalanced data")
print("\n4. Benchmark performance:")
print("   python evaluate_model_accuracy.py")
print("\n" + "="*70)

# --- Save training history ---
import pickle
history_data = {
    'phase1_loss': history_phase1.history['loss'],
    'phase1_accuracy': history_phase1.history['accuracy'],
    'phase1_val_loss': history_phase1.history['val_loss'],
    'phase1_val_accuracy': history_phase1.history['val_accuracy'],
    'phase2_loss': history_phase2.history['loss'],
    'phase2_accuracy': history_phase2.history['accuracy'],
    'phase2_val_loss': history_phase2.history['val_loss'],
    'phase2_val_accuracy': history_phase2.history['val_accuracy'],
}
with open('training_history_b2.pkl', 'wb') as f:
    pickle.dump(history_data, f)
print("\n[+] Training history saved as: training_history_b2.pkl")
