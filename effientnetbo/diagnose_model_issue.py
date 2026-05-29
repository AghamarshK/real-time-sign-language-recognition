"""
Diagnostic script to identify why the model always predicts 'A'
"""

import numpy as np
import os
import cv2
import tensorflow as tf
from collections import Counter
from hyde_net import build_efficientnet_model, preprocess_for_efficientnet

# Configuration
MODEL_WEIGHTS = "efficientnet_asl.h5"
LABELS_FILE = "label_classes.npy"
TRAIN_DIR = "dataset/train"

print("="*70)
print("MODEL DIAGNOSTIC REPORT")
print("="*70)

# Load labels
label_classes = np.load(LABELS_FILE, allow_pickle=True)
num_classes = len(label_classes)
print(f"\n[1] Classes: {list(label_classes)} (Total: {num_classes})")

# Build and load model
print(f"\n[2] Building model with {num_classes} classes...")
model = build_efficientnet_model(num_classes=num_classes)

if os.path.exists(MODEL_WEIGHTS):
    model.load_weights(MODEL_WEIGHTS)
    print(f"    ✓ Model weights loaded from: {MODEL_WEIGHTS}")
else:
    print(f"    ✗ ERROR: Model weights not found: {MODEL_WEIGHTS}")
    exit(1)

# Test on training data samples
print(f"\n[3] Testing model on TRAINING DATA samples...")
print("    Sampling 5 random images from each class...")

predictions_by_class = {}

for class_label in label_classes:
    class_dir = os.path.join(TRAIN_DIR, class_label)
    files = os.listdir(class_dir)[:5]  # Get first 5 files
    
    class_predictions = {}
    print(f"\n    Class '{class_label}' ({len(files)} samples):")
    
    for filename in files:
        filepath = os.path.join(class_dir, filename)
        
        # Load and preprocess image
        image = cv2.imread(filepath)
        if image is None:
            continue
            
        input_data = preprocess_for_efficientnet(image, size=(224, 224))
        input_batch = np.expand_dims(input_data, axis=0)
        
        # Get prediction
        pred = model.predict(input_batch, verbose=0)[0]
        pred_class = label_classes[np.argmax(pred)]
        confidence = np.max(pred)
        
        # Store
        class_predictions[pred_class] = class_predictions.get(pred_class, 0) + 1
        print(f"         → Predicted: {pred_class} (conf: {confidence:.3f})")
    
    predictions_by_class[class_label] = class_predictions

# Analyze model biases
print(f"\n[4] PREDICTION ANALYSIS")
print("    What does the model predict for each class?")

for true_class in label_classes:
    if true_class in predictions_by_class:
        predictions = predictions_by_class[true_class]
        print(f"\n    True Class '{true_class}':")
        for pred_class, count in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            pct = (count / sum(predictions.values())) * 100
            print(f"       - Predicts '{pred_class}': {count}/{sum(predictions.values())} ({pct:.0f}%)")

# Check model output layers
print(f"\n[5] MODEL ARCHITECTURE CHECK")
print(f"    Output layer: {model.layers[-1].name}")
print(f"    Output shape: {model.layers[-1].output_shape}")
print(f"    Activation: {model.layers[-1].activation}")

# Check for dead neurons or weird weights
print(f"\n[6] MODEL WEIGHT ANALYSIS")
weights = model.layers[-1].get_weights()[0]  # Dense layer weights
biases = model.layers[-1].get_weights()[1]   # Biases
print(f"    Final layer weights shape: {weights.shape}")
print(f"    Final layer biases: {biases}")
print(f"    Bias variance: {np.var(biases):.6f}")
print(f"    Weight ranges per class:")
for i, class_label in enumerate(label_classes):
    w_range = (np.min(weights[:, i]), np.max(weights[:, i]))
    print(f"       - Class '{class_label}': [{w_range[0]:.4f}, {w_range[1]:.4f}]")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
