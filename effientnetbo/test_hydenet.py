# test_hydenet.py
import numpy as np
import cv2
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from hyde_net import preprocess_for_efficientnet

# Load model and labels
label_classes = np.load("label_classes.npy", allow_pickle=True)
num_classes = len(label_classes)

# Rebuild model and load weights to bypass serialization issues
from hyde_net import build_efficientnet_model
model = build_efficientnet_model(num_classes=num_classes)
model.load_weights("efficientnet_asl.h5")

# Load sample of 16 images
test_dir = "dataset/test/demo"
image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files = sorted(image_files)[:16] # Take first 16 images

if not image_files:
    print(f"Error: No images found in {test_dir}")
    exit(1)

print(f"Found {len(image_files)} images for testing.")

# ---------------------------------------------------------
# Monte Carlo Inference for Uncertainty Estimation
# ---------------------------------------------------------
N_PASSES = 50

for img_name in image_files:
    img_path = os.path.join(test_dir, img_name)
    bgr = cv2.imread(img_path)

    if bgr is None:
        print(f"Error: Could not read image at {img_path}")
        continue

    # Preprocess for EfficientNet
    eff_input = preprocess_for_efficientnet(bgr, size=(224, 224))
    input_batch = np.expand_dims(eff_input, axis=0)

    # Predict N times
    predictions = []
    for _ in range(N_PASSES):
        # The model has Dropout(training=True), so each pass yields a slightly different result
        pred_probs = model.predict(input_batch, verbose=0)
        predictions.append(pred_probs[0]) # store the probability vector for the single image

    predictions = np.array(predictions) # Shape: (N_PASSES, NUM_CLASSES)

    # Calculate mean probabilities and uncertainty
    mean_probs = np.mean(predictions, axis=0)
    std_probs = np.std(predictions, axis=0)

    # Final prediction is the class with the highest mean probability
    pred_class_idx = np.argmax(mean_probs)
    pred_label = label_classes[pred_class_idx]
    pred_confidence = mean_probs[pred_class_idx]
    pred_uncertainty = std_probs[pred_class_idx]

    print(f"\n--- Inference Results for {img_name} ---")
    print(f"Predicted class: {pred_label}")
    print(f"Confidence (Mean Prob): {pred_confidence:.4f}")
    print(f"Uncertainty (Std Dev) : {pred_uncertainty:.4f}")

