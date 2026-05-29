# realtime_gesture_recognition.py
import numpy as np
import cv2
import tensorflow as tf
import os
import time
from collections import deque
from hyde_net import preprocess_for_efficientnet, build_efficientnet_model

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_WEIGHTS = "efficientnet_asl.h5"
LABELS_FILE = "label_classes.npy"
SAVE_DIR = "dataset/test/captured_gestures"
CONFIDENCE_THRESHOLD = 0.70 # Lowered slightly to see predictions while debugging
CONSECUTIVE_FRAMES = 5
MC_PASSES = 10 

# Stabilization Configuration
BUFFER_SIZE = 8 
prob_buffer = deque(maxlen=BUFFER_SIZE)

# ROI Configuration
ROI_SIZE = 250 # Slightly smaller ROI to capture hand more precisely
ROI_COLOR = (255, 255, 0) 

# ---------------------------------------------------------
# Load model and labels
# ---------------------------------------------------------
if not os.path.exists(LABELS_FILE):
    print(f"Error: {LABELS_FILE} not found.")
    exit(1)

label_classes = np.load(LABELS_FILE, allow_pickle=True)
num_classes = len(label_classes)

print(f"Loading model with {num_classes} classes...")
model = build_efficientnet_model(num_classes=num_classes)
if os.path.exists(MODEL_WEIGHTS):
    model.load_weights(MODEL_WEIGHTS)
    print(f"Weights loaded from {MODEL_WEIGHTS}")
else:
    print(f"Warning: {MODEL_WEIGHTS} not found. Running with uninitialized weights.")

# Ensure save directory exists
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"Created directory: {SAVE_DIR}")

# ---------------------------------------------------------
# Camera setup
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit(1)

# Get camera resolution to center ROI
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
x1, y1 = (width - ROI_SIZE) // 2, (height - ROI_SIZE) // 2
x2, y2 = x1 + ROI_SIZE, y1 + ROI_SIZE

print("Starting real-time recognition. Press 'q' to quit, 's' to manually save frame.")

consecutive_match_count = 0
last_pred_label = None
prev_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # CRITICAL: We process the original 'frame' (not flipped) for the AI
    # because the training data is usually not mirrored.
    # We only flip for the 'display_frame' to feel natural to the user.
    
    # 1. Extract ROI from original frame
    roi = frame[y1:y2, x1:x2]
    
    # 2. Preprocess ROI
    eff_input = preprocess_for_efficientnet(roi, size=(224, 224))
    input_batch = np.expand_dims(eff_input, axis=0)

    # 3. Optimized Batch Monte Carlo Inference
    mc_batch = np.repeat(input_batch, MC_PASSES, axis=0)
    all_preds = model.predict(mc_batch, verbose=0)
    
    current_mean_probs = np.mean(all_preds, axis=0)
    
    # 4. Temporal Smoothing
    prob_buffer.append(current_mean_probs)
    smoothed_probs = np.mean(prob_buffer, axis=0)
    
    pred_idx = np.argmax(smoothed_probs)
    confidence = smoothed_probs[pred_idx]
    label = label_classes[pred_idx]
    uncertainty = np.std(all_preds, axis=0)[pred_idx]

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = new_frame_time

    # 5. Prepare Display
    display_frame = cv2.flip(frame, 1) # Mirror only for user feedback
    
    # Since display is flipped, we must adjust ROI box coordinates for drawing
    # Original ROI [x1, x2] in width W -> Flipped ROI is [W-x2, W-x1]
    dx1, dx2 = width - x2, width - x1
    cv2.rectangle(display_frame, (dx1, y1), (dx2, y2), ROI_COLOR, 2)
    
    is_confident = confidence > CONFIDENCE_THRESHOLD
    color = (0, 255, 0) if is_confident else (0, 0, 255)
    
    # Text background
    cv2.rectangle(display_frame, (5, 5), (320, 150), (0, 0, 0), -1)
    alpha = 0.4
    cv2.addWeighted(display_frame, alpha, display_frame, 1 - alpha, 0, display_frame)

    cv2.putText(display_frame, f"Label: {label}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(display_frame, f"Conf: {confidence:.2f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(display_frame, f"Uncert: {uncertainty:.2f}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Show ROI being sent to model (Debug window)
    cv2.imshow("Model Input (Original Orientation)", roi)
    cv2.imshow("Gesture Recognition (Mirrored)", display_frame)

    # Auto-save logic
    if is_confident:
        if label == last_pred_label:
            consecutive_match_count += 1
        else:
            consecutive_match_count = 1
            last_pred_label = label
        
        if consecutive_match_count >= CONSECUTIVE_FRAMES:
            timestamp = int(time.time() * 1000)
            filename = f"auto_{label}_{timestamp}.jpg"
            save_path = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(save_path, roi)
            print(f"Auto-captured stabilized ROI: {save_path}")
            consecutive_match_count = 0 
    else:
        consecutive_match_count = 0

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        timestamp = int(time.time() * 1000)
        filename = f"manual_{timestamp}.jpg"
        save_path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(save_path, roi)
        print(f"Manually saved ROI: {save_path}")

cap.release()
cv2.destroyAllWindows()
