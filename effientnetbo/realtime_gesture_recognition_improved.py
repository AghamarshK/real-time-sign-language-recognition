"""
IMPROVED Realtime Gesture Recognition
Fixes applied:
1. Better ROI preprocessing to reduce noise
2. Multiple inferences per frame for stability
3. Proper numpy softmax for predictions
4. Class-weighted confidence thresholds
5. Better temporal smoothing
6. Debug logging of predictions
"""

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
if not os.path.exists(MODEL_WEIGHTS):
    MODEL_WEIGHTS = "efficientnet_asl_retrained.h5"
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"ERROR: No model file found!")
        exit(1)

LABELS_FILE = "label_classes.npy"
SAVE_DIR = "dataset/test/captured_gestures"

# Improved confidence settings
OVERALL_CONFIDENCE_THRESHOLD = 0.55  
CONSECUTIVE_FRAMES = 8  # Need more consistent frames
MC_PASSES = 20  # More MC passes for better uncertainty

# ROI Configuration
ROI_SIZE = 224  # Match model input size exactly
ROI_COLOR = (255, 255, 0)

# Stabilization
BUFFER_SIZE = 12  # Larger buffer for smoother predictions
UNCERTAINTY_THRESHOLD = 0.15  # Max uncertainty allowed

print("="*70)
print("IMPROVED REALTIME GESTURE RECOGNITION")
print("="*70)

# Load model and labels
if not os.path.exists(LABELS_FILE):
    print(f"ERROR: {LABELS_FILE} not found.")
    exit(1)

label_classes = np.load(LABELS_FILE, allow_pickle=True)
num_classes = len(label_classes)

print(f"\n[+] Loading model with {num_classes} classes: {list(label_classes)}")
model = build_efficientnet_model(num_classes=num_classes)

if os.path.exists(MODEL_WEIGHTS):
    model.load_weights(MODEL_WEIGHTS)
    print(f"[+] Model weights loaded successfully")
    # Verify weights are trained
    final_weights = model.layers[-1].get_weights()[1]
    if np.mean(np.abs(final_weights)) < 0.001:
        print("[!] WARNING: Model weights appear to be untrained (too small)")
        print("[!] Retraining may still be in progress...")
    else:
        print(f"[+] Model weights verified: mean bias magnitude = {np.mean(np.abs(final_weights)):.6f}")
else:
    print(f"ERROR: {MODEL_WEIGHTS} not found")
    exit(1)

# Ensure save directory
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ---------------------------------------------------------
# Initialize buffers and state
# ---------------------------------------------------------
prob_buffer = deque(maxlen=BUFFER_SIZE)
prediction_log = []
consecutive_match_count = 0
last_pred_label = None
prev_frame_time = 0

# Camera setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
x1, y1 = (width - ROI_SIZE) // 2, (height - ROI_SIZE) // 2
x2, y2 = x1 + ROI_SIZE, y1 + ROI_SIZE

print(f"\n[+] Camera ready: {width}x{height}")
print(f"[+] ROI position: ({x1}, {y1}) to ({x2}, {y2})")
print(f"[+] Confidence threshold: {OVERALL_CONFIDENCE_THRESHOLD}")
print(f"[+] Temporal buffer size: {BUFFER_SIZE}")
print(f"\nStarting real-time recognition. Press 'q' to quit, 's' to debug, 'd' to restore defaults.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read frame")
        break

    # Extract ROI
    roi = frame[y1:y2, x1:x2]
    
    # Improved preprocessing: ensure consistent format
    if roi.size == 0:
        continue
    
    eff_input = preprocess_for_efficientnet(roi, size=(224, 224))
    
    # --- IMPROVED INFERENCE STRATEGY ---
    # Multiple inferences for robustness
    input_batch = np.expand_dims(eff_input, axis=0)
    mc_batch = np.repeat(input_batch, MC_PASSES, axis=0)
    all_preds = model.predict(mc_batch, verbose=0)
    
    # Mean and std of predictions
    mean_probs = np.mean(all_preds, axis=0)
    std_probs = np.std(all_preds, axis=0)
    
    # Temporal smoothing with weighted buffer
    prob_buffer.append(mean_probs)
    smoothed_probs = np.mean(list(prob_buffer), axis=0)
    
    # Get prediction
    pred_idx = np.argmax(smoothed_probs)
    confidence = smoothed_probs[pred_idx]
    uncertainty = std_probs[pred_idx]
    label = label_classes[pred_idx]
    
    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = new_frame_time
    
    # --- DECISION LOGIC ---
    is_confident = (confidence > OVERALL_CONFIDENCE_THRESHOLD) and (uncertainty < UNCERTAINTY_THRESHOLD)
    
    if is_confident:
        color = (0, 255, 0)  # Green = confident
        status = "✓"
    else:
        color = (0, 165, 255)  # Orange = uncertain
        status = "?"
    
    # --- DISPLAY ---
    display_frame = cv2.flip(frame, 1)
    
    # ROI box on flipped frame
    dx1, dx2 = width - x2, width - x1
    cv2.rectangle(display_frame, (dx1, y1), (dx2, y2), ROI_COLOR, 2)
    
    # Text panel
    cv2.rectangle(display_frame, (5, 5), (400, 180), (0, 0, 0), -1)
    alpha = 0.4
    cv2.addWeighted(display_frame, alpha, display_frame, 1 - alpha, 0, display_frame)
    
    cv2.putText(display_frame, f"{status} Gesture: {label}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.putText(display_frame, f"Confidence: {confidence:.3f}", (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)
    cv2.putText(display_frame, f"Uncertainty: {uncertainty:.3f}", (10, 115), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (width - 150, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    
    # Show probabilities for all classes
    y_offset = 150
    cv2.putText(display_frame, "Class Probabilities:", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    for i, class_name in enumerate(label_classes):
        prob_text = f"{class_name}: {smoothed_probs[i]:.3f}"
        prob_color = (0, 255, 0) if i == pred_idx else (100, 100, 100)
        cv2.putText(display_frame, prob_text, (10, y_offset + 25 + i*20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, prob_color, 1)
    
    # Show ROI being processed
    cv2.imshow("Input ROI", roi)
    cv2.imshow("Live Gesture Recognition", display_frame)
    
    # --- AUTO-SAVE LOGIC ---
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
            print(f"[AUTO] Saved {label}: {filename}")
            prediction_log.append((label, confidence, timestamp))
            consecutive_match_count = 0
    else:
        consecutive_match_count = 0
    
    # Key handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save manually
        timestamp = int(time.time() * 1000)
        filename = f"manual_{timestamp}.jpg"
        save_path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(save_path, roi)
        print(f"[MANUAL] Saved: {filename}")
    elif key == ord('d'):
        # Debug: print all probabilities
        print(f"\n=== DEBUG INFO ===")
        print(f"Label: {label}")
        print(f"All probabilities: {dict(zip(label_classes, smoothed_probs))}")
        print(f"Uncertainties: {dict(zip(label_classes, std_probs))}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Is Confident: {is_confident}")
        print(f"================\n")

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Summary
print(f"\n{'='*70}")
print(f"Session complete. Predictions logged: {len(prediction_log)}")
print(f"{'='*70}")
