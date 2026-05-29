# mediapipe_gesture_recognition.py
"""
Real-time Gesture Recognition with MediaPipe Hand Detection + EfficientNet-B0 Classification

Features:
- Detects hand using MediaPipe Hands
- Extracts hand bounding box from landmarks
- Crops and preprocesses hand region to 224x224
- Classifies gesture using EfficientNet-B0 model
- Displays prediction, confidence, and bounding box in real-time
- Temporal smoothing for stable predictions
"""

import numpy as np
import cv2
import tensorflow as tf
import os
import time
from collections import deque
from mediapipe import solutions
from hyde_net import preprocess_for_efficientnet, build_efficientnet_model

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_WEIGHTS = "efficientnet_asl.h5"
LABELS_FILE = "label_classes.npy"
SAVE_DIR = "dataset/test/captured_gestures"
CONFIDENCE_THRESHOLD = 0.60
CONSECUTIVE_FRAMES = 5
MC_PASSES = 10

# MediaPipe Configuration
MEDIAPIPE_CONFIDENCE = 0.7  # Minimum confidence for hand detection
HAND_DETECTION_MODEL = "full"  # "lite" or "full"

# Stabilization Configuration
BUFFER_SIZE = 8
prob_buffer = deque(maxlen=BUFFER_SIZE)

# Visualization Configuration
HAND_BOX_COLOR = (0, 255, 0)  # Green for hand bounding box
HAND_BOX_THICKNESS = 2
LANDMARK_COLOR = (0, 0, 255)  # Red for landmarks
LANDMARK_RADIUS = 4

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
    print(f"[+] Weights loaded from {MODEL_WEIGHTS}")
else:
    print(f"[!] Warning: {MODEL_WEIGHTS} not found. Running with uninitialized weights.")

# Ensure save directory exists
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"[+] Created directory: {SAVE_DIR}")

# ---------------------------------------------------------
# Initialize MediaPipe Hands
# ---------------------------------------------------------
print(f"\nInitializing MediaPipe Hands (model: {HAND_DETECTION_MODEL})...")

hands = solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Process only one hand
    min_detection_confidence=MEDIAPIPE_CONFIDENCE,
    min_tracking_confidence=0.5,
    model_complexity=1 if HAND_DETECTION_MODEL == "full" else 0
)
print("[+] MediaPipe Hands initialized")

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def get_hand_bounding_box(landmarks, frame_width, frame_height, padding=20):
    """
    Extract bounding box from MediaPipe hand landmarks.
    
    Args:
        landmarks: List of hand landmarks (21 points)
        frame_width: Width of the frame
        frame_height: Height of the frame
        padding: Extra padding around the bounding box (in pixels)
    
    Returns:
        (x1, y1, x2, y2): Bounding box coordinates, or None if invalid
    """
    if not landmarks:
        return None
    
    # Extract x, y coordinates from landmarks
    x_coords = [lm.x * frame_width for lm in landmarks]
    y_coords = [lm.y * frame_height for lm in landmarks]
    
    # Find min/max
    x_min = int(min(x_coords)) - padding
    x_max = int(max(x_coords)) + padding
    y_min = int(min(y_coords)) - padding
    y_max = int(max(y_coords)) + padding
    
    # Clamp to frame bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame_width, x_max)
    y_max = min(frame_height, y_max)
    
    # Ensure minimum size
    if (x_max - x_min) < 50 or (y_max - y_min) < 50:
        return None
    
    return (x_min, y_min, x_max, y_max)

def crop_and_preprocess_hand(frame, bbox):
    """
    Crop the hand region from the frame and preprocess it.
    
    Args:
        frame: Original frame (BGR)
        bbox: Bounding box tuple (x1, y1, x2, y2)
    
    Returns:
        Preprocessed image (224x224, float32) ready for EfficientNet-B0
    """
    if bbox is None:
        return None
    
    x1, y1, x2, y2 = bbox
    cropped = frame[y1:y2, x1:x2]
    
    if cropped.size == 0:
        return None
    
    # Preprocess using the existing function
    preprocessed = preprocess_for_efficientnet(cropped, size=(224, 224))
    return preprocessed

def draw_hand_landmarks(frame, landmarks, bbox, frame_width, frame_height):
    """
    Draw hand landmarks and bounding box on the frame.
    
    Args:
        frame: Frame to draw on
        landmarks: MediaPipe hand landmarks
        bbox: Bounding box (x1, y1, x2, y2)
        frame_width: Width of the frame
        frame_height: Height of the frame
    """
    if bbox is None:
        return
    
    x1, y1, x2, y2 = bbox
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), HAND_BOX_COLOR, HAND_BOX_THICKNESS)
    
    # Draw landmarks
    if landmarks:
        for lm in landmarks:
            x = int(lm.x * frame_width)
            y = int(lm.y * frame_height)
            cv2.circle(frame, (x, y), LANDMARK_RADIUS, LANDMARK_COLOR, -1)

def display_prediction(frame, label, confidence, uncertainty, fps):
    """
    Display prediction information on the frame.
    
    Args:
        frame: Frame to draw on
        label: Predicted label
        confidence: Confidence score
        uncertainty: Uncertainty estimate
        fps: Frames per second
    """
    is_confident = confidence > CONFIDENCE_THRESHOLD
    color = (0, 255, 0) if is_confident else (0, 0, 255)
    
    # Semi-transparent background for text
    h, w = frame.shape[:2]
    text_bg = frame[0:160, 0:350].copy()
    cv2.rectangle(frame, (0, 0), (350, 160), (0, 0, 0), -1)
    frame[0:160, 0:350] = cv2.addWeighted(text_bg, 0.3, frame[0:160, 0:350], 0.7, 0)
    
    # Draw text
    cv2.putText(frame, f"Gesture: {label}", (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Uncertainty: {uncertainty:.2f}", (10, 115), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Status: {'DETECTED' if is_confident else 'LOW CONF'}", (10, 155), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # FPS in corner
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

# ---------------------------------------------------------
# Camera setup
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("âœ— Error: Could not open camera.")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"âœ“ Camera opened: {width}x{height}")

print("\n" + "="*60)
print("Real-time Gesture Recognition with MediaPipe Hand Detection")
print("="*60)
print("Controls:")
print("  'q'     - Quit application")
print("  's'     - Manually save current frame")
print("  'd'     - Toggle detection visualization (landmarks)")
print("="*60 + "\n")

consecutive_match_count = 0
last_pred_label = None
prev_frame_time = 0
show_landmarks = True

while True:
    ret, frame = cap.read()
    if not ret:
        print("âœ— Error: Could not read frame.")
        break
    
    # Flip frame horizontally for natural mirror view
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run MediaPipe hand detection
    results = hands.process(frame_rgb)
    
    # Default values (no hand detected)
    label = "No Hand"
    confidence = 0.0
    uncertainty = 0.0
    
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
        # Get first hand detected
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get bounding box
        bbox = get_hand_bounding_box(hand_landmarks.landmark, width, height, padding=20)
        
        if bbox is not None:
            # Crop and preprocess the hand region
            preprocessed_hand = crop_and_preprocess_hand(frame, bbox)
            
            if preprocessed_hand is not None:
                # Prepare batch for model
                input_batch = np.expand_dims(preprocessed_hand, axis=0)
                
                # Monte Carlo inference
                mc_batch = np.repeat(input_batch, MC_PASSES, axis=0)
                all_preds = model.predict(mc_batch, verbose=0)
                
                # Average predictions
                current_mean_probs = np.mean(all_preds, axis=0)
                
                # Temporal smoothing
                prob_buffer.append(current_mean_probs)
                smoothed_probs = np.mean(prob_buffer, axis=0)
                
                # Get prediction
                pred_idx = np.argmax(smoothed_probs)
                confidence = smoothed_probs[pred_idx]
                label = label_classes[pred_idx]
                uncertainty = np.std(all_preds, axis=0)[pred_idx]
                
                # Draw landmarks and bounding box
                draw_hand_landmarks(frame, hand_landmarks.landmark if show_landmarks else None, 
                                  bbox, width, height)
                
                # Draw bounding box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                            HAND_BOX_COLOR, HAND_BOX_THICKNESS)
            else:
                label = "Hand Too Small"
        else:
            label = "Hand Landmarks Invalid"
    
    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = new_frame_time
    
    # Display prediction info
    display_prediction(frame, label, confidence, uncertainty, fps)
    
    # Show frame
    cv2.imshow("MediaPipe Hand Detection + EfficientNet Classification", frame)
    
    # Auto-save logic (stable predictions)
    if confidence > CONFIDENCE_THRESHOLD and label != "No Hand":
        if label == last_pred_label:
            consecutive_match_count += 1
        else:
            consecutive_match_count = 1
            last_pred_label = label
        
        if consecutive_match_count >= CONSECUTIVE_FRAMES:
            timestamp = int(time.time() * 1000)
            filename = f"auto_{label}_{timestamp}.jpg"
            save_path = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(save_path, frame)
            print(f"âœ“ Auto-captured: {filename}")
            consecutive_match_count = 0
    else:
        consecutive_match_count = 0
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nâœ“ Exiting application...")
        break
    elif key == ord('s'):
        timestamp = int(time.time() * 1000)
        filename = f"manual_{timestamp}.jpg"
        save_path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(save_path, frame)
        print(f"âœ“ Manually saved: {filename}")
    elif key == ord('d'):
        show_landmarks = not show_landmarks
        print(f"{'âœ“' if show_landmarks else 'âœ—'} Landmarks visualization: {'ON' if show_landmarks else 'OFF'}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
print("\nâœ“ Application closed successfully")

