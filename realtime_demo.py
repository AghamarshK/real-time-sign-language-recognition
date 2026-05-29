import sys
sys.modules['tensorflow'] = None

import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import time
import os

# ============================================================
# CLASSES & DEFINITIONS
# ============================================================
DATASET_PATH = os.path.join("ASL_dataset_split", "train")
classes = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
print(f"[SUCCESS] Classes loaded: {len(classes)} -> {classes}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[SUCCESS] Device: {device}")

class LandmarkClassifier(nn.Module):
    def __init__(self, num_classes, input_dim=126):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)

def normalize_landmarks(landmarks_np):
    lm1    = landmarks_np[:63].reshape(21, 3)
    wrist1 = lm1[0:1, :]
    lm1    = lm1 - wrist1
    scale1 = np.linalg.norm(lm1[9, :]) + 1e-6
    lm1    = lm1 / scale1

    lm2 = landmarks_np[63:].reshape(21, 3)
    if np.any(lm2 != 0):
        wrist2 = lm2[0:1, :]
        lm2    = lm2 - wrist2
        scale2 = np.linalg.norm(lm2[9, :]) + 1e-6
        lm2    = lm2 / scale2

    return np.concatenate([lm1.flatten(), lm2.flatten()])

# ============================================================
# LOAD SAVED MODEL
# ============================================================
lm_model = LandmarkClassifier(num_classes=len(classes), input_dim=126).to(device)

if not os.path.exists("asl_landmark_classifier.pth"):
    print("[ERROR] asl_landmark_classifier.pth not found. Train the model first.")
    exit(1)

checkpoint = torch.load("asl_landmark_classifier.pth", map_location=device, weights_only=True)
lm_model.load_state_dict(checkpoint["model_state_dict"])
lm_model.eval()

print(f"[SUCCESS] Model loaded — Val Acc: {checkpoint.get('val_acc', 0):.2f}%")
print("[INFO] Ready! Opening webcam.")

# ============================================================
# REALTIME DEMO LOOP
# ============================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

prediction_buffer = deque(maxlen=30)
confidence_buffer = deque(maxlen=15)

last_prediction   = ""
stable_prediction = ""
stable_start      = None

STABILITY_TIME = 0.8
MIN_CONFIDENCE = 0.45

cap = cv2.VideoCapture(0)
print("[INFO] Webcam started — Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame    = cv2.flip(frame, 1)
    rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results  = hands.process(rgb_full)

    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        # Extract 126 values
        lm_list = []
        for hand_lm in results.multi_hand_landmarks:
            for lm in hand_lm.landmark:
                lm_list.extend([lm.x, lm.y, lm.z])

        # Pad to 126 if one hand
        while len(lm_list) < 126:
            lm_list.extend([0.0] * 63)
            
        lm_list = lm_list[:126]

        lm_array      = np.array(lm_list, dtype=np.float32)
        lm_normalized = normalize_landmarks(lm_array)

        lm_tensor = torch.tensor(
            lm_normalized, dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs   = lm_model(lm_tensor)
            probs     = F.softmax(outputs, dim=1)
            conf, idx = probs.max(1)

        pred_class = classes[idx.item()]
        conf_val   = conf.item()

        prediction_buffer.append(pred_class)
        confidence_buffer.append(conf_val)

        avg_conf      = sum(confidence_buffer) / len(confidence_buffer)
        majority_pred = max(set(prediction_buffer), key=prediction_buffer.count)

        if majority_pred != last_prediction:
            stable_start    = time.time()
            last_prediction = majority_pred

        if stable_start and (time.time() - stable_start > STABILITY_TIME):
            stable_prediction = majority_pred

        display_text = stable_prediction if avg_conf >= MIN_CONFIDENCE else "Unknown"

        num_hands = len(results.multi_hand_landmarks)

        # Display overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 170), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, f"Gesture: {display_text}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {avg_conf:.2f}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Hands: {num_hands}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        # Confidence bar
        bar_width = int(avg_conf * 200)
        bar_color = ((0, 255, 0) if avg_conf > 0.7 else (0, 165, 255) if avg_conf > 0.45 else (0, 0, 255))

        cv2.rectangle(frame, (10, 140), (10 + bar_width, 158), bar_color, -1)
        cv2.rectangle(frame, (10, 140), (210, 158), (255, 255, 255), 1)

    else:
        prediction_buffer.clear()
        confidence_buffer.clear()
        stable_prediction = ""
        stable_start      = None

        cv2.putText(frame, "No Hand Detected", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("ASL Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Session ended")
