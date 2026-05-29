import sys
sys.modules['tensorflow'] = None

import os
import cv2
import mediapipe as mp
import csv

def extract_landmarks(dataset_path, output_csv):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )

    data = []
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

    for label in classes:
        print(f"Processing class: {label} for {dataset_path}")
        label_path = os.path.join(dataset_path, label)
        
        for img_name in os.listdir(label_path):
            if img_name.startswith("."):
                continue

            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                lm_list = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        lm_list.extend([lm.x, lm.y, lm.z])
                
                # Pad to 126 if only one hand is detected
                while len(lm_list) < 126:
                    lm_list.extend([0.0] * 63)
                
                # If more than 2 hands detected, truncate to 126
                lm_list = lm_list[:126]
                
                row = [label] + lm_list
                data.append(row)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    
    print(f"{output_csv} generated with {len(data)} samples.")

if __name__ == "__main__":
    train_dir = os.path.join("ASL_dataset_split", "train")
    test_dir = os.path.join("ASL_dataset_split", "test")
    
    extract_landmarks(train_dir, "train_landmarks.csv")
    extract_landmarks(test_dir, "test_landmarks.csv")
