# demo_dap.py
# Demo script for original DAP+MaXNet implementation

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from hyde_net import preprocess_for_maxnet, preprocess_for_dap, dap_for_image

def main():
    # Configuration
    MODEL_PATH = 'hydenet_asl.h5'
    DEMO_DIR = 'dataset/test/demo'
    CLASS_NAMES = ['A', 'B', 'C', 'D']
    N_CLUSTERS = 100  # Original DAP uses 100 clusters
    
    # For metrics calculation
    correct = 0
    total = 0
    confidences = []  # Store all confidence scores
    class_correct = {c: 0 for c in CLASS_NAMES}  # Correct per class
    class_total = {c: 0 for c in CLASS_NAMES}    # Total per class
    
    print("Loading DAP+MaXNet model...")
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!\n")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Process each demo image
    demo_images = [f for f in os.listdir(DEMO_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    demo_images.sort()  # Process in alphabetical order
    
    print("DAP+MaXNet Model Demo")
    print("=" * 40)
    
    for img_name in demo_images:
        img_path = os.path.join(DEMO_DIR, img_name)
        true_label = img_name[0]  # Assuming filename starts with class label (A, B, C, D)
        
        print(f"\nProcessing: {img_name}")
        print(f"True class: {true_label}")
        
        try:
            # Load and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not load image: {img_path}")
            
            # Preprocess for both paths
            maxnet_input = preprocess_for_maxnet(img, (224, 224))
            dap_img = preprocess_for_dap(img, (80, 80))
            
            # Extract DAP features
            dap_codes = dap_for_image(dap_img)
            dap_codes = np.array(dap_codes).reshape(-1, 1)
            
            # Cluster DAP codes
            from sklearn.cluster import AgglomerativeClustering
            clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS)
            labels = clustering.fit_predict(dap_codes)
            
            # Create histogram features
            dap_features = np.bincount(labels, minlength=N_CLUSTERS).astype(np.float32)
            if dap_features.sum() > 0:
                dap_features = dap_features / dap_features.sum()  # normalize
            
            # Make prediction
            pred = model.predict([
                maxnet_input.reshape(1, 224, 224, 3),
                dap_features.reshape(1, -1)
            ], verbose=0)
            
            # Get prediction results
            pred_class = np.argmax(pred[0])
            pred_class_name = CLASS_NAMES[pred_class]
            confidence = pred[0][pred_class]
            
            # Print results
            print(f"→ Predicted: {CLASS_NAMES[pred_class]}")
            print(f"→ Confidence: {confidence:.2%}")
            
            # Update metrics
            true_class = CLASS_NAMES.index(true_label)
            if pred_class == true_class:
                correct += 1
                class_correct[true_label] += 1
            total += 1
            class_total[true_label] += 1
            confidences.append(confidence)
            
            print("-" * 40)
            
        except Exception as e:
            print(f"Error processing image: {str(e)}\n")
            print("-" * 40)
            total += 1
            class_total[true_label] += 1
    
    # Print final metrics
    if total > 0:
        accuracy = (correct / total) * 100
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        print("\nFinal Results:")
        print("=" * 40)
        print(f"Overall Accuracy: {accuracy:.1f}% ({correct}/{total} correct)")
        print(f"Average Confidence: {avg_confidence:.1%}")
        
        print("\nPer-Class Accuracy:")
        print("-" * 40)
        for class_name in CLASS_NAMES:
            if class_total[class_name] > 0:
                class_acc = (class_correct[class_name] / class_total[class_name]) * 100
                print(f"Class {class_name}: {class_acc:.1f}% ({class_correct[class_name]}/{class_total[class_name]})")

if __name__ == "__main__":
    main()