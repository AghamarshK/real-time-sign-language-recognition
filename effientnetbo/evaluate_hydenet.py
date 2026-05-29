# evaluate_hydenet.py
import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from hyde_net import build_efficientnet_model, preprocess_for_efficientnet

# Constants
IMG_SIZE = (224, 224)
TEST_DIR = "dataset/test/demo"
N_PASSES = 10 # Reduced for speed during evaluation, can be increased for higher precision

def load_data():
    print(f"Loading test images from {TEST_DIR}...")
    images = []
    y_true = []
    filenames = []
    
    for filename in os.listdir(TEST_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Assume label is the first character of the filename (e.g., 'A1.jpg' -> 'A')
            label = filename[0].upper()
            img_path = os.path.join(TEST_DIR, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img_processed = preprocess_for_efficientnet(img, size=IMG_SIZE)
                images.append(img_processed)
                y_true.append(label)
                filenames.append(filename)
                
    return np.array(images), np.array(y_true), filenames

def evaluate():
    # Load labels
    label_classes = np.load("label_classes.npy", allow_pickle=True)
    class_to_idx = {label: i for i, label in enumerate(label_classes)}
    idx_to_class = {i: label for i, label in enumerate(label_classes)}
    
    # Load data
    x_test, y_true_labels, filenames = load_data()
    y_true_idx = [class_to_idx[label] for label in y_true_labels]
    
    # Build model and load weights
    num_classes = len(label_classes)
    model = build_efficientnet_model(num_classes=num_classes)
    model.load_weights("efficientnet_asl.h5")
    print("Model loaded successfully.")
    
    # Monte Carlo Inference
    print(f"Running Monte Carlo inference ({N_PASSES} passes per image)...")
    all_preds_mean = []
    all_preds_std = []
    
    for i in range(len(x_test)):
        img_batch = np.expand_dims(x_test[i], axis=0)
        passes = []
        for _ in range(N_PASSES):
            pred = model.predict(img_batch, verbose=0)
            passes.append(pred[0])
        
        passes = np.array(passes)
        all_preds_mean.append(np.mean(passes, axis=0))
        all_preds_std.append(np.std(passes, axis=0))
        
    all_preds_mean = np.array(all_preds_mean)
    y_pred_idx = np.argmax(all_preds_mean, axis=1)
    y_pred_labels = [idx_to_class[i] for i in y_pred_idx]
    
    # 1. Classification Report
    print("\nClassification Report:")
    report = classification_report(y_true_idx, y_pred_idx, target_names=label_classes)
    print(report)
    with open("classification_report.txt", "w") as f:
        f.write(report)
        
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_classes, yticklabels=label_classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    # 3. Sample Predictions with Uncertainty
    plt.figure(figsize=(15, 10))
    num_samples = min(4, len(x_test))
    for i in range(num_samples):
        plt.subplot(2, 2, i + 1)
        # Rescale for display (EfficientNet expects [0, 255])
        display_img = x_test[i].astype(np.uint8)
        plt.imshow(display_img)
        
        true_label = y_true_labels[i]
        pred_label = y_pred_labels[i]
        confidence = all_preds_mean[i][y_pred_idx[i]]
        uncertainty = all_preds_std[i][y_pred_idx[i]]
        
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {true_label} | Pred: {pred_label}\nConf: {confidence:.2f} | Unc: {uncertainty:.4f}", color=color)
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    print("Sample predictions saved as 'sample_predictions.png'")

if __name__ == "__main__":
    evaluate()
