# evaluate_model_accuracy.py
# Evaluate model accuracy with detailed metrics

import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from hyde_net import build_efficientnet_model, preprocess_for_efficientnet
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
MODEL_WEIGHTS = "efficientnet_asl_improved.h5"  # Change to "efficientnet_b2_advanced.h5" for B2
LABELS_FILE = "label_classes.npy"
TEST_DIR = "dataset/test"
IMG_SIZE = (224, 224)

print("="*70)
print("Model Accuracy Evaluation")
print("="*70)
print(f"Model: {MODEL_WEIGHTS}")
print(f"Test Directory: {TEST_DIR}\n")

# Load labels
if not os.path.exists(LABELS_FILE):
    print(f"[X] Error: {LABELS_FILE} not found")
    exit(1)

label_classes = np.load(LABELS_FILE, allow_pickle=True)
num_classes = len(label_classes)
print(f"Classes: {list(label_classes)}\n")

# Build and load model
print("Loading model...")
model = build_efficientnet_model(num_classes=num_classes)

if os.path.exists(MODEL_WEIGHTS):
    model.load_weights(MODEL_WEIGHTS)
    print(f"[+] Model weights loaded\n")
else:
    print(f"[X] Model weights not found: {MODEL_WEIGHTS}")
    exit(1)

# Load test data
print("Loading test data...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=16,
    shuffle=False,
    label_mode='categorical'
)

# Get predictions
print("Running inference on test set...")
y_true = []
y_pred = []
confidences = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))
    confidences.extend(np.max(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)
confidences = np.array(confidences)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
low_conf_mask = confidences < 0.7
low_conf_accuracy = accuracy_score(y_true[low_conf_mask], y_pred[low_conf_mask]) if np.sum(low_conf_mask) > 0 else np.nan
high_conf_accuracy = accuracy_score(y_true[~low_conf_mask], y_pred[~low_conf_mask]) if np.sum(~low_conf_mask) > 0 else np.nan

print("\n" + "="*70)
print("OVERALL PERFORMANCE")
print("="*70)
print(f"Total Samples: {len(y_true)}")
print(f"Overall Accuracy: {accuracy*100:.2f}%")
print(f"Average Confidence: {np.mean(confidences):.4f}")
print(f"Confidence Range: [{np.min(confidences):.4f}, {np.max(confidences):.4f}]")

if not np.isnan(high_conf_accuracy):
    print(f"\nHigh Confidence (>=0.7) Samples: {np.sum(~low_conf_mask)}")
    print(f"High Confidence Accuracy: {high_conf_accuracy*100:.2f}%")

if not np.isnan(low_conf_accuracy):
    print(f"\nLow Confidence (<0.7) Samples: {np.sum(low_conf_mask)}")
    print(f"Low Confidence Accuracy: {low_conf_accuracy*100:.2f}%")

# Per-class metrics
print("\n" + "="*70)
print("PER-CLASS PERFORMANCE")
print("="*70)
print(classification_report(y_true, y_pred, target_names=label_classes, digits=4))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Confusion Matrix Heatmap
ax = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_classes, yticklabels=label_classes,
            ax=ax, cbar=True)
ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')

# 2. Accuracy per class
ax = axes[0, 1]
per_class_accuracy = np.diag(cm) / cm.sum(axis=1)
colors = ['green' if acc > 0.8 else 'orange' if acc > 0.6 else 'red' for acc in per_class_accuracy]
ax.bar(label_classes, per_class_accuracy * 100, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=accuracy*100, color='blue', linestyle='--', label=f'Overall: {accuracy*100:.1f}%')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Per-Class Accuracy', fontsize=12, fontweight='bold')
ax.set_ylim([0, 110])
ax.legend()
for i, (label, acc) in enumerate(zip(label_classes, per_class_accuracy)):
    ax.text(i, acc*100 + 2, f'{acc*100:.1f}%', ha='center', fontsize=9, fontweight='bold')

# 3. Confidence distribution
ax = axes[1, 0]
ax.hist(confidences, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
ax.axvline(x=np.mean(confidences), color='red', linestyle='--', 
           label=f'Mean: {np.mean(confidences):.3f}')
ax.axvline(x=0.7, color='green', linestyle='--', label='Threshold: 0.7')
ax.set_xlabel('Confidence Score')
ax.set_ylabel('Frequency')
ax.set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
ax.legend()

# 4. Correct vs Incorrect by Confidence
ax = axes[1, 1]
correct_mask = y_true == y_pred
correct_confidences = confidences[correct_mask]
incorrect_confidences = confidences[~correct_mask]

parts = ax.violinplot([correct_confidences, incorrect_confidences], 
                       positions=[1, 2], showmeans=True, showmedians=True)
ax.set_xticks([1, 2])
ax.set_xticklabels(['Correct', 'Incorrect'])
ax.set_ylabel('Confidence Score')
ax.set_title('Confidence Score: Correct vs Incorrect', fontsize=12, fontweight='bold')
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
print("\n[+] Visualization saved as: model_evaluation.png")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"Correct Predictions: {np.sum(y_true == y_pred)} / {len(y_true)}")
print(f"Incorrect Predictions: {np.sum(y_true != y_pred)} / {len(y_true)}")
print(f"Mean Confidence (Correct): {np.mean(confidences[y_true == y_pred]):.4f}")
print(f"Mean Confidence (Incorrect): {np.mean(confidences[y_true != y_pred]):.4f}")

# Recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS FOR IMPROVEMENT")
print("="*70)

if accuracy < 0.80:
    print("[!] Accuracy is below 80% - Consider these improvements:")
    print("    1. Collect more training data (target: 300+ per class)")
    print("    2. Try larger model: train_hydenet_advanced.py")
    print("    3. Improve data augmentation")
    print("    4. Check for class imbalance in training data")
elif accuracy < 0.90:
    print("[!] Accuracy is below 90% - Good progress! Next steps:")
    print("    1. Collect more challenging samples")
    print("    2. Switch to EfficientNet-B3")
    print("    3. Try ensemble of multiple models")
    print("    4. Fine-tune last layers of base model")
elif accuracy < 0.95:
    print("[+] Accuracy is 90-95% - Excellent! To reach 95%+:")
    print("    1. Collect 100+ more edge-case samples")
    print("    2. Use EfficientNet-B3 or B4")
    print("    3. Ensemble multiple model architectures")
else:
    print("[+] Accuracy is 95%+ - Outstanding performance!")
    print("    Model is production-ready for most applications.")

print("\n" + "="*70)
print("Evaluation complete!")
print("="*70)
