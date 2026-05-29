# compare_models.py
# Compare accuracy between different models

import os
import numpy as np
import tensorflow as tf
from hyde_net import build_efficientnet_model
import matplotlib.pyplot as plt

# Configuration
MODELS = [
    ("efficientnet_asl.h5", "Baseline (B0)"),
    ("efficientnet_asl_improved.h5", "Improved (B0)"),
    ("efficientnet_b2_advanced.h5", "Advanced (B2)"),
]

LABELS_FILE = "label_classes.npy"
TEST_DIR = "dataset/test"
IMG_SIZE = (224, 224)

print("="*70)
print("Model Comparison Tool")
print("="*70 + "\n")

# Load labels
if not os.path.exists(LABELS_FILE):
    print(f"[X] {LABELS_FILE} not found")
    exit(1)

label_classes = np.load(LABELS_FILE, allow_pickle=True)
num_classes = len(label_classes)

# Load test dataset
print("Loading test dataset...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=16,
    shuffle=False,
    label_mode='categorical'
)

# Prepare test data
y_true_list = []
images_list = []
for images, labels in test_ds:
    images_list.append(images.numpy())
    y_true_list.append(np.argmax(labels.numpy(), axis=1))

images = np.concatenate(images_list, axis=0)
y_true = np.concatenate(y_true_list, axis=0)

print(f"Test samples: {len(y_true)}\n")

# Evaluate each model
results = {}

for model_file, model_name in MODELS:
    print(f"Evaluating: {model_name}")
    
    if not os.path.exists(model_file):
        print(f"  [!] Model file not found: {model_file}")
        continue
    
    try:
        # Build and load model
        model = build_efficientnet_model(num_classes=num_classes)
        model.load_weights(model_file)
        
        # Get predictions
        predictions = model.predict(images, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(y_true == y_pred)
        
        # Per-class accuracy
        per_class_acc = []
        for class_idx in range(num_classes):
            mask = y_true == class_idx
            if np.sum(mask) > 0:
                class_acc = np.mean(y_pred[mask] == y_true[mask])
                per_class_acc.append(class_acc)
            else:
                per_class_acc.append(0)
        
        results[model_name] = {
            'accuracy': accuracy,
            'per_class': per_class_acc
        }
        
        print(f"  ✓ Accuracy: {accuracy*100:.2f}%")
        print(f"    Per-class accuracy:")
        for label, acc in zip(label_classes, per_class_acc):
            print(f"    - {label}: {acc*100:.1f}%")
        print()
        
    except Exception as e:
        print(f"  ✗ Error: {e}\n")

# Display comparison
print("="*70)
print("COMPARISON SUMMARY")
print("="*70 + "\n")

print(f"{'Model':<30} {'Overall Accuracy':<20} {'Improvement':<15}")
print("-" * 70)

baseline_acc = None
for model_name, data in results.items():
    acc = data['accuracy']
    if baseline_acc is None:
        baseline_acc = acc
        improvement = "Baseline"
    else:
        improvement = f"+{(acc - baseline_acc)*100:.2f}%"
    
    print(f"{model_name:<30} {acc*100:<19.2f}% {improvement:<15}")

# Visualization
if len(results) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Overall accuracy comparison
    ax = axes[0]
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy']*100 for m in model_names]
    colors = ['red' if i == 0 else 'orange' if i == 1 else 'green' for i in range(len(model_names))]
    
    bars = ax.bar(range(len(model_names)), accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Overall Accuracy Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 105])
    
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Per-class accuracy comparison
    ax = axes[1]
    x = np.arange(len(label_classes))
    width = 1 / (len(results) + 1)
    
    for i, model_name in enumerate(model_names):
        per_class = [acc*100 for acc in results[model_name]['per_class']]
        ax.bar(x + i*width, per_class, width, label=model_name, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Gesture Class')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(label_classes)
    ax.legend()
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("\n[+] Comparison chart saved as: model_comparison.png")

# Recommendation
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70 + "\n")

if results:
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_name, best_data = best_model
    best_acc = best_data['accuracy']
    
    print(f"Best model: {best_name}")
    print(f"Accuracy: {best_acc*100:.2f}%")
    
    if best_acc < 0.80:
        print("\n[!] Accuracy < 80% - Collect more training data and retrain")
    elif best_acc < 0.90:
        print("\n[+] Good accuracy - Consider using this model in production")
        print("    Or collect more data for 90%+ accuracy")
    elif best_acc < 0.95:
        print("\n[+] Excellent accuracy - Production ready!")
        print("    Model performs well enough for most applications")
    else:
        print("\n[++] Outstanding accuracy - Excellent performance!")
        print("    Model is suitable for high-accuracy requirements")

print("\n" + "="*70)
