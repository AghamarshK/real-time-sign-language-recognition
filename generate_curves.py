import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

# Import the existing model and dataset definition from train.py
from train import LandmarkClassifier, LandmarkDataset

def quick_train_and_plot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load dataset classes
    dataset_path = "ASL_dataset_split/train"
    if not os.path.exists(dataset_path):
        classes = list("abcdefghijklmnopqrstuvwxyz")
    else:
        classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    num_classes = len(classes)
    print(f"[INFO] Found {num_classes} classes.")

    train_dataset = LandmarkDataset('train_landmarks.csv', classes)
    test_dataset = LandmarkDataset('test_landmarks.csv', classes)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # We need a non-shuffled train_loader for the ROC evaluation
    train_eval_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    # Initialize a FRESH model specifically for generating curves, 
    # without affecting asl_landmark_classifier.pth
    model = LandmarkClassifier(num_classes=num_classes, input_dim=126).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 25 # Quick training specifically to generate visualization curves
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"[INFO] Starting independent quick training for {epochs} epochs to generate curves...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_losses.append(train_loss / total)
        train_accs.append(100. * correct / total)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_losses.append(val_loss / val_total)
        val_accs.append(100. * val_correct / val_total)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"       Epoch [{epoch+1}/{epochs}] | Train Loss: {train_losses[-1]:.4f} | "
                  f"Val Loss: {val_losses[-1]:.4f} | Train Acc: {train_accs[-1]:.2f}% | Val Acc: {val_accs[-1]:.2f}%")

    print("[INFO] Training complete. Generating curve visualizations...")

    sns_style_installed = False
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        sns_style_installed = True
    except ImportError:
        pass

    # 1. Plot Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss', color='blue', marker='o')
    plt.plot(range(1, epochs+1), val_losses, label='Test Loss', color='red', marker='s')
    plt.title('Training and Testing Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if not sns_style_installed: plt.grid(True)
    plt.savefig('loss_curve.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 2. Plot Accuracy Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_accs, label='Train Accuracy', color='blue', marker='o')
    plt.plot(range(1, epochs+1), val_accs, label='Test Accuracy', color='red', marker='s')
    plt.title('Training and Testing Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    if not sns_style_installed: plt.grid(True)
    plt.savefig('accuracy_curve.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 3. Generate ROC Curves
    def get_preds(loader):
        model.eval()
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return np.array(all_labels), np.array(all_probs)

    y_train, y_train_prob = get_preds(train_eval_loader)
    y_test, y_test_prob = get_preds(test_loader)

    # Binarize labels for roc curve
    y_train_bin = label_binarize(y_train, classes=range(num_classes))
    y_test_bin = label_binarize(y_test, classes=range(num_classes))

    def plot_roc(y_true, y_prob, title, filename):
        fpr, tpr, _ = roc_curve(y_true.ravel(), y_prob.ravel())
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Micro-average ROC (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        if not sns_style_installed: plt.grid(True)
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

    plot_roc(y_train_bin, y_train_prob, 'ROC Curve (Train Data)', 'roc_curve_train.png')
    plot_roc(y_test_bin, y_test_prob, 'ROC Curve (Test Data)', 'roc_curve_test.png')

    print("[SUCCESS] All curves generated successfully:")
    print(" - loss_curve.png")
    print(" - accuracy_curve.png")
    print(" - roc_curve_train.png")
    print(" - roc_curve_test.png")

if __name__ == "__main__":
    quick_train_and_plot()
