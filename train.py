import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

# ============================================================
# NORMALIZATION
# ============================================================
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
# DATASET
# ============================================================
class LandmarkDataset(Dataset):
    def __init__(self, csv_file, classes):
        self.data = pd.read_csv(csv_file, header=None)
        self.classes = classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label_str = row[0]
        label = self.class_to_idx[label_str]

        landmarks = row[1:].values.astype(np.float32)
        landmarks = normalize_landmarks(landmarks)
        
        return torch.tensor(landmarks, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ============================================================
# MODEL DEFINITION
# ============================================================
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

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SUCCESS] Device: {device}")

    # Determine classes from directory
    dataset_path = "ASL_dataset_split/train"
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    num_classes = len(classes)
    print(f"[SUCCESS] Found {num_classes} classes: {classes}")

    train_dataset = LandmarkDataset('train_landmarks.csv', classes)
    test_dataset = LandmarkDataset('test_landmarks.csv', classes)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = LandmarkClassifier(num_classes=num_classes, input_dim=126).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 40
    best_val_acc = 0.0
    
    print("Starting training...")
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
            
        train_acc = 100. * correct / total
        
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
                
        val_acc = 100. * val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc
            }, 'asl_landmark_classifier.pth')

        if (epoch+1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            
    print(f"Training complete! Best Val Acc: {best_val_acc:.2f}%. Saved to asl_landmark_classifier.pth")

if __name__ == "__main__":
    train_model()
