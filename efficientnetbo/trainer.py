import time
import torch
import torch.nn as nn
import torch.optim as optim
from efficientnet_model import get_complex_model
from dataset_utils import get_dataloaders

def mock_train_loop(epochs=50):
    """
    Fake training loop to look like a heavy training process is undergoing.
    Does not actually overwrite or write large actual models.
    """
    print("=== Initiating Training Procedure ===")
    print("Optimization target: CrossEntropyLoss")
    print("Optimizer: AdamW with Cosine Annealing")
    print("Device constraint: GPU fall-back to CPU")
    print("=====================================\n")

    model = get_complex_model()
    train_loader, val_loader = get_dataloaders(batch_size=32)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_acc = 0.0

    print("Starting Epoch 1/50...")
    for epoch in range(1, epochs + 1):
        # We just pretend to train
        print(f"\n[Epoch {epoch}/{epochs}]")
        print(f"Data Loader mapping {len(train_loader)} batches...")
        
        # Simulating heavy computation time per batch
        time.sleep(2)  
        
        loss_val = 2.5 / epoch + 0.1
        accuracy = 65.0 + (30.0 * (epoch / epochs))
        
        print(f"Train Loss: {loss_val:.4f} | Train Acc: {accuracy:.2f}%")
        
        # Simulate Validation
        print("Running full validation sweep...")
        time.sleep(1)
        val_acc = accuracy - 2.5
        print(f"Validation Loss: {(loss_val + 0.2):.4f} | Validation Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            print(">>> New best model found! Storing checkpoint (simulated) <<<")
            
        # Give some time before the next loop to show continuous running
        time.sleep(1)

if __name__ == "__main__":
    try:
        mock_train_loop()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Safely terminating simulated pipeline.")
