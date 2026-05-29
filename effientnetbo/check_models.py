# check_models.py
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import pickle

def check_model_files():
    print("\nChecking model files...")
    files_to_check = {
        'DAP Model': 'hydenet_asl.h5',
        'LDA Model': 'hydenet_lda.h5',
        'LDA Descriptor': 'lda_model.pkl',
        'Label Classes': 'label_classes.npy'
    }
    
    all_present = True
    for name, file in files_to_check.items():
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"✓ {name:<15} found ({size_mb:.1f} MB)")
        else:
            print(f"✗ {name:<15} missing!")
            all_present = False
    
    return all_present

def check_dataset():
    print("\nChecking dataset structure...")
    train_dir = "dataset/train"
    test_dir = "dataset/test"
    
    # Check train directory
    if os.path.exists(train_dir):
        train_classes = [d for d in os.listdir(train_dir) 
                        if os.path.isdir(os.path.join(train_dir, d))]
        print(f"Training classes found: {train_classes}")
        
        # Count training images
        total_train = 0
        for cls in train_classes:
            class_path = os.path.join(train_dir, cls)
            n_images = len([f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  Class {cls}: {n_images} images")
            total_train += n_images
        print(f"Total training images: {total_train}")
    else:
        print("✗ Training dataset not found!")
    
    # Check test directory
    if os.path.exists(test_dir):
        test_files = os.listdir(test_dir)
        print(f"\nTest files found: {test_files}")
    else:
        print("✗ Test dataset not found!")

def main():
    print("=== Model Status Check ===")
    
    # Check if all required files exist
    if not check_model_files():
        print("\n⚠️ Some model files are missing. You may need to train the models first.")
        print("\nTo train DAP model:")
        print("  python train_hydenet.py")
        print("\nTo train LDA model:")
        print("1. First fit LDA:")
        print("  python fit_lda.py")
        print("2. Then train neural network:")
        print("  python train_hydenet_lda.py")
        return
    
    # Check dataset
    check_dataset()
    
    # Try to load models
    print("\nTrying to load models...")
    try:
        dap_model = load_model("hydenet_asl.h5")
        print("✓ DAP model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading DAP model: {str(e)}")
    
    try:
        lda_model = load_model("hydenet_lda.h5")
        print("✓ LDA neural network loaded successfully")
    except Exception as e:
        print(f"✗ Error loading LDA model: {str(e)}")
    
    try:
        with open('lda_model.pkl', 'rb') as f:
            lda = pickle.load(f)
        print("✓ LDA descriptor loaded successfully")
    except Exception as e:
        print(f"✗ Error loading LDA descriptor: {str(e)}")
    
    print("\nNext steps:")
    print("1. To test DAP model:")
    print("   python test_hydenet.py")
    print("2. To test LDA model:")
    print("   python test_hydenet_lda.py")
    print("3. To compare both models:")
    print("   python quick_compare.py")

if __name__ == "__main__":
    main()