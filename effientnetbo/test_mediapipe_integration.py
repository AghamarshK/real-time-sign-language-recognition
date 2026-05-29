# test_mediapipe_integration.py
"""
Test script to verify MediaPipe integration and compare with existing system.
This script tests the MediaPipe hand detection and preprocessing pipeline.
"""

import numpy as np
import cv2
import os
from hyde_net import preprocess_for_efficientnet, build_efficientnet_model

try:
    from mediapipe import solutions
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("âš ï¸  MediaPipe not installed. Install with: pip install mediapipe")

# ---------------------------------------------------------
# Test 1: MediaPipe Installation Check
# ---------------------------------------------------------
def test_mediapipe_installation():
    print("\n" + "="*60)
    print("Test 1: MediaPipe Installation Check")
    print("="*60)
    
    if not MEDIAPIPE_AVAILABLE:
        print("âœ— MediaPipe is NOT installed")
        print("  Install with: pip install mediapipe")
        return False
    
    print("âœ“ MediaPipe successfully imported")
    
    # Check version
    try:
        import mediapipe
        mp_version = mediapipe.__version__
        print(f"âœ“ MediaPipe version: {mp_version}")
    except:
        print("âš ï¸  Could not determine MediaPipe version")
    
    # Check Hands module
    try:
        hands = solutions.hands.Hands()
        print("âœ“ MediaPipe Hands module available")
        hands.close()
        return True
    except Exception as e:
        print(f"âœ— Error loading Hands module: {e}")
        return False

# ---------------------------------------------------------
# Test 2: Model Loading Check
# ---------------------------------------------------------
def test_model_loading():
    print("\n" + "="*60)
    print("Test 2: Model & Labels Loading")
    print("="*60)
    
    # Check label file
    if not os.path.exists("label_classes.npy"):
        print("âœ— label_classes.npy not found")
        return False
    
    try:
        labels = np.load("label_classes.npy", allow_pickle=True)
        num_classes = len(labels)
        print(f"âœ“ Labels loaded: {num_classes} classes")
        print(f"  Classes: {list(labels)}")
    except Exception as e:
        print(f"âœ— Error loading labels: {e}")
        return False
    
    # Check model file
    if not os.path.exists("efficientnet_asl.h5"):
        print("âš ï¸  efficientnet_asl.h5 not found - will create model without weights")
        model_exists = False
    else:
        model_exists = True
    
    # Try to build and load model
    try:
        model = build_efficientnet_model(num_classes=num_classes)
        print(f"âœ“ Model built successfully")
        
        if model_exists:
            model.load_weights("efficientnet_asl.h5")
            print(f"âœ“ Model weights loaded")
        else:
            print(f"âš ï¸  Model created but weights not loaded")
        
        return True
    except Exception as e:
        print(f"âœ— Error building/loading model: {e}")
        return False

# ---------------------------------------------------------
# Test 3: Preprocessing Pipeline
# ---------------------------------------------------------
def test_preprocessing_pipeline():
    print("\n" + "="*60)
    print("Test 3: Hand Cropping & Preprocessing Pipeline")
    print("="*60)
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    print(f"âœ“ Created dummy image: {dummy_image.shape}")
    
    # Test crop
    crop = dummy_image[50:250, 50:250]
    print(f"âœ“ Cropped region: {crop.shape}")
    
    # Test preprocessing
    try:
        processed = preprocess_for_efficientnet(crop, size=(224, 224))
        print(f"âœ“ Preprocessed image: {processed.shape}")
        print(f"  Data type: {processed.dtype}")
        print(f"  Value range: [{processed.min():.1f}, {processed.max():.1f}]")
        
        assert processed.shape == (224, 224, 3), "Shape mismatch!"
        assert processed.dtype == np.float32, "Data type should be float32"
        
        print("âœ“ All preprocessing checks passed")
        return True
    except Exception as e:
        print(f"âœ— Preprocessing error: {e}")
        return False

# ---------------------------------------------------------
# Test 4: Hand Detection Simulation
# ---------------------------------------------------------
def test_hand_detection_simulation():
    print("\n" + "="*60)
    print("Test 4: Hand Bounding Box Extraction")
    print("="*60)
    
    if not MEDIAPIPE_AVAILABLE:
        print("âš ï¸  Skipping (MediaPipe not available)")
        return False
    
    # Import the function
    from mediapipe_gesture_recognition import get_hand_bounding_box
    
    # Create mock landmarks
    class MockLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
    # Create mock hand (closed fist-like bounding box)
    landmarks = []
    for i in range(21):  # 21 landmarks
        x = 0.4 + np.random.uniform(-0.1, 0.1)
        y = 0.5 + np.random.uniform(-0.1, 0.1)
        landmarks.append(MockLandmark(x, y))
    
    # Test bbox extraction
    try:
        bbox = get_hand_bounding_box(landmarks, frame_width=640, frame_height=480, padding=20)
        
        if bbox is None:
            print("âš ï¸  Bounding box returned None (possible invalid landmarks)")
            return False
        
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        print(f"âœ“ Bounding box extracted:")
        print(f"  Position: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"  Size: {width}x{height}")
        
        assert width > 0 and height > 0, "Invalid bbox dimensions"
        assert width < 640 and height < 480, "Bbox larger than frame"
        
        print("âœ“ Bounding box validation passed")
        return True
    except Exception as e:
        print(f"âœ— Bounding box error: {e}")
        return False

# ---------------------------------------------------------
# Test 5: Camera Check
# ---------------------------------------------------------
def test_camera_check():
    print("\n" + "="*60)
    print("Test 5: Camera Availability")
    print("="*60)
    
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("âœ— Camera not available (index 0)")
            print("  Try a different index or check camera connection")
            return False
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"âœ“ Camera detected on index 0")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        
        # Try to capture a frame
        ret, frame = cap.read()
        if ret:
            print(f"âœ“ Successfully captured frame: {frame.shape}")
        else:
            print("âœ— Could not capture frame")
            return False
        
        cap.release()
        print("âœ“ Camera check passed")
        return True
    except Exception as e:
        print(f"âœ— Camera error: {e}")
        return False

# ---------------------------------------------------------
# Test 6: End-to-End Pipeline (if all pass)
# ---------------------------------------------------------
def test_end_to_end():
    print("\n" + "="*60)
    print("Test 6: End-to-End Pipeline Simulation")
    print("="*60)
    
    if not MEDIAPIPE_AVAILABLE:
        print("âš ï¸  Skipping (MediaPipe not available)")
        return False
    
    try:
        import mediapipe as mp
        from mediapipe_gesture_recognition import get_hand_bounding_box, crop_and_preprocess_hand
        
        # Load labels
        labels = np.load("label_classes.npy", allow_pickle=True)
        
        # Create dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"âœ“ Created dummy frame: {frame.shape}")
        
        # Create mock hand
        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        
        landmarks = []
        for i in range(21):
            x = 0.5 + np.random.uniform(-0.1, 0.1)
            y = 0.5 + np.random.uniform(-0.1, 0.1)
            landmarks.append(MockLandmark(np.clip(x, 0.1, 0.9), np.clip(y, 0.1, 0.9)))
        
        # Get bounding box
        bbox = get_hand_bounding_box(landmarks, 640, 480, padding=20)
        print(f"âœ“ Extracted bounding box: {bbox}")
        
        # Crop and preprocess
        preprocessed = crop_and_preprocess_hand(frame, bbox)
        print(f"âœ“ Preprocessed image: {preprocessed.shape if preprocessed is not None else 'None'}")
        
        # Build model
        model = build_efficientnet_model(len(labels))
        print(f"âœ“ Model built with {len(labels)} output classes")
        
        # Test prediction
        batch = np.expand_dims(preprocessed, axis=0)
        preds = model.predict(batch, verbose=0)
        print(f"âœ“ Model prediction: {preds.shape}")
        
        pred_class = np.argmax(preds)
        confidence = preds[0, pred_class]
        print(f"âœ“ Predicted class: {labels[pred_class]} (confidence: {confidence:.3f})")
        
        print("âœ“ End-to-end pipeline successful")
        return True
    except Exception as e:
        print(f"âœ— End-to-end error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ---------------------------------------------------------
# Main Test Runner
# ---------------------------------------------------------
def main():
    print("\n" + "="*60)
    print("MediaPipe Gesture Recognition - Integration Test Suite")
    print("="*60)
    
    results = {}
    
    # Run tests
    results["MediaPipe Installation"] = test_mediapipe_installation()
    results["Model & Labels"] = test_model_loading()
    results["Preprocessing Pipeline"] = test_preprocessing_pipeline()
    results["Hand Detection"] = test_hand_detection_simulation()
    results["Camera"] = test_camera_check()
    results["End-to-End"] = test_end_to_end()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status = "âœ“ PASS" if result else "âœ— FAIL"
        print(f"{status:8} | {test_name}")
    
    all_passed = all(results.values())
    
    print("="*60)
    if all_passed:
        print("âœ“ ALL TESTS PASSED - System is ready!")
        print("\nNext step: Run the main application")
        print("  python mediapipe_gesture_recognition.py")
    else:
        print("âœ— Some tests failed - See details above")
        print("\nCommon fixes:")
        print("  1. Install MediaPipe: pip install mediapipe")
        print("  2. Check camera connection")
        print("  3. Verify model files exist (efficientnet_asl.h5, label_classes.npy)")
    
    print("="*60)

if __name__ == "__main__":
    main()

