# MEDIAPIPE_SETUP_GUIDE.md

## MediaPipe Hand Detection Integration Guide

This guide explains how to use the new **MediaPipe Gesture Recognition** system with your trained EfficientNet-B0 model.

---

## What's New

### System Architecture

```
Webcam Frame (captured)
        ↓
   [MediaPipe Hands Detection]
        ↓
   Extract Hand Landmarks
        ↓
   Compute Bounding Box
        ↓
   Crop Hand Region
        ↓
   Preprocess to 224×224
        ↓
   [EfficientNet-B0 Classification]
        ↓
   Get Prediction + Confidence
        ↓
   Temporal Smoothing (optional)
        ↓
   Display on Screen with Visualization
```

### Key Features

1. **MediaPipe Hand Detection**
   - Detects hand position and 21 landmarks in real-time
   - Works reliably under various lighting conditions
   - Supports single-hand detection

2. **Automatic Hand Cropping**
   - Computes bounding box from landmarks
   - Crops the hand region automatically
   - Resizes to 224×224 for EfficientNet-B0
   - Applies proper normalization

3. **Real-time Classification**
   - Uses your trained EfficientNet-B0 model
   - Monte Carlo Dropout for uncertainty estimation
   - Temporal smoothing for stable predictions

4. **Visual Feedback**
   - Green bounding box around detected hand
   - Red landmarks showing key hand points
   - Gesture label, confidence score, and uncertainty displayed
   - FPS counter
   - Auto-save on stable predictions

---

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `mediapipe` - Hand detection framework
- `opencv-python` - Video capture and visualization
- `tensorflow` - Model inference
- Other dependencies for your EfficientNet model

### 2. Verify Model Files

Make sure you have:
- `efficientnet_asl.h5` - Trained model weights
- `label_classes.npy` - Class labels

---

## Usage

### Run the MediaPipe Gesture Recognition System

```bash
python mediapipe_gesture_recognition.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `s` | Manually save current frame |
| `d` | Toggle landmark visualization |

### Console Output

```
✓ Weights loaded from efficientnet_asl.h5
✓ MediaPipe Hands initialized
✓ Camera opened: 640x480

============================================================
Real-time Gesture Recognition with MediaPipe Hand Detection
============================================================
Controls:
  'q'     - Quit application
  's'     - Manually save current frame
  'd'     - Toggle detection visualization
============================================================
```

---

## How It Works

### Step-by-Step Pipeline

#### 1. Frame Capture
```python
ret, frame = cap.read()
frame = cv2.flip(frame, 1)  # Mirror for natural view
```

#### 2. Hand Detection with MediaPipe
```python
results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
hand_landmarks = results.multi_hand_landmarks[0]  # Get first hand
```

#### 3. Bounding Box Extraction
```python
bbox = get_hand_bounding_box(
    hand_landmarks.landmark,
    frame_width, frame_height,
    padding=20
)
# Returns: (x_min, y_min, x_max, y_max)
```

#### 4. Hand Cropping & Preprocessing
```python
cropped = frame[y_min:y_max, x_min:x_max]
preprocessed = preprocess_for_efficientnet(cropped, size=(224, 224))
```

#### 5. Classification
```python
# Monte Carlo Dropout for uncertainty
mc_batch = np.repeat(input_batch, MC_PASSES, axis=0)
all_preds = model.predict(mc_batch, verbose=0)

# Temporal smoothing
smoothed_probs = np.mean(prob_buffer, axis=0)
prediction = np.argmax(smoothed_probs)
```

#### 6. Visualization
```python
# Draw bounding box
cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)

# Draw landmarks
for landmark in hand_landmarks:
    cv2.circle(frame, (x, y), radius, color, -1)

# Display text
cv2.putText(frame, f"Gesture: {label}", ...)
```

---

## Configuration Options

Open `mediapipe_gesture_recognition.py` and modify these settings:

### Model & Detection
```python
MODEL_WEIGHTS = "efficientnet_asl.h5"          # Model file
LABELS_FILE = "label_classes.npy"              # Label classes
CONFIDENCE_THRESHOLD = 0.70                    # Min confidence for "confident" prediction
MC_PASSES = 10                                 # Monte Carlo inference passes
```

### MediaPipe
```python
MEDIAPIPE_CONFIDENCE = 0.7                     # Hand detection confidence
HAND_DETECTION_MODEL = "full"                  # "lite" (faster) or "full" (more accurate)
```

### Smoothing & Auto-save
```python
BUFFER_SIZE = 8                                # Temporal smoothing buffer
CONSECUTIVE_FRAMES = 5                         # Frames needed for stable prediction
```

### Visualization
```python
HAND_BOX_COLOR = (0, 255, 0)                  # Green for bounding box
LANDMARK_COLOR = (0, 0, 255)                  # Red for landmarks
HAND_BOX_THICKNESS = 2                        # Line thickness
```

---

## Expected Output

### On-screen Display

```
┌─────────────────────────────────────────────────┐
│ Gesture: A                                      │
│ Confidence: 0.95                               │
│ Uncertainty: 0.12                              │
│ Status: DETECTED                               │
│                                                 │
│ [Green bounding box around hand]               │
│ [Red dots on hand landmarks]                   │
│                                           FPS: 28.5 │
└─────────────────────────────────────────────────┘
```

### Console Feedback

```
✓ Auto-captured: auto_A_1710154623001.jpg
✓ Manually saved: manual_1710154625123.jpg
✓ Landmarks visualization: ON
✓ Exiting application...
✓ Application closed successfully
```

---

## Troubleshooting

### Issue: "No Hand Detected"
- **Cause**: Hand too small or outside frame
- **Solution**: 
  - Move hand closer to camera
  - Ensure good lighting
  - Keep hand fully visible in frame

### Issue: "Hand Too Small"
- **Cause**: Bounding box is too small for proper classification
- **Solution**: 
  - Move hand closer to camera
  - Adjust `padding` parameter in `get_hand_bounding_box()` if needed

### Issue: Low Confidence Predictions
- **Cause**: Unusual hand position or poor lighting
- **Solution**:
  - Ensure good lighting
  - Position hand naturally
  - Adjust `CONFIDENCE_THRESHOLD` (lower = more predictions)

### Issue: Jerky/Unstable Predictions
- **Cause**: Temporal buffer too small or model variance
- **Solution**:
  - Increase `BUFFER_SIZE` (more smoothing)
  - Increase `CONSECUTIVE_FRAMES` for auto-save
  - Improve lighting

### Issue: MediaPipe Not Found
- **Cause**: MediaPipe not installed
- **Solution**:
  ```bash
  pip install mediapipe
  ```

### Issue: Camera Not Working
- **Cause**: Camera not connected or in use by another app
- **Solution**:
  - Check camera connection
  - Close other apps using camera
  - Try `cv2.VideoCapture(1)` if multiple cameras exist

---

## Comparison: Old vs. New System

### Old System (ROI-based)
```
Fixed 250×250 ROI in center of frame
        ↓
Crop from center
        ↓
Classify
        ↓
Works well with constrained setup
```

### New System (MediaPipe-based)
```
Hand position anywhere in frame
        ↓
MediaPipe detects hand location
        ↓
Automatic bounding box extraction
        ↓
Intelligent cropping and preprocessing
        ↓
Classify
        ↓
Works well with natural hand movements
```

---

## Performance Metrics

### Typical Performance (On CPU)
- **Hand Detection**: ~30ms per frame
- **Image Preprocessing**: ~5ms per frame
- **EfficientNet Classification**: ~50-100ms per frame (depends on MC_PASSES)
- **Overall FPS**: 8-15 FPS (with MC_PASSES=10)

### On GPU
- **Overall FPS**: 25-30 FPS (with MC_PASSES=10)

### Optimization Tips
1. **Reduce MC_PASSES**: Use 5 instead of 10 for faster inference (accuracy trade-off)
2. **Use "lite" model**: `HAND_DETECTION_MODEL = "lite"` (faster but less accurate)
3. **Disable landmarks**: Set `show_landmarks = False` to skip drawing
4. **Reduce buffer**: Smaller `BUFFER_SIZE` for less smoothing

---

## Advanced Usage

### Custom Hand Detection Confidence
```python
# In mediapipe_gesture_recognition.py
MEDIAPIPE_CONFIDENCE = 0.5  # Lower = more detections, more false positives
```

### Adjusting Bounding Box Padding
```python
# In mediapipe_gesture_recognition.py, line that calls get_hand_bounding_box:
bbox = get_hand_bounding_box(hand_landmarks.landmark, width, height, padding=30)
# Higher padding = larger crop area
```

### Disabling Auto-save
```python
# Comment out the auto-save section or set:
CONSECUTIVE_FRAMES = float('inf')  # Never auto-save
```

---

## Files Overview

### Main Script
- **mediapipe_gesture_recognition.py** - The main real-time recognition system

### Supporting Files (Already Exists)
- **hyde_net.py** - EfficientNet model builder and preprocessing
- **efficientnet_asl.h5** - Trained model weights
- **label_classes.npy** - Class label mappings

### Configuration
- **requirements.txt** - Updated with mediapipe dependency

---

## Next Steps

1. ✅ Install MediaPipe: `pip install -r requirements.txt`
2. ✅ Run the system: `python mediapipe_gesture_recognition.py`
3. ✅ Test with hand gestures in front of camera
4. ✅ Press 's' to manually save frames, 'q' to quit
5. ✅ Check saved images in `dataset/test/captured_gestures/`

---

## Integration with Existing System

This new system **complements** your existing setup:
- ✅ Uses the same `efficientnet_asl.h5` model
- ✅ Compatible with your training pipeline
- ✅ Works alongside `realtime_gesture_recognition.py` (old ROI-based system)
- ✅ Can be extended to multi-hand detection
- ✅ Supports retraining with auto-captured data

---

## Support & Documentation

For more information about:
- **MediaPipe**: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- **EfficientNet**: https://github.com/keras-team/keras-applications
- **OpenCV**: https://docs.opencv.org/

---
