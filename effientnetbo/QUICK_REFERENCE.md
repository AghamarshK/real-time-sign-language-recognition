# QUICK_REFERENCE.md

# MediaPipe Gesture Recognition - Quick Reference Card

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test integration
python test_mediapipe_integration.py

# 3. Run application
python mediapipe_gesture_recognition.py
```

---

## 📱 Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save current frame |
| `d` | Toggle landmarks |

---

## 📊 On-Screen Display

```
┌─────────────────────────────────┐
│ Gesture: A                      │
│ Confidence: 0.95                │
│ Uncertainty: 0.12               │
│ Status: DETECTED                │
│                             FPS: 28 │
└─────────────────────────────────┘
```

---

## ⚙️ Key Configuration Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `MC_PASSES` | 10 | MC inference iterations (higher = slower, more robust) |
| `CONFIDENCE_THRESHOLD` | 0.70 | Min confidence for "DETECTED" status |
| `BUFFER_SIZE` | 8 | Temporal smoothing (higher = smoother) |
| `CONSECUTIVE_FRAMES` | 5 | Frames for auto-save stability |
| `MEDIAPIPE_CONFIDENCE` | 0.7 | Hand detection threshold |
| `HAND_DETECTION_MODEL` | "full" | "lite" (fast) or "full" (accurate) |

---

## 🎯 Common Configurations

### Fast Mode (Real-time Priority)
```python
MC_PASSES = 5
HAND_DETECTION_MODEL = "lite"
BUFFER_SIZE = 3
```
**Result**: ~25-30 FPS on GPU, ~15-20 FPS on CPU

### Accurate Mode (Quality Priority)
```python
MC_PASSES = 20
HAND_DETECTION_MODEL = "full"
BUFFER_SIZE = 12
CONFIDENCE_THRESHOLD = 0.80
```
**Result**: ~6-10 FPS on CPU, ~15-20 FPS on GPU

### Balanced Mode (Recommended)
```python
MC_PASSES = 10
HAND_DETECTION_MODEL = "full"
BUFFER_SIZE = 8
```
**Result**: ~12-20 FPS on CPU, ~25-30 FPS on GPU

---

## 📁 File Structure

```
ML/
├── mediapipe_gesture_recognition.py  ← MAIN APPLICATION
├── test_mediapipe_integration.py     ← TEST SUITE
├── hyde_net.py                       ← MODEL DEFINITION
├── efficientnet_asl.h5               ← TRAINED WEIGHTS
├── label_classes.npy                 ← CLASS LABELS
├── requirements.txt                  ← DEPENDENCIES
├── QUICKSTART_MEDIAPIPE.md           ← Quick start
├── MEDIAPIPE_SETUP_GUIDE.md          ← Detailed guide
├── CONFIGURATION_GUIDE.md            ← Config options
├── SYSTEM_ARCHITECTURE.md            ← Architecture docs
└── dataset/test/captured_gestures/   ← Auto-saved frames
```

---

## 🔍 Pipeline Stages

```
1. Capture Frame (30 FPS)
2. Hand Detection (MediaPipe)
3. Bounding Box Extraction
4. Hand Cropping & Resizing (224×224)
5. EfficientNet-B0 Classification
6. Monte Carlo Uncertainty Estimation
7. Temporal Smoothing
8. Display with Visualization
```

---

## 📈 Performance Targets

### Expected FPS
- **CPU (Intel i7)**: 10-20 FPS
- **GPU (NVIDIA)**: 25-30 FPS
- **Mobile (CPU)**: 5-10 FPS

### Expected Inference Time
- **Hand Detection**: 20-30ms
- **Classification (10 MC passes)**: 100-150ms
- **Total**: 150-200ms per frame

---

## 🛠️ Troubleshooting Checklist

| Issue | Solution |
|-------|----------|
| No hand detected | Move hand closer, improve lighting |
| Hand too small | Move hand closer to camera |
| Low confidence | Improve lighting, natural hand position |
| Slow FPS | Reduce MC_PASSES, use "lite" model |
| Unstable predictions | Increase BUFFER_SIZE |
| MediaPipe error | `pip install mediapipe` |
| Camera error | Check connection, try different index |

---

## 🎓 Key Concepts

### Monte Carlo Dropout
- Multiple predictions with dropout enabled
- Average predictions for robustness
- Std of predictions = uncertainty

### Temporal Smoothing
- Buffer of recent predictions
- Average to reduce noise/jitter
- Trade-off: smoothness vs. responsiveness

### Bounding Box
- Extracted from 21 hand landmarks
- Includes padding for complete hand region
- Adaptive to hand size and position

### MediaPipe Landmarks
```
0   = Wrist
1-4 = Thumb
5-8 = Index
9-12= Middle
13-16 = Ring
17-20 = Pinky
```

---

## 💾 File I/O

### Auto-Save Triggered When
- Confidence > CONFIDENCE_THRESHOLD
- Same gesture detected for CONSECUTIVE_FRAMES frames
- Files saved to: `dataset/test/captured_gestures/`

### Manual Save
- Press 's' key to save current frame

### Output Filename Format
- Auto: `auto_{gesture}_{timestamp}.jpg`
- Manual: `manual_{timestamp}.jpg`

---

## 📊 Model Specs

| Spec | Value |
|------|-------|
| Input size | 224×224×3 |
| Base model | EfficientNetB0 (ImageNet pretrained) |
| Output classes | Depends on training (e.g., 4 for A,B,C,D) |
| Model size | ~44 MB |
| Dropout rate | 0.2 (MC Dropout) |

---

## 🎨 Customization

### Change Colors
```python
HAND_BOX_COLOR = (0, 255, 0)    # Green box
LANDMARK_COLOR = (0, 0, 255)    # Red landmarks
```

### Change Visualization
```python
HAND_BOX_THICKNESS = 3          # Thicker box
LANDMARK_RADIUS = 5             # Bigger landmarks
```

### Change Bounding Box Padding
```python
# In mediapipe_gesture_recognition.py, line ~240:
bbox = get_hand_bounding_box(hand_landmarks.landmark, 
                             width, height, padding=30)
```

---

## 📊 Metrics Explained

### Gesture
- Predicted hand gesture class (from label_classes.npy)

### Confidence
- Probability of predicted class (0.0-1.0)
- Higher = more certain
- Used for CONFIDENCE_THRESHOLD check

### Uncertainty
- Standard deviation of MC predictions
- Lower = consistent predictions
- Inverse relationship with reliability

### Status
- **DETECTED**: confidence > CONFIDENCE_THRESHOLD (green)
- **LOW CONF**: confidence ≤ CONFIDENCE_THRESHOLD (red)

### FPS
- Frames per second being processed
- Real-time performance indicator

---

## 🐛 Debug Tips

### Enable Verbose Mode
```python
model.predict(mc_batch, verbose=1)  # Instead of verbose=0
```

### Print Debug Info
Add to `mediapipe_gesture_recognition.py`:
```python
print(f"Hand detected: {results.multi_hand_landmarks is not None}")
print(f"Bbox: {bbox}")
print(f"Preprocessed shape: {preprocessed_hand.shape}")
print(f"Predictions shape: {all_preds.shape}")
```

### Monitor Performance
```python
import time
start = time.time()
# ... do something ...
print(f"Time: {time.time() - start:.3f}s")
```

---

## 📚 Documentation Structure

| File | Purpose |
|------|---------|
| `QUICKSTART_MEDIAPIPE.md` | 3-step quick start |
| `MEDIAPIPE_SETUP_GUIDE.md` | Complete setup guide |
| `CONFIGURATION_GUIDE.md` | All configuration options |
| `SYSTEM_ARCHITECTURE.md` | System design & data flow |
| `QUICK_REFERENCE.md` | This file |

---

## 🔗 External Resources

- **MediaPipe**: https://developers.google.com/mediapipe
- **EfficientNet**: https://github.com/keras-team/keras-applications
- **OpenCV**: https://docs.opencv.org/
- **TensorFlow**: https://www.tensorflow.org/

---

## ✅ Pre-flight Checklist

Before running, verify:

- [ ] MediaPipe installed: `pip list | grep mediapipe`
- [ ] Model file exists: `efficientnet_asl.h5` (~44 MB)
- [ ] Labels file exists: `label_classes.npy`
- [ ] Camera connected and working
- [ ] Good lighting in the room
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] All Python files in same directory
- [ ] No other app using the camera

---

## 🎯 Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Run tests: `python test_mediapipe_integration.py`
3. ✅ Launch app: `python mediapipe_gesture_recognition.py`
4. ✅ Test with gestures
5. ✅ Collect dataset (auto-save or manual 's')
6. ✅ Fine-tune model with new data (if needed)

---

## 💡 Tips & Tricks

1. **Better accuracy**: Improve lighting, natural hand position
2. **Faster FPS**: Reduce MC_PASSES to 5
3. **Stable predictions**: Increase BUFFER_SIZE to 12-15
4. **Dataset collection**: Use auto-save feature for quality frames
5. **Testing**: Press 'd' to toggle landmark visualization
6. **Custom model**: Replace efficientnet_asl.h5 with your own

---

## 🎉 You're Ready!

Everything is set up. Just run:

```bash
python mediapipe_gesture_recognition.py
```

Enjoy real-time gesture recognition! 🚀

---
