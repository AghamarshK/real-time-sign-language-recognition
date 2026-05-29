# QUICKSTART_MEDIAPIPE.md

# MediaPipe Hand Detection - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install MediaPipe
```bash
pip install -r requirements.txt
```

Or specifically:
```bash
pip install mediapipe
```

### Step 2: Test the Integration
```bash
python test_mediapipe_integration.py
```

This will:
- ✓ Verify MediaPipe is installed
- ✓ Check model files
- ✓ Test preprocessing pipeline
- ✓ Verify camera access
- ✓ Run end-to-end tests

### Step 3: Run the Application
```bash
python mediapipe_gesture_recognition.py
```

---

## 🎮 How to Use

1. **Position your hand** in front of the webcam
2. **Make a gesture** (A, B, C, D, etc. depending on your training)
3. **Watch the display** showing:
   - 🟩 Green bounding box around your hand
   - 🔴 Red dots on hand landmarks
   - Label of detected gesture
   - Confidence score
   - FPS counter

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save current frame |
| `d` | Toggle landmark visualization |

---

## ✨ What's Different from the Old System

### Old System (ROI-based)
```
Fixed area in center of screen
Must keep hand in specific position
```

### New System (MediaPipe)
```
Hand detection anywhere
Automatic hand tracking
Intelligent cropping
Works with natural hand movements
```

---

## 📊 Real-time Feedback

**On-screen Information:**
```
┌─────────────────────────────┐
│ Gesture: A                  │
│ Confidence: 0.95            │
│ Uncertainty: 0.12           │
│ Status: DETECTED            │
│                      FPS: 28.5 │
└─────────────────────────────┘
```

- **Gesture**: Predicted gesture class
- **Confidence**: How sure the model is (0.0-1.0)
- **Uncertainty**: Estimation uncertainty (Monte Carlo Dropout)
- **Status**: DETECTED (high confidence) or LOW CONF (low confidence)
- **FPS**: Frames per second

---

## 🔧 Troubleshooting

### "No Hand Detected"
→ Move your hand closer to camera or ensure it's fully visible

### "Hand Too Small"
→ Move hand closer to camera

### "Low Confidence"
→ Improve lighting or position hand naturally

### "MediaPipe not found"
→ Run: `pip install mediapipe`

### Camera not working
→ Check connection and try a different camera index

---

## 📁 File Structure

```
ML/
├── mediapipe_gesture_recognition.py  ← Main application
├── test_mediapipe_integration.py     ← Test/verify setup
├── MEDIAPIPE_SETUP_GUIDE.md          ← Detailed guide
├── QUICKSTART_MEDIAPIPE.md           ← This file
├── hyde_net.py                       ← Model definition
├── efficientnet_asl.h5               ← Trained weights
├── label_classes.npy                 ← Class labels
├── requirements.txt                  ← Dependencies
└── dataset/test/captured_gestures/   ← Auto-saved frames
```

---

## 🎯 Example Session

```
$ python test_mediapipe_integration.py
✓ All tests passed!

$ python mediapipe_gesture_recognition.py
✓ Weights loaded
✓ MediaPipe initialized
✓ Camera opened: 640x480

[Press 'q' to quit, 's' to save]

✓ Landmarks visualization: ON
✓ Auto-captured: auto_A_1710154623001.jpg
✓ Landmarks visualization: OFF
✓ Manually saved: manual_1710154625123.jpg
✓ Exiting application...
```

---

## 📈 Performance Tips

**Faster (lower FPS accuracy):**
- Reduce `MC_PASSES` from 10 to 5
- Use "lite" model: `HAND_DETECTION_MODEL = "lite"`

**More Accurate (slower):**
- Increase `MC_PASSES` to 20
- Use "full" model: `HAND_DETECTION_MODEL = "full"`

**Stable Predictions:**
- Increase `BUFFER_SIZE` for more smoothing
- Increase `CONSECUTIVE_FRAMES` for auto-save threshold

---

## 🎓 How the System Works

```
1. Capture frame from webcam
                ↓
2. Detect hand using MediaPipe (21 landmarks)
                ↓
3. Calculate bounding box from landmarks
                ↓
4. Crop hand region from frame
                ↓
5. Resize to 224×224 for EfficientNet-B0
                ↓
6. Normalize pixel values
                ↓
7. Run Monte Carlo inference (10 passes)
                ↓
8. Average predictions for robustness
                ↓
9. Apply temporal smoothing (buffer)
                ↓
10. Display result with bounding box & landmarks
```

---

## 🔗 Integration with Training Pipeline

The system seamlessly integrates with your existing setup:

**Training:** `train_hydenet.py` → `efficientnet_asl.h5`
**Inference (Old):** `realtime_gesture_recognition.py` (ROI-based)
**Inference (New):** `mediapipe_gesture_recognition.py` (Hand detection)

Both use the same model and can coexist!

---

## 📚 Learn More

See `MEDIAPIPE_SETUP_GUIDE.md` for:
- Detailed architecture explanation
- Configuration options
- Advanced usage
- Performance metrics
- Troubleshooting guide

---

## ✅ Checklist Before Running

- [ ] MediaPipe installed: `pip install mediapipe`
- [ ] Model weights exist: `efficientnet_asl.h5`
- [ ] Labels file exists: `label_classes.npy`
- [ ] Camera connected and working
- [ ] Good lighting
- [ ] No other apps using camera

---

## 🆘 Still Having Issues?

1. Run the test suite: `python test_mediapipe_integration.py`
2. Check error messages carefully
3. Refer to `MEDIAPIPE_SETUP_GUIDE.md`
4. Verify all files are in the correct location
5. Try in a well-lit environment
6. Ensure camera has permission to access

---

**Ready? Let's go!**

```bash
python mediapipe_gesture_recognition.py
```

Enjoy real-time gesture recognition! 🎉

---
