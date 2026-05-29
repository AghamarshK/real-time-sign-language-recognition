# SYSTEM_ARCHITECTURE.md

# MediaPipe Gesture Recognition - System Architecture

Comprehensive overview of the system architecture, data flow, and components.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  REAL-TIME GESTURE RECOGNITION                  │
│                   MediaPipe + EfficientNet-B0                   │
└─────────────────────────────────────────────────────────────────┘

                         INPUT PIPELINE
                              │
                              ▼
                    ┌──────────────────┐
                    │   WEBCAM CAPTURE │
                    │   (30 FPS)       │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  FRAME MIRRORING │
                    │  (User feedback) │
                    └──────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
   ┌─────────────┐                        ┌──────────────────┐
   │ MediaPipe   │                        │ Input Validation │
   │   Hands     │                        │  (Frame size,    │
   │ Detection   │                        │   format check)  │
   └─────────────┘                        └──────────────────┘
        │
        ▼ (21 landmarks)
   ┌──────────────────────┐
   │  Landmark Processing │
   │  (Normalize coords)  │
   └──────────────────────┘
        │
        ▼
   ┌──────────────────────┐
   │ Bounding Box Extract │
   │ (Min/Max + padding)  │
   └──────────────────────┘
        │
        ▼
   ┌──────────────────────┐
   │  Hand Region Crop    │
   │  (Extract from frame)│
   └──────────────────────┘
        │
        ▼                           PREPROCESSING
   ┌──────────────────────┐
   │  Image Resizing      │
   │  (224×224)           │
   └──────────────────────┘
        │
        ▼
   ┌──────────────────────┐
   │ Normalization        │
   │ (BGR→RGB conversion) │
   └──────────────────────┘
        │
        ▼
   ┌──────────────────────┐
   │ Batch Creation       │
   │ (Add batch dimension)│
   └──────────────────────┘
        │
        └─────────────────────────────┐
                                      │
                      INFERENCE PIPELINE
                                      │
                                      ▼
        ┌─────────────────────────────────────────────┐
        │     Monte Carlo Dropout Inference           │
        │     (N=MC_PASSES iterations)                │
        └─────────────────────────────────────────────┘
                  │       │       │
             Pass 1   Pass 2  ...Pass N
                  │       │       │
                  ▼       ▼       ▼
         ┌──────────────────────────────┐
         │   EfficientNet-B0 Model      │
         │  ┌──────────────────────┐    │
         │  │ Input Layer (224×224)│    │
         │  ├──────────────────────┤    │
         │  │ EfficientNetB0 (     │    │
         │  │  imagenet weights)   │    │
         │  ├──────────────────────┤    │
         │  │ GlobalAvgPooling2D   │    │
         │  ├──────────────────────┤    │
         │  │ Dropout(0.2) [MC]    │    │
         │  ├──────────────────────┤    │
         │  │ Dense (softmax)      │    │
         │  │ Output: [P_class]    │    │
         │  └──────────────────────┘    │
         └──────────────────────────────┘
                  │       │       │
             Pred 1  Pred 2  ...Pred N
                  │       │       │
                  └───────┼───────┘
                          ▼
        ┌─────────────────────────────────────────────┐
        │   Prediction Averaging                      │
        │   mean(all_predictions, axis=0)             │
        │   Returns: [mean_P_class_1, ..., P_class_K] │
        └─────────────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────────┐
        │   Temporal Smoothing (Deque Buffer)         │
        │   Buffer: [smooth_pred_n-7, ..., smooth_pred_n]
        │   mean(buffer, axis=0)                      │
        │   Returns: smoothed probabilities           │
        └─────────────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────────────┐
        │   Post-processing                           │
        │   - argmax(smoothed_probs) → class index    │
        │   - confidence = max(smoothed_probs)        │
        │   - uncertainty = std(all_predictions)      │
        │   - label = label_classes[class_index]      │
        └─────────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
        ▼                                   ▼
    CONFIDENCE                         VISUALIZATION
    FILTERING                          PIPELINE
        │                                   │
        ├─► is_confident?                   ├─► Draw bounding box
        │   (conf > threshold)              │   (green rectangle)
        │                                   │
        ├─► Auto-save                       ├─► Draw landmarks
        │   (N consecutive                  │   (red circles)
        │    high-confidence frames)        │
        │                                   ├─► Display metrics
        │                                   │   (gesture, confidence,
        │                                   │    uncertainty, FPS)
        │                                   │
        │                                   ├─► Text overlay
        │                                   │   (black bg,
        │                                   │    semi-transparent)
        │                                   │
        └───────────────┬───────────────────┘
                        │
                        ▼
        ┌─────────────────────────────────────┐
        │   Display on Screen                 │
        │   (OpenCV imshow)                   │
        └─────────────────────────────────────┘
                        │
                        ▼
        ┌─────────────────────────────────────┐
        │   Keyboard Input Handling           │
        │   - 'q': quit                       │
        │   - 's': save                       │
        │   - 'd': toggle landmarks           │
        └─────────────────────────────────────┘
```

---

## Component Breakdown

### 1. Input Stage

#### Webcam Capture
```python
cap = cv2.VideoCapture(0)
ret, frame = cap.read()  # Shape: (height, width, 3) BGR
```
- Captures continuous frames from default webcam
- Frame rate: ~30 FPS (device dependent)
- Color space: BGR (OpenCV default)

#### Frame Processing
```python
frame = cv2.flip(frame, 1)  # Mirror for natural viewing
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

---

### 2. Hand Detection Stage (MediaPipe)

#### MediaPipe Hands
```
Input: RGB frame (any size)
       │
       ▼
┌──────────────────────┐
│ Hand Detection Model │ (model_complexity=0 or 1)
│ (Convolutional)      │
└──────────────────────┘
       │
       ▼
21 Hand Landmarks (normalized coordinates)
       │
       ├─ Landmark 0: Wrist
       ├─ Landmarks 1-4: Thumb
       ├─ Landmarks 5-8: Index finger
       ├─ Landmarks 9-12: Middle finger
       ├─ Landmarks 13-16: Ring finger
       └─ Landmarks 17-20: Pinky finger
```

#### Landmark Structure
```python
class Landmark:
    x: float (0.0 - 1.0)  # Normalized to frame width
    y: float (0.0 - 1.0)  # Normalized to frame height
    z: float              # Depth (relative to wrist)
```

---

### 3. Bounding Box Extraction

```python
def get_hand_bounding_box(landmarks, width, height, padding=20):
    """
    Step 1: Extract normalized coordinates
    x_coords = [lm.x * width for lm in landmarks]
    y_coords = [lm.y * height for lm in landmarks]
    
    Step 2: Find bounding box
    x_min = min(x_coords) - padding
    x_max = max(x_coords) + padding
    y_min = min(y_coords) - padding
    y_max = max(y_coords) + padding
    
    Step 3: Clamp to frame bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)
    
    Return: (x_min, y_min, x_max, y_max)
    """
```

**Example**:
- Frame: 640×480
- Landmarks range: x=[0.3-0.5], y=[0.4-0.6]
- Pixels: x=[192-320], y=[192-288]
- With padding=20: x=[172-340], y=[172-308]
- Bbox: (172, 172, 340, 308)

---

### 4. Preprocessing Stage

#### Cropping
```python
cropped = frame[y_min:y_max, x_min:x_max]  # Extract region
# Output shape: (variable height, variable width, 3)
```

#### Resizing
```python
resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)
# Output shape: (224, 224, 3)
```

#### Normalization
```python
# Convert BGR to RGB (EfficientNet convention)
stacked = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
normalized = stacked.astype(np.float32)  # Keep as [0, 255]
# Output: (224, 224, 3) float32 in range [0, 255]
```

---

### 5. Model Inference Stage

#### EfficientNet-B0 Architecture
```
Input: (224, 224, 3)
  │
  ▼
EfficientNetB0 Backbone (ImageNet pretrained)
  │
  ├─ Mobile block 1 (40 filters)
  ├─ Mobile block 2 (80 filters)
  ├─ Mobile block 3 (192 filters)
  └─ Mobile block 4 (320 filters)
  │
  ▼
GlobalAveragePooling2D()
  │
  ▼
Dropout(0.2, training=True)  ← Monte Carlo Dropout
  │
  ▼
Dense(num_classes, softmax)
  │
  ▼
Output: (num_classes,) probabilities
```

#### Monte Carlo Dropout
```python
# Single prediction with dropout OFF (standard):
pred = model.predict(input_batch)

# Monte Carlo predictions with dropout ON:
mc_predictions = []
for _ in range(MC_PASSES):
    pred = model.predict(input_batch, training=True)
    mc_predictions.append(pred)

# Ensemble average
mean_pred = np.mean(mc_predictions, axis=0)       # Mean probabilities
std_pred = np.std(mc_predictions, axis=0)         # Uncertainty
```

---

### 6. Post-processing Stage

#### Prediction Averaging
```python
current_mean_probs = np.mean(all_preds, axis=0)
# Shape: (num_classes,)
# Each value is average probability for that class
```

#### Temporal Smoothing
```python
prob_buffer = deque(maxlen=BUFFER_SIZE)
prob_buffer.append(current_mean_probs)
smoothed_probs = np.mean(prob_buffer, axis=0)
# Averages last BUFFER_SIZE frames' predictions
```

#### Classification
```python
pred_idx = np.argmax(smoothed_probs)           # Most likely class
confidence = smoothed_probs[pred_idx]          # Confidence in that class
label = label_classes[pred_idx]                # Class name
uncertainty = np.std(all_preds, axis=0)[pred_idx]  # Uncertainty
```

---

### 7. Visualization Stage

#### Bounding Box Drawing
```python
cv2.rectangle(
    frame,
    (x_min, y_min),
    (x_max, y_max),
    HAND_BOX_COLOR,        # (0, 255, 0) = Green
    HAND_BOX_THICKNESS     # 2
)
```

#### Landmark Drawing
```python
for lm in landmarks:
    x = int(lm.x * frame_width)
    y = int(lm.y * frame_height)
    cv2.circle(
        frame,
        (x, y),
        LANDMARK_RADIUS,       # 4
        LANDMARK_COLOR,        # (0, 0, 255) = Red
        -1                     # Filled circle
    )
```

#### Text Overlay
```python
# Semi-transparent background
text_bg = frame[0:160, 0:350].copy()
cv2.rectangle(frame, (0, 0), (350, 160), (0, 0, 0), -1)
frame[0:160, 0:350] = cv2.addWeighted(
    text_bg, 0.3,
    frame[0:160, 0:350], 0.7,
    0
)

# Draw text
cv2.putText(frame, f"Gesture: {label}", (10, 35), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
```

---

## Data Flow Example

### Frame-by-Frame Processing

```
┌─ Frame 1 ────────────────────────────────────────┐
│                                                  │
│ Input: (480, 640, 3) BGR frame                 │
│  │                                               │
│  ├─ MediaPipe detection: ✓ Hand found           │
│  │  └─ 21 landmarks extracted                   │
│  │                                               │
│  ├─ Bounding box: (150, 200, 380, 420)         │
│  │  └─ Cropped region: (230, 220, 3)           │
│  │                                               │
│  ├─ Preprocessed: (224, 224, 3) float32        │
│  │                                               │
│  ├─ MC Inference (10 passes):                   │
│  │  ├─ Pass 1: [0.8, 0.15, 0.05]               │
│  │  ├─ Pass 2: [0.75, 0.20, 0.05]              │
│  │  └─ Pass 10: [0.78, 0.17, 0.05]             │
│  │                                               │
│  ├─ Mean: [0.765, 0.167, 0.068]                │
│  │  └─ Confidence: 0.765 (Class 0)             │
│  │  └─ Uncertainty: 0.047                       │
│  │                                               │
│  └─ Output: "Gesture: A, Conf: 0.77"           │
│                                                  │
│ Display on screen with bounding box + landmarks│
└──────────────────────────────────────────────────┘

┌─ Frame 2 ────────────────────────────────────────┐
│ ... (similar processing)                         │
│ Temporal buffer: [frame_1_probs, frame_2_probs] │
│ Smoothed prediction: average of buffer           │
└──────────────────────────────────────────────────┘

... (frames 3-8)

┌─ Frame 9 ────────────────────────────────────────┐
│ Temporal buffer (size=8): 8 consecutive frames   │
│ Smoothed prediction: very stable                 │
│                                                  │
│ If confidence > THRESHOLD for CONSECUTIVE_FRAMES│
│  └─ Auto-save this frame                        │
└──────────────────────────────────────────────────┘
```

---

## Performance Analysis

### Latency Breakdown

```
Webcam Capture:        ~33ms (1/30 FPS)
Frame Processing:      ~5ms
MediaPipe Detection:   ~20ms (full model)
Preprocessing:         ~5ms
MC Inference (10x):    ~100ms (CPU) / ~20ms (GPU)
Post-processing:       ~2ms
Visualization:         ~10ms
Display:              ~5ms
────────────────────
Total:                ~180ms (CPU) / ~80ms (GPU)

FPS: ~5.5 (CPU) / ~12.5 (GPU)
```

### Memory Usage

```
Model Weights:
- EfficientNetB0:      ~44 MB

Runtime Memory:
- Frame buffer:        ~3 MB
- MC batch (10×):      ~30 MB
- Predictions buffer:  ~0.5 MB
- Landmarks:           ~0.01 MB
────────────────────
Total:                 ~78 MB
```

---

## Error Handling Flow

```
Start Processing Frame
        │
        ▼
Is frame valid?
    │
    ├─ NO → Skip frame, retry
    │
    └─ YES
        │
        ▼
Is hand detected?
    │
    ├─ NO → label = "No Hand"
    │        confidence = 0.0
    │
    └─ YES
        │
        ▼
Is bounding box valid?
    │
    ├─ NO (too small/invalid) → label = "Hand Too Small"
    │
    └─ YES
        │
        ▼
Is cropped region valid?
    │
    ├─ NO → label = "Hand Landmarks Invalid"
    │
    └─ YES
        │
        ▼
Preprocess & Classify
        │
        ▼
Display Result + Visualization
```

---

## Key Design Decisions

### 1. MediaPipe for Hand Detection
✅ Robust, real-time, supports various lighting conditions
✅ 21 landmarks provide detailed hand information
❌ Cannot detect multiple hands easily

### 2. EfficientNet-B0 for Classification
✅ Lightweight (~44 MB), suitable for real-time
✅ Good accuracy on gesture classification
✅ Supports fine-tuning on custom datasets

### 3. Monte Carlo Dropout for Uncertainty
✅ Simple to implement (just use training=True)
✅ Provides uncertainty estimates without ensemble
❌ Slower (requires N forward passes)

### 4. Temporal Smoothing
✅ Stabilizes jittery predictions
✅ Reduces false positives
❌ Adds latency

### 5. Automatic Bounding Box
✅ Works with any hand position
✅ Adaptive to hand size
❌ Requires landmark extraction (minor overhead)

---

## Extension Points

### Add Multi-hand Support
```python
# Current: processes first hand
hand_landmarks = results.multi_hand_landmarks[0]

# Extension: process all hands
for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
    # Process each hand separately
    # Can show multiple predictions
```

### Add Gesture Recognition History
```python
gesture_history = deque(maxlen=30)  # Last 30 gestures
gesture_history.append({
    'gesture': label,
    'confidence': confidence,
    'timestamp': time.time()
})
```

### Add Recording/Streaming
```python
out = cv2.VideoWriter('gesture_demo.mp4', 
                      fourcc, fps, (width, height))
out.write(frame)  # Write frame to file
```

---

## Summary

The system follows a clear pipeline:
1. **Capture** → Frame from webcam
2. **Detect** → Hand landmarks via MediaPipe
3. **Extract** → Bounding box from landmarks
4. **Preprocess** → Crop and resize to 224×224
5. **Classify** → EfficientNet-B0 with MC Dropout
6. **Smooth** → Temporal averaging
7. **Display** → Overlay visualization on frame

This architecture balances:
- ✅ Real-time performance (~12-30 FPS)
- ✅ Accuracy (uncertainty estimation)
- ✅ Robustness (temporal smoothing)
- ✅ Flexibility (easy configuration)

---
