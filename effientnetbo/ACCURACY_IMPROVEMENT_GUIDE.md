# Model Accuracy Improvement Guide

## Current Status
- **Model**: EfficientNet-B0 with ImageNet pre-training
- **Input**: 224×224 RGB images (hand region)
- **Training**: 20 epochs, 16 batch size, data augmentation
- **Current Performance**: Baseline with 4 gesture classes (A, B, C, D)

---

## 1. Quick Wins (Easy, High Impact)

### 1.1 Increase Training Epochs
**Impact**: +3-5% accuracy  
**Time**: 5-10 minutes

Edit `train_hydenet.py`:
```python
EPOCHS = 50  # Increase from 20 to 50
```

### 1.2 Improve Data Augmentation
**Impact**: +2-4% accuracy  
**Time**: 2 minutes

Edit `train_hydenet.py` - replace the data augmentation section:
```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.3),      # Increased from 0.2
    tf.keras.layers.RandomZoom(0.3),          # Increased from 0.2
    tf.keras.layers.RandomTranslation(0.2, 0.2),  # NEW: Add translation
    tf.keras.layers.RandomContrast(0.2),      # NEW: Add contrast
    tf.keras.layers.RandomBrightness(0.2),    # NEW: Add brightness
])
```

### 1.3 Reduce Learning Rate & Add Learning Rate Scheduling
**Impact**: +1-3% accuracy  
**Time**: 3 minutes

Edit `hyde_net.py` - modify the optimizer:
```python
# OLD:
opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

# NEW:
from tensorflow.keras.optimizers.schedules import ExponentialDecay
initial_learning_rate = 5e-4
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True
)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

### 1.4 Increase Batch Size
**Impact**: +1-2% accuracy  
**Time**: 1 minute

Edit `train_hydenet.py`:
```python
BATCH_SIZE = 32  # Increase from 16
```

---

## 2. Medium Improvements (Moderate Effort, Good Impact)

### 2.1 Use Larger Model: EfficientNet-B2 or B3
**Impact**: +5-10% accuracy  
**Time**: 10 minutes  
**Trade-off**: Slower inference (use GPU if available)

Edit `hyde_net.py`:
```python
# OLD:
from tensorflow.keras.applications import EfficientNetB0
base_model = EfficientNetB0(...)

# NEW:
from tensorflow.keras.applications import EfficientNetB2
base_model = EfficientNetB2(
    include_top=False, 
    weights='imagenet', 
    input_tensor=inputs,
    pooling=None
)

# Also update IMG_SIZE in train_hydenet.py:
IMG_SIZE = (260, 260)  # B2 expects 260x260
```

### 2.2 Add More Dense Layers Before Output
**Impact**: +2-4% accuracy  
**Time**: 3 minutes

Edit `hyde_net.py`:
```python
# OLD:
x = GlobalAveragePooling2D(name="avg_pool")(x)
x = Dropout(0.2)(x, training=True)
outputs = Dense(num_classes, activation="softmax", name="predictions")(x)

# NEW:
x = GlobalAveragePooling2D(name="avg_pool")(x)
x = Dense(256, activation='relu', name='fc1')(x)
x = Dropout(0.3)(x, training=True)
x = Dense(128, activation='relu', name='fc2')(x)
x = Dropout(0.2)(x, training=True)
outputs = Dense(num_classes, activation="softmax", name="predictions")(x)
```

### 2.3 Unfreeze & Fine-tune Base Model
**Impact**: +3-6% accuracy  
**Time**: 5 minutes  
**Requirements**: More GPU memory

Edit `hyde_net.py` - after building model:
```python
# Freeze the base model initially
base_model.trainable = False

# Train for 10 epochs with frozen base
# Then unfreeze and train for 20 more epochs with lower learning rate

# Add this training logic in train_hydenet.py after initial fit:
print("Unfreezing base model layers for fine-tuning...")
base_model.trainable = True

# Recompile with much lower learning rate
opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tune for more epochs
history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    initial_epoch=EPOCHS
)
```

---

## 3. Advanced Improvements (Higher Effort, Best Results)

### 3.1 Collect More Training Data
**Impact**: +10-20% accuracy  
**Time**: 30 minutes - 1 hour  
**Why**: The biggest factor in accuracy

Do this **immediately**:
```bash
# Run the gesture capture script multiple times
python realtime_gesture_recognition.py
# Press 's' to save frames of each gesture
# Collect 100-200 images per gesture class from different:
# - Lighting conditions
# - Hand sizes (close & far from camera)
# - Hand positions (left, center, right, top, bottom)
# - Hand orientations (rotated at different angles)
```

### 3.2 Improve Hand Region Extraction
**Impact**: +2-5% accuracy  
**Time**: 15 minutes

The current bounding box might not be optimal. Edit `mediapipe_gesture_recognition.py`:

```python
def get_hand_bounding_box_improved(landmarks, frame_width, frame_height, padding=30):
    """
    Improved bounding box extraction with aspect ratio correction
    """
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    # Convert normalized to pixel coordinates
    x_min = int(x_min * frame_width)
    y_min = int(y_min * frame_height)
    x_max = int(x_max * frame_width)
    y_max = int(y_max * frame_height)
    
    width = x_max - x_min
    height = y_max - y_min
    
    # Add padding
    padding_x = int(width * 0.2) + padding  # 20% of width + fixed padding
    padding_y = int(height * 0.2) + padding  # 20% of height + fixed padding
    
    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = min(frame_width, x_max + padding_x)
    y_max = min(frame_height, y_max + padding_y)
    
    # Make square (better for CNN)
    size = max(x_max - x_min, y_max - y_min)
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    x_min = max(0, center_x - size // 2)
    y_min = max(0, center_y - size // 2)
    x_max = min(frame_width, x_min + size)
    y_max = min(frame_height, y_min + size)
    
    return (x_min, y_min, x_max, y_max)
```

### 3.3 Use Ensemble of Models
**Impact**: +5-8% accuracy  
**Time**: 30 minutes  
**Trade-off**: Slower inference

Train multiple models with different architectures:
```python
# Train EfficientNet-B0, B1, B2 separately
# Then average their predictions
models = [
    load_model('efficientnet_b0.h5'),
    load_model('efficientnet_b1.h5'),
    load_model('efficientnet_b2.h5'),
]

predictions = []
for model in models:
    pred = model.predict(preprocessed_image)
    predictions.append(pred)

ensemble_pred = np.mean(predictions, axis=0)
final_class = np.argmax(ensemble_pred)
```

### 3.4 Add Class Weights (If Data Imbalanced)
**Impact**: +2-5% accuracy (if classes are imbalanced)  
**Time**: 5 minutes

Edit `train_hydenet.py`:
```python
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_ds.class_names),
    y=train_ds.class_names
)
class_weight = dict(enumerate(class_weights))

# Train with class weights
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight
)
```

---

## 4. Implementation Priority Order

**Phase 1 (Start Here - 10 minutes)**:
1. Increase EPOCHS to 50
2. Improve data augmentation (add translation, contrast, brightness)
3. Reduce learning rate (use schedule)

**Phase 2 (30 minutes)**:
4. Increase batch size to 32
5. Add more dense layers before output
6. Start collecting more training data

**Phase 3 (1+ hours)**:
7. Switch to EfficientNet-B2 or B3
8. Fine-tune base model
9. Create ensemble model

---

## 5. Measuring Improvement

After making changes, retrain and measure:

```python
# Run this after training
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Get predictions on validation set
y_true = []
y_pred = []
for images, labels in val_ds:
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels, axis=1))

# Print metrics
print(classification_report(y_true, y_pred, target_names=class_names))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Plot confusion matrix
import seaborn as sns
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.savefig('confusion_matrix.png')
```

---

## 6. GPU Acceleration (If Available)

To speed up training on GPU:

```bash
# Install CUDA-enabled TensorFlow
pip install tensorflow-metal  # macOS with Apple Silicon
pip install tensorflow[and-cuda]  # Linux/Windows with NVIDIA GPU
```

Check if GPU is being used:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## 7. Recommended Strategy

**Best approach for +10-15% improvement in 1 hour**:
1. ✅ Increase epochs to 50
2. ✅ Improve augmentation (add 3 new techniques)
3. ✅ Use learning rate scheduling  
4. ✅ Add dense layers (256→128)
5. ✅ Collect 50-100 more training images per class
6. ✅ Retrain model

**Expected result**: ~85-90% accuracy (if you currently have ~75-80%)

---

## 8. Real-time Performance Tips

In `mediapipe_gesture_recognition.py`, increase `MC_PASSES` for better confidence:

```python
MC_PASSES = 15  # Increase from 10
BUFFER_SIZE = 12  # Increase from 8
CONFIDENCE_THRESHOLD = 0.75  # Increase from 0.70
```

This makes predictions more stable and accurate at the cost of slightly more computation.

---

## Questions to Ask

- **How many training images do you have per class?** (target: 200+ per class)
- **What's your current validation accuracy?** (we can estimate improvement potential)
- **Do you have GPU?** (enables using larger models like B3)
- **Are you experiencing class imbalance?** (some gestures harder than others?)

---

## Next Steps

1. Choose your starting point from Phase 1 above
2. Edit the files as shown
3. Run: `python train_hydenet.py`
4. Test with: `python mediapipe_gesture_recognition.py`
5. Report back with the accuracy metrics!
