# hyde_net.py
# Implementation of EfficientNet-B0 CNN
# Replaces previous HyDeNet (LDA + MaXNet) approach.

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ==================================================================
# Implementation map (high-level)
#
# This file contains the Data Augmentation pipeline, preprocessing helpers
# for EfficientNet-B0, and the EfficientNet-B0 model constructor with
# Monte Carlo Dropout enabled for uncertainty estimation.
# ==================================================================

# -----------------------
# Preprocessing helpers
# -----------------------
def preprocess_for_efficientnet(image, size=(224, 224)):
    """
    image: BGR or grayscale -> returns float32 image ready for EfficientNet
    """
    if image.ndim == 2:
        gray = image.copy()
        # convert to 3-channel
        stacked = np.stack([gray, gray, gray], axis=-1)
    else:
        stacked = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    resized = cv2.resize(stacked, size, interpolation=cv2.INTER_AREA)
    # EfficientNetB0 expects inputs in range [0, 255] and handles normalization internally
    return resized.astype(np.float32)

# -----------------------
# EfficientNet-B0 model base
# -----------------------
def build_efficientnet_model(num_classes, input_shape=(224, 224, 3)):
    """
    Builds the EfficientNet-B0 model with Monte Carlo Dropout enabled.
    Note: Data augmentation should be applied via the dataset loader 
    (e.g., using tf.keras.Sequential in the training script) instead 
    of baking it into the model to avoid serialization errors during save.
    """
    inputs = tf.keras.Input(shape=(None, None, 3))
    
    # Base Model (EfficientNetB0)
    # We use imagenet weights, exclude top dense layers.
    base_model = EfficientNetB0(
        include_top=False, 
        weights='imagenet', 
        input_tensor=inputs,
        pooling=None
    )

    
    # Optionally freeze base model layers during initial fine-tuning phases
    # base_model.trainable = False 
    
    x = base_model.output
    
    # 3. Add custom classification head
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    
    # Monte Carlo Dropout: passing training=True forces dropout to be active
    # even during inference time, allowing us to estimate uncertainty.
    x = Dropout(0.2)(x, training=True)
    
    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = Model(inputs, outputs, name="EfficientNet_MC")
    
    # Compile
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# -----------------------
# Utilities: metrics
# -----------------------
def compute_metrics(y_true, y_pred):
    """
    y_true: integer labels
    y_pred: integer predicted labels
    returns accuracy, precision, recall, f1 (macro)
    """
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}

# -----------------------
# Example usage (skeleton)
# -----------------------
if __name__ == "__main__":
    # Example pipeline for a single image
    import os
    # load an example image (replace with your path)
    img_path = 'example.jpg'
    if not os.path.exists(img_path):
        print("Place an example.jpg in working dir to run the demo pipeline.")
    else:
        bgr = cv2.imread(img_path)
        
        # Preprocess
        eff_in = preprocess_for_efficientnet(bgr, size=(224,224))
        
        # Build model
        eff_model = build_efficientnet_model(num_classes=10)
        
        # Single forward 
        preds = eff_model.predict(np.expand_dims(eff_in, 0))
        print("Pred shape:", preds.shape)
