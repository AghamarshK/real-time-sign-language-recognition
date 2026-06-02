# Real-time ASL Gesture Recognition System

A high-performance system for translating American Sign Language (ASL) alphabet gestures into text in real-time. This project implements and compares two state-of-the-art deep learning paradigms:
1. **Geometric skeletal landmark classification** using MediaPipe and a custom MLP.
2. **Pixel-level convolutional feature extraction** using the **EfficientNet-B0** backbone.

---

## 🚀 Deep Dive: EfficientNet-B0

The "Image Classifier" pipeline is powered by **EfficientNet-B0**, a convolutional neural network architecture that achieved state-of-the-art accuracy with significantly fewer parameters than previous models.

### 1. Compound Scaling
Unlike traditional CNNs that scale only depth (ResNet), width (WideResNet), or resolution, EfficientNet scales all three dimensions together using a principled **compound coefficient**. This ensures that as the model grows deeper to capture more complex features, it also grows wider and processes higher resolution images to preserve spatial detail.

### 2. MBConv (Mobile Inverted Bottleneck Convolution)
The core building block of EfficientNet-B0 is the **MBConv** layer. It uses:
- **Depthwise Separable Convolutions**: Significantly reduces the number of parameters and FLOPs compared to standard convolutions.
- **Inverted Residuals**: Expands the input to a higher dimension, applies convolutions, and then squeezes it back, allowing for richer feature learning.
- **Squeeze-and-Excitation (SE) Blocks**: Dynamically weights different channels based on their importance, helping the model focus on the most relevant features (like hand shape) while ignoring background noise.

### 3. Custom Head for ASL
Our implementation replaces the standard 1000-class ImageNet head with a specialized classification stack:
- **Dropout (0.4)** for robust regularization.
- **Linear (1280 → 512)** layer for latent feature projection.
- **Batch Normalization & ReLU** for training stability.
- **Final Output (26 classes)** for the ASL alphabet.

---

## 🦴 Landmark-Based Pipeline (MLP)

For low-latency applications, we use **MediaPipe Hands** to extract a 21-point skeletal model of the hand. This reduces the input from 150,528 pixels (224x224x3) to just **63 coordinates (x, y, z)**.
- **Input Dimension**: 126 (Support for 2 hands).
- **Architecture**: 5-layer deep MLP with BatchNorm and Dropout.
- **Accuracy**: Achieves a superior **98% accuracy** on the test set.

---

## 📉 Monte Carlo Dropout for Uncertainty

To ensure reliability in real-world scenarios, we implemented **Monte Carlo Dropout**. 
By enabling dropout during inference and performing multiple forward passes, the system calculates the **variance** of the predictions.
- **High Confidence**: Low variance across passes.
- **Uncertainty**: High variance (e.g., when the hand is moving too fast or partially occluded). This prevents the system from making "wild guesses."

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- PyTorch
- MediaPipe
- OpenCV

### Installation
```bash
pip install torch torchvision mediapipe opencv-python pandas numpy
```

### Running the Demo
Launch the real-time webcam inference:
```bash
python realtime_demo.py
```

---

## 📊 Evaluation Summary

| Model | Accuracy | Inference Latency | Model Size |
| :--- | :--- | :--- | :--- |
| **EfficientNet-B0** | 93% | ~12ms (GPU) | 15.5 MB |
| **Landmark MLP** | **98%** | **<1ms (CPU)** | **1.3 MB** |

---

## 📂 Project Structure
- `realtime_demo.py`: Main script for live webcam translation.
- `train.py`: Training script for the Landmark-based MLP.
- `efficientnetbo/`: Implementation of the image-based classifier.
- `generate_report.py`: Automation script for technical documentation.
- `Final_ASL_Research_Report.docx`: Comprehensive 20+ page technical report.
