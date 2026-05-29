# Sign Language Translator - EfficientNet Model

This repository contains the training and inference code for the EfficientNet-B0 based Sign Language Translator model.

## Setup Instructions

1. **Prerequisites**: Make sure you have Python 3.8+ installed on your system.
2. **Install Dependencies**: Open a terminal/command prompt in this folder and run:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model
To train the EfficientNet-B0 model from scratch or fine-tune it:
1. Place your training dataset in the `dataset/train` folder. The folder structure should be:
   ```
   dataset/train/
       class_A/
           img1.jpg
           img2.jpg
       class_B/
           img1.jpg
           ...
   ```
2. Run the training script:
   ```bash
   python train_hydenet.py
   ```
This will output `efficientnet_asl.h5` (the trained model weights) and `label_classes.npy` (the class name mappings).

### 2. Testing / Inference
To run a single image through the trained model to get a prediction and uncertainty estimation (using Monte Carlo Dropout):
1. Place a sample image at `dataset/test/sample.jpg` (or edit the `img_path` variable in `test_hydenet.py` to point to your image).
2. Make sure you have the trained weights `efficientnet_asl.h5` and labels `label_classes.npy` in this directory.
3. Run the test script:
   ```bash
   python test_hydenet.py
   ```
The script will run the image through the model 50 times and output the predicted class, confidence, and uncertainty.
