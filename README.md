# DeepFake Detection

A deepfake detection system designed to classify facial images as **real** or **fake** using deep learning and transfer learning.

The project implements an end-to-end pipeline that starts with video frame extraction and face detection, followed by face preprocessing and classification using **EfficientNetB0**. The model is first trained using transfer learning with ImageNet-pretrained weights and then fine-tuned on the target dataset to improve its ability to distinguish between real and manipulated faces.

The project also includes a complete evaluation pipeline for analyzing model performance using metrics such as **Accuracy, Precision, Recall, F1-score, ROC-AUC, Loss, and Confusion Matrix**.

### Main Pipeline

**Videos → Frame Extraction → Face Detection → Face Preprocessing → Transfer Learning → Fine-Tuning → Evaluation**

The main goal of this project is to explore a practical deep learning pipeline for deepfake detection while maintaining a clean and reproducible project structure.

## Features

* **Video Frame Extraction** — Extracts representative frames from input videos for further processing.
* **Face Detection** — Detects and extracts faces from video frames using OpenCV YuNet.
* **Face Preprocessing** — Applies face cropping, padding, resizing, and image preprocessing to prepare the data for model training.
* **Transfer Learning** — Uses an ImageNet-pretrained EfficientNetB0 as the backbone for deepfake classification.
* **Fine-Tuning** — Fine-tunes selected layers of the pretrained backbone on the target deepfake dataset.
* **Binary Classification** — Classifies facial images into two categories: **Real** and **Fake**.
* **Model Evaluation** — Evaluates the trained model using Accuracy, Loss, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix.
* **Training Visualization** — Provides training and validation curves to analyze model learning and potential overfitting.
* **Modular Project Structure** — Separates data processing, model development, training, evaluation, and results into dedicated components.

## Project Architecture
The project follows a modular end-to-end pipeline for deepfake detection:

```text
Input Videos
     │
     ▼
Frame Extraction
     │
     ▼
Face Detection (YuNet)
     │
     ▼
Face Cropping & Padding
     │
     ▼
Image Preprocessing
     │
     ▼
EfficientNetB0
(ImageNet Pretrained)
     │
     ├── Transfer Learning
     │
     ▼
Fine-Tuning
     │
     ▼
Binary Classification
     │
     ├── Real
     └── Fake
     │
     ▼
Model Evaluation
     │
     ├── Accuracy
     ├── Loss
     ├── Precision
     ├── Recall
     ├── F1-Score
     ├── ROC-AUC
     └── Confusion Matrix
```

### Pipeline Overview

1. **Frame Extraction**
   Frames are extracted from the input videos to create image samples for the detection pipeline.

2. **Face Detection**
   Faces are detected from the extracted frames using **OpenCV YuNet**.

3. **Face Cropping & Padding**
   Detected face regions are cropped and padded to preserve relevant facial context before classification.

4. **Image Preprocessing**
   The processed faces are resized and prepared for input into the EfficientNetB0 model.

5. **Transfer Learning**
   An **ImageNet-pretrained EfficientNetB0** is used as the feature extraction backbone while its pretrained layers are initially frozen.

6. **Fine-Tuning**
   Selected layers of the backbone are unfrozen and trained with a lower learning rate to adapt the pretrained features to the deepfake detection task.

7. **Binary Classification**
   The model predicts whether each input face belongs to the **Real** or **Fake** class.

8. **Evaluation**
   The final model is evaluated using multiple classification metrics and visualization tools to provide a more complete assessment of its performance.



