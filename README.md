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


## Dataset

This project uses the **FaceForensics-1000** dataset available on Kaggle for training and evaluating the deepfake detection model.

### Dataset Overview

A total of **2,000 videos** were used in this project:

* **1,000 Real videos**
* **1,000 Fake videos**

This provides a balanced dataset with an equal number of real and manipulated videos.

| Class     |    Videos |
| --------- | --------: |
| Real      |     1,000 |
| Fake      |     1,000 |
| **Total** | **2,000** |

### Frame Extraction

To convert the video dataset into image samples suitable for the deep learning pipeline, **5 frames were extracted from each video**.

Therefore, the initial frame dataset contains:

**2,000 videos × 5 frames = 10,000 frames**

These frames are subsequently passed through the face detection and preprocessing pipeline before being used for model training.

```text
2,000 Videos
      │
      │ 5 frames per video
      ▼
10,000 Extracted Frames
      │
      ▼
Face Detection (YuNet)
      │
      ▼
Face Cropping & Padding
      │
      ▼
Preprocessed Face Images
```

### Train, Validation, and Test Split

The dataset is divided into **training, validation, and testing subsets**.

The original video samples are separated before the extracted frames are used for model training. This is important because frames originating from the same video should not appear in different splits.

The dataset is organized into:

| Split          | Purpose                                                                     |
| -------------- | --------------------------------------------------------------------------- |
| **Training**   | Used to train the deepfake detection model                                  |
| **Validation** | Used to monitor model performance during training and guide model selection |
| **Test**       | Used exclusively for final evaluation on unseen data                        |

The training dataset is further divided into training and validation subsets, while the test set remains isolated until the final evaluation stage.

### Dataset Classes

The dataset contains two balanced classes:

* **Real** — Original, non-manipulated video content.
* **Fake** — Video content containing manipulated facial regions.

### Face-Based Dataset

The classification model does not directly process complete video frames. Instead, facial regions are extracted from the frames and used as the model input.

The preprocessing pipeline consists of:

1. Extracting frames from videos.
2. Detecting faces using **OpenCV YuNet**.
3. Cropping the detected facial regions.
4. Applying padding around the detected face.
5. Resizing the resulting face images.
6. Preparing the images for the EfficientNetB0 model.

### Dataset Source

The dataset used in this project is available on Kaggle:

[FaceForensics-1000 — Kaggle Dataset](https://www.kaggle.com/datasets/rohingarg12/faceforensics-1000?utm_source=chatgpt.com)

> **Note:** The dataset is used for research and educational purposes. Please refer to the dataset provider's terms and licensing conditions before redistributing or using the dataset for other purposes.

## Data Pipeline

The project uses a modular data processing pipeline to transform raw videos into normalized facial images suitable for deep learning.

The complete pipeline is:

```text id="qz8k2p"
Raw Videos
    │
    ▼
Frame Extraction
    │
    ▼
Extracted Frames
    │
    ▼
Face Detection (YuNet)
    │
    ▼
Face Cropping & Padding
    │
    ▼
Processed Face Images
    │
    ▼
Train / Validation / Test
    │
    ▼
TensorFlow Data Pipeline
    │
    ▼
EfficientNetB0
```

### 1. Frame Extraction

Five frames are extracted from each video, resulting in an initial dataset of **10,000 frames from 2,000 videos**.

The extracted frames are stored separately from the original video files and serve as the input to the face detection stage.

### 2. Face Detection

Faces are detected from the extracted frames using **OpenCV YuNet**, a lightweight face detection model designed for efficient face localization.

For each frame, the detected face with the highest confidence score is selected as the primary face region.

### 3. Face Cropping and Padding

After detecting the face, the corresponding bounding box is extracted from the frame.

Additional padding is applied around the detected face to preserve useful facial context that may exist outside the original bounding box.

The resulting region is then resized to the input resolution required by EfficientNetB0.

### 4. Dataset Organization

The processed facial images are organized according to their class labels:

```text id="gk0xfr"
processed/
├── real/
│   ├── ...
│   └── ...
└── fake/
    ├── ...
    └── ...
```

The resulting samples are then separated into training, validation, and test datasets.

### 5. TensorFlow Data Pipeline

The processed images are loaded using **TensorFlow's `tf.data` API**.

The pipeline performs operations such as:

* Reading image files
* JPEG decoding
* Image preprocessing
* Label assignment
* Batching
* Prefetching

A batch size of **32** is used during training and evaluation.

Prefetching with `AUTOTUNE` is also used to improve the efficiency of data loading during model training.

### 6. Model Input

The final processed face images are resized to:

```text id="uknyxm"
224 × 224 × 3
```

These images are then passed to the EfficientNetB0 backbone for feature extraction and binary classification.

