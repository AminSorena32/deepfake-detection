# DeepFake Detection

A deepfake detection system designed to classify facial images as **real** or **fake** using deep learning and transfer learning.

The project implements an end-to-end pipeline that starts with video frame extraction and face detection, followed by face preprocessing and classification using **EfficientNetB0**. The model is first trained using transfer learning with ImageNet-pretrained weights and then fine-tuned on the target dataset to improve its ability to distinguish between real and manipulated faces.

The project also includes a complete evaluation pipeline for analyzing model performance using metrics such as **Accuracy, Precision, Recall, F1-score, ROC-AUC, Loss, and Confusion Matrix**.

### Main Pipeline

**Videos → Frame Extraction → Face Detection → Face Preprocessing → Transfer Learning → Fine-Tuning → Evaluation**

The main goal of this project is to explore a practical deep learning pipeline for deepfake detection while maintaining a clean and reproducible project structure.

## Pipeline

```text
Videos
   ↓
Frame Extraction
   ↓
Face Detection (YuNet)
   ↓
Face Cropping & Preprocessing
   ↓
EfficientNetB0
   ↓
Transfer Learning
   ↓
Fine-Tuning
   ↓
Real / Fake
   ↓
Evaluation
```

The pipeline extracts **5 frames per video**, detects and preprocesses faces using **YuNet**, and classifies them using an **ImageNet-pretrained EfficientNetB0** model.

## Dataset

The project uses the **FaceForensics-1000** dataset with **2,000 videos**:

| Class     |    Videos |
| --------- | --------: |
| Real      |     1,000 |
| Fake      |     1,000 |
| **Total** | **2,000** |

* **5 frames** are extracted from each video → **10,000 frames**
* Data is split into **Training, Validation, and Test** sets.
* The test set is kept separate for final evaluation.

**Dataset:** [FaceForensics-1000 — Kaggle](https://www.kaggle.com/datasets/rohingarg12/faceforensics-1000?utm_source=chatgpt.com)

## Model

The project uses **EfficientNetB0** with **ImageNet-pretrained weights** for binary classification.

```text id="jz4n8s"
Input (224×224×3)
       ↓
EfficientNetB0
       ↓
Custom Classification Head
       ↓
Sigmoid Output
       ↓
Real / Fake
```

* **Transfer Learning** with a frozen pretrained backbone
* **Fine-Tuning** of selected EfficientNetB0 layers
* **Binary Cross-Entropy** loss
* **Adam** optimizer

## Training

The model is trained in two stages:

1. **Transfer Learning** — The pretrained EfficientNetB0 backbone is frozen and the classification head is trained.
2. **Fine-Tuning** — Selected layers of the backbone are unfrozen and trained to adapt the model to the deepfake detection task.

### Main Configuration

* **Input Size:** 224 × 224
* **Batch Size:** 32
* **Optimizer:** Adam
* **Loss:** Binary Cross-Entropy
* **Framework:** TensorFlow / Keras

## Results

The final model was evaluated on the held-out test set using standard classification metrics.

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **66.25%** |
| Precision | **65.09%** |
| Recall    | **70.10%** |
| F1-Score  | **67.50%** |
| ROC-AUC   | **72.16%** |

The model achieves a **66.25% test accuracy** with a **ROC-AUC of 72.16%**, providing a baseline for further improvements in deepfake detection.

## Project Structure

```text id="c7k2p1"
DeepFake-Detection/
├── models/
├── notebooks/
├── results/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── evaluation/
├── .gitignore
├── requirements.txt
└── README.md
```

The project is organized into separate modules for **data processing, model development, training, evaluation, and results**.

## Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/DeepFake-Detection.git
cd DeepFake-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline

Use the notebooks in the `notebooks/` directory to follow the complete workflow from **data preprocessing to model evaluation**.

> The dataset is not included in the repository. Download it from the Kaggle link provided in the **Dataset** section.

## Future Improvements

* Improve face preprocessing and detection quality.
* Experiment with stronger CNN and transformer-based architectures.
* Increase dataset diversity and size.
* Explore video-level deepfake detection instead of frame-level classification.
* Optimize training and inference performance.



