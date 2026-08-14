# DeepFake Detection

A deepfake detection system designed to classify facial images as **real** or **fake** using deep learning and transfer learning.

The project implements an end-to-end pipeline that starts with video frame extraction and face detection, followed by face preprocessing and classification using **EfficientNetB0**. The model is first trained using transfer learning with ImageNet-pretrained weights and then fine-tuned on the target dataset to improve its ability to distinguish between real and manipulated faces.

The project also includes a complete evaluation pipeline for analyzing model performance using metrics such as **Accuracy, Precision, Recall, F1-score, ROC-AUC, Loss, and Confusion Matrix**.

### Main Pipeline

**Videos → Frame Extraction → Face Detection → Face Preprocessing → Transfer Learning → Fine-Tuning → Evaluation**

The main goal of this project is to explore a practical deep learning pipeline for deepfake detection while maintaining a clean and reproducible project structure.

