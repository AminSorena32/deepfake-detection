from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed" / "faces"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
MODEL_PATH = MODEL_DIR / "best_model.keras"
THRESHOLD = 0.5

def load_image(path):
    """
    Load an image from disk and decode it as RGB.
    """

    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    return image

def preprocess(image, label):
    """
    Apply EfficientNet preprocessing.
    """

    image = tf.keras.applications.efficientnet.preprocess_input(
        image
    )

    return image, label

def create_test_dataset():
    """
    Create the test dataset.

    Labels:
        real = 0
        fake = 1
    """

    test_dir = PROCESSED_DIR / "test"

    real_dir = test_dir / "real"
    fake_dir = test_dir / "fake"

    real_paths = tf.data.Dataset.list_files(
        str(real_dir / "*.jpg"),
        shuffle=False
    )
    fake_paths = tf.data.Dataset.list_files(
        str(fake_dir / "*.jpg"),
        shuffle=False
    )
    real_images = real_paths.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    fake_images = fake_paths.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    real_images = real_images.map(
        lambda image: (
            image,
            tf.constant(0, dtype=tf.int32)
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    fake_images = fake_images.map(
        lambda image: (
            image,
            tf.constant(1, dtype=tf.int32)
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_dataset = real_images.concatenate(
        fake_images
    )
    test_dataset = test_dataset.map(
        preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_dataset = test_dataset.batch(
        BATCH_SIZE
    )
    test_dataset = test_dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return test_dataset

def get_predictions(model, test_dataset):
    """
    Collect true labels and prediction probabilities.
    """

    y_true = []
    y_pred_prob = []

    for images, labels in test_dataset:

        predictions = model.predict(
            images,
            verbose=0
        )

        y_true.extend(
            labels.numpy()
        )

        y_pred_prob.extend(
            predictions.flatten()
        )

    y_true = np.array(
        y_true,
        dtype=np.int32
    )

    y_pred_prob = np.array(
        y_pred_prob,
        dtype=np.float32
    )

    y_pred = (
        y_pred_prob >= THRESHOLD
    ).astype(np.int32)

    return (
        y_true,
        y_pred,
        y_pred_prob
    )

def calculate_metrics(
    model,
    test_dataset,
    y_true,
    y_pred,
    y_pred_prob
):
    """
    Calculate all evaluation metrics.
    """
    loss, accuracy = model.evaluate(
        test_dataset,
        verbose=0
    )
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )
    roc_auc = roc_auc_score(
        y_true,
        y_pred_prob
    )
    cm = confusion_matrix(
        y_true,
        y_pred
    )
    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }

def save_confusion_matrix(cm):
    """
    Save the confusion matrix figure.
    """

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(cm)

    plt.title(
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.xticks(
        [0, 1],
        ["Real", "Fake"]
    )

    plt.yticks(
        [0, 1],
        ["Real", "Fake"]
    )

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

def save_roc_curve(
    y_true,
    y_pred_prob
):
    """
    Save the ROC curve figure.
    """

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fpr, tpr, _ = roc_curve(
        y_true,
        y_pred_prob
    )

    auc = roc_auc_score(
        y_true,
        y_pred_prob
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "roc_curve.png",
        dpi=300
    )

    plt.close()

def save_metrics(metrics):
    """
    Save evaluation metrics to a text file.
    """

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        METRICS_DIR /
        "evaluation_results.txt"
    )

    cm = metrics["confusion_matrix"]

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "DeepFake Detection - Evaluation Results\n"
        )

        file.write(
            "=" * 50 + "\n\n"
        )

        file.write(
            f"Loss      : {metrics['loss']:.4f}\n"
        )

        file.write(
            f"Accuracy  : {metrics['accuracy']:.4f}\n"
        )

        file.write(
            f"Precision : {metrics['precision']:.4f}\n"
        )

        file.write(
            f"Recall    : {metrics['recall']:.4f}\n"
        )

        file.write(
            f"F1-Score  : {metrics['f1_score']:.4f}\n"
        )

        file.write(
            f"ROC-AUC   : {metrics['roc_auc']:.4f}\n"
        )

        file.write(
            "\nConfusion Matrix:\n"
        )

        file.write(
            f"[[{cm[0, 0]}, {cm[0, 1]}],\n"
            f" [{cm[1, 0]}, {cm[1, 1]}]]\n"
        )

    return output_file

def print_results(metrics):
    """
    Print evaluation results.
    """

    cm = metrics["confusion_matrix"]

    print("\n")
    print("=" * 70)
    print("DEEPFAKE DETECTION - EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"\nLoss      : {metrics['loss']:.4f}"
    )

    print(
        f"Accuracy  : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : {metrics['precision']:.4f}"
    )

    print(
        f"Recall    : {metrics['recall']:.4f}"
    )

    print(
        f"F1-Score  : {metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC   : {metrics['roc_auc']:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    print("=" * 70)

def evaluate_model():
    """
    Run the complete model evaluation pipeline.
    """
    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )
    print(
        f"Loading model:\n{MODEL_PATH}"
    )
    model = tf.keras.models.load_model(
        MODEL_PATH
    )
    print(
        "\nCreating test dataset..."
    )
    test_dataset = create_test_dataset()
    print(
        "Generating predictions..."
    )
    (
        y_true,
        y_pred,
        y_pred_prob
    ) = get_predictions(
        model,
        test_dataset
    )
    metrics = calculate_metrics(
        model=model,
        test_dataset=test_dataset,
        y_true=y_true,
        y_pred=y_pred,
        y_pred_prob=y_pred_prob
    )

    save_confusion_matrix(
        metrics["confusion_matrix"]
    )
    save_roc_curve(
        y_true,
        y_pred_prob
    )
    metrics_file = save_metrics(
        metrics
    )
    print_results(
        metrics
    )
    print(
        f"\nMetrics saved to:\n"
        f"{metrics_file}"
    )
    print(
        f"\nFigures saved to:\n"
        f"{FIGURES_DIR}"
    )
    return metrics

def main():

    print("=" * 70)
    print("DEEPFAKE DETECTION - MODEL EVALUATION")
    print("=" * 70)

    evaluate_model()

    print("\n")
    print("=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
