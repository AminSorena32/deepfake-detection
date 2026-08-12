from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.models.build_model import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed" / "faces"

MODEL_DIR = PROJECT_ROOT / "models"

TRANSFER_LEARNING_MODEL = (
    MODEL_DIR / "transfer_learning_best.keras"
)


INPUT_SHAPE = (224, 224, 3)

BATCH_SIZE = 32
EPOCHS = 10

SHUFFLE_BUFFER_SIZE = 1000

LEARNING_RATE = 1e-4

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
    Apply EfficientNet preprocessing to an image.
    """

    image = preprocess_input(image)

    return image, label

def create_dataset(
    real_dir,
    fake_dir,
    shuffle=False
):
    """
    Create a labeled tf.data.Dataset.

    Labels:
        real = 0
        fake = 1
    """

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

    dataset = real_images.concatenate(
        fake_images
    )

    dataset = dataset.map(
        preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=SHUFFLE_BUFFER_SIZE
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset

def create_training_datasets():
    """
    Create training and validation datasets.
    """

    train_dir = PROCESSED_DIR / "train"
    validation_dir = PROCESSED_DIR / "val"

    train_dataset = create_dataset(
        real_dir=train_dir / "real",
        fake_dir=train_dir / "fake",
        shuffle=True
    )

    validation_dataset = create_dataset(
        real_dir=validation_dir / "real",
        fake_dir=validation_dir / "fake",
        shuffle=False
    )

    return (
        train_dataset,
        validation_dataset
    )

def train_transfer_learning():
    """
    Train the DeepFake detector using transfer learning.

    EfficientNetB0 is loaded with ImageNet pretrained
    weights and its backbone remains frozen.

    Only the classification head is trained.
    """
    (
        train_dataset,
        validation_dataset
    ) = create_training_datasets()

    model = build_model(
        input_shape=INPUT_SHAPE,
        learning_rate=LEARNING_RATE
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=TRANSFER_LEARNING_MODEL,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[
            early_stopping,
            checkpoint
        ]
    )

    print("\n" + "=" * 70)
    print("TRANSFER LEARNING FINISHED")
    print("=" * 70)

    print(
        f"\nBest model saved to:\n"
        f"{TRANSFER_LEARNING_MODEL}"
    )

    return model, history

def main():

    print("=" * 70)
    print("DEEPFAKE DETECTION - TRANSFER LEARNING")
    print("=" * 70)

    print(
        f"\nProcessed data:\n"
        f"{PROCESSED_DIR}"
    )

    print(
        f"\nModel output:\n"
        f"{TRANSFER_LEARNING_MODEL}"
    )

    train_transfer_learning()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()

