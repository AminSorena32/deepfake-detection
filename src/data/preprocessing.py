from pathlib import Path
import json
import gc
import cv2 as cv
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

FRAME_DIR = DATA_DIR / "interim" / "frames"
DETECTION_DIR = DATA_DIR / "interim" / "detections"

FACE_DIR = DATA_DIR / "processed" / "faces"

TARGET_SIZE = (224, 224)

BATCH_SIZE = 50

MIN_CONFIDENCE = 0.4


# Padding ratios
PAD_LEFT = 0.20
PAD_RIGHT = 0.20
PAD_TOP = 0.30
PAD_BOTTOM = 0.35

def load_detections(
    detection_file
):
    """
    Load bounding boxes from a JSON file.
    """

    if not detection_file.exists():

        raise FileNotFoundError(
            f"Detection file not found:\n"
            f"{detection_file}"
        )

    with open(
        detection_file,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)

def preprocess_face(
    image,
    bounding_box,
    target_size=TARGET_SIZE
):
    """
    Apply padding, crop, aspect-ratio preservation,
    and resize.

    Returns
    -------
    np.ndarray or None
    """

    if image is None:
        return None

    height, width = image.shape[:2]

    x = bounding_box["x"]
    y = bounding_box["y"]

    box_width = bounding_box["width"]
    box_height = bounding_box["height"]

    confidence = bounding_box["confidence"]

    if confidence < MIN_CONFIDENCE:
        return None

    pad_left = int(
        box_width * PAD_LEFT
    )

    pad_right = int(
        box_width * PAD_RIGHT
    )

    pad_top = int(
        box_height * PAD_TOP
    )

    pad_bottom = int(
        box_height * PAD_BOTTOM
    )

    x1 = max(
        0,
        x - pad_left
    )

    y1 = max(
        0,
        y - pad_top
    )

    x2 = min(
        width,
        x + box_width + pad_right
    )

    y2 = min(
        height,
        y + box_height + pad_bottom
    )

    face_crop = image[
        y1:y2,
        x1:x2
    ]

    if face_crop.size == 0:
        return None

    crop_height, crop_width = (
        face_crop.shape[:2]
    )

    target_width, target_height = (
        target_size
    )

    scale = min(
        target_width / crop_width,
        target_height / crop_height
    )

    new_width = max(
        1,
        int(crop_width * scale)
    )

    new_height = max(
        1,
        int(crop_height * scale)
    )

    resized = cv.resize(
        face_crop,
        (new_width, new_height),
        interpolation=cv.INTER_AREA
    )

    canvas = np.zeros(
        (
            target_height,
            target_width,
            3
        ),
        dtype=np.uint8
    )

    x_offset = (
        target_width - new_width
    ) // 2

    y_offset = (
        target_height - new_height
    ) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    return canvas

def process_dataset(
    detections,
    output_dir,
    batch_size=BATCH_SIZE
):
    """
    Apply preprocessing to images using
    previously generated bounding boxes.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    saved = 0
    skipped = 0
    errors = 0

    items = list(
        detections.items()
    )

    print(
        f"Images to process: {len(items)}"
    )

    for start in range(
        0,
        len(items),
        batch_size
    ):

        batch = items[
            start:start + batch_size
        ]

        for relative_path, bounding_box in tqdm(
            batch,
            desc="Preprocessing",
            unit="image"
        ):

            try:

                image_path = (
                    FRAME_DIR /
                    relative_path
                )

                if not image_path.exists():

                    skipped += 1
                    continue

            
                image = cv.imread(
                    str(image_path)
                )

                if image is None:

                    skipped += 1
                    continue

                face = preprocess_face(
                    image=image,
                    bounding_box=bounding_box,
                    target_size=TARGET_SIZE
                )

                if face is None:

                    skipped += 1
                    continue

                output_path = (
                    output_dir /
                    relative_path
                )

                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                success = cv.imwrite(
                    str(output_path),
                    face
                )

                if success:
                    saved += 1
                else:
                    errors += 1

                del image
                del face

            except Exception as error:

                print(
                    f"\nError processing "
                    f"{relative_path}: {error}"
                )

                errors += 1

        gc.collect()

        print(
            f"Saved={saved} | "
            f"Skipped={skipped} | "
            f"Errors={errors}"
        )

    return (
        saved,
        skipped,
        errors
    )

def main():

    datasets = {

        "train": (
            DETECTION_DIR / "train.json",
            FACE_DIR / "train"
        ),

        "val": (
            DETECTION_DIR / "val.json",
            FACE_DIR / "val"
        ),

        "test": (
            DETECTION_DIR / "test.json",
            FACE_DIR / "test"
        )
    }

    total_saved = 0
    total_skipped = 0
    total_errors = 0

    for dataset_name, (
        detection_file,
        output_dir
    ) in datasets.items():

        print("\n" + "=" * 70)
        print(
            f"PREPROCESSING - "
            f"{dataset_name.upper()}"
        )
        print("=" * 70)

        detections = load_detections(
            detection_file
        )

        saved, skipped, errors = (
            process_dataset(
                detections=detections,
                output_dir=output_dir,
                batch_size=BATCH_SIZE
            )
        )

        total_saved += saved
        total_skipped += skipped
        total_errors += errors

        gc.collect()

    print("\n")
    print("=" * 70)
    print("PREPROCESSING FINISHED")
    print("=" * 70)

    print(f"Saved   : {total_saved}")
    print(f"Skipped : {total_skipped}")
    print(f"Errors  : {total_errors}")


if __name__ == "__main__":
    main()

