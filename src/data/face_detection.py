from pathlib import Path
import json
import urllib.request
import cv2 as cv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

FRAME_DIR = DATA_DIR / "interim" / "frames"
DETECTION_DIR = DATA_DIR / "interim" / "detections"

MODEL_DIR = PROJECT_ROOT / "models" / "face_detection"
MODEL_PATH = (
    MODEL_DIR /
    "face_detection_yunet_2023mar.onnx"
)

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)

INPUT_SIZE = (320, 320)

MIN_CONFIDENCE = 0.4
NMS_THRESHOLD = 0.3
TOP_K = 5000

def download_yunet():
    """
    Download YuNet ONNX model if it does not already exist.
    """

    if MODEL_PATH.exists():
        print("YuNet model already exists.")
        return

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Downloading YuNet model...")

    urllib.request.urlretrieve(
        YUNET_URL,
        MODEL_PATH
    )

    print("YuNet model downloaded successfully.")

def create_detector():
    """
    Create and return an OpenCV YuNet detector.
    """

    if not hasattr(cv, "FaceDetectorYN"):
        raise RuntimeError(
            "FaceDetectorYN is not available in "
            "this OpenCV installation."
        )

    download_yunet()

    detector = cv.FaceDetectorYN.create(
        model=str(MODEL_PATH),
        config="",
        input_size=INPUT_SIZE,
        score_threshold=MIN_CONFIDENCE,
        nms_threshold=NMS_THRESHOLD,
        top_k=TOP_K,
        backend_id=cv.dnn.DNN_BACKEND_OPENCV,
        target_id=cv.dnn.DNN_TARGET_CPU
    )

    print("YuNet detector loaded.")

    return detector

def detect_face(
    image,
    detector,
    min_confidence=MIN_CONFIDENCE
):
    """
    Detect the highest-confidence face.

    Parameters
    ----------
    image : np.ndarray
        Input image.

    detector :
        YuNet detector.

    Returns
    -------
    dict or None

        {
            "x": int,
            "y": int,
            "width": int,
            "height": int,
            "confidence": float
        }
    """

    if image is None:
        return None

    height, width = image.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(image)

    if faces is None:
        return None

    best_face = None
    best_score = -1.0

    for face in faces:

        score = float(face[-1])

        if (
            score >= min_confidence
            and score > best_score
        ):
            best_face = face
            best_score = score

    if best_face is None:
        return None

    x, y, box_width, box_height = (
        best_face[:4].astype(int)
    )

    return {
        "x": int(x),
        "y": int(y),
        "width": int(box_width),
        "height": int(box_height),
        "confidence": float(best_score)
    }

def process_dataset(
    image_paths,
    output_file,
    detector
):
    """
    Run face detection on all images and save
    bounding boxes to a JSON file.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    detections = {}

    skipped = 0
    errors = 0

    for image_path in tqdm(
        image_paths,
        desc=f"Detecting faces - {output_file.stem}",
        unit="image"
    ):

        try:

            image = cv.imread(
                str(image_path)
            )

            if image is None:

                skipped += 1
                continue

            bounding_box = detect_face(
                image=image,
                detector=detector
            )

            if bounding_box is None:

                skipped += 1
                continue

            # Store path relative to frame directory
            relative_path = str(
                image_path.relative_to(
                    FRAME_DIR
                )
            )

            detections[relative_path] = (
                bounding_box
            )

            del image

        except Exception as error:

            print(
                f"\nError processing "
                f"{image_path}: {error}"
            )

            errors += 1

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            detections,
            file,
            indent=4
        )

    print(
        f"\nSaved detections: "
        f"{output_file}"
    )

    print(
        f"Detected : {len(detections)}"
    )

    print(
        f"Skipped  : {skipped}"
    )

    print(
        f"Errors   : {errors}"
    )

def get_dataset_paths():

    return {

        "train": (
            [
                *(
                    FRAME_DIR /
                    "train" /
                    "real"
                ).glob("*.jpg")
            ],
            [
                *(
                    FRAME_DIR /
                    "train" /
                    "fake"
                ).glob("*.jpg")
            ],
            DETECTION_DIR / "train.json"
        ),

        "val": (
            [
                *(
                    FRAME_DIR /
                    "val" /
                    "real"
                ).glob("*.jpg")
            ],
            [
                *(
                    FRAME_DIR /
                    "val" /
                    "fake"
                ).glob("*.jpg")
            ],
            DETECTION_DIR / "val.json"
        ),

        "test": (
            [
                *(
                    FRAME_DIR /
                    "test" /
                    "real"
                ).glob("*.jpg")
            ],
            [
                *(
                    FRAME_DIR /
                    "test" /
                    "fake"
                ).glob("*.jpg")
            ],
            DETECTION_DIR / "test.json"
        )
    }

def main():

    print(
        "OpenCV version:",
        cv.__version__
    )

    detector = create_detector()

    datasets = get_dataset_paths()

    for dataset_name, (
        real_images,
        fake_images,
        output_file
    ) in datasets.items():

        print("\n" + "=" * 70)
        print(
            f"FACE DETECTION - "
            f"{dataset_name.upper()}"
        )
        print("=" * 70)

        image_paths = (
            real_images +
            fake_images
        )

        image_paths = sorted(
            image_paths
        )

        print(
            f"Images: {len(image_paths)}"
        )

        process_dataset(
            image_paths=image_paths,
            output_file=output_file,
            detector=detector
        )

    print("\n")
    print("=" * 70)
    print("FACE DETECTION FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()

