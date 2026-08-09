from pathlib import Path
import random
import cv2 as cv
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

VIDEO_DIR = DATA_DIR / "raw" / "faceforensics-1000"

FRAME_DIR = DATA_DIR / "interim" / "frames"

NUM_FRAMES = 5
TRAIN_RATIO = 0.80
VAL_RATIO = 0.80

RANDOM_SEED = 42


REAL_VIDEO_DIR = VIDEO_DIR / "ffpp_real"
FAKE_VIDEO_DIR = VIDEO_DIR / "ffpp_fake"

def split_videos(video_paths):
    """
    Split videos into train, validation and test sets.

    The split is performed at video level to prevent frames
    from the same video appearing in different datasets.
    """

    video_paths = list(video_paths)

    random.shuffle(video_paths)

    # First split: Train + Validation / Test
    train_size = int(len(video_paths) * TRAIN_RATIO)

    train_val_videos = video_paths[:train_size]
    test_videos = video_paths[train_size:]

    # Second split: Train / Validation
    train_size = int(len(train_val_videos) * VAL_RATIO)

    train_videos = train_val_videos[:train_size]
    val_videos = train_val_videos[train_size:]

    return train_videos, val_videos, test_videos

def extract_frames(video_paths, output_dir, dataset_name):
    """
    Extract uniformly distributed frames from videos.

    Parameters
    ----------
    video_paths : list[Path]
        List of video file paths.

    output_dir : Path
        Directory where extracted frames are saved.

    dataset_name : str
        Name displayed in the progress bar.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    for video_path in tqdm(
        video_paths,
        desc=f"Extracting {dataset_name}",
        unit="video"
    ):

        video_path = Path(video_path)

        cap = cv.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Warning: Could not open {video_path}")
            continue

        total_frames = int(
            cap.get(cv.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            cap.release()
            continue

        # Select uniformly distributed frames
        frame_indices = np.linspace(
            0,
            total_frames - 1,
            NUM_FRAMES,
            dtype=int
        )

        video_name = video_path.stem

        for frame_number, frame_index in enumerate(frame_indices):

            cap.set(
                cv.CAP_PROP_POS_FRAMES,
                int(frame_index)
            )

            ret, frame = cap.read()

            if not ret:
                continue

            output_path = (
                output_dir /
                f"{video_name}_{frame_number}.jpg"
            )

            cv.imwrite(
                str(output_path),
                frame
            )

        cap.release()

def main():

    # Make experiments reproducible
    random.seed(RANDOM_SEED)

    # Check dataset directories
    if not REAL_VIDEO_DIR.exists():
        raise FileNotFoundError(
            f"Real video directory not found:\n"
            f"{REAL_VIDEO_DIR}"
        )

    if not FAKE_VIDEO_DIR.exists():
        raise FileNotFoundError(
            f"Fake video directory not found:\n"
            f"{FAKE_VIDEO_DIR}"
        )

    # Collect videos
    real_videos = list(
        REAL_VIDEO_DIR.glob("*.mp4")
    )

    fake_videos = list(
        FAKE_VIDEO_DIR.glob("*.mp4")
    )

    print(f"Real videos: {len(real_videos)}")
    print(f"Fake videos: {len(fake_videos)}")

    # Split datasets
    (
        train_real,
        val_real,
        test_real
    ) = split_videos(real_videos)

    (
        train_fake,
        val_fake,
        test_fake
    ) = split_videos(fake_videos)

    print("\nDataset split:")
    print(
        f"Train: "
        f"{len(train_real) + len(train_fake)} videos"
    )

    print(
        f"Validation: "
        f"{len(val_real) + len(val_fake)} videos"
    )

    print(
        f"Test: "
        f"{len(test_real) + len(test_fake)} videos"
    )

    # Output directories
    train_real_dir = FRAME_DIR / "train" / "real"
    train_fake_dir = FRAME_DIR / "train" / "fake"

    val_real_dir = FRAME_DIR / "val" / "real"
    val_fake_dir = FRAME_DIR / "val" / "fake"

    test_real_dir = FRAME_DIR / "test" / "real"
    test_fake_dir = FRAME_DIR / "test" / "fake"

    # Extract Train frames
    extract_frames(
        train_real,
        train_real_dir,
        "Train - Real"
    )

    extract_frames(
        train_fake,
        train_fake_dir,
        "Train - Fake"
    )

    # Extract Validation frames
    extract_frames(
        val_real,
        val_real_dir,
        "Validation - Real"
    )

    extract_frames(
        val_fake,
        val_fake_dir,
        "Validation - Fake"
    )

    # Extract Test frames
    extract_frames(
        test_real,
        test_real_dir,
        "Test - Real"
    )

    extract_frames(
        test_fake,
        test_fake_dir,
        "Test - Fake"
    )

    print("\nFrame extraction completed successfully.")


if __name__ == "__main__":
    main()

