import subprocess
from pathlib import Path


DATA_DIR = Path("data/raw")
KAGGLE_DIR = Path.home() / ".kaggle"
KAGGLE_FILE = KAGGLE_DIR / "kaggle.json"


def setup_kaggle():
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

    if not KAGGLE_FILE.exists():
        raise FileNotFoundError(
            "kaggle.json was not found. "
            "Please place it in ~/.kaggle/"
        )

    KAGGLE_FILE.chmod(0o600)


def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            "rohingarg12/faceforensics-1000",
            "-p",
            str(DATA_DIR),
        ],
        check=True,
    )


def extract_dataset():
    zip_file = DATA_DIR / "faceforensics-1000.zip"

    if not zip_file.exists():
        raise FileNotFoundError(f"{zip_file} not found.")

    subprocess.run(
        [
            "unzip",
            "-q",
            str(zip_file),
            "-d",
            str(DATA_DIR),
        ],
        check=True,
    )


if __name__ == "__main__":
    setup_kaggle()
    download_dataset()
    extract_dataset()

    print("Dataset downloaded and extracted successfully.")