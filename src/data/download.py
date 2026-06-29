"""
Download the Healthcare Provider Fraud Detection dataset from Kaggle.
Requires KAGGLE_USERNAME and KAGGLE_KEY in the environment (or .env).

Falls back to printing manual-download instructions if credentials are absent.
"""
import os
import sys
import zipfile
from pathlib import Path

# Allow running as a script directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import DATA_RAW, KAGGLE_DATASET, KAGGLE_KEY, KAGGLE_USERNAME


EXPECTED_FILES = [
    "Train-1542865627584.csv",
    "Train_Inpatientdata-1542865627584.csv",
    "Train_Outpatientdata-1542865627584.csv",
    "Train_Beneficiarydata-1542865627584.csv",
]


def already_downloaded() -> bool:
    return all((DATA_RAW / f).exists() for f in EXPECTED_FILES)


def download_via_kaggle_api() -> bool:
    """Use the kaggle Python package to download. Returns True on success."""
    try:
        import kaggle  # noqa: F401 – triggers auth from env vars
        from kaggle.api.kaggle_api_extended import KaggleApiExtended

        api = KaggleApiExtended()
        api.authenticate()
        print(f"Downloading dataset '{KAGGLE_DATASET}' …")
        api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_RAW), unzip=True)
        print("Download complete.")
        return True
    except Exception as exc:
        print(f"[download] Kaggle API failed: {exc}")
        return False


def print_manual_instructions() -> None:
    print(
        "\n─────────────────────────────────────────────────────────────\n"
        "Manual download instructions\n"
        "─────────────────────────────────────────────────────────────\n"
        "1. Go to:\n"
        f"   https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
        "2. Click 'Download' and unzip all CSVs into:\n"
        f"   {DATA_RAW}\n"
        "3. Re-run this script or continue to the next step.\n"
        "\nAlternatively, run the synthetic fallback:\n"
        "   python src/data/synthetic.py\n"
        "─────────────────────────────────────────────────────────────\n"
    )


def main() -> None:
    if already_downloaded():
        print("Dataset already present in data/raw/. Skipping download.")
        return

    if KAGGLE_USERNAME and KAGGLE_KEY:
        # Inject credentials so kaggle SDK can find them
        os.environ.setdefault("KAGGLE_USERNAME", KAGGLE_USERNAME)
        os.environ.setdefault("KAGGLE_KEY", KAGGLE_KEY)
        if download_via_kaggle_api():
            return

    print_manual_instructions()


if __name__ == "__main__":
    main()
