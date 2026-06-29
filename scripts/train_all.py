"""
One-shot training pipeline.

1. Generates synthetic data (if raw data is absent)
2. Builds features
3. Trains FWA classifier
4. Trains prediction models
5. Trains clustering
6. Runs anomaly detection
7. Computes SHAP global importances

Run:
    python scripts/train_all.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_RAW, DATA_PROC


def ensure_data() -> None:
    """Auto-generate synthetic data if the raw CSVs are not present."""
    needed = "Train-1542865627584.csv"
    if not (DATA_RAW / needed).exists():
        print("Raw data not found. Generating synthetic dataset…")
        from src.data.synthetic import generate_and_save
        generate_and_save()
    else:
        print("Raw data found.")


def run_features() -> None:
    print("\n[1/5] Feature Engineering")
    from src.data.features import run
    run()


def run_classification() -> None:
    print("\n[2/5] FWA Classifier")
    from src.models.classification import train
    train()


def run_prediction() -> None:
    print("\n[3/5] Prediction Models")
    from src.models.prediction import train
    train()


def run_clustering() -> None:
    print("\n[4/5] Provider Clustering")
    from src.models.clustering import train
    train()


def run_anomaly() -> None:
    print("\n[4b] Time-Series Anomaly Detection")
    from src.models.anomaly import run
    run()


def run_shap() -> None:
    print("\n[5/5] SHAP Global Importances")
    from src.models.inference import compute_global_shap
    compute_global_shap()


if __name__ == "__main__":
    print("=" * 60)
    print("TPO Sentinel - Training Pipeline")
    print("=" * 60)

    ensure_data()
    run_features()
    run_classification()
    run_prediction()
    run_clustering()
    run_anomaly()
    run_shap()

    print("\n" + "=" * 60)
    print("All models trained successfully.")
    print("   Launch dashboard: streamlit run src/app/streamlit_app.py")
    print("=" * 60)
