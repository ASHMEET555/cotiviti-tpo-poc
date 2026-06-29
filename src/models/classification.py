"""
FWA (Fraud, Waste, Abuse) Classifier.

Trains an XGBoost binary classifier on provider-level features and saves:
  - data/processed/models/clf_model.joblib
  - data/processed/models/clf_metrics.json
  - data/processed/models/clf_feature_cols.json
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_PROC, MODELS_DIR, CLF_N_ESTIMATORS, CLF_MAX_DEPTH, CLF_RANDOM_STATE

MODEL_PATH = MODELS_DIR / "clf_model.joblib"
SCALER_PATH = MODELS_DIR / "clf_scaler.joblib"
METRICS_PATH = MODELS_DIR / "clf_metrics.json"
FEATURES_PATH = MODELS_DIR / "clf_feature_cols.json"

EXCLUDE_COLS = {"Provider", "PotentialFraud", "FraudLabel"}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS
            and df[c].dtype in (np.float64, np.int64, np.float32, np.int32, float, int)]


def train(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = pd.read_parquet(DATA_PROC / "provider_features.parquet")

    feature_cols = get_feature_cols(df)
    X = df[feature_cols].values
    y = df["FraudLabel"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=CLF_RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=CLF_N_ESTIMATORS,
            max_depth=CLF_MAX_DEPTH,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=CLF_RANDOM_STATE,
            n_jobs=-1,
        )
    except ImportError:
        model = RandomForestClassifier(
            n_estimators=CLF_N_ESTIMATORS,
            max_depth=CLF_MAX_DEPTH,
            random_state=CLF_RANDOM_STATE,
            n_jobs=-1,
        )

    model.fit(X_train_s, y_train)

    proba = model.predict_proba(X_test_s)[:, 1]
    preds = (proba >= 0.5).astype(int)

    roc = float(roc_auc_score(y_test, proba))
    ap = float(average_precision_score(y_test, proba))
    cm = confusion_matrix(y_test, preds).tolist()
    report = classification_report(y_test, preds, output_dict=True)

    metrics = {
        "roc_auc": roc,
        "avg_precision": ap,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "positive_rate_train": float(y_train.mean()),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_cols, f)

    print(f"[clf] ROC-AUC={roc:.4f}  Avg-Precision={ap:.4f}")
    return metrics


def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH) as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols


def predict_provider(provider_id: str, df: pd.DataFrame | None = None) -> dict:
    """Return fraud probability and raw score for a single provider."""
    if df is None:
        df = pd.read_parquet(DATA_PROC / "provider_features.parquet")
    model, scaler, feature_cols = load_model()
    row = df[df["Provider"] == provider_id]
    if row.empty:
        return {"error": f"Provider {provider_id} not found"}
    X = scaler.transform(row[feature_cols].values)
    proba = float(model.predict_proba(X)[0, 1])
    return {
        "provider_id": provider_id,
        "fraud_probability": round(proba, 4),
        "fraud_flag": proba >= 0.5,
        "true_label": int(row["FraudLabel"].iloc[0]),
    }


if __name__ == "__main__":
    train()
