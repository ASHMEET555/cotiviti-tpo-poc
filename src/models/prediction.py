"""
Clinical Risk & Reimbursement Prediction (Treatment pillar).

Trains two gradient-boosted regression models:
  1. Expected total reimbursement per provider
  2. Average length-of-stay (LOS) per provider

Saves models and metrics under data/processed/models/.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_PROC, MODELS_DIR, CLF_RANDOM_STATE

EXCLUDE_COLS = {"Provider", "PotentialFraud", "FraudLabel"}
TARGETS = {
    "reimbursement": "IP_TotalReimbursed",
    "los": "AvgLOS",
}


def get_feature_cols(df: pd.DataFrame, target_col: str) -> list[str]:
    drop = EXCLUDE_COLS | {target_col}
    return [c for c in df.columns if c not in drop
            and df[c].dtype in (np.float64, np.int64, np.float32, np.int32, float, int)]


def train_single(df: pd.DataFrame, target_name: str, target_col: str) -> dict:
    df = df.copy()
    if target_col not in df.columns:
        print(f"[pred] Target '{target_col}' not found – skipping.")
        return {}

    df = df[df[target_col].notna()]
    feature_cols = get_feature_cols(df, target_col)
    X = df[feature_cols].values
    y = df[target_col].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=CLF_RANDOM_STATE
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=CLF_RANDOM_STATE
    )
    model.fit(X_tr_s, y_tr)
    preds = model.predict(X_te_s)

    mae = float(mean_absolute_error(y_te, preds))
    rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
    r2 = float(r2_score(y_te, preds))

    metrics = {"mae": mae, "rmse": rmse, "r2": r2,
               "n_train": int(X_tr.shape[0]), "n_test": int(X_te.shape[0])}

    model_path = MODELS_DIR / f"pred_{target_name}_model.joblib"
    scaler_path = MODELS_DIR / f"pred_{target_name}_scaler.joblib"
    feat_path = MODELS_DIR / f"pred_{target_name}_features.json"
    metrics_path = MODELS_DIR / f"pred_{target_name}_metrics.json"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[pred:{target_name}] MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return metrics


def train(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = pd.read_parquet(DATA_PROC / "provider_features.parquet")
    results = {}
    for name, col in TARGETS.items():
        results[name] = train_single(df, name, col)
    return results


def predict_provider(provider_id: str, target_name: str = "reimbursement",
                     df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = pd.read_parquet(DATA_PROC / "provider_features.parquet")
    model = joblib.load(MODELS_DIR / f"pred_{target_name}_model.joblib")
    scaler = joblib.load(MODELS_DIR / f"pred_{target_name}_scaler.joblib")
    with open(MODELS_DIR / f"pred_{target_name}_features.json") as f:
        feature_cols = json.load(f)

    row = df[df["Provider"] == provider_id]
    if row.empty:
        return {"error": f"Provider {provider_id} not found"}
    X = scaler.transform(row[feature_cols].values)
    pred = float(model.predict(X)[0])
    return {
        "provider_id": provider_id,
        "target": target_name,
        "predicted_value": round(pred, 2),
    }


if __name__ == "__main__":
    train()
