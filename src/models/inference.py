"""
SHAP Explainability (Inference pillar).

Provides global feature importance and per-provider local SHAP explanations
for the FWA classification model.

Saves:
  - data/processed/models/shap_global.json  (feature -> mean |SHAP value|)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_PROC, MODELS_DIR
from src.models.classification import load_model

SHAP_GLOBAL_PATH = MODELS_DIR / "shap_global.json"


def compute_global_shap(df: pd.DataFrame | None = None) -> dict:
    """Compute global mean |SHAP| values and save to file."""
    if df is None:
        df = pd.read_parquet(DATA_PROC / "provider_features.parquet")

    model, scaler, feature_cols = load_model()
    if len(df) > 500:
        df = df.sample(500, random_state=42)
    X = scaler.transform(df[feature_cols].values)

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        # For binary classification: shap_values may be a list [neg_class, pos_class]
        if isinstance(shap_vals, list):
            shap_matrix = shap_vals[1]
        else:
            shap_matrix = shap_vals

        mean_abs = np.abs(shap_matrix).mean(axis=0)
        importance = dict(zip(feature_cols, mean_abs.tolist()))
        # Sort by importance descending
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        print(f"[shap] SHAP computation failed ({e}), using feature importances instead.")
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            importance = dict(sorted(
                zip(feature_cols, imp.tolist()),
                key=lambda x: x[1], reverse=True
            ))
        else:
            importance = {c: 1.0 / len(feature_cols) for c in feature_cols}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SHAP_GLOBAL_PATH, "w") as f:
        json.dump(importance, f, indent=2)
    print(f"[shap] Global SHAP saved → {SHAP_GLOBAL_PATH}")
    return importance


def explain_provider(provider_id: str, df: pd.DataFrame | None = None) -> dict:
    """Return local SHAP values for a single provider."""
    if df is None:
        df = pd.read_parquet(DATA_PROC / "provider_features.parquet")

    model, scaler, feature_cols = load_model()
    row = df[df["Provider"] == provider_id]
    if row.empty:
        return {"error": f"Provider {provider_id} not found"}

    X_raw = row[feature_cols].values
    X = scaler.transform(X_raw)

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            local_vals = shap_vals[1][0]
        else:
            local_vals = shap_vals[0]
        explanation = dict(zip(feature_cols, local_vals.tolist()))
    except Exception:
        if hasattr(model, "feature_importances_"):
            proba = model.predict_proba(X)[0, 1]
            imp = model.feature_importances_
            # Scale by predicted probability as rough surrogate
            explanation = {c: float(v * proba) for c, v in zip(feature_cols, imp)}
        else:
            explanation = {c: 0.0 for c in feature_cols}

    # Top 10 by absolute value
    top10 = dict(sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
    return {
        "provider_id": provider_id,
        "top_features": top10,
        "all_features": explanation,
    }


def load_global_shap() -> dict:
    if not SHAP_GLOBAL_PATH.exists():
        return {}
    with open(SHAP_GLOBAL_PATH) as f:
        return json.load(f)


if __name__ == "__main__":
    compute_global_shap()
