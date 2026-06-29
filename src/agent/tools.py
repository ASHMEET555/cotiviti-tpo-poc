"""
Tool registry for the chain-of-thought agent.

Each tool wraps a model module and returns a structured dict so the agent
can include the results in its reasoning prompt.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_PROC


def _load_provider_df() -> pd.DataFrame:
    return pd.read_parquet(DATA_PROC / "provider_features.parquet")


# ── Tool functions ─────────────────────────────────────────────────────────────

def tool_classify(provider_id: str, df: pd.DataFrame | None = None) -> dict:
    """Return fraud probability from the FWA classifier."""
    from src.models.classification import predict_provider
    return predict_provider(provider_id, df)


def tool_anomalies(provider_id: str) -> dict:
    """Return time-series anomaly summary for a provider."""
    from src.models.anomaly import get_provider_anomalies
    anom_df = get_provider_anomalies(provider_id)
    if anom_df.empty:
        return {"provider_id": provider_id, "anomaly_months": 0, "anomaly_details": []}
    flags = anom_df[anom_df["AnomalyFlag"] == 1]
    details = flags[["YearMonth", "TotalReimbursed", "ClaimCount", "AnomalyScore"]].to_dict(orient="records")
    return {
        "provider_id": provider_id,
        "total_months": len(anom_df),
        "anomaly_months": len(flags),
        "anomaly_rate": round(len(flags) / len(anom_df), 3),
        "anomaly_details": details[:5],  # top 5 for prompt brevity
    }


def tool_cluster(provider_id: str) -> dict:
    """Return peer-group cluster info."""
    from src.models.clustering import get_provider_cluster
    return get_provider_cluster(provider_id)


def tool_explain(provider_id: str, df: pd.DataFrame | None = None) -> dict:
    """Return top SHAP feature explanations."""
    from src.models.inference import explain_provider
    result = explain_provider(provider_id, df)
    if "top_features" in result:
        # Keep only top 5 for prompt brevity
        top5 = dict(list(result["top_features"].items())[:5])
        result["top_features"] = top5
    return result


def tool_predict_reimbursement(provider_id: str, df: pd.DataFrame | None = None) -> dict:
    """Return predicted reimbursement for the provider."""
    from src.models.prediction import predict_provider
    return predict_provider(provider_id, "reimbursement", df)


TOOL_REGISTRY = {
    "classify": tool_classify,
    "anomalies": tool_anomalies,
    "cluster": tool_cluster,
    "explain": tool_explain,
    "predict_reimbursement": tool_predict_reimbursement,
}
