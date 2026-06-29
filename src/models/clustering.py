"""
Provider Peer-Group Clustering (Operations pillar).

Uses KMeans to segment providers into peer cohorts based on billing behaviour.
Flags providers that are statistical outliers within their cluster.

Saves:
  - data/processed/models/cluster_model.joblib
  - data/processed/models/cluster_scaler.joblib
  - data/processed/provider_clusters.parquet
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DATA_PROC, MODELS_DIR, CLUSTER_K, CLUSTER_RANDOM_STATE

CLUSTER_FEATURES = [
    "IP_ClaimCount", "OP_ClaimCount",
    "IP_TotalReimbursed", "OP_TotalReimbursed",
    "IP_AvgReimbursed", "OP_AvgReimbursed",
    "UniquePhysicians", "IP_UniqueBenes", "OP_UniqueBenes",
    "AvgLOS", "AvgBeneAge", "AvgChronicCount",
]

CLUSTERS_PATH = DATA_PROC / "provider_clusters.parquet"


def train(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        df = pd.read_parquet(DATA_PROC / "provider_features.parquet")

    feat_cols = [c for c in CLUSTER_FEATURES if c in df.columns]
    X_raw = df[feat_cols].fillna(0).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # Determine best K via silhouette (search 2..8, cap at n_samples//2)
    max_k = min(8, len(df) // 2)
    best_k = min(CLUSTER_K, max_k)
    best_sil = -1.0
    if len(df) > 10:
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=CLUSTER_RANDOM_STATE, n_init=10)
            labels_tmp = km.fit_predict(X)
            sil = float(silhouette_score(X, labels_tmp))
            if sil > best_sil:
                best_sil = sil
                best_k = k

    kmeans = KMeans(n_clusters=best_k, random_state=CLUSTER_RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # Distance to own centroid (outlier score)
    centroids = kmeans.cluster_centers_
    distances = np.linalg.norm(X - centroids[cluster_labels], axis=1)

    out = df[["Provider", "FraudLabel"]].copy() if "FraudLabel" in df.columns else df[["Provider"]].copy()
    out["Cluster"] = cluster_labels
    out["DistanceToCentroid"] = distances

    # Flag providers > 2 std above their cluster mean distance
    cluster_stats = out.groupby("Cluster")["DistanceToCentroid"].agg(["mean", "std"]).reset_index()
    out = out.merge(cluster_stats, on="Cluster")
    out["OutlierFlag"] = (out["DistanceToCentroid"] > out["mean"] + 2 * out["std"]).astype(int)
    out = out.drop(columns=["mean", "std"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, MODELS_DIR / "cluster_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "cluster_scaler.joblib")
    with open(MODELS_DIR / "cluster_features.json", "w") as f:
        json.dump(feat_cols, f)
    out.to_parquet(CLUSTERS_PATH, index=False)

    metrics = {
        "n_clusters": best_k,
        "silhouette_score": best_sil,
        "n_outliers": int(out["OutlierFlag"].sum()),
    }
    print(f"[cluster] k={best_k}  silhouette={best_sil:.4f}  outliers={metrics['n_outliers']}")
    return metrics


def get_provider_cluster(provider_id: str) -> dict:
    clusters = pd.read_parquet(CLUSTERS_PATH)
    row = clusters[clusters["Provider"] == provider_id]
    if row.empty:
        return {"error": f"Provider {provider_id} not found"}
    r = row.iloc[0]
    return {
        "provider_id": provider_id,
        "cluster": int(r["Cluster"]),
        "distance_to_centroid": round(float(r["DistanceToCentroid"]), 4),
        "outlier_flag": bool(r["OutlierFlag"]),
    }


def get_cluster_summary() -> pd.DataFrame:
    clusters = pd.read_parquet(CLUSTERS_PATH)
    return clusters.groupby("Cluster").agg(
        Count=("Provider", "count"),
        AvgDistance=("DistanceToCentroid", "mean"),
        Outliers=("OutlierFlag", "sum"),
        FraudRate=("FraudLabel", "mean") if "FraudLabel" in clusters.columns else ("Provider", "count"),
    ).reset_index()


if __name__ == "__main__":
    train()
