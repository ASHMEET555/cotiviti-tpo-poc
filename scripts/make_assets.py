"""
Export static chart PNGs used in the PowerPoint deck.

Charts exported:
  1. fraud_distribution.png     – provider fraud label distribution
  2. feature_importance.png     – top 15 SHAP features
  3. fraud_probability_dist.png – histogram of fraud probabilities
  4. cluster_scatter.png        – cluster distance scatter
  5. timeseries_sample.png      – time-series for a sample provider
  6. reimbursement_boxplot.png  – reimbursement by fraud status

Run after train_all.py:
    python scripts/make_assets.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import DATA_PROC, MODELS_DIR, ASSETS_DIR

ASSETS_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "fraud": "#e74c3c",
    "legit": "#2ecc71",
    "primary": "#2980b9",
    "warning": "#e67e22",
    "neutral": "#95a5a6",
}


def _save(fig, name: str) -> None:
    path = ASSETS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {path}")


def chart_fraud_distribution(df: pd.DataFrame) -> None:
    counts = df["FraudLabel"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Legitimate", "Fraud"],
                  [counts.get(0, 0), counts.get(1, 0)],
                  color=[COLORS["legit"], COLORS["fraud"]], width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():,}", ha="center", fontweight="bold")
    ax.set_title("Provider Fraud Label Distribution", fontweight="bold")
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "fraud_distribution.png")


def chart_feature_importance() -> None:
    import json
    shap_path = MODELS_DIR / "shap_global.json"
    if not shap_path.exists():
        print("  [skip] shap_global.json not found")
        return
    with open(shap_path) as f:
        importance = json.load(f)
    top15 = dict(list(importance.items())[:15])
    names = list(reversed(list(top15.keys())))
    vals = list(reversed(list(top15.values())))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, vals, color=COLORS["warning"])
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Top 15 Features – Global SHAP Importance", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "feature_importance.png")


def chart_fraud_prob_dist(df: pd.DataFrame) -> None:
    model_path = MODELS_DIR / "clf_model.joblib"
    if not model_path.exists():
        print("  [skip] clf_model.joblib not found")
        return
    import json, joblib
    with open(MODELS_DIR / "clf_feature_cols.json") as f:
        feature_cols = json.load(f)
    model = joblib.load(model_path)
    scaler = joblib.load(MODELS_DIR / "clf_scaler.joblib")
    X = scaler.transform(df[feature_cols].fillna(0).values)
    probas = model.predict_proba(X)[:, 1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(probas, bins=40, color=COLORS["primary"], edgecolor="white", alpha=0.8)
    ax.axvline(0.5, color=COLORS["fraud"], linestyle="--", label="Threshold 0.5")
    ax.set_xlabel("P(fraud)")
    ax.set_ylabel("Number of Providers")
    ax.set_title("Distribution of Fraud Probability Scores", fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "fraud_probability_dist.png")


def chart_cluster_scatter() -> None:
    cluster_path = DATA_PROC / "provider_clusters.parquet"
    if not cluster_path.exists():
        print("  [skip] provider_clusters.parquet not found")
        return
    clusters = pd.read_parquet(cluster_path)
    colors_map = {0: "#3498db", 1: "#e74c3c"}

    fig, ax = plt.subplots(figsize=(7, 4))
    for flag, grp in clusters.groupby("OutlierFlag"):
        label = "Outlier" if flag else "Normal"
        ax.scatter(grp["DistanceToCentroid"], grp["Cluster"],
                   c=colors_map[flag], label=label, alpha=0.7, s=30)
    ax.set_xlabel("Distance to Cluster Centroid")
    ax.set_ylabel("Cluster ID")
    ax.set_title("Provider Peer-Group Clustering", fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "cluster_scatter.png")


def chart_timeseries_sample() -> None:
    anom_path = DATA_PROC / "provider_anomalies.parquet"
    if not anom_path.exists():
        print("  [skip] provider_anomalies.parquet not found")
        return
    anom = pd.read_parquet(anom_path)
    # Pick the provider with the most anomaly flags
    top_prov = (
        anom.groupby("Provider")["AnomalyFlag"].sum()
        .sort_values(ascending=False)
        .index[0]
    )
    pdata = anom[anom["Provider"] == top_prov].sort_values("YearMonth")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(pdata["YearMonth"], pdata["TotalReimbursed"],
            color=COLORS["primary"], marker="o", markersize=4, label="Monthly Total")
    anom_pts = pdata[pdata["AnomalyFlag"] == 1]
    if not anom_pts.empty:
        ax.scatter(anom_pts["YearMonth"], anom_pts["TotalReimbursed"],
                   color=COLORS["fraud"], zorder=5, s=80, marker="X", label="Anomaly")
    ax.set_title(f"Monthly Billing Time-Series – {top_prov}", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Reimbursed ($)")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.autofmt_xdate()
    _save(fig, "timeseries_sample.png")


def chart_reimbursement_box(df: pd.DataFrame) -> None:
    if "IP_TotalReimbursed" not in df.columns:
        print("  [skip] IP_TotalReimbursed not in features")
        return
    fraud = df[df["FraudLabel"] == 1]["IP_TotalReimbursed"].dropna()
    legit = df[df["FraudLabel"] == 0]["IP_TotalReimbursed"].dropna()

    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(
        [legit, fraud],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
    )
    bp["boxes"][0].set_facecolor(COLORS["legit"])
    bp["boxes"][1].set_facecolor(COLORS["fraud"])
    ax.set_xticklabels(["Legitimate", "Fraud"])
    ax.set_ylabel("Total IP Reimbursement ($)")
    ax.set_title("Reimbursement Distribution by Fraud Status", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "reimbursement_boxplot.png")


def main() -> None:
    print("Generating chart assets…")
    feat_path = DATA_PROC / "provider_features.parquet"
    if not feat_path.exists():
        print("ERROR: provider_features.parquet not found. Run train_all.py first.")
        return

    df = pd.read_parquet(feat_path)
    chart_fraud_distribution(df)
    chart_feature_importance()
    chart_fraud_prob_dist(df)
    chart_cluster_scatter()
    chart_timeseries_sample()
    chart_reimbursement_box(df)
    print(f"\nAll charts saved to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
