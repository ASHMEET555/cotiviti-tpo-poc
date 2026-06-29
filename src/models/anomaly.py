"""
Time-Series Anomaly Detection (Operations pillar).

For each provider's monthly billing series, applies complementary
methods and returns a combined anomaly score:
  1. Rolling z-score on TotalReimbursed
  2. Detrended residual z-score
  3. Global IsolationForest scores (one model on all monthly records)

Saves:
  - data/processed/provider_anomalies.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (DATA_PROC, ANOMALY_CONTAMINATION,
                         ANOMALY_ZSCORE_THRESHOLD)

ANOMALIES_PATH = DATA_PROC / "provider_anomalies.parquet"


def _rolling_zscore(series: pd.Series, window: int = 6) -> pd.Series:
    mu = series.rolling(window, min_periods=2).mean()
    sigma = series.rolling(window, min_periods=2).std().replace(0, 1e-9)
    return ((series - mu) / sigma).fillna(0)


def _stl_residual_zscore(series: pd.Series) -> pd.Series:
    """Detrended residual z-score (fast; suitable for thousands of providers)."""
    if len(series) < 3:
        return pd.Series(0.0, index=series.index)
    trend = series.rolling(3, min_periods=1, center=True).mean()
    resid = series - trend
    z = (resid - resid.mean()) / (resid.std() + 1e-9)
    return z.fillna(0)


def detect_for_provider(
    provider_ts: pd.DataFrame,
    iso_flag: pd.Series | None = None,
    iso_score: pd.Series | None = None,
) -> pd.DataFrame:
    """
    provider_ts: DataFrame with columns [YearMonth, ClaimCount, TotalReimbursed]
    sorted by YearMonth.
    """
    df = provider_ts.copy().sort_values("YearMonth").reset_index(drop=True)

    df["zscore_reimbursed"] = _rolling_zscore(df["TotalReimbursed"])
    df["stl_zscore"] = _stl_residual_zscore(df["TotalReimbursed"])

    if iso_flag is not None and iso_score is not None:
        df["iso_flag"] = iso_flag.values
        df["iso_score"] = iso_score.values
    else:
        cc_z = (df["ClaimCount"] - df["ClaimCount"].mean()) / (df["ClaimCount"].std() + 1e-9)
        df["iso_flag"] = (cc_z.abs() > ANOMALY_ZSCORE_THRESHOLD).astype(int)
        df["iso_score"] = cc_z.abs()

    df["AnomalyScore"] = (
        df["zscore_reimbursed"].abs() * 0.35
        + df["stl_zscore"].abs() * 0.35
        + df["iso_score"] * 0.30
    )
    df["AnomalyFlag"] = (
        (df["zscore_reimbursed"].abs() > ANOMALY_ZSCORE_THRESHOLD)
        | (df["stl_zscore"].abs() > ANOMALY_ZSCORE_THRESHOLD)
        | (df["iso_flag"] == 1)
    ).astype(int)

    return df


def run(ts: pd.DataFrame | None = None) -> pd.DataFrame:
    if ts is None:
        ts = pd.read_parquet(DATA_PROC / "timeseries.parquet")

    ts = ts.sort_values(["Provider", "YearMonth"]).reset_index(drop=True)

    # One global IsolationForest (fast vs per-provider fits)
    X = ts[["ClaimCount", "TotalReimbursed"]].fillna(0).values
    iso = IsolationForest(
        contamination=ANOMALY_CONTAMINATION, random_state=42, n_jobs=-1
    )
    iso_flags = (iso.fit_predict(X) == -1).astype(int)
    iso_scores = -iso.score_samples(X)
    ts = ts.copy()
    ts["_iso_flag"] = iso_flags
    ts["_iso_score"] = iso_scores

    records = []
    for provider_id, grp in ts.groupby("Provider", sort=False):
        result = detect_for_provider(
            grp.drop(columns=["_iso_flag", "_iso_score"]),
            iso_flag=grp["_iso_flag"],
            iso_score=grp["_iso_score"],
        )
        result["Provider"] = provider_id
        records.append(result)

    if not records:
        return pd.DataFrame()

    out = pd.concat(records, ignore_index=True)
    out.to_parquet(ANOMALIES_PATH, index=False)
    n_anom = int(out["AnomalyFlag"].sum())
    print(f"[anomaly] Processed {ts['Provider'].nunique()} providers, "
          f"{n_anom} anomalous months detected.")
    return out


def get_provider_anomalies(provider_id: str) -> pd.DataFrame:
    df = pd.read_parquet(ANOMALIES_PATH)
    return df[df["Provider"] == provider_id].reset_index(drop=True)


if __name__ == "__main__":
    run()
